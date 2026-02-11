import os
import warnings
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, render_template, request, jsonify
import sklearn
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Loading environment variables from .env file
env_path = os.path.join(BASE_DIR, '.env')
print(f"Loading .env from: {env_path}")
print(f".env file exists: {os.path.exists(env_path)}")
load_dotenv(env_path)

# Debug: Check if credentials are loaded
print(f"AWS_ACCESS_KEY_ID loaded: {os.getenv('AWS_ACCESS_KEY_ID') is not None}")
print(f"AWS_SECRET_ACCESS_KEY loaded: {os.getenv('AWS_SECRET_ACCESS_KEY') is not None}")
print(f"GROQ_API_KEY loaded: {os.getenv('GROQ_API_KEY') is not None}")

# Silence joblib/loky physical core detection warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='Could not find the number of physical cores.*'
)

# Suppress sklearn warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='This Pipeline instance is not fitted yet.*'
)

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='X does not have valid feature names, but .* was fitted with feature names'
)

warnings.filterwarnings(
    'ignore',
    category=UserWarning
)

warnings.filterwarnings(
    'ignore',
    category=FutureWarning
)

sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])

# Imporing custom classes and functions used in the pipelines
import custom_func_and_class
from custom_func_and_class import (
    PredictProbaTransformer, RegPredictProbaTransformer, IntRatePredictProbaTransformer,
    NewFeatureGenerator, NewFeatureAddingTransformer, OutlierCapper,
    TermTransformer, IssueDTransformer, EmpLengthTransformer, EarliestCrLineTransformer,
    ColumnDroppingTransformer, bin_pub_rec, bin_emp_length, bin_delinq_2yrs,
    bin_fico_range_low, binarize_revol_util, binarize_bc_util, apply_log1p_df,
    transform_home_ownership, transform_purpose
)

# Additional compatibility patches for custom classes
for cls in [PredictProbaTransformer, RegPredictProbaTransformer, IntRatePredictProbaTransformer,
            NewFeatureGenerator, NewFeatureAddingTransformer, OutlierCapper,
            TermTransformer, IssueDTransformer, EmpLengthTransformer, 
            EarliestCrLineTransformer, ColumnDroppingTransformer]:
    if sklearn_version >= (1, 6) and not hasattr(cls, '__sklearn_tags__'):
        try:
            from sklearn.utils._tags import _DEFAULT_TAGS
            cls.__sklearn_tags__ = lambda self: _DEFAULT_TAGS
        except:
            pass

# Download models from S3 if not present locally
print("Checking and downloading models from S3...")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "final_inference_pipeline.pkl")
REG_MODEL_PATH = os.path.join(MODEL_DIR, "reg_final_inference_pipeline.pkl")

# Only download if models don't exist locally
if not os.path.exists(CLASS_MODEL_PATH) or not os.path.exists(REG_MODEL_PATH):
    try:
        import boto3
        
        # Get AWS credentials from environment
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "eu-north-1")
        bucket_name = os.getenv("S3_BUCKET_NAME", "mlmodel.pklfiles")
        
        # Debug: Check if credentials are loaded
        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS credentials not found in environment variables. Please check your .env file.")
        
        print(f"Using S3 bucket: {bucket_name} in region: {aws_region}")
        
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        if not os.path.exists(CLASS_MODEL_PATH):
            print(f"Downloading classification model from S3...")
            s3.download_file(bucket_name, "models/final_inference_pipeline.pkl", CLASS_MODEL_PATH)
            print("✓ Classification model downloaded")
        
        if not os.path.exists(REG_MODEL_PATH):
            print(f"Downloading regression model from S3...")
            s3.download_file(bucket_name, "models/reg_final_inference_pipeline.pkl", REG_MODEL_PATH)
            print("✓ Regression model downloaded")
            
    except Exception as e:
        print(f"⚠ Warning: Could not download models from S3: {e}")
        print("  Will attempt to load local models if available...")
else:
    print("✓ Models already exist locally")

# Loading the trained pipelines
print("Loading models...")
try:
    proba_pipeline = joblib.load(CLASS_MODEL_PATH)
    print("✓ Loaded proba_pipeline")
except Exception as e:
    print(f"Error loading proba_pipeline: {e}")
    raise

try:
    reg_pipeline = joblib.load(REG_MODEL_PATH)
    print("✓ Loaded reg_pipeline")
except Exception as e:
    print(f"Error loading reg_pipeline: {e}")
    raise


def _build_ct_column_names(ct):
    """
    Build the prefixed column names a ColumnTransformer would produce
    if set_output(transform='pandas') actually worked.  We compute them
    from the transformer names, input column names and output shapes.
    This is needed to patch the transform() method of ColumnTransformer
    to return DataFrames with the correct column names that downstream models expect.
    """
    from sklearn.preprocessing import OneHotEncoder
    col_names = []
    for name, trans, cols in ct.transformers_:
        if trans == 'drop':
            continue
        # Figure out how many output columns this transformer produces
        # by looking at the last step (OneHotEncoder produces multiple columns)
        last_step = trans
        if hasattr(trans, 'steps'):
            last_step = trans.steps[-1][1]

        if isinstance(last_step, OneHotEncoder) and hasattr(last_step, 'categories_'):
            # OneHotEncoder – one column per category
            for cat in last_step.categories_[0]:
                col_names.append(f'{name}__{cols[0]}_{cat}')
        else:
            # Single-column output – standard prefix (OrdinalEncoder, etc.)
            col_names.append(f'{name}__{cols[0]}')
    return col_names


def _patch_column_transformers(pipeline):
    """
    Walk the pipeline tree, find every ColumnTransformer named
    'initial_feature_processing' and wrap its transform() so that it
    returns a pandas DataFrame with the correct prefixed column names
    that downstream models were trained to expect.
    """


    def _wrap_ct(ct):
        if getattr(ct, '_transform_patched', False):
            return
        col_names = _build_ct_column_names(ct)
        original_transform = ct.transform

        def _patched_transform(X, **kwargs):
            out = original_transform(X, **kwargs)
            if isinstance(out, np.ndarray):
                out = pd.DataFrame(out, columns=col_names)
            elif isinstance(out, pd.DataFrame):
                out.columns = col_names
            return out

        ct.transform = _patched_transform
        ct._transform_patched = True

    def _walk(estimator):
        if isinstance(estimator, ColumnTransformer):
            _wrap_ct(estimator)
        if hasattr(estimator, 'steps'):
            for _, step in estimator.steps:
                if step is not None and step != 'passthrough':
                    _walk(step)
        if hasattr(estimator, 'transformer_list'):
            for _, trans in estimator.transformer_list:
                if trans is not None:
                    _walk(trans)
        for attr in ('transformers_', 'transformers'):
            if hasattr(estimator, attr):
                for item in getattr(estimator, attr):
                    if len(item) >= 2 and item[1] not in ('drop', 'passthrough', None):
                        _walk(item[1])
        for attr in ('estimator', 'base_estimator', 'final_estimator',
                     'new_feature_generator'):
            if hasattr(estimator, attr):
                nested = getattr(estimator, attr)
                if nested is not None:
                    _walk(nested)

    _walk(pipeline)

_patch_column_transformers(proba_pipeline)
_patch_column_transformers(reg_pipeline)

print("All models loaded successfully!\n")

# Initialize Groq client for LLM-powered risk explanations
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    groq_client = Groq(api_key=api_key)
    print("✓ Groq client initialized")
except Exception as e:
    print(f"⚠ Warning: Groq client not initialized: {e}")
    print("  Risk descriptions will use fallback static text.")
    groq_client = None

# Flask app
app = Flask(__name__)

def final_prediction_pipeline(X):
    """Run all models and return expected loss, probability of default and loss amount"""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    proba_pred = proba_pipeline.predict_proba(X)
    
    # Handle array output properly
    if hasattr(proba_pred, 'values'):
        proba_pred = proba_pred.values
    probability_of_default = 1 - proba_pred[:, 1].reshape(-1, 1)
    
    reg_pred = reg_pipeline.predict(X)
    
    # Handle array output properly
    if hasattr(reg_pred, 'values'):
        reg_pred = reg_pred.values
    if reg_pred.ndim == 1:
        reg_pred = reg_pred.reshape(-1, 1)
    
    loss_amount = np.expm1(reg_pred)
    
    expected_loss = probability_of_default * loss_amount
    
    return expected_loss, probability_of_default, loss_amount

def generate_llm_risk_explanation(probability_of_default, fico_range_low, bc_util, revol_util, installment_to_income_ratio, loan_amnt, annual_inc, dti, risk_level):
    """Generate AI-powered risk explanation using Groq LLM client, with fallback to static description if LLM fails or is unavailable"""
    try:
        if groq_client is None:
            # Fallback to static description
            if probability_of_default > 0.75:
                return "The model predicts a high probability of default (>75%). This loan carries significant risk."
            elif probability_of_default >= 0.3:
                return "The model predicts a moderate probability of default (30-75%). This loan carries moderate risk."
            else:
                return "The model predicts a low probability of default (<30%). This loan carries relatively low risk."
        
        prompt = f"""Analyze this loan application and explain the risk in simple, point-wise format:

- Default Risk: {probability_of_default*100:.2f}% chance of not repaying
- Risk Category: {risk_level}
- Credit Score (FICO): {fico_range_low}
- Debt-to-Income Ratio: {dti:.1f}%
- Bank Card Usage: {bc_util:.1f}% of available credit
- Revolving Credit Usage: {revol_util:.1f}% of available credit
- Monthly Payment vs Income: {installment_to_income_ratio*100:.1f}% of monthly income
- Loan Amount: ${loan_amnt:,.0f}
- Annual Income: ${annual_inc:,.0f}

Provide 3-4 bullet points explaining:
1. Overall risk assessment in simple terms
2. Key positive factors (if any)
3. Key risk factors or concerns (if any)
4. Brief recommendation

Use bullet points (•) and keep each point short and easy to understand."""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful financial advisor who explains loan risks in simple, clear bullet points that anyone can understand. Use • for bullets. Be conversational and avoid jargon."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=250
        )
        
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating LLM explanation: {e}")
        # Fallback to static description
        if probability_of_default > 0.75:
            return "The model predicts a high probability of default (>75%). This loan carries significant risk."
        elif probability_of_default >= 0.3:
            return "The model predicts a moderate probability of default (30-75%). This loan carries moderate risk."
        else:
            return "The model predicts a low probability of default (<30%). This loan carries relatively low risk."

def analyze_risk_factors(fico_range_low, bc_util, revol_util, installment_to_income_ratio, probability_of_default, loan_amnt, annual_inc, dti):
    """Analyze risk factors and generate warnings and risk level"""
    warnings = []
    
    # Check FICO score
    if fico_range_low < 600:
        warnings.append("Credit score is low (below 600), indicating high risk of default.")
    
    # Check bc_util
    if bc_util > 90:
        warnings.append(f"Bank card utilization is very high ({bc_util:.1f}%). This indicates the borrower is using over 90% of their available credit on bank cards, which suggests high credit dependency and potential financial stress.")
    
    # Check revol_util
    if revol_util > 90:
        warnings.append(f"Revolving credit utilization is very high ({revol_util:.1f}%). This indicates the borrower is using over 90% of their available revolving credit, which is a strong indicator of financial distress and increases default risk significantly.")
    
    # Check installment to income ratio
    if installment_to_income_ratio > 0.5:
        warnings.append(f"Installment to monthly income ratio is very high ({installment_to_income_ratio*100:.1f}%). This means more than 50% of monthly income would go towards this loan payment, leaving limited funds for other expenses and increasing default risk.")
    
    # Determine risk level based on probability of default
    if probability_of_default > 0.75:
        risk_level = "High Risk"
    elif probability_of_default >= 0.3:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    # Generate LLM-powered risk description
    risk_description = generate_llm_risk_explanation(
        probability_of_default=probability_of_default,
        fico_range_low=fico_range_low,
        bc_util=bc_util,
        revol_util=revol_util,
        installment_to_income_ratio=installment_to_income_ratio,
        loan_amnt=loan_amnt,
        annual_inc=annual_inc,
        dti=dti,
        risk_level=risk_level
    )
    
    return {
        'warnings': warnings,
        'risk_level': risk_level,
        'risk_description': risk_description
    }

def predict_loan_outcomes(
    loan_amnt,
    term,
    installment,
    emp_length,
    home_ownership,
    annual_inc,
    verification_status,
    issue_d,
    purpose,
    dti,
    delinq_2yrs,
    earliest_cr_line,
    fico_range_low,
    pub_rec,
    revol_util,
    bc_util
):
    """Make predictions for a single loan application"""
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'loan_amnt': [loan_amnt],
        'term': [term],
        'installment': [installment],
        'emp_length': [emp_length],
        'home_ownership': [home_ownership],
        'annual_inc': [annual_inc],
        'verification_status': [verification_status],
        'issue_d': [issue_d],
        'purpose': [purpose],
        'dti': [dti],
        'delinq_2yrs': [delinq_2yrs],
        'earliest_cr_line': [earliest_cr_line],
        'fico_range_low': [fico_range_low],
        'pub_rec': [pub_rec],
        'revol_util': [revol_util],
        'bc_util': [bc_util]
    })

    # Pass the input data to the final prediction pipeline
    expected_loss, probability_of_default, loss_amnt = final_prediction_pipeline(input_data)

    # Extract scalar values from the predictions
    expected_loss_val = expected_loss.flatten()[0]
    probability_of_default_val = probability_of_default.flatten()[0]
    loss_amnt_val = loss_amnt.flatten()[0]
    loan_amount_val = loan_amnt

    # Calculate additional ratios
    expected_loss_to_loan_amount_ratio = expected_loss_val / loan_amount_val if loan_amount_val != 0 else np.nan
    monthly_income = annual_inc / 12
    installment_to_monthly_income_ratio = installment / monthly_income if monthly_income != 0 else np.nan

    # Analyze risk factors
    risk_analysis = analyze_risk_factors(
        fico_range_low=fico_range_low,
        bc_util=bc_util,
        revol_util=revol_util,
        installment_to_income_ratio=installment_to_monthly_income_ratio,
        probability_of_default=probability_of_default_val,
        loan_amnt=loan_amnt,
        annual_inc=annual_inc,
        dti=dti
    )

    return {
        'probability_of_default': float(probability_of_default_val),
        'loss_amount': float(loss_amnt_val),
        'expected_loss': float(expected_loss_val),
        'el_to_loan_ratio': float(expected_loss_to_loan_amount_ratio),
        'inst_to_income_ratio': float(installment_to_monthly_income_ratio),
        'risk_level': risk_analysis['risk_level'],
        'risk_description': risk_analysis['risk_description'],
        'warnings': risk_analysis['warnings']
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.json
        
        # Make prediction
        result = predict_loan_outcomes(
            loan_amnt=float(data['loan_amnt']),
            term=data['term'],
            installment=float(data['installment']),
            emp_length=data['emp_length'],
            home_ownership=data['home_ownership'],
            annual_inc=float(data['annual_inc']),
            verification_status=data['verification_status'],
            issue_d=data['issue_d'],
            purpose=data['purpose'],
            dti=float(data['dti']),
            delinq_2yrs=int(data['delinq_2yrs']),
            earliest_cr_line=data['earliest_cr_line'],
            fico_range_low=int(data['fico_range_low']),
            pub_rec=int(data['pub_rec']),
            revol_util=float(data['revol_util']),
            bc_util=float(data['bc_util'])
        )
        
        return jsonify({
            'success': True,
            'results': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
