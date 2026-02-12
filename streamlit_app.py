import os
import warnings
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
import sklearn
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# Page configuration
st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Loading environment variables from .env file
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(env_path)

# Silence joblib/loky physical core detection warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])

# Importing custom classes and functions used in the pipelines
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

@st.cache_resource
def download_and_load_models():
    """Download models from S3 if not present locally and load them"""
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
            
            if not aws_access_key or not aws_secret_key:
                st.error("AWS credentials not found in environment variables. Please check your .env file.")
                st.stop()
            
            s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            
            with st.spinner("Downloading models from S3..."):
                if not os.path.exists(CLASS_MODEL_PATH):
                    s3.download_file(bucket_name, "models/final_inference_pipeline.pkl", CLASS_MODEL_PATH)
                
                if not os.path.exists(REG_MODEL_PATH):
                    s3.download_file(bucket_name, "models/reg_final_inference_pipeline.pkl", REG_MODEL_PATH)
                    
        except Exception as e:
            st.warning(f"Could not download models from S3: {e}")
            st.info("Attempting to load local models...")

    # Loading the trained pipelines
    try:
        proba_pipeline = joblib.load(CLASS_MODEL_PATH)
        reg_pipeline = joblib.load(REG_MODEL_PATH)
        
        # Patch column transformers
        _patch_column_transformers(proba_pipeline)
        _patch_column_transformers(reg_pipeline)
        
        return proba_pipeline, reg_pipeline
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

def _build_ct_column_names(ct):
    """Build the prefixed column names a ColumnTransformer would produce"""
    from sklearn.preprocessing import OneHotEncoder
    col_names = []
    for name, trans, cols in ct.transformers_:
        if trans == 'drop':
            continue
        last_step = trans
        if hasattr(trans, 'steps'):
            last_step = trans.steps[-1][1]

        if isinstance(last_step, OneHotEncoder) and hasattr(last_step, 'categories_'):
            for cat in last_step.categories_[0]:
                col_names.append(f'{name}__{cols[0]}_{cat}')
        else:
            col_names.append(f'{name}__{cols[0]}')
    return col_names

def _patch_column_transformers(pipeline):
    """Walk the pipeline tree and patch ColumnTransformers"""
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
        for attr in ('estimator', 'base_estimator', 'final_estimator', 'new_feature_generator'):
            if hasattr(estimator, attr):
                nested = getattr(estimator, attr)
                if nested is not None:
                    _walk(nested)

    _walk(pipeline)

@st.cache_resource
def initialize_groq_client():
    """Initialize Groq client for LLM-powered risk explanations"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.warning("GROQ_API_KEY not found. Risk descriptions will use fallback static text.")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.warning(f"Groq client not initialized: {e}")
        return None

def final_prediction_pipeline(X, proba_pipeline, reg_pipeline):
    """Run all models and return expected loss, probability of default and loss amount"""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    proba_pred = proba_pipeline.predict_proba(X)
    
    if hasattr(proba_pred, 'values'):
        proba_pred = proba_pred.values
    probability_of_default = 1 - proba_pred[:, 1].reshape(-1, 1)
    
    reg_pred = reg_pipeline.predict(X)
    
    if hasattr(reg_pred, 'values'):
        reg_pred = reg_pred.values
    if reg_pred.ndim == 1:
        reg_pred = reg_pred.reshape(-1, 1)
    
    loss_amount = np.expm1(reg_pred)
    expected_loss = probability_of_default * loss_amount
    
    return expected_loss, probability_of_default, loss_amount

def generate_llm_risk_explanation(groq_client, probability_of_default, fico_range_low, bc_util, revol_util, 
                                   installment_to_income_ratio, loan_amnt, annual_inc, dti, risk_level):
    """Generate AI-powered risk explanation using Groq LLM client"""
    try:
        if groq_client is None:
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
        if probability_of_default > 0.75:
            return "The model predicts a high probability of default (>75%). This loan carries significant risk."
        elif probability_of_default >= 0.3:
            return "The model predicts a moderate probability of default (30-75%). This loan carries moderate risk."
        else:
            return "The model predicts a low probability of default (<30%). This loan carries relatively low risk."

def analyze_risk_factors(groq_client, fico_range_low, bc_util, revol_util, installment_to_income_ratio, 
                        probability_of_default, loan_amnt, annual_inc, dti):
    """Analyze risk factors and generate warnings and risk level"""
    warnings = []
    
    if fico_range_low < 600:
        warnings.append("Credit score is low (below 600), indicating high risk of default.")
    
    if bc_util > 90:
        warnings.append(f"Bank card utilization is very high ({bc_util:.1f}%). This indicates the borrower is using over 90% of their available credit on bank cards.")
    
    if revol_util > 90:
        warnings.append(f"Revolving credit utilization is very high ({revol_util:.1f}%). This is a strong indicator of financial distress.")
    
    if installment_to_income_ratio > 0.5:
        warnings.append(f"Installment to monthly income ratio is very high ({installment_to_income_ratio*100:.1f}%). More than 50% of monthly income would go towards this loan payment.")
    
    if probability_of_default > 0.75:
        risk_level = "High Risk"
    elif probability_of_default >= 0.3:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    risk_description = generate_llm_risk_explanation(
        groq_client=groq_client,
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

def predict_loan_outcomes(proba_pipeline, reg_pipeline, groq_client, loan_amnt, term, installment, 
                         emp_length, home_ownership, annual_inc, verification_status, issue_d, 
                         purpose, dti, delinq_2yrs, earliest_cr_line, fico_range_low, pub_rec, 
                         revol_util, bc_util):
    """Make predictions for a single loan application"""
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

    expected_loss, probability_of_default, loss_amnt = final_prediction_pipeline(input_data, proba_pipeline, reg_pipeline)

    expected_loss_val = expected_loss.flatten()[0]
    probability_of_default_val = probability_of_default.flatten()[0]
    loss_amnt_val = loss_amnt.flatten()[0]
    loan_amount_val = loan_amnt

    expected_loss_to_loan_amount_ratio = expected_loss_val / loan_amount_val if loan_amount_val != 0 else np.nan
    monthly_income = annual_inc / 12
    installment_to_monthly_income_ratio = installment / monthly_income if monthly_income != 0 else np.nan

    risk_analysis = analyze_risk_factors(
        groq_client=groq_client,
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

# Main App
def main():
    st.title("💰 Loan Risk Assessment System")
    st.markdown("### Predict loan default risk and expected loss using AI")
    
    # Load models and initialize Groq
    with st.spinner("Loading models..."):
        proba_pipeline, reg_pipeline = download_and_load_models()
        groq_client = initialize_groq_client()
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar for input
    st.sidebar.header("📋 Loan Application Details")
    
    # Loan Information
    st.sidebar.subheader("Loan Details")
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=1000, max_value=50000, value=10000, step=500)
    term = st.sidebar.selectbox("Loan Term", [" 36 months", " 60 months"])
    installment = st.sidebar.number_input("Monthly Installment ($)", min_value=0.0, value=300.0, step=10.0)
    purpose = st.sidebar.selectbox("Loan Purpose", 
        ['credit_card', 'debt_consolidation', 'home_improvement', 'house', 
         'major_purchase', 'medical', 'moving', 'other', 'renewable_energy', 
         'small_business', 'vacation', 'wedding'])
    issue_d = st.sidebar.selectbox("Issue Month", 
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Borrower Information
    st.sidebar.subheader("Borrower Details")
    annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=1000, max_value=500000, value=50000, step=1000)
    emp_length = st.sidebar.selectbox("Employment Length", 
        ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', 
         '6 years', '7 years', '8 years', '9 years', '10+ years'])
    home_ownership = st.sidebar.selectbox("Home Ownership", ['MORTGAGE', 'OWN', 'RENT', 'OTHER'])
    verification_status = st.sidebar.selectbox("Income Verification", ['Not Verified', 'Source Verified', 'Verified'])
    
    # Credit Information
    st.sidebar.subheader("Credit Details")
    fico_range_low = st.sidebar.slider("FICO Score (Low Range)", min_value=300, max_value=850, value=680, step=5)
    dti = st.sidebar.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
    revol_util = st.sidebar.number_input("Revolving Credit Utilization (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    bc_util = st.sidebar.number_input("Bank Card Utilization (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    delinq_2yrs = st.sidebar.number_input("Delinquencies (Past 2 Years)", min_value=0, max_value=20, value=0, step=1)
    pub_rec = st.sidebar.number_input("Public Records", min_value=0, max_value=20, value=0, step=1)
    earliest_cr_line = st.sidebar.selectbox("Earliest Credit Line", 
        [f"{month}-{year}" for year in range(1970, 2026) for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']][-120:])
    
    # Predict button
    if st.sidebar.button("🔮 Predict Risk", type="primary"):
        with st.spinner("Analyzing loan application..."):
            try:
                result = predict_loan_outcomes(
                    proba_pipeline=proba_pipeline,
                    reg_pipeline=reg_pipeline,
                    groq_client=groq_client,
                    loan_amnt=loan_amnt,
                    term=term,
                    installment=installment,
                    emp_length=emp_length,
                    home_ownership=home_ownership,
                    annual_inc=annual_inc,
                    verification_status=verification_status,
                    issue_d=issue_d,
                    purpose=purpose,
                    dti=dti,
                    delinq_2yrs=delinq_2yrs,
                    earliest_cr_line=earliest_cr_line,
                    fico_range_low=fico_range_low,
                    pub_rec=pub_rec,
                    revol_util=revol_util,
                    bc_util=bc_util
                )
                
                # Display results
                st.header("📊 Risk Assessment Results")
                
                # Risk Level Badge
                risk_color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}
                st.markdown(f"## {risk_color.get(result['risk_level'], '⚪')} {result['risk_level']}")
                
                # Key Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Probability of Default",
                        value=f"{result['probability_of_default']*100:.2f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Expected Loss",
                        value=f"${result['expected_loss']:,.2f}",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        label="Loss Amount",
                        value=f"${result['loss_amount']:,.2f}",
                        delta=None
                    )
                
                # Additional Metrics
                st.subheader("📈 Additional Metrics")
                col4, col5 = st.columns(2)
                
                with col4:
                    st.metric(
                        label="Expected Loss to Loan Ratio",
                        value=f"{result['el_to_loan_ratio']*100:.2f}%"
                    )
                
                with col5:
                    st.metric(
                        label="Installment to Income Ratio",
                        value=f"{result['inst_to_income_ratio']*100:.2f}%"
                    )
                
                # Risk Description
                st.subheader("🤖 AI Risk Analysis")
                st.info(result['risk_description'])
                
                # Warnings
                if result['warnings']:
                    st.subheader("⚠️ Risk Warnings")
                    for warning in result['warnings']:
                        st.warning(warning)
                else:
                    st.success("✅ No major risk warnings detected")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)
    
    # Instructions
    else:
        st.info("👈 Fill in the loan application details in the sidebar and click 'Predict Risk' to get started")
        
        # Display some information about the app
        st.markdown("""
        ### How it works:
        1. **Enter loan details** in the sidebar (loan amount, term, purpose, etc.)
        2. **Provide borrower information** (income, employment, home ownership)
        3. **Add credit details** (FICO score, debt-to-income ratio, credit utilization)
        4. **Click 'Predict Risk'** to get comprehensive risk assessment
        
        ### What you'll get:
        - **Probability of Default**: Likelihood that the borrower won't repay
        - **Expected Loss**: Estimated financial loss if default occurs
        - **Risk Level**: Overall risk classification (Low/Medium/High)
        - **AI Risk Analysis**: Detailed explanation powered by LLM
        - **Risk Warnings**: Specific factors that increase risk
        """)

if __name__ == "__main__":
    main()
