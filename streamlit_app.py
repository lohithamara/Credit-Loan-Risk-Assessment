import streamlit as st
import os
import joblib
import boto3
import __main__

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Credit Loan Risk Assessment", layout="wide")
st.title("💳 Credit Loan Risk Assessment System")

# ==============================
# LOAD CUSTOM TRANSFORMERS
# ==============================
from custom_func_and_class import *

# Fix for models saved with __main__ references
__main__.PredictProbaTransformer = PredictProbaTransformer
__main__.RegPredictProbaTransformer = RegPredictProbaTransformer
__main__.IntRatePredictProbaTransformer = IntRatePredictProbaTransformer
__main__.NewFeatureGenerator = NewFeatureGenerator
__main__.NewFeatureAddingTransformer = NewFeatureAddingTransformer
__main__.OutlierCapper = OutlierCapper
__main__.TermTransformer = TermTransformer
__main__.IssueDTransformer = IssueDTransformer
__main__.EmpLengthTransformer = EmpLengthTransformer
__main__.EarliestCrLineTransformer = EarliestCrLineTransformer
__main__.ColumnDroppingTransformer = ColumnDroppingTransformer
__main__.bin_pub_rec = bin_pub_rec
__main__.bin_emp_length = bin_emp_length
__main__.bin_delinq_2yrs = bin_delinq_2yrs
__main__.bin_fico_range_low = bin_fico_range_low
__main__.binarize_revol_util = binarize_revol_util
__main__.binarize_bc_util = binarize_bc_util
__main__.apply_log1p_df = apply_log1p_df
__main__.transform_home_ownership = transform_home_ownership
__main__.transform_purpose = transform_purpose

# ==============================
# S3 CONFIG (FROM SECRETS)
# ==============================
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
region = st.secrets.get("AWS_REGION", "eu-north-1")
bucket_name = st.secrets.get("BUCKET_NAME", "mlmodel.pklfiles")

# ==============================
# MODEL PATHS
# ==============================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "final_inference_pipeline.pkl")
REG_MODEL_PATH = os.path.join(MODEL_DIR, "reg_finalInference_pipeline.pkl")

# ==============================
# DOWNLOAD MODELS FROM S3
# ==============================
@st.cache_resource
def load_models():
    st.info("Connecting to S3...")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )

    if not os.path.exists(CLASS_MODEL_PATH):
        st.info("Downloading classification model...")
        s3.download_file(bucket_name, "models/final_inference_pipeline.pkl", CLASS_MODEL_PATH)

    if not os.path.exists(REG_MODEL_PATH):
        st.info("Downloading regression model...")
        s3.download_file(bucket_name, "models/reg_finalInference_pipeline.pkl", REG_MODEL_PATH)

    st.info("Loading models into memory...")
    clf_model = joblib.load(CLASS_MODEL_PATH)
    reg_model = joblib.load(REG_MODEL_PATH)

    return clf_model, reg_model


# Load once
clf_model, reg_model = load_models()
st.success("✅ Models Loaded Successfully")

# ==============================
# USER INPUT SECTION
# ==============================
st.header("Enter Loan Details")

loan_amount = st.number_input("Loan Amount", min_value=1000, value=10000)
annual_income = st.number_input("Annual Income", min_value=1000, value=50000)
fico_score = st.number_input("FICO Score", min_value=300, max_value=850, value=700)

# Replace below with full feature input structure you actually use
input_data = [[loan_amount, annual_income, fico_score]]

# ==============================
# PREDICTION
# ==============================
if st.button("Predict Risk"):
    try:
        default_prob = clf_model.predict_proba(input_data)[0][1]
        expected_loss = reg_model.predict(input_data)[0]

        st.subheader("📊 Prediction Results")
        st.write(f"Default Probability: **{default_prob:.2%}**")
        st.write(f"Expected Loss: **{expected_loss:.2f}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ==============================
# OPTIONAL: GROQ AI SECTION
# ==============================
if "GROQ_API_KEY" in st.secrets:
    from groq import Groq
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    st.header("🤖 AI Loan Summary")

    if st.button("Generate AI Explanation"):
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Explain loan risk assessment in simple terms."}
            ],
            model="mixtral-8x7b-32768"
        )
        st.write(response.choices[0].message.content)
