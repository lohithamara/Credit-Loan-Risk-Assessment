import streamlit as st
import joblib
import boto3
import os

st.title("Credit Loan Risk Assessment")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "final_inference_pipeline.pkl")
REG_MODEL_PATH = os.path.join(MODEL_DIR, "reg_finalInference_pipeline.pkl")

# AWS credentials from secrets
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
region = "eu-north-1"  # change if needed

s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

BUCKET_NAME = "mlmodel.pklfiles"

# Download only if not exists
if not os.path.exists(CLASS_MODEL_PATH):
    s3.download_file(BUCKET_NAME, "final_inference_pipeline.pkl", CLASS_MODEL_PATH)

if not os.path.exists(REG_MODEL_PATH):
    s3.download_file(BUCKET_NAME, "reg_finalInference_pipeline.pkl", REG_MODEL_PATH)

proba_pipeline = joblib.load(CLASS_MODEL_PATH)
reg_pipeline = joblib.load(REG_MODEL_PATH)

st.success("Models loaded successfully!")

# Simple test UI
loan_amount = st.number_input("Loan Amount", value=10000)

if st.button("Predict"):
    # Replace with real preprocessing
    prediction = proba_pipeline.predict([[loan_amount]])
    st.write("Prediction:", prediction)
