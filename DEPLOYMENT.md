# 🚀 Deployment Guide

This guide will help you deploy your Loan Risk Assessment System using AWS S3 for model storage and Render for hosting.

---

## 📋 Prerequisites

1. **AWS Account** - For S3 bucket to store model files
2. **GitHub Account** - To push your code
3. **Render Account** - For deployment (free tier available)
4. **Groq API Key** - Already configured

---

## Step 1️⃣: Upload Models to S3

### 1.1 Create S3 Bucket (if not already created)
```bash
# Your bucket: mlmodel.pklfiles
# Region: eu-north-1 (or your preferred region)
```

### 1.2 Upload Model Files to S3

Upload these files to your S3 bucket:
- `models/final_inference_pipeline.pkl`
- `models/reg_final_inference_pipeline.pkl`

**S3 Structure:**
```
mlmodel.pklfiles/
├── models/
│   ├── final_inference_pipeline.pkl
│   └── reg_final_inference_pipeline.pkl
```

### 1.3 Keep S3 Bucket Private ✅

**Important:** Keep your bucket **private** (not public). The app will use AWS credentials to download models securely.

### 1.4 Get AWS Credentials

1. Go to AWS Console → IAM → Users
2. Create a new user or use existing
3. Attach policy: `AmazonS3ReadOnlyAccess` (or custom policy for your bucket)
4. Generate Access Keys:
   - **AWS_ACCESS_KEY_ID**
   - **AWS_SECRET_ACCESS_KEY**
5. Save these credentials securely

---

## Step 2️⃣: Prepare for GitHub

### 2.1 Verify .gitignore
Make sure `.gitignore` includes:
```
.env
.venv/
__pycache__/
*.pyc
models/*.pkl
```

This ensures:
- ✅ API keys are NOT pushed to GitHub
- ✅ Model files are NOT pushed (will download from S3)
- ✅ Virtual environment is excluded

### 2.2 Update .env.template
Verify `.env.template` has all required variables:
```bash
GROQ_API_KEY=your_groq_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
AWS_REGION=eu-north-1
S3_BUCKET_NAME=mlmodel.pklfiles
```

### 2.3 Initialize Git Repository
```bash
# If not already initialized
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Loan Risk Assessment System"
```

### 2.4 Push to GitHub
```bash
# Create a new repository on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

---

## Step 3️⃣: Deploy on Render

### 3.1 Create New Web Service

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select your loan risk assessment repository

### 3.2 Configure Build Settings

**Basic Settings:**
- **Name:** `loan-risk-assessment` (or your choice)
- **Region:** Choose closest to your users
- **Branch:** `main`
- **Root Directory:** Leave blank
- **Runtime:** `Python 3`

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
gunicorn -w 4 -b 0.0.0.0:$PORT final_app:app
```

### 3.3 Configure Environment Variables

Add these in Render Dashboard → Environment Variables:

| Key | Value | Notes |
|-----|-------|-------|
| `GROQ_API_KEY` | `gsk_...` | Your Groq API key |
| `AWS_ACCESS_KEY_ID` | Your AWS access key | From Step 1.4 |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | From Step 1.4 |
| `AWS_REGION` | `eu-north-1` | Your S3 bucket region |
| `S3_BUCKET_NAME` | `mlmodel.pklfiles` | Your bucket name |
| `PYTHON_VERSION` | `3.11.9` | Match your local version |

### 3.4 Advanced Settings

- **Instance Type:** Free tier is sufficient for testing
- **Auto-Deploy:** Enable (automatically deploys on git push)

### 3.5 Deploy!

Click **"Create Web Service"**

Render will:
1. ✅ Clone your GitHub repository
2. ✅ Install dependencies from `requirements.txt`
3. ✅ Download models from S3 (on first startup)
4. ✅ Start your Flask app with Gunicorn
5. ✅ Provide you with a public URL

---

## Step 4️⃣: Verify Deployment

### 4.1 Check Deployment Logs

In Render dashboard, check logs for:
```
Checking and downloading models from S3...
Downloading classification model from S3...
✓ Classification model downloaded
Downloading regression model from S3...
✓ Regression model downloaded
Loading models...
✓ Loaded proba_pipeline
✓ Loaded reg_pipeline
✓ Groq client initialized
```

### 4.2 Test Your Application

1. Open the Render-provided URL (e.g., `https://loan-risk-assessment.onrender.com`)
2. Fill in a loan application
3. Click "Analyze Loan Risk"
4. Verify AI-generated risk assessment appears

---

## 🔧 Troubleshooting

### Models Not Downloading

**Error:** `Could not download models from S3`

**Solutions:**
1. Verify AWS credentials are correct in Render environment variables
2. Check S3 bucket name and region
3. Verify IAM permissions allow S3 read access
4. Check S3 file paths match: `models/final_inference_pipeline.pkl`

### Groq API Error

**Error:** `Groq client not initialized`

**Solution:** Verify `GROQ_API_KEY` is set correctly in Render environment variables

### Application Crashes

**Solution:** Check Render logs for specific error messages

---

## 🔄 Updating Your Application

### After Making Code Changes:

```bash
# Commit changes
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin main
```

Render will automatically:
- Detect the push
- Rebuild and redeploy
- Reuse existing models (won't re-download if already present)

### Updating Models:

1. Upload new `.pkl` files to S3 (overwrites old ones)
2. In Render Dashboard:
   - Go to your service → **Manual Deploy** → **Clear build cache & deploy**
   - This forces fresh model download

---

## 💰 Cost Considerations

### Free Tier Services:
- ✅ **Render:** Free tier available (750 hours/month)
- ✅ **Groq:** Free tier with generous limits
- ⚠️ **AWS S3:** Pay as you go (minimal cost for 2 model files)

### S3 Costs (Approximate):
- **Storage:** ~$0.023/GB/month (Stockholm region)
- **Data Transfer:** Free inbound, $0.09/GB outbound
- **Your models:** ~200MB = **<$0.01/month storage**
- **Downloads:** Only on deployment = **minimal cost**

**Estimated Monthly Cost:** < $1 USD 🎉

---

## 🎉 Success!

Your Loan Risk Assessment System is now:
- ✅ Deployed to the cloud
- ✅ Automatically pulls models from S3
- ✅ Uses AI explanations via Groq
- ✅ Accessible via public URL
- ✅ Auto-deploys on git push

**Share your URL and start analyzing loans!** 🏦

---

## 📞 Need Help?

Common issues and solutions are in the Troubleshooting section above. For specific errors, check:
1. Render deployment logs
2. AWS CloudWatch (if enabled)
3. Browser console (F12) for frontend issues
