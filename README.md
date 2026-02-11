# 🏦 Loan Risk Assessment System

An advanced machine learning-based web application for predicting loan default risk and expected losses. The system uses ensemble models (XGBoost, LightGBM) combined with AI-powered explanations via Groq's Llama 3.3 to provide comprehensive, easy-to-understand loan risk assessments.

## 📋 Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Technologies Used](#technologies-used)
- [Model Information](#model-information)
- [Development Notebooks](#development-notebooks)
- [Screenshots](#screenshots)

---

## ✨ Features

### Core Functionality
- **Default Probability Prediction**: ML models predict the likelihood of loan default
- **Loss Amount Estimation**: Predicts potential loss if default occurs
- **Expected Loss Calculation**: Computes probability-weighted expected loss
- **Risk Categorization**: Automatically classifies loans as Low, Medium, or High Risk

### AI-Powered Explanations
- **LLM Integration**: Uses Groq's Llama 3.3-70B model for natural language risk explanations
- **Point-wise Analysis**: Generates 3-4 bullet points explaining:
  - Overall risk assessment in simple terms
  - Key positive factors
  - Risk factors and concerns
  - Brief recommendations
- **Contextual Intelligence**: AI analyzes all risk factors including FICO score, DTI ratio, utilization rates, and income ratios

### User Interface
- **Modern Web Interface**: Beautiful gradient-styled responsive design
- **Calendar Date Pickers**: Month/year selection for loan issue date and earliest credit line
- **Flexible Loan Terms**: Support for 3, 6, 9, 12, 18, 24, 36, and 60-month loans
- **Interactive Forms**: Real-time validation with helpful tooltips
- **Visual Risk Indicators**: Color-coded risk badges (Green/Orange/Red)
- **Comprehensive Results**: 5-card dashboard showing key metrics
- **Risk Factor Warnings**: Automatic detection and display of high-risk conditions

### Advanced Features
- **Custom Transformers**: Specialized sklearn transformers for feature engineering
- **Ensemble Predictions**: Combines multiple ML models for robust predictions
- **Automatic Fallbacks**: Gracefully handles API failures with static descriptions
- **Cross-platform**: Works on Windows, Linux, and macOS

---

## 🏗️ System Architecture

```
User Input → Flask Backend → Preprocessing Pipeline → ML Models → Risk Calculation
                   ↓                                                    ↓
              Groq LLM API ← Risk Data ← Risk Analysis ← Model Outputs
                   ↓
           AI Explanation → Frontend Display
```

### Pipeline Components

1. **Data Preprocessing**
   - Term transformation (months to numeric)
   - Date parsing (issue date, earliest credit line)
   - Employment length normalization
   - Feature engineering (new ratios, binning)

2. **Model Prediction**
   - Classification pipeline: Probability of default
   - Regression pipeline: Loss amount estimation

3. **Risk Analysis**
   - Calculate expected loss (probability × loss amount)
   - Evaluate risk factors (FICO, DTI, utilization rates)
   - Generate warnings for high-risk conditions

4. **LLM Enhancement**
   - Send risk data to Groq's Llama 3.3-70B
   - Generate human-readable explanations
   - Format as bullet points for clarity

---

## 📁 Project Structure

```
project/
├── final_app.py              # Main Flask application with ML inference
├── custom_func_and_class.py  # Custom sklearn transformers and functions
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create from example)
│
├── models/                   # Trained ML models
│   ├── final_inference_pipeline.pkl        # Classification pipeline
│   └── reg_final_inference_pipeline.pkl    # Regression pipeline
│
├── scripts/                  # Frontend files
│   ├── index.html           # Main web interface
│   └── frontend.html        # Additional frontend assets
│
├── datasets/                 # Training/validation data
│   ├── df_classification.csv
│   └── df_regression.csv
│
└── notebooks/                # Jupyter notebooks for development
    ├── Data_Gathering.ipynb
    ├── Data_Exploration.ipynb
    ├── Feature_Engineering_Classification.ipynb
    ├── Feature_Engineering_Regression.ipynb
    ├── Model_Buliding_Classification.ipynb
    └── Model_Building_Regression.ipynb
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone or Download the Project
```bash
cd C:\Users\lohit\OneDrive\Desktop\project
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- Flask 3.0.2 - Web framework
- scikit-learn 1.6.1 - ML framework
- pandas 2.2.2 - Data manipulation
- numpy 2.0.2 - Numerical computing
- xgboost 3.1.3 - Gradient boosting
- lightgbm 4.6.0 - Gradient boosting
- groq 0.11.0 - LLM API client
- python-dotenv 1.0.0 - Environment management
- joblib 1.5.3 - Model serialization
- scipy 1.16.3 - Scientific computing

---

## ⚙️ Configuration

### 1. Create Environment File
Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Get Groq API Key
1. Visit [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up for a free account (no credit card required)
3. Create an API key
4. Copy the key into your `.env` file

### 3. Verify Models
Ensure these files exist in the `models/` directory:
- `final_inference_pipeline.pkl`
- `reg_final_inference_pipeline.pkl`

---

## 🎮 Running the Application

### Start the Flask Server

```bash
python final_app.py
```

You should see:
```
Loading models...
✓ Loaded proba_pipeline
✓ Loaded reg_pipeline
All models loaded successfully!

✓ Groq client initialized
 * Running on http://127.0.0.1:5000
```

### Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Using the Application

1. **Fill in Loan Application Details:**
   - Loan Amount: $25,000
   - Loan Term: Select from dropdown (3, 6, 9, 12, 18, 24, 36, 60 months)
   - Monthly Installment: $450
   - Annual Income: $100,000
   - Employment Length: 10+ years
   - Home Ownership: Mortgage/Rent/Own
   - Verification Status: Verified/Not Verified
   - Issue Date: Use calendar picker
   - Loan Purpose: Select from dropdown
   - DTI Ratio: 25.0%
   - Delinquencies: 1
   - Earliest Credit Line: Use calendar picker
   - FICO Score: 720
   - Public Records: 0
   - Revolving Utilization: 60%
   - Bankcard Utilization: 70%

2. **Click "Analyze Loan Risk"**

3. **View Results:**
   - Default Probability with percentage
   - Expected Loss amount
   - Loss Amount if default occurs
   - Expected Loss Ratio (% of loan)
   - Installment to Income ratio
   - AI-Generated Risk Analysis with bullet points
   - Color-coded risk badge (Low/Medium/High)
   - Specific risk factor warnings (if any)

---

## 📡 API Documentation

### Endpoints

#### `GET /`
Returns the main web interface (index.html)

**Response:** HTML page

---

#### `POST /predict`
Predicts loan risk based on application details

**Request Body (JSON):**
```json
{
  "loan_amnt": 25000.0,
  "term": " 60 months",
  "installment": 450.0,
  "emp_length": "10+ years",
  "home_ownership": "MORTGAGE",
  "annual_inc": 100000.0,
  "verification_status": "Not Verified",
  "issue_d": "Jan-2016",
  "purpose": "credit_card",
  "dti": 25.0,
  "delinq_2yrs": 1,
  "earliest_cr_line": "Mar-2000",
  "fico_range_low": 720,
  "pub_rec": 0,
  "revol_util": 60.0,
  "bc_util": 70.0
}
```

**Response (JSON):**
```json
{
  "success": true,
  "results": {
    "probability_of_default": 0.1534,
    "loss_amount": 8234.56,
    "expected_loss": 1263.42,
    "el_to_loan_ratio": 0.0505,
    "inst_to_income_ratio": 0.0540,
    "risk_level": "Low Risk",
    "risk_description": "• This loan appears relatively safe with a 15.34% default probability...\n• The borrower has a strong credit score of 720...",
    "warnings": []
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message details"
}
```

---

## 🔧 Technologies Used

### Backend
- **Flask** - Python web framework
- **scikit-learn** - Machine learning library with custom transformers
- **XGBoost** - Gradient boosting for classification
- **LightGBM** - Gradient boosting for regression
- **pandas & numpy** - Data manipulation and numerical computing

### Machine Learning
- **Classification Pipeline**: Predicts loan status (default/non-default) using probability
- **Regression Pipeline**: Predicts loss amount using log-transformed targets
- **Custom Transformers**: 
  - `TermTransformer` - Converts term strings to numeric
  - `IssueDTransformer` - Parses issue dates
  - `EmpLengthTransformer` - Normalizes employment length
  - `EarliestCrLineTransformer` - Parses earliest credit line dates
  - `NewFeatureGenerator` - Creates derived features
  - `OutlierCapper` - Caps extreme values
  - `ColumnDroppingTransformer` - Removes unnecessary features
  - Various binning and transformation functions

### AI Enhancement
- **Groq API** - Fast LLM inference using Llama 3.3-70B
- **Llama 3.3-70B-Versatile** - Large language model for generating explanations
- **python-dotenv** - Environment variable management

### Frontend
- **HTML5/CSS3** - Modern, responsive design
- **JavaScript (Vanilla)** - Form handling and API communication
- **Month Input Widget** - HTML5 date picker for calendar selection
- **Gradient UI** - Purple gradient theme with glass morphism effects

### Development
- **Jupyter Notebooks** - Data exploration and model development
- **joblib** - Model serialization and loading
- **Virtual Environment** - Isolated dependency management

---

## 🤖 Model Information

### Classification Pipeline (Stacking Ensemble)
**Purpose:** Predict probability of loan default using ensemble learning

**Architecture:**
```
Input Data → Base Models Layer → Probability Extraction → Meta Model → Final Prediction
```

**Detailed Steps:**
1. **Initial Feature Processing** (ColumnTransformer)
   - Term transformation: Convert "36 months" → 36 (numeric)
   - Date parsing: Parse issue_d and earliest_cr_line dates
   - Employment length normalization: "10+ years" → 10
   - One-Hot Encoding for categorical features (home_ownership, verification_status, purpose)
   - Standard scaling for numerical features

2. **Feature Engineering**
   - Generate derived features:
     - `cr_history`: Credit history length (current_year - earliest_cr_line)
     - `installment_to_income_ratio`: Monthly payment / annual income
     - `loan_to_inc_ratio`: Loan amount / annual income

3. **Outlier Capping**
   - Apply statistical capping to prevent extreme values from affecting predictions

4. **Base Models** (4 models trained on different undersampled datasets)
   - RandomForest Classifier
   - XGBoost Classifier
   - LightGBM Classifier
   - GradientBoosting Classifier

5. **Feature Union**
   - Extract probability predictions from each base model
   - Combine into single feature vector [rf_prob, xgb_prob, lgb_prob, gb_prob]

6. **Meta Model** (Level 2)
   - Logistic Regression trained on base model probabilities
   - Final ensemble prediction

**Output:** Probability of default (0.0 to 1.0)

---

### Regression Pipeline (Stacking Ensemble)
**Purpose:** Predict loss amount if default occurs using ensemble learning

**Architecture:**
```
Input Data → Base Models Layer → Regression Predictions → Meta Model → Log Transform Reversal
```

**Detailed Steps:**
1. **Initial Feature Processing** (ColumnTransformer)
   - Same transformations as classification pipeline
   - Term, date, employment length, encoding, and scaling

2. **Feature Engineering**
   - Same derived features:
     - Credit history length
     - Installment to income ratio
     - Loan to income ratio

3. **Outlier Capping**
   - Statistical capping for extreme values

4. **Base Models** (4 regressors predicting log-transformed loss)
   - RandomForest Regressor
   - XGBoost Regressor
   - LightGBM Regressor
   - GradientBoosting Regressor
   - Each predicts log(loss_amount)

5. **Feature Union**
   - Extract predictions from each base model
   - Combine into single feature vector [rf_pred, xgb_pred, lgb_pred, gb_pred]

6. **Meta Model** (Level 2)
   - Linear Regression trained on base model predictions
   - Outputs log(loss_amount)

7. **Exponential Transformation**
   - Apply `np.expm1()` to reverse log transformation
   - Converts log(loss) → actual dollar amount

**Output:** Predicted loss amount in dollars

### Risk Calculation Formula
The system combines both pipelines to calculate comprehensive risk metrics:

```python
# Step 1: Get default probability from classification pipeline
proba_pred = proba_pipeline.predict_proba(X)
probability_of_default = 1 - proba_pred[:, 1]  # Invert to get default risk

# Step 2: Get loss amount from regression pipeline
reg_pred = reg_pipeline.predict(X)
loss_amount = np.expm1(reg_pred)  # Reverse log transformation

# Step 3: Calculate expected loss (key risk metric)
expected_loss = probability_of_default * loss_amount

# Step 4: Additional risk ratios
expected_loss_ratio = expected_loss / loan_amount  # % of loan at risk
installment_to_income = installment / (annual_income / 12)  # Monthly burden
```

**Example:**
- Loan Amount: $25,000
- Probability of Default: 15.34% (0.1534)
- Loss Amount if Default: $8,234.56
- **Expected Loss**: $1,263.42 (15.34% × $8,234.56)
- **Expected Loss Ratio**: 5.05% of loan amount

### Input Features
- **loan_amnt**: Loan amount requested ($)
- **term**: Loan term (3, 6, 9, 12, 18, 24, 36, 60 months)
- **installment**: Monthly payment amount ($)
- **emp_length**: Employment length (years)
- **home_ownership**: MORTGAGE, RENT, OWN, OTHER
- **annual_inc**: Annual income ($)
- **verification_status**: Verified, Not Verified, Source Verified
- **issue_d**: Loan issue date (MMM-YYYY)
- **purpose**: Loan purpose (debt_consolidation, credit_card, etc.)
- **dti**: Debt-to-income ratio (%)
- **delinq_2yrs**: Number of 30+ day delinquencies in past 2 years
- **earliest_cr_line**: Date of earliest credit line (MMM-YYYY)
- **fico_range_low**: FICO credit score (300-850)
- **pub_rec**: Number of derogatory public records
- **revol_util**: Revolving credit utilization (%)
- **bc_util**: Bankcard utilization (%)

---

## 📊 Development Notebooks

The `notebooks/` directory contains Jupyter notebooks documenting the complete ML pipeline:

1. **Data_Gathering.ipynb** - Data collection and initial loading
2. **Data_Exploration.ipynb** - Exploratory data analysis and visualization
3. **Feature_Engineering_Classification.ipynb** - Feature engineering for classification task
4. **Feature_Engineering_Regression.ipynb** - Feature engineering for regression task
5. **Model_Buliding_Classification.ipynb** - Classification model training and evaluation
6. **Model_Building_Regression.ipynb** - Regression model training and evaluation

These notebooks show the complete data science workflow from raw data to production models.

---

## 🎨 Screenshots

### Main Application Interface
- Clean, modern gradient design
- Responsive form with 16 input fields
- Calendar pickers for dates
- Dropdown menus for categorical fields
- Real-time validation
- Helpful tooltips with information icons

### Results Display
- 5-card metric dashboard
- Risk badge prominently displayed
- AI-generated bullet-point explanation
- Color-coded risk indicators:
  - 🟢 Green - Low Risk
  - 🟠 Orange - Medium Risk
  - 🔴 Red - High Risk
- Automatic warnings for concerning factors
- "New Analysis" button for additional predictions

---

## 🛡️ Risk Assessment Logic

### Risk Levels
- **Low Risk**: Default probability < 30%
- **Medium Risk**: Default probability 30-75%
- **High Risk**: Default probability > 75%

### Automatic Warnings
The system generates warnings for:
- FICO score below 600
- Bankcard utilization > 90%
- Revolving utilization > 90%
- Installment to income ratio > 50%

### LLM Explanation
The AI analyzes:
- Default probability percentage
- Credit score quality
- Debt-to-income ratio
- Credit utilization patterns
- Payment to income ratio
- Loan amount relative to income

And generates clear, actionable explanations in plain English with bullet points.

---

## 🐛 Troubleshooting

### Common Issues

**1. Groq API Error**
```
⚠ Warning: Groq client not initialized
```
**Solution:** Ensure `.env` file exists with valid `GROQ_API_KEY`

**2. Model Loading Error**
```
Error loading proba_pipeline: [Errno 2] No such file or directory
```
**Solution:** Verify `.pkl` files are in `models/` directory

**3. sklearn Compatibility Warning**
```
This Pipeline instance is not fitted yet
```
**Solution:** These warnings are suppressed automatically; models work correctly

**4. Virtual Environment Not Activated**
**Solution:** Run `.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (Linux/Mac)

---

## 📝 License

This project is for educational and demonstration purposes.

---

## 👨‍💻 Development

### Local Development
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run with debug mode
python final_app.py
```

The Flask app runs in debug mode by default, enabling:
- Auto-reload on code changes
- Detailed error messages
- Interactive debugger

### Adding New Features
1. Update `custom_func_and_class.py` for new transformers
2. Retrain models using notebooks
3. Save updated `.pkl` files to `models/`
4. Modify `final_app.py` for new endpoints
5. Update `index.html` for UI changes

---

## 🙏 Acknowledgments

- **Groq** - For fast LLM inference API
- **Llama 3.3** - For natural language explanations
- **scikit-learn** - For ML framework
- **XGBoost & LightGBM** - For gradient boosting models
- **Flask** - For web framework

---

## 📞 Support

For issues, questions, or contributions, please check the project documentation or create an issue in the repository.

---

**Built with ❤️ using Machine Learning & AI**
