# 💸 Credit Risk Prediction Project

This project is an end-to-end machine learning pipeline for predicting credit risk (loan default) using real-world borrower data. It includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, interpretability with SHAP and LIME, and deployment as a web application using Streamlit.

---

## 🚀 Project Features

- 📊 **Data Cleaning**:
  - Missing values handled
  - Outliers clipped using IQR
  - Inconsistent values fixed (e.g., unrealistic ages)
  - Duplicates removed

- 🛠️ **Feature Engineering**:
  - `loan_percent_income`: ratio of loan amount to income
  - `loan_grade_num`: numeric mapping of loan grades
  - One-hot encoding for categorical variables
  - `income_times_employment` interaction feature

- 📈 **Exploratory Data Analysis**:
  - Univariate and bivariate analysis
  - Correlation heatmap

- 🤖 **Modeling**:
  - Logistic Regression, Random Forest, and XGBoost
  - Hyperparameter tuning using `RandomizedSearchCV`
  - Final model: **Tuned XGBoost**

- 📊 **Model Evaluation**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix and classification report
  - ROC-AUC Score: **0.95**

- 🧠 **Model Explainability**:
  - SHAP summary and force plots (global + local interpretation)
  - LIME for individual prediction explanation

- 💻 **Deployment**:
  - Streamlit web application
  - Takes user input and returns prediction
  - Fully interactive frontend with custom logic and styling

---

## 📁 Project Structure

```
credit_risk_app/
│
├── app.py                          # Streamlit app code
├── credit_risk_xgboost_model.pkl   # Trained model
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview (this file)
```

---

## ⚙️ Installation

1. Clone this repo
```bash
git clone https://github.com/yourusername/credit-risk-app.git
cd credit-risk-app
```

2. Create virtual environment (optional but recommended)
```bash
python -m venv credit-env
source credit-env/bin/activate  # or credit-env\Scripts\activate on Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🌐 Deployment

You can deploy this app easily using [Streamlit Cloud](https://streamlit.io/cloud):

- Push this project to GitHub
- Go to Streamlit Cloud, connect your GitHub repo
- Set `app.py` as the entry point
- Done!

---


