import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('credit_risk_xgboost_model.pkl')

# App title
st.title("💸 Credit Risk Prediction App")
st.write("Enter borrower information to predict risk of default.")

# Input form
with st.form(key='credit_form'):
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Monthly Income", value=50000)
    person_emp_length = st.number_input("Employment Length (Years)", min_value=-1.0, max_value=50.0, value=5.0)
    loan_amnt = st.number_input("Loan Amount", value=10000)
    loan_int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=25.0, value=12.5)
    
    loan_intent = st.selectbox("Loan Purpose", ['PERSONAL', 'EDUCATION', 'VENTURE', 'HOMEIMPROVEMENT', 'MEDICAL', 'DEBTCONSOLIDATION'])
    loan_grade_num = st.slider("Loan Grade (A=1 to G=7)", min_value=1, max_value=7, value=3)
    home_ownership = st.selectbox("Home Ownership", ['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
    cb_person_default_on_file = st.selectbox("Defaulted on Credit File?", ['Y', 'N'])
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", value=5)

    submit = st.form_submit_button("Predict")

# Prediction logic
if submit:
    # Manual preprocessing (must match your training set)
    input_data = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_length': person_emp_length,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_grade_num': loan_grade_num,
        'loan_percent_income': loan_amnt / person_income,
        'cb_person_default_on_file': 1 if cb_person_default_on_file == 'Y' else 0,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'income_times_employment': person_income * person_emp_length,
        'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == 'HOMEIMPROVEMENT' else 0,
        'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
        'loan_intent_PERSONAL': 1 if loan_intent == 'PERSONAL' else 0,
        'loan_intent_VENTURE': 1 if loan_intent == 'VENTURE' else 0,
        'person_home_ownership_OWN': 1 if home_ownership == 'OWN' else 0,
        'person_home_ownership_RENT': 1 if home_ownership == 'RENT' else 0,
        'person_home_ownership_OTHER': 1 if home_ownership == 'OTHER' else 0,
        'emp_length_missing': 0,
        'int_rate_missing': 0
    }])

    # Ensure column order matches model training
    input_data = input_data[model.feature_names_in_]

    # Now predict safely
    prediction = model.predict(input_data)[0]

    
    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ Prediction: High Risk — Likely to Default")
    else:
        st.success("✅ Prediction: Low Risk — Not Likely to Default")
