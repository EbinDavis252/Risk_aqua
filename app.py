import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import base64

st.set_page_config(page_title="Aqua Loan Risk Dashboard", layout="wide")

# ---------------------------
# Custom Backgrounds (Sidebar & Page)
# ---------------------------
sidebar_bg_color = "#003f5c"
page_bg_color = "#f4f4f4"

def set_custom_backgrounds():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {page_bg_color};
        }}
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg_color};
        }}
        .big-font {{
            font-size:40px !important;
            font-weight: bold;
            color: #ffffff;
        }}
        .section-title {{
            font-size: 30px;
            font-weight: 600;
            color: #333333;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_custom_backgrounds()

# ---------------------------
# Sidebar Options
# ---------------------------
st.sidebar.markdown("<div class='big-font'>🌊 Aqua Risk AI</div>", unsafe_allow_html=True)
section = st.sidebar.radio("Go to", ["🔹 Risk Assessment", "🔹 Water Quality", "🔸 Combined Analysis"])

# ---------------------------
# Risk Assessment Section
# ---------------------------
if section == "🔹 Risk Assessment":
    st.title("💸 Loan Default Risk Assessment")
    uploaded_loan = st.file_uploader("Upload Loan Dataset CSV", type=["csv"], key="loan_csv")
    
    if uploaded_loan:
        loan_df = pd.read_csv(uploaded_loan)
        loan_df.columns = [col.strip().lower() for col in loan_df.columns]
        required_loan_cols = ['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income', 'defaulted']

        if all(col in loan_df.columns for col in required_loan_cols):
            st.success("✅ Valid loan dataset uploaded.")
            X = loan_df[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']]
            y = loan_df['defaulted']
            
            model = RandomForestClassifier()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.write("### 🧮 Loan Risk Prediction Summary")
            st.text(classification_report(y, y_pred))
        else:
            st.error(f"Missing columns in uploaded dataset. Required: {required_loan_cols}")

# ---------------------------
# Water Quality Section
# ---------------------------
elif section == "🔹 Water Quality":
    st.title("💧 Water Quality Risk Detection")
    uploaded_sensor = st.file_uploader("Upload Sensor Dataset CSV", type=["csv"], key="sensor_csv")

    if uploaded_sensor:
        sensor_df = pd.read_csv(uploaded_sensor)
        sensor_df.columns = [col.strip().lower() for col in sensor_df.columns]
        required_sensor_cols = ['temp', 'ph', 'do', 'ammonia', 'farm_failed']

        if all(col in sensor_df.columns for col in required_sensor_cols):
            st.success("✅ Valid sensor dataset uploaded.")
            X = sensor_df[['temp', 'ph', 'do', 'ammonia']]
            y = sensor_df['farm_failed']

            model = RandomForestClassifier()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.write("### 🐟 Farm Failure Risk Summary")
            st.text(classification_report(y, y_pred))
        else:
            st.error(f"Missing columns in uploaded dataset. Required: {required_sensor_cols}")

# ---------------------------
# Combined Section
# ---------------------------
elif section == "🔸 Combined Analysis":
    st.title("🧠 Combined Risk & Quality Analysis")
    
    uploaded_loan_combined = st.file_uploader("Upload Loan Dataset CSV", type=["csv"], key="loan_combined")
    uploaded_sensor_combined = st.file_uploader("Upload Sensor Dataset CSV", type=["csv"], key="sensor_combined")

    if uploaded_loan_combined and uploaded_sensor_combined:
        loan_df = pd.read_csv(uploaded_loan_combined)
        sensor_df = pd.read_csv(uploaded_sensor_combined)

        loan_df.columns = [col.strip().lower() for col in loan_df.columns]
        sensor_df.columns = [col.strip().lower() for col in sensor_df.columns]

        loan_cols = ['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income', 'defaulted']
        sensor_cols = ['temp', 'ph', 'do', 'ammonia', 'farm_failed']

        if all(col in loan_df.columns for col in loan_cols) and all(col in sensor_df.columns for col in sensor_cols):
            st.success("✅ Both datasets validated successfully.")
            
            # Combine X features (only for demonstration)
            X_combined = pd.concat([
                loan_df[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']],
                sensor_df[['temp', 'ph', 'do', 'ammonia']]
            ], axis=1).dropna()
            y_combined = loan_df['defaulted'][:len(X_combined)]

            model = RandomForestClassifier()
            model.fit(X_combined, y_combined)
            y_pred = model.predict(X_combined)

            st.write("### 📊 Combined Risk Summary")
            st.text(classification_report(y_combined, y_pred))
        else:
            st.error("Please make sure both datasets have all required columns.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<center>Built with ❤️ for Aqua Finance by Ebin Davis</center>", unsafe_allow_html=True)
