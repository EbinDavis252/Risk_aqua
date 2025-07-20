import streamlit as st
import pandas as pd
import sqlite3
import joblib

# Load models
loan_model = joblib.load("loan_default_model.pkl")
farm_model = joblib.load("farm_failure_model.pkl")

st.set_page_config(page_title="Aqua Loan Risk Dashboard", layout="wide")
st.title("💧 AI-Driven Risk Assessment System for Aqua Loan Providers")

# Connect to SQLite
conn = sqlite3.connect("aqua_risk.db")

# Upload CSVs
st.sidebar.header("📤 Upload CSVs")
loan_csv = st.sidebar.file_uploader("Loan Data CSV", type="csv")
sensor_csv = st.sidebar.file_uploader("Farm Sensor Data CSV", type="csv")

if st.sidebar.button("📥 Load Data into Database"):
    if loan_csv:
        loan_df = pd.read_csv(loan_csv)
        loan_df.to_sql("loan_data", conn, if_exists="replace", index=False)
        st.success("Loan data uploaded.")
    if sensor_csv:
        sensor_df = pd.read_csv(sensor_csv)
        sensor_df.to_sql("farm_sensor_data", conn, if_exists="replace", index=False)
        st.success("Sensor data uploaded.")

# Dashboard
st.header("📊 Risk Dashboard")

col1, col2 = st.columns(2)
try:
    loan_data = pd.read_sql("SELECT * FROM loan_data", conn)
    col1.metric("Total Farmers", loan_data["farmer_id"].nunique())
    col1.metric("Defaults", loan_data["defaulted"].sum())
except:
    col1.warning("No loan data yet.")

try:
    sensor_data = pd.read_sql("SELECT * FROM farm_sensor_data", conn)
    col2.metric("Sensor Records", len(sensor_data))
    col2.metric("Farm Failures", sensor_data["farm_failure"].sum())
except:
    col2.warning("No sensor data yet.")

# Prediction Form
st.header("🤖 Risk Prediction")
with st.form("prediction_form"):
    st.subheader("Loan Features")
    age = st.number_input("Age", 18, 70, 35)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    income = st.number_input("Monthly Income", 1000, 100000, 15000)
    loan_amount = st.number_input("Loan Amount", 5000, 500000, 50000)
    loan_term = st.slider("Loan Term (months)", 6, 36, 24)
    emi_paid = st.slider("EMIs Paid on Time", 0, 24, 12)
    emi_missed = st.slider("EMIs Missed", 0, 10, 2)

    st.subheader("Sensor Readings")
    temp = st.slider("Water Temp (°C)", 20.0, 40.0, 28.0)
    ph = st.slider("pH Level", 5.0, 9.0, 7.0)
    do = st.slider("Dissolved Oxygen (mg/L)", 2.0, 12.0, 6.0)
    ammonia = st.slider("Ammonia (ppm)", 0.0, 5.0, 1.0)
    mortality = st.slider("Mortality Count", 0, 100, 10)

    submitted = st.form_submit_button("🚨 Predict Risk")

if submitted:
    loan_input = [[loan_amount, loan_term, emi_paid, emi_missed, age, credit_score, income]]
    sensor_input = [[temp, ph, do, ammonia, mortality]]

    loan_risk = loan_model.predict_proba(loan_input)[0][1]
    farm_risk = farm_model.predict_proba(sensor_input)[0][1]

    st.subheader("📌 Predictions")
    st.write(f"**Loan Default Probability:** {loan_risk:.2%}")
    st.write(f"**Farm Failure Probability:** {farm_risk:.2%}")

    if loan_risk > 0.6 or farm_risk > 0.6:
        st.error("🚨 High Risk - Field visit needed")
    elif loan_risk > 0.3 or farm_risk > 0.3:
        st.warning("⚠️ Medium Risk - Monitor this case")
    else:
        st.success("✅ Low Risk")
