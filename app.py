import streamlit as st
import pandas as pd
import sqlite3
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# SQLite setup
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

def register_user(username, password):
    cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()

# Session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Sidebar Menu
st.set_page_config(page_title="Aqua Risk System", layout="wide")
st.title("🐟 AI-Driven Risk Assessment System for Aqua Loan Providers")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.sidebar.subheader("Create Account")
    new_user = st.sidebar.text_input("Username")
    new_pass = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Register"):
        register_user(new_user, new_pass)
        st.sidebar.success("Registered! Please login.")

if choice == "Login" or st.session_state.logged_in:
    if not st.session_state.logged_in:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}")
            else:
                st.sidebar.error("Invalid login.")

    if st.session_state.logged_in:
        st.markdown("---")
        st.subheader("📂 Upload Datasets")

        loan_file = st.sidebar.file_uploader("Upload Loan Data CSV", type=["csv"])
        sensor_file = st.sidebar.file_uploader("Upload Sensor Data CSV", type=["csv"])

        if loan_file and sensor_file:
            loan_df = pd.read_csv(loan_file).dropna()
            sensor_df = pd.read_csv(sensor_file).dropna()

            st.success("✅ Files uploaded successfully!")

            # Loan Model Training
            X_loan = loan_df[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']]
            y_loan = loan_df['defaulted']
            loan_model = XGBClassifier()
            loan_model.fit(X_loan, y_loan)

            # Farm Model Training
            X_farm = sensor_df[['temp', 'ph', 'do', 'ammonia', 'mortality']]
            y_farm = sensor_df['farm_failure']
            farm_model = RandomForestClassifier()
            farm_model.fit(X_farm, y_farm)

            st.markdown("---")
            st.subheader("📊 Risk Prediction Form")

            col1, col2, col3 = st.columns(3)
            with col1:
                loan_amount = st.number_input("Loan Amount", 5000, 100000)
                age = st.slider("Age", 18, 80, 35)
            with col2:
                loan_term = st.number_input("Loan Term (months)", 6, 60, 12)
                credit_score = st.slider("Credit Score", 300, 900, 650)
            with col3:
                emi_paid = st.slider("EMIs Paid", 0, 24, 6)
                emi_missed = st.slider("EMIs Missed", 0, 10, 1)
                income = st.number_input("Monthly Income", 1000, 50000, 15000)

            st.markdown("### Farm Sensor Values")
            col4, col5, col6 = st.columns(3)
            with col4:
                temp = st.slider("Temperature (°C)", 20.0, 35.0, 28.0)
            with col5:
                ph = st.slider("pH Level", 5.0, 9.0, 7.0)
            with col6:
                do = st.slider("Dissolved Oxygen (mg/L)", 2.0, 10.0, 6.0)

            col7, col8 = st.columns(2)
            with col7:
                ammonia = st.slider("Ammonia (ppm)", 0.0, 3.0, 1.0)
            with col8:
                mortality = st.slider("Mortality Rate (%)", 0, 100, 10)

            if st.button("🔍 Predict Risk"):
                pred_loan = pd.DataFrame([[loan_amount, loan_term, emi_paid, emi_missed, age, credit_score, income]],
                                         columns=X_loan.columns)
                pred_farm = pd.DataFrame([[temp, ph, do, ammonia, mortality]],
                                         columns=X_farm.columns)

                loan_risk = loan_model.predict_proba(pred_loan)[0][1]
                farm_risk = farm_model.predict_proba(pred_farm)[0][1]

                st.metric("Loan Default Risk", f"{loan_risk*100:.2f}%")
                st.metric("Farm Failure Risk", f"{farm_risk*100:.2f}%")

                if loan_risk > 0.7 and farm_risk > 0.7:
                    st.error("🚨 HIGH RISK: Immediate action needed!")
                elif loan_risk > 0.5 or farm_risk > 0.5:
                    st.warning("⚠️ MEDIUM RISK: Field check recommended.")
                else:
                    st.success("✅ LOW RISK: No intervention required.")
        else:
            st.info("📁 Please upload both datasets.")
else:
    st.warning("🔒 Please login to continue.")
