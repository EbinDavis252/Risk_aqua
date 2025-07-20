import streamlit as st
import pandas as pd
import sqlite3
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Database setup
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

# ---------------------- Auth Functions ---------------------- #
def register_user(username, password):
    cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()

# ---------------------- Streamlit UI ---------------------- #
st.set_page_config(page_title="Aqua Risk Assessment System", layout="wide")
st.title("🐟 AI-Driven Risk Assessment System for Aqua Loan Providers")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.sidebar.subheader("Create New Account")
    new_user = st.sidebar.text_input("Username")
    new_pass = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Register"):
        register_user(new_user, new_pass)
        st.sidebar.success("Account created! Go to Login.")

elif choice == "Login":
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    
    if st.sidebar.button("Login"):
        user = login_user(username, password)
        if user:
            st.success(f"Welcome {username}!")
            st.markdown("---")
            st.subheader("📂 Upload Your Datasets")

            loan_csv = st.sidebar.file_uploader("Upload Loan Data CSV", type=["csv"])
            sensor_csv = st.sidebar.file_uploader("Upload Farm Sensor Data CSV", type=["csv"])

            if loan_csv and sensor_csv:
                loan_df = pd.read_csv(loan_csv).dropna()
                sensor_df = pd.read_csv(sensor_csv).dropna()

                st.success("✅ Data Uploaded Successfully!")
                
                # ------------------ ML Training ------------------ #
                # Loan Model
                X_loan = loan_df[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']]
                y_loan = loan_df['defaulted']
                loan_model = XGBClassifier()
                loan_model.fit(X_loan, y_loan)

                # Farm Model
                X_sensor = sensor_df[['temp', 'ph', 'do', 'ammonia', 'mortality']]
                y_sensor = sensor_df['farm_failure']
                farm_model = RandomForestClassifier()
                farm_model.fit(X_sensor, y_sensor)

                # ------------------ Prediction Input ------------------ #
                st.subheader("📊 Predict Risk for New Farmer")
                st.markdown("### Loan Info")
                col1, col2, col3 = st.columns(3)
                with col1:
                    loan_amount = st.number_input("Loan Amount", 5000, 100000)
                    age = st.slider("Age", 18, 80, 35)
                with col2:
                    loan_term = st.number_input("Loan Term (months)", 6, 60, 24)
                    credit_score = st.slider("Credit Score", 300, 900, 650)
                with col3:
                    emi_paid = st.slider("EMIs Paid", 0, 24, 12)
                    emi_missed = st.slider("EMIs Missed", 0, 10, 1)
                    income = st.number_input("Monthly Income", 1000, 100000, 15000)

                st.markdown("### Farm Sensor Info")
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
                    mortality = st.slider("Recent Mortality Rate (%)", 0, 100, 10)

                if st.button("🔍 Predict Risk"):
                    X_new_loan = pd.DataFrame([[loan_amount, loan_term, emi_paid, emi_missed, age, credit_score, income]],
                                              columns=X_loan.columns)
                    X_new_sensor = pd.DataFrame([[temp, ph, do, ammonia, mortality]],
                                                columns=X_sensor.columns)

                    loan_risk = loan_model.predict_proba(X_new_loan)[0][1]
                    farm_risk = farm_model.predict_proba(X_new_sensor)[0][1]

                    st.markdown("---")
                    st.subheader("📈 Risk Assessment Results")
                    st.metric("Loan Default Probability", f"{loan_risk*100:.2f}%")
                    st.metric("Farm Failure Probability", f"{farm_risk*100:.2f}%")

                    # Alert System
                    if loan_risk > 0.7 and farm_risk > 0.7:
                        st.error("🚨 HIGH RISK: Immediate field intervention needed!")
                    elif loan_risk > 0.5 or farm_risk > 0.5:
                        st.warning("⚠️ MEDIUM RISK: Schedule follow-up visit.")
                    else:
                        st.success("✅ LOW RISK: No immediate action required.")
        else:
            st.sidebar.error("Invalid credentials. Try again.")
