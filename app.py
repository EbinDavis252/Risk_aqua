import streamlit as st
import os
import pandas as pd
import joblib
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Styling the sidebar ---
sidebar_style = """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #003366, #004080);
    padding: 2rem 1rem;
}
[data-testid="stSidebar"] * {
    color: #ffcc00 !important;
    font-family: 'Segoe UI', sans-serif !important;
}
section[data-testid="stSidebar"] input {
    background-color: white !important;
    color: black !important;
    border-radius: 5px !important;
}
section[data-testid="stSidebar"] button {
    background-color: #0066cc !important;
    color: white !important;
    font-weight: bold;
    border-radius: 5px !important;
    padding: 0.4rem 1rem;
}
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# --- Functions ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_user(username, password):
    with open("users.csv", "a") as f:
        f.write(f"{username},{hash_password(password)}\n")

def validate_user(username, password):
    if not os.path.exists("users.csv"):
        return False
    hashed = hash_password(password)
    with open("users.csv", "r") as f:
        for line in f:
            u, p = line.strip().split(",")
            if u == username and p == hashed:
                return True
    return False

def save_uploaded_file(uploaded_file, username, prefix):
    user_dir = f"userdata/{username}"
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, f"{prefix}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- App title ---
st.title("🐟 AI-Driven Aqua Loan Risk Assessment Dashboard")

# --- Login/Register System ---
auth_status = False
menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Login / Register", menu)

if choice == "Register":
    st.sidebar.subheader("Create New Account")
    new_user = st.sidebar.text_input("New Username")
    new_pass = st.sidebar.text_input("New Password", type='password')
    if st.sidebar.button("Register"):
        save_user(new_user, new_pass)
        st.sidebar.success("Registered! Please login.")

if choice == "Login":
    st.sidebar.subheader("Login to Continue")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Login"):
        if validate_user(username, password):
            st.session_state['username'] = username
            st.success(f"Welcome {username}!")
            auth_status = True
        else:
            st.error("Invalid credentials")

# --- If Authenticated ---
if 'username' in st.session_state:
    username = st.session_state['username']
    tab = st.sidebar.radio("Select Section", ["📉 Risk Assessment", "🌊 Water Quality", "🧪 Combined Analysis"])

    if tab == "📉 Risk Assessment":
        st.header("📉 Loan Default Risk Prediction")

        uploaded_loan = st.file_uploader("Upload Loan Dataset", type=["csv"], key="loan_upload")
        if uploaded_loan:
            path = save_uploaded_file(uploaded_loan, username, "loan")
            df_loan = pd.read_csv(path)
            st.dataframe(df_loan.head())

            # Train model on uploaded data
            try:
                X = df_loan[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']]
                y = df_loan['default']
                model = RandomForestClassifier()
                model.fit(X, y)
                df_loan["default_probability"] = model.predict_proba(X)[:,1]
                st.success("Model trained and predictions added!")
                st.dataframe(df_loan[['loan_amount', 'default_probability']].head())
            except KeyError as e:
                st.error(f"Missing columns: {e}")

    elif tab == "🌊 Water Quality":
        st.header("🌊 Water Quality Risk (Farm Failure)")

        uploaded_sensor = st.file_uploader("Upload Sensor Data", type=["csv"], key="sensor_upload")
        if uploaded_sensor:
            path = save_uploaded_file(uploaded_sensor, username, "sensor")
            df_sensor = pd.read_csv(path)
            st.dataframe(df_sensor.head())

            # Train model on uploaded sensor data
            try:
                X = df_sensor[['temp', 'ph', 'do', 'ammonia']]
                y = df_sensor['mortality']
                model = RandomForestClassifier()
                model.fit(X, y)
                df_sensor["failure_risk"] = model.predict_proba(X)[:,1]
                st.success("Farm failure risk predicted.")
                st.dataframe(df_sensor[['ph', 'ammonia', 'failure_risk']].head())
            except KeyError as e:
                st.error(f"Missing columns: {e}")

    elif tab == "🧪 Combined Analysis":
        st.header("🧪 Combined Loan & Water Risk Analysis")

        uploaded_loan = st.file_uploader("Upload Loan Dataset", type=["csv"], key="combo_loan")
        uploaded_sensor = st.file_uploader("Upload Sensor Dataset", type=["csv"], key="combo_sensor")

        if uploaded_loan and uploaded_sensor:
            path1 = save_uploaded_file(uploaded_loan, username, "combo_loan")
            path2 = save_uploaded_file(uploaded_sensor, username, "combo_sensor")

            df_loan = pd.read_csv(path1)
            df_sensor = pd.read_csv(path2)

            try:
                X_loan = df_loan[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']]
                y_loan = df_loan['default']
                model_loan = RandomForestClassifier()
                model_loan.fit(X_loan, y_loan)
                loan_probs = model_loan.predict_proba(X_loan)[:,1]

                X_sensor = df_sensor[['temp', 'ph', 'do', 'ammonia']]
                y_sensor = df_sensor['mortality']
                model_sensor = RandomForestClassifier()
                model_sensor.fit(X_sensor, y_sensor)
                farm_probs = model_sensor.predict_proba(X_sensor)[:,1]

                df_combined = pd.DataFrame({
                    "loan_default_prob": loan_probs[:len(farm_probs)],
                    "farm_failure_prob": farm_probs
                })
                df_combined["combined_risk"] = (df_combined["loan_default_prob"] + df_combined["farm_failure_prob"]) / 2

                st.success("Combined risk score generated.")
                st.dataframe(df_combined.head())
            except KeyError as e:
                st.error(f"Missing columns in datasets: {e}")
