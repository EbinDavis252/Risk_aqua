import streamlit as st
import pandas as pd
import sqlite3
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.set_page_config(page_title="Aqua Risk Platform", layout="wide")

# -------------------- UI Setup -------------------- #
sidebar_bg_color = "#003f5c"
page_bg_color = "#f4f4f4"

def set_backgrounds():
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {page_bg_color}; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar_bg_color}; }}
    .big-font {{ font-size:40px !important; font-weight: bold; color: white; }}
    </style>
    """, unsafe_allow_html=True)

set_backgrounds()
st.sidebar.markdown("<div class='big-font'>🌊 Aqua Risk AI</div>", unsafe_allow_html=True)

# -------------------- DB Setup -------------------- #
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
conn.commit()

# -------------------- Authentication -------------------- #
def register_user(username, password):
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    if c.fetchone():
        return False
    c.execute("INSERT INTO users (username, password) VALUES (?,?)", (username, password))
    conn.commit()
    os.makedirs(f"userdata/{username}", exist_ok=True)
    return True

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone()

# -------------------- Auth UI -------------------- #
st.title("🧠 Aqua Loan Risk Dashboard")

auth_choice = st.sidebar.selectbox("Login or Register", ["Login", "Register"])
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

user_authenticated = False
if auth_choice == "Register":
    if st.sidebar.button("Register"):
        if register_user(username, password):
            st.sidebar.success("🎉 Registered! You can now log in.")
        else:
            st.sidebar.error("⚠️ Username already exists.")

if auth_choice == "Login":
    if st.sidebar.button("Login"):
        if login_user(username, password):
            st.session_state['user'] = username
            st.sidebar.success(f"✅ Logged in as {username}")
            user_authenticated = True
        else:
            st.sidebar.error("🚫 Incorrect username or password")

# -------------------- If Logged In -------------------- #
if 'user' in st.session_state:
    user_authenticated = True
    username = st.session_state['user']

if user_authenticated:
    section = st.sidebar.radio("Go to", ["🔹 Risk Assessment", "🔹 Water Quality", "🔸 Combined Analysis"])

    # -------------------- Risk Assessment -------------------- #
    if section == "🔹 Risk Assessment":
        st.subheader("💸 Loan Default Risk Assessment")
        uploaded_loan = st.file_uploader("Upload Loan Dataset CSV", type=["csv"], key="loan_csv")

        if uploaded_loan:
            loan_path = f"userdata/{username}/loan_data.csv"
            with open(loan_path, "wb") as f:
                f.write(uploaded_loan.getbuffer())
            loan_df = pd.read_csv(loan_path)
            loan_df.columns = [col.strip().lower() for col in loan_df.columns]

            required_loan_cols = ['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income', 'defaulted']
            if all(col in loan_df.columns for col in required_loan_cols):
                X = loan_df[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']]
                y = loan_df['defaulted']
                model = RandomForestClassifier().fit(X, y)
                y_pred = model.predict(X)
                st.success("✅ Loan risk model trained!")
                st.text(classification_report(y, y_pred))
            else:
                st.error(f"Missing columns: {required_loan_cols}")

    # -------------------- Water Quality -------------------- #
    elif section == "🔹 Water Quality":
        st.subheader("💧 Water Quality Risk Detection")
        uploaded_sensor = st.file_uploader("Upload Sensor Dataset CSV", type=["csv"], key="sensor_csv")

        if uploaded_sensor:
            sensor_path = f"userdata/{username}/sensor_data.csv"
            with open(sensor_path, "wb") as f:
                f.write(uploaded_sensor.getbuffer())
            sensor_df = pd.read_csv(sensor_path)
            sensor_df.columns = [col.strip().lower() for col in sensor_df.columns]

            required_sensor_cols = ['temp', 'ph', 'do', 'ammonia', 'farm_failed']
            if all(col in sensor_df.columns for col in required_sensor_cols):
                X = sensor_df[['temp', 'ph', 'do', 'ammonia']]
                y = sensor_df['farm_failed']
                model = RandomForestClassifier().fit(X, y)
                y_pred = model.predict(X)
                st.success("✅ Water quality model trained!")
                st.text(classification_report(y, y_pred))
            else:
                st.error(f"Missing columns: {required_sensor_cols}")

    # -------------------- Combined Analysis -------------------- #
    elif section == "🔸 Combined Analysis":
        st.subheader("🔗 Combined Loan & Sensor Risk Prediction")
        uploaded_loan_combined = st.file_uploader("Upload Loan Dataset", type=["csv"], key="loan_comb")
        uploaded_sensor_combined = st.file_uploader("Upload Sensor Dataset", type=["csv"], key="sensor_comb")

        if uploaded_loan_combined and uploaded_sensor_combined:
            loan_path = f"userdata/{username}/loan_combined.csv"
            sensor_path = f"userdata/{username}/sensor_combined.csv"
            with open(loan_path, "wb") as f:
                f.write(uploaded_loan_combined.getbuffer())
            with open(sensor_path, "wb") as f:
                f.write(uploaded_sensor_combined.getbuffer())

            loan_df = pd.read_csv(loan_path)
            sensor_df = pd.read_csv(sensor_path)

            loan_df.columns = [col.strip().lower() for col in loan_df.columns]
            sensor_df.columns = [col.strip().lower() for col in sensor_df.columns]

            loan_cols = ['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income', 'defaulted']
            sensor_cols = ['temp', 'ph', 'do', 'ammonia', 'farm_failed']

            if all(col in loan_df.columns for col in loan_cols) and all(col in sensor_df.columns for col in sensor_cols):
                X_combined = pd.concat([
                    loan_df[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']],
                    sensor_df[['temp', 'ph', 'do', 'ammonia']]
                ], axis=1).dropna()
                y_combined = loan_df['defaulted'][:len(X_combined)]

                model = RandomForestClassifier().fit(X_combined, y_combined)
                y_pred = model.predict(X_combined)
                st.success("✅ Combined model trained successfully.")
                st.text(classification_report(y_combined, y_pred))
            else:
                st.error("Required columns missing in either dataset.")

    # -------------------- Footer -------------------- #
    st.markdown("---")
    st.markdown("<center>🔐 You are logged in as <b>{}</b></center>".format(username), unsafe_allow_html=True)
    st.markdown("<center>Built with ❤️ by Ebin Davis @ Grant Thornton Bharat LLP</center>", unsafe_allow_html=True)

else:
    st.warning("🔒 Please register or login to continue.")
