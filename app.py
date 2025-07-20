import streamlit as st
import pandas as pd
import os
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ---------- Database & Auth Setup ----------
conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

def save_uploaded_file(uploadedfile, username, filename):
    user_dir = os.path.join("userdata", username)
    os.makedirs(user_dir, exist_ok=True)
    filepath = os.path.join(user_dir, filename)
    with open(filepath, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return filepath

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="Aqua Risk Assessment", layout="wide")

# ---------- Custom Sidebar Styling ----------
sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        background-image: linear-gradient(to bottom, #003366, #004080);
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
        font-family: 'Segoe UI', sans-serif !important;
    }
    </style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# ---------- Session State ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

# ---------- Registration/Login ----------
menu = st.sidebar.radio("Login / Register", ["Login", "Register"])

if menu == "Register":
    st.sidebar.subheader("Create New Account")
    new_user = st.sidebar.text_input("Username")
    new_pass = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Register"):
        c.execute("SELECT * FROM users WHERE username=?", (new_user,))
        if c.fetchone():
            st.sidebar.warning("Username already exists.")
        else:
            c.execute("INSERT INTO users VALUES (?, ?)", (new_user, new_pass))
            conn.commit()
            st.sidebar.success("Registration successful! Please log in.")

elif menu == "Login":
    st.sidebar.subheader("Login to your account")
    user = st.sidebar.text_input("Username")
    passwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (user, passwd))
        if c.fetchone():
            st.session_state.authenticated = True
            st.session_state.username = user
            st.sidebar.success(f"Welcome, {user}!")
        else:
            st.sidebar.error("Invalid username or password.")

# ---------- Main App ----------
if st.session_state.authenticated:
    st.title("🌊 AI-Driven Risk Assessment System for Aqua Loan Providers")

    tab = st.sidebar.radio("Choose Analysis Section", ["📉 Risk Assessment", "💧 Water Quality", "🔗 Combined Analysis"])

    if tab == "📉 Risk Assessment":
        st.header("Loan Default Risk Prediction")
        uploaded_file = st.file_uploader("Upload Loan Dataset (.csv)", type=["csv"], key="loan_file")
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file, st.session_state.username, "loan_data.csv")
            loan_df = pd.read_csv(file_path)

            try:
                X = loan_df[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']]
                y = loan_df['defaulted']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier().fit(X_train, y_train)
                preds = model.predict(X_test)

                st.subheader("Classification Report")
                st.text(classification_report(y_test, preds))
                joblib.dump(model, os.path.join("userdata", st.session_state.username, "loan_model.pkl"))

            except Exception as e:
                st.error(f"Error processing loan data: {e}")

    elif tab == "💧 Water Quality":
        st.header("Fish Farm Failure Prediction")
        uploaded_sensor = st.file_uploader("Upload Sensor Dataset (.csv)", type=["csv"], key="sensor_file")
        if uploaded_sensor:
            file_path = save_uploaded_file(uploaded_sensor, st.session_state.username, "sensor_data.csv")
            sensor_df = pd.read_csv(file_path)
            try:
                sensor_df.columns = [col.strip().lower() for col in sensor_df.columns]
                required_cols = ['temp', 'ph', 'do', 'ammonia', 'mortality']

                if all(col in sensor_df.columns for col in required_cols):
                    X = sensor_df[['temp', 'ph', 'do', 'ammonia']]
                    y = sensor_df['mortality']

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    model = RandomForestClassifier().fit(X_train, y_train)
                    preds = model.predict(X_test)

                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, preds))
                    joblib.dump(model, os.path.join("userdata", st.session_state.username, "sensor_model.pkl"))
                else:
                    st.error(f"CSV must include: {required_cols}")
            except Exception as e:
                st.error(f"Error processing sensor data: {e}")

    elif tab == "🔗 Combined Analysis":
        st.header("Combined Risk Prediction (Loan + Sensor)")
        loan_file = st.file_uploader("Upload Loan Dataset (.csv)", type=["csv"], key="combo_loan")
        sensor_file = st.file_uploader("Upload Sensor Dataset (.csv)", type=["csv"], key="combo_sensor")

        if loan_file and sensor_file:
            try:
                loan_df = pd.read_csv(loan_file)
                sensor_df = pd.read_csv(sensor_file)
                sensor_df.columns = [col.strip().lower() for col in sensor_df.columns]

                # Match and combine datasets by row
                loan_df = loan_df.reset_index(drop=True)
                sensor_df = sensor_df.reset_index(drop=True)
                combined_df = pd.concat([loan_df, sensor_df], axis=1)

                st.write("🔍 Combined Data Preview:")
                st.dataframe(combined_df.head())

                X = combined_df[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income', 'temp', 'ph', 'do', 'ammonia']]
                y = combined_df['defaulted']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier().fit(X_train, y_train)
                preds = model.predict(X_test)

                st.subheader("Combined Classification Report")
                st.text(classification_report(y_test, preds))

            except Exception as e:
                st.error(f"Error during combined analysis: {e}")
else:
    st.warning("Please login or register to access the dashboard.")
