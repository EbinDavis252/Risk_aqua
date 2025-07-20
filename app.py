import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------------------
# 🎨 Per-tab Background Colors
def set_background(color):
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        .banner {{
            font-size: 40px;
            text-align: center;
            color: #000;
            font-weight: bold;
            margin-top: 20px;
        }}
        </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 📦 SQLite DB Setup
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)''')
conn.commit()

# -----------------------------
# 🔐 Login/Register System
def register_user(username, password):
    c.execute("INSERT INTO users VALUES (?, ?)", (username, password))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone()

# -----------------------------
# 📤 Sidebar: File Upload + Login
menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    set_background("#e2f7e1")  # light green
    st.sidebar.subheader("Create Account")
    new_user = st.sidebar.text_input("Username")
    new_pass = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Register"):
        try:
            register_user(new_user, new_pass)
            st.success("✅ Account created! Please log in.")
        except sqlite3.IntegrityError:
            st.error("❌ Username already exists.")

elif choice == "Login":
    set_background("#d4e8fc")  # soft blue
    st.sidebar.subheader("Login to App")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Login"):
        result = login_user(username, password)
        if result:
            set_background("#fdf6cc")  # soft yellow
            st.markdown('<div class="banner">🐟 Aqua Loan Risk Assessment System</div>', unsafe_allow_html=True)
            st.success(f"Welcome {username} 👋")

            st.subheader("📂 Upload CSV Data")
            loan_file = st.file_uploader("Upload Loan Data CSV", type=["csv"], key="loan")
            sensor_file = st.file_uploader("Upload Sensor Data CSV", type=["csv"], key="sensor")

            if loan_file:
                loan_df = pd.read_csv(loan_file)
                st.write("📊 Loan Data Preview", loan_df.head())

                loan_df.columns = [col.strip().lower() for col in loan_df.columns]
                expected_loan_cols = ['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income', 'defaulted']
                if all(col in loan_df.columns for col in expected_loan_cols):
                    X_loan = loan_df[expected_loan_cols[:-1]]
                    y_loan = loan_df['defaulted']
                    model_loan = RandomForestClassifier()
                    model_loan.fit(X_loan, y_loan)
                    preds_loan = model_loan.predict(X_loan)
                    st.success("✅ Loan Default Model Trained")
                    st.write(classification_report(y_loan, preds_loan, output_dict=True))
                else:
                    st.error(f"Loan data must contain columns: {expected_loan_cols}")

            if sensor_file:
                sensor_df = pd.read_csv(sensor_file)
                st.write("📊 Sensor Data Preview", sensor_df.head())

                sensor_df.columns = [col.strip().lower() for col in sensor_df.columns]
                expected_sensor_cols = ['temp', 'ph', 'do', 'ammonia', 'mortality']
                if all(col in sensor_df.columns for col in expected_sensor_cols):
                    X_farm = sensor_df[expected_sensor_cols[:-1]]
                    y_farm = sensor_df['mortality']
                    model_farm = RandomForestClassifier()
                    model_farm.fit(X_farm, y_farm)
                    preds_farm = model_farm.predict(X_farm)
                    st.success("✅ Farm Failure Model Trained")
                    st.write(classification_report(y_farm, preds_farm, output_dict=True))
                else:
                    st.error(f"Sensor data must contain columns: {expected_sensor_cols}")

        else:
            st.error("❌ Invalid Username or Password")
