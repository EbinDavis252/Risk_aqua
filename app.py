import streamlit as st
import pandas as pd
import os
import joblib
import sqlite3
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------- DATABASE SETUP -----------------
conn = sqlite3.connect("aqua_users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS uploads (
    username TEXT,
    section TEXT,
    filename TEXT,
    data BLOB
)
""")
conn.commit()

# ----------------- SIDEBAR STYLE -----------------
st.markdown("""
    <style>
    .css-1544g2n {padding: 2rem 1rem 1rem 1rem;}
    .css-1d391kg {background-color: #003566;}
    .css-1cypcdb {color: gold !important;}
    .stTextInput label, .stPassword label, .stSelectbox label, .stRadio label {
        color: gold !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- LOGIN & REGISTER -----------------
st.sidebar.image("https://i.imgur.com/I80W1Q0.png", use_column_width=True)
auth_action = st.sidebar.selectbox("Login / Register", ["Login", "Register"])

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if auth_action == "Register":
    if st.sidebar.button("Register"):
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        if cursor.fetchone():
            st.sidebar.warning("Username already exists.")
        else:
            cursor.execute("INSERT INTO users VALUES (?, ?)", (username, password))
            conn.commit()
            st.sidebar.success("Registered successfully! Please login.")

elif auth_action == "Login":
    if st.sidebar.button("Login"):
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        if cursor.fetchone():
            st.session_state["user"] = username
            st.sidebar.success(f"Welcome {username}!")
        else:
            st.sidebar.error("Invalid credentials")

# ----------------- MAIN FUNCTIONALITY -----------------
if "user" in st.session_state:
    section = st.sidebar.radio("Select Section", ["📉 Risk Assessment", "💧 Water Quality", "🧪 Combined Analysis"])

    def save_uploaded_file(uploaded_file, section):
        file_data = uploaded_file.read()
        cursor.execute("INSERT INTO uploads VALUES (?, ?, ?, ?)", (st.session_state["user"], section, uploaded_file.name, file_data))
        conn.commit()

    if section == "📉 Risk Assessment":
        st.header("Loan Default Risk Assessment")
        loan_file = st.file_uploader("Upload Loan Dataset", type=['csv'], key="loan")
        if loan_file:
            save_uploaded_file(loan_file, "risk")
            df = pd.read_csv(loan_file)
            st.dataframe(df)

            try:
                X = df[['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']]
                y = df['default']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                st.subheader("📊 Prediction Insights")
                fig = px.histogram(df, x="credit_score", color="default", barmode="overlay", title="Credit Score vs Default")
                st.plotly_chart(fig)

                st.info("✅ **Recommendation**: Loans with low credit scores and high EMI missed count are high risk. Consider stricter eligibility or collateral-backed lending.")

            except Exception as e:
                st.error("Upload valid dataset with required columns.")

    elif section == "💧 Water Quality":
        st.header("Water Quality Monitoring for Fish Farm")
        sensor_file = st.file_uploader("Upload Sensor Dataset", type=['csv'], key="sensor")
        if sensor_file:
            save_uploaded_file(sensor_file, "water")
            df = pd.read_csv(sensor_file)
            st.dataframe(df)

            try:
                X = df[['temp', 'ph', 'do', 'ammonia']]
                y = df['mortality']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                st.subheader("📊 Environmental Metrics")
                fig = px.scatter(df, x='ph', y='ammonia', color='mortality', size='do', title='Mortality vs pH & Ammonia')
                st.plotly_chart(fig)

                alert = df[df['ammonia'] > 0.25]
                if not alert.empty:
                    st.warning("⚠️ High Ammonia Detected! Take corrective actions immediately.")
                st.success("✅ Recommendation: Keep ammonia below 0.25 ppm and pH between 6.5-7.5.")

            except Exception as e:
                st.error("Upload valid dataset with required columns.")

    elif section == "🧪 Combined Analysis":
        st.header("Combined Risk & Water Quality Analysis")
        loan_file = st.file_uploader("Upload Loan Dataset", type=['csv'], key="loan_combined")
        sensor_file = st.file_uploader("Upload Sensor Dataset", type=['csv'], key="sensor_combined")
        if loan_file and sensor_file:
            save_uploaded_file(loan_file, "combined_loan")
            save_uploaded_file(sensor_file, "combined_water")
            loan_df = pd.read_csv(loan_file)
            sensor_df = pd.read_csv(sensor_file)

            st.subheader("📉 Loan vs Mortality Combined Insight")
            try:
                loan_df['mortality_rate'] = sensor_df['mortality']
                combined = pd.concat([loan_df, sensor_df], axis=1)
                fig = px.density_heatmap(combined, x='credit_score', y='ammonia', z='mortality', nbinsx=10, nbinsy=10, title='Mortality by Credit Score & Ammonia')
                st.plotly_chart(fig)

                st.info("💡 Insight: Poor financial health (low credit score) + poor water quality (high ammonia) significantly increases mortality risk.")

                st.success("✅ Recommendation: Provide financial training & water testing kits to fish farmers with high risk profiles.")

            except Exception as e:
                st.error("Ensure both datasets are valid and aligned properly.")

else:
    st.warning("Please login or register to continue.")
