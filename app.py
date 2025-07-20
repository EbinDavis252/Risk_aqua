import streamlit as st
import sqlite3
import os
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ---------------------------- SETUP ----------------------------
st.set_page_config(layout="wide", page_title="AI-Driven Aqua Risk System")

# Database setup
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS audit_logs (
    username TEXT,
    section TEXT,
    model_accuracy TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# Create directory for uploaded data
if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# ---------------------------- SIDEBAR UI ----------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #003366;
        }
        .sidebar-text input, .sidebar-text label {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Water_icon.svg", width=80)
st.sidebar.title("🌊 Aqua Risk AI App")

auth_option = st.sidebar.selectbox("🔐 Login / Register", ["Login", "Register"])
username = st.sidebar.text_input("👤 Username")
password = st.sidebar.text_input("🔒 Password", type="password")

def register_user():
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        st.sidebar.error("Username already exists.")
    else:
        cursor.execute("INSERT INTO users VALUES (?,?)", (username, password))
        conn.commit()
        st.sidebar.success("Registration successful. Please log in.")

def login_user():
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone() is not None

if auth_option == "Register":
    if st.sidebar.button("Register"):
        register_user()
    st.stop()

if not login_user():
    st.sidebar.warning("Login to Continue")
    st.stop()

# ---------------------------- PAGE TABS ----------------------------
st.title("🧠 AI-Driven Risk Assessment System for Aqua Loan Providers")
tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Loan Risk Assessment",
    "🔵 Water Quality Monitoring",
    "🟢 Combined Aqua Risk Analysis",
    "📊 Reporting & Audit Logs"
])

def load_data(file, name):
    if file:
        df = pd.read_csv(file)
        path = f"saved_user_data/{username}_{name}.csv"
        df.to_csv(path, index=False)
        return df
    elif os.path.exists(f"saved_user_data/{username}_{name}.csv"):
        return pd.read_csv(f"saved_user_data/{username}_{name}.csv")
    else:
        return None

# ---------------------------- TAB 1: Loan Risk ----------------------------
with tab1:
    st.header("📊 Farmer Loan Default Risk Prediction")
    file = st.file_uploader("Upload Farmer & Loan Profile Data", type=["csv"], key="loan")
    df = load_data(file, "loan")

    if df is not None:
        st.dataframe(df.head())

        if 'default' in df.columns:
            X = df.drop("default", axis=1).select_dtypes(include='number')
            y = df['default']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)
            acc = round(report['accuracy'] * 100, 2)
            st.success(f"✅ Model Trained - Accuracy: {acc}%")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "loan_model.pkl")

            # Save result in DB
            cursor.execute("INSERT INTO audit_logs (username, section, model_accuracy) VALUES (?, ?, ?)",
                           (username, "Loan Risk", f"{acc}%"))
            conn.commit()

            st.info("💡 Default risk rises with high loan amount & low income.")

# ---------------------------- TAB 2: Water Quality ----------------------------
with tab2:
    st.header("🌊 Water Quality Risk Monitoring")
    file = st.file_uploader("Upload Water Quality Data", type=["csv"], key="water")
    df = load_data(file, "water")

    if df is not None:
        st.dataframe(df.head())

        if {'pH', 'temperature', 'ammonia', 'dissolved_oxygen'}.issubset(df.columns):
            fig = px.scatter_matrix(df, dimensions=['pH', 'temperature', 'ammonia', 'dissolved_oxygen'],
                                    title="Water Quality Metrics")
            st.plotly_chart(fig, use_container_width=True)

            df['risk'] = ((df['pH'] < 6.5) | (df['ammonia'] > 0.5) | (df['dissolved_oxygen'] < 4)).astype(int)
            risk_count = df['risk'].sum()
            st.warning(f"⚠️ {risk_count} farms are at water quality risk.")

            st.success("✅ Advice: Use aeration & monitor ammonia levels.")

# ---------------------------- TAB 3: Combined Analysis ----------------------------
with tab3:
    st.header("🧬 Combined Loan + Water Quality Risk")
    file1 = st.file_uploader("Upload Loan Data", type=["csv"], key="combine1")
    file2 = st.file_uploader("Upload Water Quality Data", type=["csv"], key="combine2")
    df1 = load_data(file1, "combine1")
    df2 = load_data(file2, "combine2")

    if df1 is not None and df2 is not None:
        df_combined = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        st.dataframe(df_combined.head())

        if 'default' in df_combined.columns:
            X = df_combined.drop("default", axis=1).select_dtypes(include='number')
            y = df_combined["default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)
            acc = round(report['accuracy'] * 100, 2)
            st.success(f"✅ Combined Model Accuracy: {acc}%")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "combined_model.pkl")

            cursor.execute("INSERT INTO audit_logs (username, section, model_accuracy) VALUES (?, ?, ?)",
                           (username, "Combined Risk", f"{acc}%"))
            conn.commit()

            st.info("📉 Farms with poor water & financial health are high-risk.")

# ---------------------------- TAB 4: Reports ----------------------------
with tab4:
    st.header("📑 AI Model Reports and Logs")
    df_log = pd.read_sql_query("SELECT * FROM audit_logs WHERE username=? ORDER BY timestamp DESC", conn, params=(username,))
    if not df_log.empty:
        st.dataframe(df_log)
    else:
        st.info("No audit logs yet. Train a model first.")
