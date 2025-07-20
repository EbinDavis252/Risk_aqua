import streamlit as st
import pandas as pd
import os
import sqlite3
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from fpdf import FPDF
import smtplib
from email.message import EmailMessage
from datetime import datetime

# --------------- Config -------------------
st.set_page_config(layout="wide", page_title="Aqua Finance AI System")

# Create data directory if not exists
if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# Create user db
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

# --------------- Stylish Sidebar + Banner -------------------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#001f3f, #003366);
        color: white;
    }
    .main {
        background: linear-gradient(135deg, #dff6f0 10%, #ffffff 90%);
        font-family: 'Arial Rounded MT Bold';
    }
    .welcome-banner {
        padding: 10px;
        background-color: #004080;
        color: white;
        font-size: 22px;
        text-align: center;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Water_icon.svg", width=80)
st.sidebar.title("🌊 Aqua Finance Risk App")

# --------------- Login / Register -------------------
auth = st.sidebar.selectbox("🔐 Authentication", ["Login", "Register"])
username = st.sidebar.text_input("👤 Username")
password = st.sidebar.text_input("🔒 Password", type="password")

def register():
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        st.sidebar.error("Username already exists.")
    else:
        cursor.execute("INSERT INTO users VALUES (?,?)", (username, password))
        conn.commit()
        st.sidebar.success("Registered successfully!")

def login():
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()

if auth == "Register":
    if st.sidebar.button("Register"):
        register()
    st.stop()

if not login():
    st.sidebar.warning("Login to continue")
    st.stop()

# Show welcome banner
st.markdown(f"<div class='welcome-banner'>👋 Welcome, {username}!</div>", unsafe_allow_html=True)
st.markdown("---")

# --------------- Navigation Tabs -------------------
tabs = st.sidebar.radio("📁 Navigate", [
    "🔴 Risk Assessment", 
    "🔵 Water Quality", 
    "🟢 Combined Analysis", 
    "📧 Export & Email Report"
])

# --------------- Data Handling -------------------
def load_data(file, name):
    if file:
        df = pd.read_csv(file)
        path = f"saved_user_data/{username}_{name}.csv"
        df.to_csv(path, index=False)
        return df
    elif os.path.exists(f"saved_user_data/{username}_{name}.csv"):
        return pd.read_csv(f"saved_user_data/{username}_{name}.csv")
    return None

# --------------- Risk Assessment -------------------
if tabs == "🔴 Risk Assessment":
    st.subheader("📊 Loan Risk Assessment")
    file = st.file_uploader("Upload Risk Dataset", type=["csv"], key="risk")
    df = load_data(file, "risk")

    if df is not None:
        st.dataframe(df.head())
        if 'loan_amount' in df.columns and 'default' in df.columns:
            fig = px.histogram(df, x="loan_amount", color="default")
            st.plotly_chart(fig)

        if 'default' in df.columns:
            X = df.drop("default", axis=1).select_dtypes(include='number')
            y = df['default']
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.text("Model Results:\n" + classification_report(y_test, preds))
            joblib.dump(model, "risk_model.pkl")
            st.success("✅ Model Trained")

# --------------- Water Quality -------------------
elif tabs == "🔵 Water Quality":
    st.subheader("🌊 Water Quality Monitor")
    file = st.file_uploader("Upload Water Dataset", type=["csv"], key="water")
    df = load_data(file, "water")

    if df is not None:
        st.dataframe(df.head())
        if {'pH', 'ammonia', 'dissolved_oxygen'}.issubset(df.columns):
            fig = px.scatter_matrix(df, dimensions=['pH', 'ammonia', 'dissolved_oxygen'])
            st.plotly_chart(fig)

            df['risk'] = ((df['pH'] < 6.5) | (df['ammonia'] > 0.5) | (df['dissolved_oxygen'] < 4)).astype(int)
            st.warning(f"⚠️ {df['risk'].sum()} out of {len(df)} show poor quality")
            st.info("💡 Suggestion: Check aeration and ammonia levels regularly.")

# --------------- Combined Analysis -------------------
elif tabs == "🟢 Combined Analysis":
    st.subheader("🔗 Joint Farm & Water Risk")
    risk_file = st.file_uploader("Risk Data", type=["csv"], key="comb_risk")
    water_file = st.file_uploader("Water Data", type=["csv"], key="comb_water")

    df1 = load_data(risk_file, "comb_risk")
    df2 = load_data(water_file, "comb_water")

    if df1 is not None and df2 is not None:
        df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        st.dataframe(df.head())

        if 'default' in df.columns:
            X = df.drop("default", axis=1).select_dtypes(include='number')
            y = df['default']
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.text("Model Results:\n" + classification_report(y_test, preds))
            st.info("📌 Farms with poor water and credit scores are high risk.")

# --------------- Export & Email Report -------------------
elif tabs == "📧 Export & Email Report":
    st.subheader("📤 Generate & Email Report")

    df = None
    path1 = f"saved_user_data/{username}_risk.csv"
    path2 = f"saved_user_data/{username}_water.csv"

    if os.path.exists(path1):
        df = pd.read_csv(path1)

    st.write("📁 Data preview")
    if df is not None:
        st.dataframe(df.head())
    else:
        st.warning("Upload a dataset first in Risk tab.")

    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Aqua Finance Report", ln=True, align='C')
        pdf.set_font("Arial", "", 10)
        if df is not None:
            for i, row in df.head(10).iterrows():
                pdf.cell(200, 8, str(row.values), ln=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        pdf.output(f"report_{username}_{timestamp}.pdf")
        st.success("✅ Report Saved!")

    email = st.text_input("📧 Enter your email:")
    if st.button("✉️ Send Email"):
        try:
            msg = EmailMessage()
            msg.set_content("Hi, find your Aqua Finance report attached.")
            msg['Subject'] = 'Aqua Report'
            msg['From'] = "your_email@gmail.com"
            msg['To'] = email

            filename = f"report_{username}_{timestamp}.pdf"
            with open(filename, 'rb') as f:
                file_data = f.read()
            msg.add_attachment(file_data, maintype='application', subtype='pdf', filename=filename)

            # Simulated SMTP - change to real creds for prod
            # smtp = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            # smtp.login("your_email@gmail.com", "your_app_password")
            # smtp.send_message(msg)
            # smtp.quit()

            st.success("📩 Email simulated (SMTP commented for security)")
        except Exception as e:
            st.error(f"❌ Email failed: {e}")
