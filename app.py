import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sqlite3
from datetime import datetime
from fpdf import FPDF

# Page config
st.set_page_config(page_title="Aqua Risk Intelligence", page_icon="🌊", layout="wide")

# Branding
st.markdown("""
    <style>
    body {
        background-color: #f4f8fb;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #e0f7fa, #fce4ec);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(#0f2027, #203a43, #2c5364);
        color: white;
    }
    .big-title {
        font-size: 38px;
        color: #003366;
        text-align: center;
        font-weight: bold;
        padding: 10px;
    }
    .stTextInput>div>div>input {
        background-color: white;
        color: black;
    }
    </style>
    <div class="big-title">🌊 Aqua Risk Intelligence Platform</div>
""", unsafe_allow_html=True)

# Ensure folder exists
if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# Database connection
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT,
    role TEXT DEFAULT 'user'
)
''')
conn.commit()

# Functions
def get_user_role(username):
    cursor.execute("SELECT role FROM users WHERE username=?", (username,))
    result = cursor.fetchone()
    return result[0] if result else "user"

def login(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()

def register_user(username, password):
    role = "admin" if username == "admin" else "user"
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        return False
    cursor.execute("INSERT INTO users VALUES (?, ?, ?)", (username, password, role))
    conn.commit()
    return True

# Login/Registration
st.sidebar.title("🔐 User Login")
auth_mode = st.sidebar.radio("Choose Action", ["Login", "Register"])
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_status = False

if st.sidebar.button(auth_mode):
    if auth_mode == "Register":
        if register_user(username, password):
            st.sidebar.success("Registered! Please log in.")
        else:
            st.sidebar.error("User already exists.")
    elif auth_mode == "Login":
        user = login(username, password)
        if user:
            login_status = True
            st.session_state["user"] = username
            st.session_state["role"] = get_user_role(username)
            st.sidebar.success(f"Welcome, {username}!")
        else:
            st.sidebar.error("Invalid credentials")

if "user" in st.session_state:
    username = st.session_state["user"]
    role = st.session_state["role"]

    # Sidebar Options
    if role == "admin":
        tab = st.sidebar.radio("📁 Menu", ["Admin Dashboard", "All User Files", "Risk Assessment", "Water Quality", "Combined Analysis"])
    else:
        tab = st.sidebar.radio("📁 Menu", ["Risk Assessment", "Water Quality", "Combined Analysis"])

    user_folder = os.path.join("saved_user_data", username)
    os.makedirs(user_folder, exist_ok=True)

    # Upload
    st.sidebar.markdown("### 📤 Upload Your CSV")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (.csv)", type=["csv"])

    if uploaded_file:
        file_path = os.path.join(user_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"{uploaded_file.name} saved!")

    def load_latest_csv(folder):
        try:
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
            return pd.read_csv(sorted(files, key=os.path.getmtime)[-1]) if files else None
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

    if tab == "Admin Dashboard":
        st.subheader("👥 Registered Users")
        cursor.execute("SELECT username, role FROM users")
        df_users = pd.DataFrame(cursor.fetchall(), columns=["Username", "Role"])
        st.table(df_users)

    elif tab == "All User Files":
        st.subheader("📁 All Uploaded Files")
        all_users = os.listdir("saved_user_data")
        for user in all_users:
            st.markdown(f"**{user}**:")
            files = os.listdir(os.path.join("saved_user_data", user))
            for f in files:
                st.markdown(f"- 📄 `{f}`")

    elif tab == "Risk Assessment":
        st.header("📊 Loan Default Risk Analysis")
        df = load_latest_csv(user_folder)
        if df is not None:
            st.dataframe(df.head())
            if "loan_amount" in df.columns:
                fig = px.histogram(df, x="loan_amount", color="default")
                st.plotly_chart(fig)
            if "income" in df.columns:
                fig = px.box(df, x="default", y="income", color="default")
                st.plotly_chart(fig)
            st.info("💡 Recommendation: High loan amount with low income shows more default risk.")

    elif tab == "Water Quality":
        st.header("💧 Fish Farm Risk Based on Water Quality")
        df = load_latest_csv(user_folder)
        if df is not None:
            st.dataframe(df.head())
            if {"temp", "ph", "do", "mortality"}.issubset(df.columns):
                fig = px.scatter_matrix(df, dimensions=["temp", "ph", "do", "mortality"], color="mortality")
                st.plotly_chart(fig)
            st.warning("⚠️ Alert: High ammonia or low DO increases fish death rate.")

    elif tab == "Combined Analysis":
        st.header("🔗 Combined Aqua Risk Intelligence")
        df = load_latest_csv(user_folder)
        if df is not None:
            st.dataframe(df.head())
            if "loan_amount" in df.columns and "mortality" in df.columns:
                fig = px.scatter(df, x="loan_amount", y="mortality", color="default", title="Loan vs Mortality")
                st.plotly_chart(fig)
            st.success("✅ Recommendation: Avoid giving large loans to farms with high fish mortality rates.")
else:
    st.warning("🔐 Please login to access the dashboard.")
