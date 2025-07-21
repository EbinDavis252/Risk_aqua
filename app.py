import streamlit as st
import os
import sqlite3
import pandas as pd
from PIL import Image
import base64

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="Aqua Finance Risk App", layout="wide")

# ---------- CUSTOM BACKGROUND ----------
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    page_bg_img = f"""
    <style>
    [data-testid="stApp"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("assets/background.jpg")

# ---------- DATABASE SETUP ----------
os.makedirs("saved_user_data", exist_ok=True)

conn_users = sqlite3.connect("users.db", check_same_thread=False)
conn_feedback = sqlite3.connect("feedback_data.db", check_same_thread=False)
cursor_users = conn_users.cursor()
cursor_feedback = conn_feedback.cursor()

cursor_users.execute("""CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT)""")

cursor_feedback.execute("""CREATE TABLE IF NOT EXISTS feedback (
    username TEXT,
    comment TEXT)""")

conn_users.commit()
conn_feedback.commit()

# ---------- SESSION STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ---------- LOGIN & REGISTER ----------
def login_register():
    st.sidebar.title("🔐 Login / Register")

    choice = st.sidebar.selectbox("Login/Register", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Continue"):
        if choice == "Register":
            cursor_users.execute("SELECT * FROM users WHERE username=?", (username,))
            if cursor_users.fetchone():
                st.sidebar.warning("User already exists!")
            else:
                cursor_users.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn_users.commit()
                st.sidebar.success("Registered Successfully!")
        elif choice == "Login":
            cursor_users.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            if cursor_users.fetchone():
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"👋 Welcome, {username}!")
            else:
                st.sidebar.error("Incorrect username or password!")

# ---------- FILE UPLOAD ----------
def upload_data():
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        filepath = f"saved_user_data/{st.session_state.username}_data.csv"
        df.to_csv(filepath, index=False)
        st.success(f"File saved as: `{filepath}`")
        st.dataframe(df)

# ---------- DASHBOARD ----------
def dashboard():
    st.header("📊 Dashboard")
    filepath = f"saved_user_data/{st.session_state.username}_data.csv"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        st.subheader("Data Overview")
        st.dataframe(df.head())

        if "loan_default" in df.columns:
            st.bar_chart(df["loan_default"].value_counts())
        if "ph" in df.columns:
            st.line_chart(df["ph"])
    else:
        st.warning("No uploaded data found.")

# ---------- PREDICTION ----------
def prediction():
    st.header("🤖 Risk Prediction")
    st.write("Here you can integrate your ML models.")
    st.info("Model loading and prediction code goes here.")

# ---------- FEEDBACK ----------
def feedback():
    st.header("💬 Feedback")
    comment = st.text_area("Leave your feedback")
    if st.button("Submit Feedback"):
        cursor_feedback.execute("INSERT INTO feedback (username, comment) VALUES (?, ?)", (st.session_state.username, comment))
        conn_feedback.commit()
        st.success("Thanks for your feedback!")

# ---------- ADMIN PANEL ----------
def admin_panel():
    st.header("🛡️ Admin Panel")
    admin_username = "admin"
    if st.session_state.username == admin_username:
        st.subheader("All Users")
        users = cursor_users.execute("SELECT * FROM users").fetchall()
        st.table(pd.DataFrame(users, columns=["Username", "Password"]))

        st.subheader("Feedback")
        feedbacks = cursor_feedback.execute("SELECT * FROM feedback").fetchall()
        st.table(pd.DataFrame(feedbacks, columns=["Username", "Comment"]))
    else:
        st.error("Access Denied. Admins only.")

# ---------- MAIN ----------
def main():
    if not st.session_state.logged_in:
        login_register()
    else:
        st.sidebar.markdown("---")
        st.sidebar.success(f"Logged in as `{st.session_state.username}`")

        tab = st.sidebar.radio("Navigation", ["Upload Data", "Dashboard", "Prediction", "Feedback", "Admin Panel"])

        if tab == "Upload Data":
            upload_data()
        elif tab == "Dashboard":
            dashboard()
        elif tab == "Prediction":
            prediction()
        elif tab == "Feedback":
            feedback()
        elif tab == "Admin Panel":
            admin_panel()

if __name__ == "__main__":
    main()
