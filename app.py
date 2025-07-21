import streamlit as st
import pandas as pd
import sqlite3
import os
import joblib
import os
if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------- PAGE SETUP ----------------------------
st.set_page_config(layout="wide", page_title="Aqua Risk System")

st.markdown("""
    <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
            background-size: cover;
            background-attachment: fixed;
        }
        [data-testid="stSidebar"] {
            background-image: url("https://images.unsplash.com/photo-1519638399535-1b036603ac77");
            background-size: cover;
        }
        .welcome-banner {
            font-size: 30px;
            padding: 10px;
            text-align: center;
            background-color: #ffffffcc;
            border-radius: 10px;
            font-weight: bold;
            color: #003366;
        }
        .stAlert > div {
            background-color: #ffffcc !important;
            color: #333 !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- DB SETUP ----------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        username TEXT,
        comment TEXT,
        submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

conn.commit()

if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# ---------------------------- AUTH ----------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Water_icon.svg", width=80)
st.sidebar.title("🌊 Aqua Finance Risk App")

auth_option = st.sidebar.selectbox("🔐 Login / Register", ["Login", "Register"])
username_input = st.sidebar.text_input("👤 Username")
password_input = st.sidebar.text_input("🔒 Password", type="password")

def register_user():
    cursor.execute("SELECT * FROM users WHERE username=?", (username_input,))
    if cursor.fetchone():
        st.sidebar.error("Username already exists.")
    else:
        cursor.execute("INSERT INTO users VALUES (?,?)", (username_input, password_input))
        conn.commit()
        st.sidebar.success("✅ Registered successfully. Please log in.")

def login_user():
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username_input, password_input))
    return cursor.fetchone() is not None

if auth_option == "Register":
    if st.sidebar.button("Register"):
        register_user()
    st.stop()

if not login_user():
    st.sidebar.warning("Login to Continue")
    st.stop()

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if login_user(username, password):  # <-- This checks if login is correct
        st.session_state['username'] = username  # <-- Save username AFTER successful login
        st.success(f"Welcome, {username}!")
    else:
        st.error("Invalid username or password")


# ---------------------------- SIDEBAR NAV ----------------------------
menu = st.sidebar.radio("📁 Navigation", [
    "Upload Data", "Dashboard", "Prediction", "Feedback", "Admin Panel"
])

st.markdown(f"<div class='welcome-banner'>👋 Welcome, {username}!</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------- UTILS ----------------------------
def save_uploaded_file(uploaded_file, name):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        file_path = f"saved_user_data/{username}_{name}.csv"
        df.to_csv(file_path, index=False)
        return df
    elif os.path.exists(f"saved_user_data/{username}_{name}.csv"):
        return pd.read_csv(f"saved_user_data/{username}_{name}.csv")
    return None

# ---------------------------- 1. UPLOAD DATA ----------------------------
if menu == "Upload Data":
    st.header("📤 Upload Your Datasets")

    col1, col2 = st.columns(2)
    with col1:
        risk_file = st.file_uploader("Upload Risk Dataset", type=["csv"], key="risk")
        if risk_file:
            save_uploaded_file(risk_file, "risk")
            st.success("✅ Risk dataset saved.")
    with col2:
        water_file = st.file_uploader("Upload Water Quality Dataset", type=["csv"], key="water")
        if water_file:
            save_uploaded_file(water_file, "water")
            st.success("✅ Water quality dataset saved.")

# ---------------------------- 2. DASHBOARD ----------------------------
elif menu == "Dashboard":
    st.header("📊 Dashboard")

    risk_df = save_uploaded_file(None, "risk")
    water_df = save_uploaded_file(None, "water")

    if risk_df is not None:
        st.subheader("📌 Risk Data Overview")
        st.dataframe(risk_df.head())
        if 'loan_amount' in risk_df.columns and 'default' in risk_df.columns:
            st.bar_chart(risk_df.groupby('default')['loan_amount'].mean())

    if water_df is not None:
        st.subheader("💧 Water Quality Overview")
        st.dataframe(water_df.head())
        if {'pH', 'ammonia', 'dissolved_oxygen'}.issubset(water_df.columns):
            st.line_chart(water_df[['pH', 'ammonia', 'dissolved_oxygen']])

# ---------------------------- 3. PREDICTION ----------------------------
elif menu == "Prediction":
    st.header("🤖 AI-Based Prediction")

    risk_df = save_uploaded_file(None, "risk")
    if risk_df is not None and 'default' in risk_df.columns:
        X = risk_df.drop('default', axis=1).select_dtypes(include='number')
        y = risk_df['default']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier().fit(X_train, y_train)
        preds = model.predict(X_test)

        st.success("✅ Model Trained Successfully!")
        st.text("📄 Classification Report:")
        st.text(classification_report(y_test, preds))
        joblib.dump(model, f"saved_user_data/{username}_risk_model.pkl")
    else:
        st.warning("Please upload a valid Risk dataset with 'default' column.")

# ---------------------------- 4. FEEDBACK ----------------------------
elif menu == "Feedback":
    st.header("📝 Feedback")
    comment = st.text_area("💬 Share your feedback about the app:")
    if st.button("Submit Feedback"):
        cursor.execute("INSERT INTO feedback (username, comment) VALUES (?, ?)", (username, comment))
        conn.commit()
        st.success("✅ Feedback submitted. Thank you!")

# ---------------------------- 5. ADMIN PANEL ----------------------------
elif menu == "Admin Panel":
    if username != "admin":
        st.warning("⛔ Access Denied: Admins Only")
    else:
        st.header("🛠 Admin Panel")

        st.subheader("👥 Registered Users")
        users = pd.read_sql("SELECT * FROM users", conn)
        st.dataframe(users)

        st.subheader("💬 Feedbacks")
        feedbacks = pd.read_sql("SELECT * FROM feedback", conn)
        st.dataframe(feedbacks)
