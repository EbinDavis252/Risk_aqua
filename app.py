
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
st.set_page_config(layout="wide", page_title="Aqua Risk System")

# Apply Custom Style
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
            background-position: center;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: black !important;
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #333;
            border-radius: 5px;
            font-weight: bold;
        }

        [data-testid="stSidebar"] input::placeholder {
            color: #666666 !important;
        }

        [data-testid="stSidebar"] label {
            color: #ffffff !important;
            font-weight: 600;
        }

        .stButton > button {
            color: white;
            background-color: #006699;
            border-radius: 10px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
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

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] .css-10trblm {
            color: #ffffff !important;
            text-shadow: 1px 1px 3px black;
        }

        .stAlert > div {
            background-color: #ffffcc !important;
            color: #333 !important;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- DATABASE ----------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Water_icon.svg", width=80)
st.sidebar.title("🌊 Aqua Finance Risk App")
auth_option = st.sidebar.selectbox("🔐 Login / Register", ["Login", "Register"])
username = st.sidebar.text_input("👤 Username", key="username")
password = st.sidebar.text_input("🔒 Password", type="password", key="password")

# ---------------------------- AUTH ----------------------------
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
else:
    st.session_state['username'] = username

# ---------------------------- WELCOME BANNER ----------------------------
st.markdown(f"<div class='welcome-banner'>👋 Welcome, {username}! You're logged in.</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------- MAIN NAV ----------------------------
section = st.sidebar.radio("📁 Select Section", [
    "📤 Upload Data", "📊 Dashboard", "🤖 Prediction", "📝 Feedback", "🛡️ Admin Panel"
])

# ---------------------------- DATA LOADING UTILITY ----------------------------
def load_data(file, name):
    if file:
        df = pd.read_csv(file)
        path = f"saved_user_data/{username}_{name}.csv"
        df.to_csv(path, index=False)
        return df
    elif os.path.exists(f"saved_user_data/{username}_{name}.csv"):
        return pd.read_csv(f"saved_user_data/{username}_{name}.csv")
    return None

# ---------------------------- SECTION: UPLOAD DATA ----------------------------
if section == "📤 Upload Data":
    st.subheader("📤 Upload Risk Dataset")
    risk_file = st.file_uploader("Upload Risk Dataset", type=["csv"], key="upload_risk")
    if risk_file:
        df = load_data(risk_file, "risk")
        st.success("✅ Risk dataset uploaded and saved.")
        st.dataframe(df.head())

    st.subheader("📤 Upload Water Quality Dataset")
    water_file = st.file_uploader("Upload Water Dataset", type=["csv"], key="upload_water")
    if water_file:
        df = load_data(water_file, "water")
        st.success("✅ Water dataset uploaded and saved.")
        st.dataframe(df.head())

# ---------------------------- SECTION: DASHBOARD ----------------------------
elif section == "📊 Dashboard":
    st.subheader("📊 Risk Dataset Overview")
    df_risk = load_data(None, "risk")
    if df_risk is not None:
        st.dataframe(df_risk.head())
        if 'loan_amount' in df_risk.columns and 'default' in df_risk.columns:
            fig = px.histogram(df_risk, x='loan_amount', color='default', title="Loan Amount vs Default")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌊 Water Quality Overview")
    df_water = load_data(None, "water")
    if df_water is not None:
        st.dataframe(df_water.head())
        if {'pH', 'temperature', 'ammonia', 'dissolved_oxygen'}.issubset(df_water.columns):
            fig = px.scatter_matrix(df_water, dimensions=['pH', 'temperature', 'ammonia', 'dissolved_oxygen'],
                                    title="Water Quality Correlation")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------- SECTION: PREDICTION ----------------------------
elif section == "🤖 Prediction":
    df_risk = load_data(None, "risk")
    df_water = load_data(None, "water")
    if df_risk is not None and df_water is not None:
        combined = pd.concat([df_risk.reset_index(drop=True), df_water.reset_index(drop=True)], axis=1)
        if 'default' in combined.columns:
            X = combined.drop("default", axis=1).select_dtypes(include='number')
            y = combined["default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "combined_model.pkl")
            st.success("✅ Combined Risk Model Trained Successfully.")

# ---------------------------- SECTION: FEEDBACK ----------------------------
elif section == "📝 Feedback":
    st.subheader("📝 Share your Feedback")
    feedback = st.text_area("💬 Your feedback")
    rating = st.slider("⭐ Rating (1-5)", 1, 5, 3)
    if st.button("Submit Feedback"):
        with sqlite3.connect("aqua_risk.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedbacks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    feedback TEXT,
                    rating INTEGER,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute("INSERT INTO feedbacks (username, feedback, rating) VALUES (?, ?, ?)", (username, feedback, rating))
            conn.commit()
            st.success("✅ Thank you for your feedback!")

# ---------------------------- SECTION: ADMIN PANEL ----------------------------
elif section == "🛡️ Admin Panel":
    st.header("🛡️ Admin Dashboard")
    if st.session_state.get('username') != "admin":
        st.error("🚫 Access Denied. Admins only.")
        st.stop()
    st.success("✅ Welcome Admin! You have access to all user data and feedback.")
    st.subheader("👥 Registered Users")
    with sqlite3.connect("users.db") as conn:
        user_df = pd.read_sql_query("SELECT username FROM users", conn)
        st.dataframe(user_df)
    st.subheader("📝 User Feedback")
    with sqlite3.connect("aqua_risk.db") as conn:
        feedback_df = pd.read_sql_query("SELECT * FROM feedbacks ORDER BY submitted_at DESC", conn)
        st.dataframe(feedback_df)
