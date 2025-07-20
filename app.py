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
st.set_page_config(layout="wide", page_title="AI-Driven Risk Assessment")

# DB Initialization
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()

# Folder for uploaded user data
if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# ---------------------------- CUSTOM STYLING ----------------------------
st.markdown("""
    <style>
        /* Page background */
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1504274066651-8d31a536b11a?auto=format&fit=crop&w=1950&q=80');
            background-size: cover;
        }

        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(135deg, #003366, #005599);
            color: white;
        }

        /* Sidebar labels */
        [data-testid="stSidebar"] label {
            color: #ffffff !important;
            font-weight: 600;
        }

        /* Fix login input text visibility */
        input {
            color: black !important;
            font-weight: bold;
        }

        /* Fix sidebar warning message visibility */
        [data-testid="stSidebar"] .stAlert {
            background-color: rgba(255, 255, 255, 0.9);
            color: black !important;
            border-left: 5px solid orange;
            font-weight: bold;
        }

        /* Buttons */
        .stButton > button {
            background-color: #00A8E8;
            color: white;
            font-weight: bold;
        }

        /* Welcome Banner */
        .welcome-banner {
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.85);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Water_icon.svg", width=80)
st.sidebar.title("🌊 Aqua Finance AI App")

auth_option = st.sidebar.selectbox("🔐 Login / Register", ["Login", "Register"])
username = st.sidebar.text_input("👤 Username")
password = st.sidebar.text_input("🔒 Password", type="password")

# ---------------------------- AUTHENTICATION ----------------------------
def register_user():
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        st.sidebar.error("❌ Username already exists.")
    else:
        cursor.execute("INSERT INTO users VALUES (?,?)", (username, password))
        conn.commit()
        st.sidebar.success("✅ Registration successful! Please login.")

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

# ---------------------------- WELCOME BANNER ----------------------------
st.markdown(f"""
<div class="welcome-banner">
    <h2>👋 Welcome, <span style='color:#003366'>{username}</span>!</h2>
    <p>This is your personalized AI dashboard for Aqua Finance Risk Insights.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------- NAVIGATION ----------------------------
section = st.sidebar.radio("📁 Sections", ["🔴 Risk Assessment", "🔵 Water Quality", "🟢 Combined Analysis"])

# ---------------------------- DATA LOADING FUNCTION ----------------------------
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

# ---------------------------- RISK ASSESSMENT TAB ----------------------------
if section == "🔴 Risk Assessment":
    st.header("📊 Loan Risk Assessment")
    risk_file = st.file_uploader("Upload Risk Dataset", type=["csv"], key="risk_upload")
    risk_df = load_data(risk_file, "risk")

    if risk_df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(risk_df.head())

        if 'loan_amount' in risk_df.columns and 'default' in risk_df.columns:
            st.subheader("Loan Distribution by Default")
            fig = px.histogram(risk_df, x="loan_amount", color="default", title="Loan Amount Distribution")
            st.plotly_chart(fig, use_container_width=True)

        if 'default' in risk_df.columns:
            X = risk_df.drop("default", axis=1).select_dtypes(include='number')
            y = risk_df["default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)

            st.subheader("Model Performance")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "risk_model.pkl")
            st.success("✅ Risk prediction model trained successfully!")

# ---------------------------- WATER QUALITY TAB ----------------------------
elif section == "🔵 Water Quality":
    st.header("🌊 Water Quality Analysis")
    water_file = st.file_uploader("Upload Water Dataset", type=["csv"], key="water_upload")
    water_df = load_data(water_file, "water")

    if water_df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(water_df.head())

        if {'pH', 'temperature', 'ammonia', 'dissolved_oxygen'}.issubset(water_df.columns):
            st.subheader("Correlation Between Parameters")
            fig = px.scatter_matrix(water_df, dimensions=['pH', 'temperature', 'ammonia', 'dissolved_oxygen'],
                                    title="Parameter Correlation")
            st.plotly_chart(fig, use_container_width=True)

            water_df['risk'] = ((water_df['pH'] < 6.5) | 
                                (water_df['ammonia'] > 0.5) | 
                                (water_df['dissolved_oxygen'] < 4)).astype(int)
            risky_count = water_df['risk'].sum()
            st.warning(f"⚠️ {risky_count} out of {len(water_df)} records indicate poor water quality.")
            st.info("💡 Recommendation: Ensure DO > 4, Ammonia < 0.5, pH 6.5–8.5.")

# ---------------------------- COMBINED ANALYSIS TAB ----------------------------
elif section == "🟢 Combined Analysis":
    st.header("🔗 Combined Risk & Water Analysis")
    comb_file1 = st.file_uploader("Upload Risk Dataset", type=["csv"], key="comb_risk")
    comb_file2 = st.file_uploader("Upload Water Quality Dataset", type=["csv"], key="comb_water")

    df1 = load_data(comb_file1, "comb_risk")
    df2 = load_data(comb_file2, "comb_water")

    if df1 is not None and df2 is not None:
        st.success("✅ Both datasets loaded successfully.")
        combined_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        st.dataframe(combined_df.head())

        if 'default' in combined_df.columns:
            X = combined_df.drop("default", axis=1).select_dtypes(include='number')
            y = combined_df["default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)

            st.subheader("Combined Risk Classification Report")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "combined_model.pkl")

            st.info("📊 Farms with both poor financial and water quality indicators are high-risk.")

