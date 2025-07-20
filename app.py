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

# Initialize DB
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()

# Directory for saving data
if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# ---------------------------- SIDEBAR ----------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #003366;
        }
        .sidebar-text input, .sidebar-text label {
            color: white !important;
            font-family: 'Arial Rounded MT Bold', sans-serif;
        }
        div[data-testid="stSidebar"] span, div[data-testid="stSidebar"] label {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Water_icon.svg", width=80)
st.sidebar.title("🌊 Aqua Finance Risk App")

auth_option = st.sidebar.selectbox("🔐 Login / Register", ["Login", "Register"])
username = st.sidebar.text_input("👤 Username", key="username")
password = st.sidebar.text_input("🔒 Password", type="password", key="password")

# ---------------------------- AUTHENTICATION ----------------------------
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

# ---------------------------- MAIN APP ----------------------------

section = st.sidebar.radio("📁 Select Section", ["🔴 Risk Assessment", "🔵 Water Quality", "🟢 Combined Analysis"])

st.title("🧠 AI-Driven Risk Assessment System for Aqua Loan Providers")
st.markdown("---")

# ---------------------------- DATA LOADER ----------------------------

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

# ---------------------------- RISK ASSESSMENT ----------------------------
if section == "🔴 Risk Assessment":
    st.header("📊 Risk Assessment Dataset")
    risk_file = st.file_uploader("Upload Risk Dataset", type=["csv"], key="risk_upload")
    risk_df = load_data(risk_file, "risk")

    if risk_df is not None:
        st.dataframe(risk_df.head())

        # Visualization
        if 'loan_amount' in risk_df.columns and 'default' in risk_df.columns:
            fig = px.histogram(risk_df, x="loan_amount", color="default", title="Loan Amount Distribution by Default")
            st.plotly_chart(fig, use_container_width=True)

        # Model Training
        if 'default' in risk_df.columns:
            X = risk_df.drop("default", axis=1).select_dtypes(include='number')
            y = risk_df['default']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "risk_model.pkl")

            st.success("✅ Model Trained Successfully!")
            st.info("💡 Tip: High risk loans often correlate with high loan amounts or low income levels.")

# ---------------------------- WATER QUALITY ----------------------------
elif section == "🔵 Water Quality":
    st.header("🌊 Water Quality Analysis")
    water_file = st.file_uploader("Upload Water Quality Dataset", type=["csv"], key="water_upload")
    water_df = load_data(water_file, "water")

    if water_df is not None:
        st.dataframe(water_df.head())

        if {'pH', 'temperature', 'ammonia', 'dissolved_oxygen'}.issubset(water_df.columns):
            fig = px.scatter_matrix(water_df, dimensions=['pH', 'temperature', 'ammonia', 'dissolved_oxygen'],
                                    title="Water Quality Parameters Correlation")
            st.plotly_chart(fig, use_container_width=True)

            # Early Warning System
            water_df['risk'] = ((water_df['pH'] < 6.5) | 
                                (water_df['ammonia'] > 0.5) | 
                                (water_df['dissolved_oxygen'] < 4)).astype(int)

            risky_count = water_df['risk'].sum()
            st.warning(f"⚠️ {risky_count} out of {len(water_df)} entries have poor water quality.")
            st.success("✅ AI Advice: Add aeration and test for nitrogen compounds regularly.")

# ---------------------------- COMBINED ANALYSIS ----------------------------
elif section == "🟢 Combined Analysis":
    st.header("🔗 Combined Risk & Water Analysis")
    comb_file1 = st.file_uploader("Upload Risk Data", type=["csv"], key="comb_risk")
    comb_file2 = st.file_uploader("Upload Water Quality Data", type=["csv"], key="comb_water")
    df1 = load_data(comb_file1, "comb_risk")
    df2 = load_data(comb_file2, "comb_water")

    if df1 is not None and df2 is not None:
        st.success("✅ Both datasets loaded.")
        combined_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        st.dataframe(combined_df.head())

        if 'default' in combined_df.columns:
            X = combined_df.drop("default", axis=1).select_dtypes(include='number')
            y = combined_df["default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.text("Combined Classification Report:")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "combined_model.pkl")

            st.info("📈 Recommendation: Farms with both low water quality and poor financial metrics are at high risk.")
