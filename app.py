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

# Create user data folder if not exists
if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# ---------------------------- CUSTOM CSS ----------------------------
st.markdown("""
    <style>
        body {
            background-image: url('https://img.freepik.com/free-photo/water-texture-background_23-2148964083.jpg');
            background-size: cover;
        }

        [data-testid="stSidebar"] {
            background-image: url('https://img.freepik.com/free-vector/abstract-background-with-water-drops_23-2148402960.jpg');
            background-size: cover;
            color: white;
        }

        input, textarea {
            color: black !important;
        }

        .welcome-banner {
            background-color: rgba(255,255,255,0.85);
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
        }

        h1, h2, h3, h4 {
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/5763/5763132.png", width=80)
st.sidebar.markdown("<h2 style='color:white;'>🌊 Aqua Risk System</h2>", unsafe_allow_html=True)

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
        st.sidebar.success("Registered successfully! Please log in.")

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

# ---------------------------- MAIN ----------------------------

# 🎉 Stylish Welcome Banner
st.markdown(f"""
<div class='welcome-banner'>
    <h2>👋 Welcome, <i>{username}</i>!</h2>
    <p>You're logged in to the Aqua Risk Assessment System.</p>
</div>
""", unsafe_allow_html=True)

# 🧠 Big Banner Heading
st.markdown("""
<div style='
    background-color: rgba(255, 255, 255, 0.85);
    padding: 15px 25px;
    border-radius: 10px;
    margin-top: 10px;
    margin-bottom: 20px;
'>
<h1 style='color: #003366; text-align: center;'>🧠 Aqua Risk System Dashboard</h1>
</div>
""", unsafe_allow_html=True)

section = st.sidebar.radio("📁 Select Section", ["🔴 Risk Assessment", "🔵 Water Quality", "🟢 Combined Analysis"])

# ---------------------------- LOAD DATA ----------------------------
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
    st.header("📊 Loan Default Risk Analysis")
    risk_file = st.file_uploader("Upload Farmer Loan Profile Dataset", type=["csv"], key="risk_upload")
    risk_df = load_data(risk_file, "risk")

    if risk_df is not None:
        st.subheader("📌 Dataset Preview")
        st.dataframe(risk_df.head())

        # Visuals
        if 'loan_amount' in risk_df.columns and 'default' in risk_df.columns:
            fig = px.histogram(risk_df, x="loan_amount", color="default", title="Loan Amount vs Default")
            st.plotly_chart(fig, use_container_width=True)

        if 'default' in risk_df.columns:
            X = risk_df.drop("default", axis=1).select_dtypes(include='number')
            y = risk_df["default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "risk_model.pkl")
            st.success("✅ Loan Risk Model Trained")

# ---------------------------- WATER QUALITY ----------------------------
elif section == "🔵 Water Quality":
    st.header("🌊 Water Quality Monitoring")
    water_file = st.file_uploader("Upload Water Quality Dataset", type=["csv"], key="water_upload")
    water_df = load_data(water_file, "water")

    if water_df is not None:
        st.subheader("📌 Dataset Preview")
        st.dataframe(water_df.head())

        if {'pH', 'temperature', 'ammonia', 'dissolved_oxygen'}.issubset(water_df.columns):
            fig = px.scatter_matrix(water_df, dimensions=['pH', 'temperature', 'ammonia', 'dissolved_oxygen'],
                                    title="Water Quality Parameter Relationships")
            st.plotly_chart(fig, use_container_width=True)

            water_df['risk'] = ((water_df['pH'] < 6.5) |
                                (water_df['ammonia'] > 0.5) |
                                (water_df['dissolved_oxygen'] < 4)).astype(int)

            risky_count = water_df['risk'].sum()
            st.warning(f"⚠️ {risky_count} out of {len(water_df)} entries flagged for poor water quality.")
            st.success("💡 Tip: Low DO and high ammonia can cause fish death. Apply aeration and filtration.")

# ---------------------------- COMBINED ANALYSIS ----------------------------
elif section == "🟢 Combined Analysis":
    st.header("🔗 Integrated Aqua Risk Assessment")
    comb_file1 = st.file_uploader("Upload Risk Data", type=["csv"], key="comb_risk")
    comb_file2 = st.file_uploader("Upload Water Quality Data", type=["csv"], key="comb_water")
    df1 = load_data(comb_file1, "comb_risk")
    df2 = load_data(comb_file2, "comb_water")

    if df1 is not None and df2 is not None:
        st.success("✅ Datasets Loaded Successfully")
        combined_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        st.subheader("📌 Merged Dataset")
        st.dataframe(combined_df.head())

        if 'default' in combined_df.columns:
            X = combined_df.drop("default", axis=1).select_dtypes(include='number')
            y = combined_df["default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.text("📊 Combined Classification Report:")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "combined_model.pkl")

            st.info("📈 Insight: Loan default chances increase with poor water + financial health.")

