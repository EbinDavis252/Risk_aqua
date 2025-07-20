import streamlit as st
import sqlite3
import os
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ---------------------------- PAGE SETUP ----------------------------
st.set_page_config(page_title="AI Aqua Risk System", layout="wide")

# Custom CSS for full background and sidebar
st.markdown("""
    <style>
        /* Full App Background */
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
            background-size: cover;
            background-attachment: fixed;
        }

        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background-image: url("https://images.unsplash.com/photo-1519638399535-1b036603ac77");
            background-size: cover;
            background-position: center;
        }

        /* Make text black inside input fields */
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

        /* Fix sidebar warning message visibility */
[data-testid="stSidebar"] .stAlert {
    background-color: rgba(255, 255, 255, 0.9);
    color: black !important;
    border-left: 5px solid orange;
    font-weight: bold;
        }

        /* Stylish button */
        .stButton > button {
            color: white;
            background-color: #006699;
            border-radius: 10px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
        }

        /* Welcome Banner */
        .welcome-banner {
            font-size: 30px;
            padding: 10px;
            text-align: center;
            background-color: #ffffffcc;
            border-radius: 10px;
            font-weight: bold;
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- DATABASE SETUP ----------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
''')
conn.commit()

# Create upload directory
if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# ---------------------------- SIDEBAR LOGIN ----------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2798/2798007.png", width=80)
st.sidebar.title("🌊 Aqua Risk System")

auth_option = st.sidebar.selectbox("Login/Register", ["Login", "Register"])
username = st.sidebar.text_input("👤 Username")
password = st.sidebar.text_input("🔒 Password", type="password")

def register_user():
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        st.sidebar.error("Username already exists.")
    else:
        cursor.execute("INSERT INTO users VALUES (?, ?)", (username, password))
        conn.commit()
        st.sidebar.success("Registered successfully. Please log in.")

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

# ---------------------------- AFTER LOGIN ----------------------------
st.markdown(f"""<div class="welcome-banner">👋 Welcome, <span style="color:#004488">{username}</span>!</div>""", unsafe_allow_html=True)
section = st.sidebar.radio("📁 Select Section", ["🔴 Risk Assessment", "🔵 Water Quality", "🟢 Combined Analysis"])
st.markdown("---")

# ---------------------------- HELPER FUNCTION ----------------------------
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

# ---------------------------- SECTION 1: RISK ASSESSMENT ----------------------------
if section == "🔴 Risk Assessment":
    st.header("📊 Loan Default Risk Assessment")
    risk_file = st.file_uploader("Upload Farmer Loan Dataset", type=["csv"], key="risk_upload")
    risk_df = load_data(risk_file, "risk")

    if risk_df is not None:
        st.subheader("🔍 Preview")
        st.dataframe(risk_df.head())

        if 'loan_amount' in risk_df.columns and 'default' in risk_df.columns:
            fig = px.histogram(risk_df, x="loan_amount", color="default", title="Loan Amount vs Default")
            st.plotly_chart(fig, use_container_width=True)

        if 'default' in risk_df.columns:
            X = risk_df.drop("default", axis=1).select_dtypes(include='number')
            y = risk_df['default']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.text("📈 Classification Report:")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "risk_model.pkl")
            st.success("✅ Model Trained Successfully!")

# ---------------------------- SECTION 2: WATER QUALITY ----------------------------
elif section == "🔵 Water Quality":
    st.header("🌊 Water Quality Risk Analysis")
    water_file = st.file_uploader("Upload Water Quality Dataset", type=["csv"], key="water_upload")
    water_df = load_data(water_file, "water")

    if water_df is not None:
        st.subheader("🔍 Preview")
        st.dataframe(water_df.head())

        if {'pH', 'temperature', 'ammonia', 'dissolved_oxygen'}.issubset(water_df.columns):
            fig = px.scatter_matrix(water_df, dimensions=['pH', 'temperature', 'ammonia', 'dissolved_oxygen'],
                                    title="Water Parameters Correlation")
            st.plotly_chart(fig, use_container_width=True)

            water_df['risk'] = ((water_df['pH'] < 6.5) | 
                                (water_df['ammonia'] > 0.5) | 
                                (water_df['dissolved_oxygen'] < 4)).astype(int)
            risky = water_df['risk'].sum()
            st.warning(f"⚠️ {risky} of {len(water_df)} samples indicate poor water quality.")
            st.success("💡 Recommendation: Improve aeration and reduce ammonia levels.")

# ---------------------------- SECTION 3: COMBINED ANALYSIS ----------------------------
elif section == "🟢 Combined Analysis":
    st.header("🔗 Combined Risk Analysis")
    comb_file1 = st.file_uploader("Upload Farmer Dataset", type=["csv"], key="comb_risk")
    comb_file2 = st.file_uploader("Upload Water Dataset", type=["csv"], key="comb_water")
    df1 = load_data(comb_file1, "comb_risk")
    df2 = load_data(comb_file2, "comb_water")

    if df1 is not None and df2 is not None:
        st.success("✅ Both datasets loaded.")
        combined_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        st.subheader("🔍 Merged View")
        st.dataframe(combined_df.head())

        if 'default' in combined_df.columns:
            X = combined_df.drop("default", axis=1).select_dtypes(include='number')
            y = combined_df["default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.text("📈 Combined Classification Report:")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "combined_model.pkl")
            st.info("💬 Insight: Risk is highest when water is poor and loan amount is high.")
