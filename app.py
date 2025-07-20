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

        /* Title heading override */
        h1 {
            color: #ffffff;
            text-shadow: 2px 2px #00000088;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- DATABASE ----------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
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

# ---------------------------- WELCOME BANNER ----------------------------
st.markdown(f"<div class='welcome-banner'>👋 Welcome, {username}! You're logged in.</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------- MAIN APP ----------------------------
section = st.sidebar.radio("📁 Select Section", ["🔴 Risk Assessment", "🔵 Water Quality", "🟢 Combined Analysis"])

st.title("🧠 Aqua Risk System – AI-Driven Insights for Lenders & Insurers")
st.markdown("---")

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

# ---------------------------- RISK TAB ----------------------------
if section == "🔴 Risk Assessment":
    st.header("📊 Farmer Loan Risk Dataset")
    risk_file = st.file_uploader("Upload Risk Dataset", type=["csv"], key="risk_upload")
    risk_df = load_data(risk_file, "risk")

    if risk_df is not None:
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
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "risk_model.pkl")
            st.success("✅ Model Trained Successfully!")

# ---------------------------- WATER QUALITY TAB ----------------------------
elif section == "🔵 Water Quality":
    st.header("🌊 Fish Farm Water Quality")
    water_file = st.file_uploader("Upload Water Quality Dataset", type=["csv"], key="water_upload")
    water_df = load_data(water_file, "water")

    if water_df is not None:
        st.dataframe(water_df.head())

        if {'pH', 'temperature', 'ammonia', 'dissolved_oxygen'}.issubset(water_df.columns):
            fig = px.scatter_matrix(water_df, dimensions=['pH', 'temperature', 'ammonia', 'dissolved_oxygen'],
                                    title="Water Quality Parameter Correlations")
            st.plotly_chart(fig, use_container_width=True)

            water_df['risk'] = ((water_df['pH'] < 6.5) | 
                                (water_df['ammonia'] > 0.5) | 
                                (water_df['dissolved_oxygen'] < 4)).astype(int)

            risky_count = water_df['risk'].sum()
            st.warning(f"⚠️ {risky_count} out of {len(water_df)} entries indicate poor water conditions.")
            st.success("✅ Tip: Regular oxygenation & ammonia checks help lower risk.")

# ---------------------------- COMBINED TAB ----------------------------
elif section == "🟢 Combined Analysis":
    st.header("🔗 Combined Risk + Water Analysis")
    comb_file1 = st.file_uploader("Upload Risk Dataset", type=["csv"], key="comb_risk")
    comb_file2 = st.file_uploader("Upload Water Quality Dataset", type=["csv"], key="comb_water")
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

            st.info("📈 Farms with poor water and financial conditions = highest risk.")
