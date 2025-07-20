import streamlit as st
import sqlite3
import os
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ---------------------------- CONFIG ----------------------------
st.set_page_config(page_title="AI Aqua Risk System", layout="wide")

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

# ---------------------------- FOLDER SETUP ----------------------------
if not os.path.exists("saved_user_data"):
    os.makedirs("saved_user_data")

# ---------------------------- SIDEBAR STYLE ----------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #002244;
        }
        .css-10trblm, .st-bz, .st-cz, .st-dz, .st-em {
            color: white !important;
            font-weight: 500;
        }
        input, label, .stTextInput label, .stSelectbox label {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Water_icon.svg", width=80)
st.sidebar.title("🌊 Aqua Risk System")

# ---------------------------- AUTH ----------------------------
auth_option = st.sidebar.selectbox("Login/Register", ["Login", "Register"])
username = st.sidebar.text_input("👤 Username", key="username")
password = st.sidebar.text_input("🔒 Password", type="password", key="password")

def register_user():
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone():
        st.sidebar.error("Username already exists.")
    else:
        cursor.execute("INSERT INTO users VALUES (?,?)", (username, password))
        conn.commit()
        st.sidebar.success("Registered! Please log in.")

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

# ---------------------------- PAGE SELECT ----------------------------
section = st.sidebar.radio("📁 Select Tab", ["🐟 Farm Risk", "💰 Loan Default", "🔗 Combined Analysis"])

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

# ---------------------------- FARM RISK TAB ----------------------------
if section == "🐟 Farm Risk":
    st.title("🐟 Fish Farm Failure Risk (Water Quality Based)")
    water_file = st.file_uploader("Upload Water Quality CSV", type=["csv"], key="farm_file")
    df = load_data(water_file, "farm_risk")

    if df is not None:
        st.subheader("📋 Uploaded Water Quality Data")
        st.dataframe(df.head())

        if {'temperature', 'pH', 'ammonia', 'dissolved_oxygen', 'mortality'}.issubset(df.columns):
            fig = px.scatter_matrix(df, dimensions=['temperature', 'pH', 'ammonia', 'dissolved_oxygen'],
                                    title="Water Quality Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

            df['risk'] = ((df['pH'] < 6.5) | (df['ammonia'] > 0.5) | (df['dissolved_oxygen'] < 4)).astype(int)
            risk_count = df['risk'].sum()

            st.warning(f"⚠️ {risk_count} out of {len(df)} farms show poor water conditions.")
            st.success("✅ Recommendation: Monitor pH and ammonia weekly. Add aeration to boost DO.")

# ---------------------------- LOAN DEFAULT TAB ----------------------------
elif section == "💰 Loan Default":
    st.title("💰 Loan Default Risk (Farmer & Farm Profile Based)")
    profile_file = st.file_uploader("Upload Farmer Profile CSV", type=["csv"], key="loan_file")
    df = load_data(profile_file, "loan_risk")

    if df is not None:
        st.subheader("📋 Uploaded Farmer Loan Data")
        st.dataframe(df.head())

        if 'default' in df.columns:
            if 'loan_amount' in df.columns:
                fig = px.histogram(df, x='loan_amount', color='default', barmode='overlay',
                                   title="Loan Amount Distribution by Default Status")
                st.plotly_chart(fig, use_container_width=True)

            X = df.drop("default", axis=1).select_dtypes(include='number')
            y = df['default']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)
            st.subheader("📊 Loan Default Classification Report")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "loan_model.pkl")

            st.info("📌 Tip: Higher risk is correlated with high loan amount & low income levels.")

# ---------------------------- COMBINED TAB ----------------------------
elif section == "🔗 Combined Analysis":
    st.title("🔗 Combined Water & Loan Risk Assessment")
    comb1 = st.file_uploader("Upload Farmer Loan Data", type=["csv"], key="c1")
    comb2 = st.file_uploader("Upload Water Quality Data", type=["csv"], key="c2")

    df1 = load_data(comb1, "combined_loan")
    df2 = load_data(comb2, "combined_water")

    if df1 is not None and df2 is not None:
        st.success("✅ Both datasets loaded and merged.")
        combined_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        st.dataframe(combined_df.head())

        if 'default' in combined_df.columns:
            X = combined_df.drop("default", axis=1).select_dtypes(include='number')
            y = combined_df["default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier().fit(X_train, y_train)
            preds = model.predict(X_test)

            st.subheader("📈 Combined Risk Prediction Report")
            st.text(classification_report(y_test, preds))
            joblib.dump(model, "combined_model.pkl")

            st.info("📌 Insight: Farms with poor water quality and low income have the highest default risk.")
            st.warning("⚠️ Consider regular quality monitoring and farm audits for high-risk borrowers.")
