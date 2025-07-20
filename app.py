import streamlit as st
import os
import sqlite3
from pathlib import Path

# ----------------------
# Database setup
# ----------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

# ----------------------
# File storage
# ----------------------
Path("saved_user_data").mkdir(exist_ok=True)

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Aqua Risk System", layout="wide")

# ----------------------
# Sidebar Styling
# ----------------------
with st.sidebar:
    st.markdown("""
        <style>
            .css-1d391kg {background-color: #002244;}
            .stSelectbox > div {color: white;}
            .stTextInput > div > div > input {
                background-color: #ffffff10;
                color: white;
                border: 1px solid #ccc;
            }
            .stButton button {
                background-color: #00aaff;
                color: white;
                width: 100%;
                border-radius: 8px;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    st.image("https://cdn-icons-png.flaticon.com/512/4278/4278449.png", width=60)
    st.markdown("<h2 style='color:white;'>🌊 Aqua Risk Intelligence</h2>", unsafe_allow_html=True)

    auth_choice = st.selectbox("Login/Register", ["Login", "Register"])

    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    if auth_choice == "Register":
        if st.button("Create Account"):
            cursor.execute("INSERT OR REPLACE INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            st.success("✅ Account created successfully! Please log in.")
    else:
        if st.button("Login to Continue"):
            cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            result = cursor.fetchone()
            if result:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"✅ Welcome {username}!")
            else:
                st.error("❌ Invalid credentials")

# ----------------------
# Main App Content
# ----------------------
if st.session_state.get("logged_in"):

    st.sidebar.markdown("---")
    section = st.sidebar.radio("📂 Select Module", ["📊 Risk Assessment", "💧 Water Quality", "📈 Combined Insights"])

    st.title(f"{section} Dashboard")

    if section == "📊 Risk Assessment":
        st.subheader("🧠 Predict Loan Default Risk")
        uploaded_file = st.file_uploader("Upload Farmer & Loan Data", type=["csv"], key="loan_data")
        if uploaded_file:
            import pandas as pd
            import plotly.express as px

            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head())

            # Example analysis
            st.plotly_chart(px.histogram(data, x='loan_amount', color='loan_default', title="Loan Amount vs Default"))
            st.info("📌 Consider monitoring farms with high loan amount and low credit score for better default prediction.")
    
    elif section == "💧 Water Quality":
        st.subheader("💧 Analyze Water Indicators for Fish Farm")
        water_file = st.file_uploader("Upload Water Quality Data", type=["csv"], key="water_data")
        if water_file:
            import pandas as pd
            import plotly.express as px

            water_df = pd.read_csv(water_file)
            st.dataframe(water_df.head())

            # Example chart
            st.plotly_chart(px.line(water_df, x='date', y='pH', title="pH Trend Over Time"))
            st.plotly_chart(px.box(water_df, y='ammonia_level', title="Ammonia Level Spread"))

            st.warning("⚠️ pH levels outside the 6.5–8.5 range indicate increased risk to aquatic life.")
            st.success("✅ Stable DO and Ammonia levels indicate good farm health.")
    
    elif section == "📈 Combined Insights":
        st.subheader("📊 Combined Risk Monitoring System")
        col1, col2 = st.columns(2)

        with col1:
            loan_file = st.file_uploader("Loan Data", type="csv", key="combo_loan")
        with col2:
            water_file = st.file_uploader("Water Quality", type="csv", key="combo_water")

        if loan_file and water_file:
            import pandas as pd
            loan_df = pd.read_csv(loan_file)
            water_df = pd.read_csv(water_file)

            st.markdown("### 🧠 AI Recommendations")
            if loan_df['loan_default'].mean() > 0.5:
                st.error("🔴 High default rate! Suggest risk-based loan approval system.")
            if water_df['pH'].mean() < 6.5 or water_df['pH'].mean() > 8.5:
                st.warning("⚠️ Water pH out of optimal range. Recommend immediate field testing.")
            st.success("🧪 Integrating loan data with water quality helps prioritize high-risk farms.")

else:
    st.warning("🔐 Please log in from the sidebar to use the Aqua Risk System.")

