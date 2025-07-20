import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Sidebar Styling
sidebar_style = """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    padding: 2rem 1rem;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #ffffff;
    font-weight: bold;
}
section[data-testid="stSidebar"] input {
    background-color: #ffffff;
    color: #000000;
    border-radius: 6px;
}
section[data-testid="stSidebar"] button {
    background-color: #1e90ff;
    color: #ffffff;
    font-weight: bold;
    border-radius: 8px;
    margin-top: 0.5rem;
}
section[data-testid="stSidebar"] button:hover {
    background-color: #4682B4;
}
section[data-testid="stSidebar"] .stRadio > div {
    color: #ffcc00;
    font-weight: 500;
}
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# User Database
USERS_FILE = 'users.pkl'

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'wb') as f:
        pickle.dump({}, f)

def load_users():
    with open(USERS_FILE, 'rb') as f:
        return pickle.load(f)

def save_users(users):
    with open(USERS_FILE, 'wb') as f:
        pickle.dump(users, f)

users = load_users()

# Login/Register
st.sidebar.title("🔐 Login / Register")
auth_mode = st.sidebar.selectbox("Select Mode", ["Login", "Register"])
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

login_status = False

if auth_mode == "Register":
    if st.sidebar.button("Register"):
        if username in users:
            st.sidebar.warning("User already exists.")
        else:
            users[username] = password
            save_users(users)
            st.sidebar.success("Registered successfully! Now login.")

elif auth_mode == "Login":
    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.sidebar.success("Logged in successfully!")
            login_status = True
        else:
            st.sidebar.error("Invalid credentials.")

# Proceed if logged in
if login_status:
    st.sidebar.markdown("---")
    section = st.sidebar.radio("📊 Select Section", 
        ["📉 Risk Assessment", "🌊 Water Quality", "🧪 Combined Analysis"])

    # Create user directory
    user_dir = f"user_data/{username}"
    os.makedirs(user_dir, exist_ok=True)

    st.title("💼 Aqua Loan Risk Management Dashboard")

    def train_model(data, features, target):
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return model, acc, classification_report(y_test, y_pred, output_dict=True)

    if section == "📉 Risk Assessment":
        st.subheader("Upload Risk Assessment Dataset")
        risk_file = st.file_uploader("Upload CSV for Loan Risk", type=['csv'], key="risk")
        if risk_file:
            loan_df = pd.read_csv(risk_file)
            loan_df.to_csv(f"{user_dir}/loan_data.csv", index=False)
            st.success("File uploaded and saved!")
            st.write(loan_df.head())

            try:
                features = ['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 'age', 'credit_score', 'income']
                model, acc, report = train_model(loan_df, features, 'default')
                st.write("✅ Model Accuracy:", acc)
                st.json(report)
            except Exception as e:
                st.error(f"Error: {e}")

    elif section == "🌊 Water Quality":
        st.subheader("Upload Water Quality Dataset")
        water_file = st.file_uploader("Upload CSV for Water Quality", type=['csv'], key="water")
        if water_file:
            water_df = pd.read_csv(water_file)
            water_df.to_csv(f"{user_dir}/water_data.csv", index=False)
            st.success("File uploaded and saved!")
            st.write(water_df.head())

            try:
                features = ['temp', 'ph', 'do', 'ammonia']
                model, acc, report = train_model(water_df, features, 'mortality')
                st.write("✅ Model Accuracy:", acc)
                st.json(report)
            except Exception as e:
                st.error(f"Error: {e}")

    elif section == "🧪 Combined Analysis":
        st.subheader("Upload Combined Dataset")
        combined_file = st.file_uploader("Upload Combined Dataset", type=['csv'], key="combined")
        if combined_file:
            combined_df = pd.read_csv(combined_file)
            combined_df.to_csv(f"{user_dir}/combined_data.csv", index=False)
            st.success("File uploaded and saved!")
            st.write(combined_df.head())

            try:
                features = ['loan_amount', 'loan_term', 'emi_paid', 'emi_missed', 
                            'age', 'credit_score', 'income', 'temp', 'ph', 'do', 'ammonia']
                model, acc, report = train_model(combined_df, features, 'default')
                st.write("✅ Combined Model Accuracy:", acc)
                st.json(report)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.warning("Please login to continue.")
