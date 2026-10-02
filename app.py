import streamlit as st
import pandas as pd
import joblib

# ---------- 1) Page ka title aur description ----------
st.set_page_config(page_title="Bank Transaction Predictor", page_icon="🏦")

st.title("🏦 Bank Transaction Predictor")
st.write(
    "Ye app ek bank transaction ki details dekh kar andaza lagata hai "
    "ke transaction **Withdrawal** hai ya nahi (nahi hai to Deposit hai). "
    "Neeche details bharein aur **Predict** dabayein."
)

# ---------- 2) Saved model load karna ----------
@st.cache_resource
def load_model():
    return joblib.load("bank_transaction.pkl")

model = load_model()

# ---------- 3) User se inputs lena ----------
st.subheader("Transaction ki details")

amount = st.number_input(
    "Amount (Rs.)", min_value=0, max_value=200000, value=30000, step=500
)

tx_date = st.date_input("Transaction date")

account_id = st.selectbox("Account ID", [f"A{i:03d}" for i in range(1, 11)])

branch = st.selectbox("Branch", ["Karachi", "Lahore", "Islamabad"])

account_type = st.selectbox("Account Type", ["Saving", "Current"])

# ---------- 4) Predict button ----------
if st.button("Predict"):
    # Model ko bilkul wahi columns chahiye jo training mein the
    new_data = pd.DataFrame([{
        "Amount": amount,
        "Month": tx_date.month,
        "Day": tx_date.day,
        "DayOfWeek": tx_date.weekday(),     # 0 = Monday ... 6 = Sunday
        "Account_ID": account_id,
        "Branch": branch,
        "Account_Type": account_type,
    }])

    prediction = model.predict(new_data)[0]
    withdrawal_chance = model.predict_proba(new_data)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error("Withdrawal predicted?  **YES** (ye Withdrawal hai)")
    else:
        st.success("Withdrawal predicted?  **NO** (ye Deposit hai)")

    # Bonus: confidence
    st.write(f"Withdrawal ka chance: **{withdrawal_chance:.0%}**")
    st.write(f"Deposit ka chance: **{1 - withdrawal_chance:.0%}**")
    st.progress(float(withdrawal_chance))