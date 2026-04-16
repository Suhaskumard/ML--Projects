import streamlit as st
import numpy as np
import pickle

st.title("📊 Customer Churn Prediction")

model = pickle.load(open("model.pkl", "rb"))

tenure = st.slider("Tenure (months)", 0, 72)
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")
contract = st.selectbox("Contract (0=Month,1=OneYear,2=TwoYear)", [0,1,2])

if st.button("Predict"):
    avg = total / (tenure + 1)

    data = np.array([[tenure, monthly, total, contract, avg]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Customer will CHURN ❌")
    else:
        st.success("Customer will STAY ✅")

