import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    model = joblib.load('lgb_model.pkl')
    return model

model = load_model()

feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it's fraudulent.")

st.header("Transaction Details")

with st.form("prediction_form"):
    user_inputs = {}
    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            user_inputs[feature] = st.number_input(label=feature, format="%.6f", step=0.01)
            
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([user_inputs])
    
    input_data['Amount'] = np.log1p(input_data['Amount'])
    
    prediction = model.predict(input_data)[0]
    
    st.header("Prediction Result")
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")