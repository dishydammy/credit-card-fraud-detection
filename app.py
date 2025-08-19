import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = joblib.load('lgb_model.pkl')
    return model

model = load_model()

# Define feature names based on the dataset description
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# App title and description
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it's fraudulent.")

# Create input fields for each feature
st.header("Transaction Details")

# Using a form for better user experience and to control when prediction runs
with st.form("prediction_form"):
    user_inputs = {}
    cols = st.columns(2) # Create two columns for a more compact layout
    for i, feature in enumerate(feature_names):
        # Alternate columns for inputs
        with cols[i % 2]:
            # Use number_input for numerical features. Adjust format or step if needed.
            user_inputs[feature] = st.number_input(label=feature, format="%.6f", step=0.01)
            
    # Submit button for the form
    submitted = st.form_submit_button("Predict")

# Perform prediction if the form is submitted
if submitted:
    # Convert user inputs into a DataFrame
    input_data = pd.DataFrame([user_inputs])
    
    # Apply log transformation to the 'Amount' feature
    # np.log1p is used for log(1 + x) to handle zero values safely
    input_data['Amount'] = np.log1p(input_data['Amount'])
    
    # Make prediction
    # Assuming the model's predict method returns 0 (Not Fraud) or 1 (Fraud)
    prediction = model.predict(input_data)[0]
    
    # Display the result
    st.header("Prediction Result")
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")