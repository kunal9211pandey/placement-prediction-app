import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title
st.title("Placement Prediction App")

st.write("Enter CGPA and IQ to check placement prediction")

# User input
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ", min_value=0, max_value=300, step=1)

# Predict button
if st.button("Predict"):
    # Convert input to array
    input_data = np.array([[cgpa, iq]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)[0]
    
    # Output
    if prediction == 1:
        st.success("Placed")
    else:
        st.error("Not Placed")
