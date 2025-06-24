import streamlit as st
import numpy as np
import pickle

# Load the saved model and scaler
model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Page settings
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")

st.markdown("Enter the patient details below to predict if they are diabetic.")

# Input fields
Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose", min_value=0, step=1)
BloodPressure = st.number_input("Blood Pressure", min_value=0, step=1)
SkinThickness = st.number_input("Skin Thickness", min_value=0, step=1)
Insulin = st.number_input("Insulin", min_value=0, step=1)
BMI = st.number_input("BMI", min_value=0.0, step=0.1, format="%.1f")
DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.001, format="%.3f")
Age = st.number_input("Age", min_value=0, step=1)

# Prediction logic
if st.button("Predict"):
    # Format the input into a NumPy array
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])

    # Standardize the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Show result
    if prediction[0] == 1:
        st.error("🚨 The person is **diabetic**.")
    else:
        st.success("✅ The person is **not diabetic**.")
