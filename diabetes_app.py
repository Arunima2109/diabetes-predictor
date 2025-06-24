import streamlit as st
import numpy as np
import pickle

# Load saved model and scaler
model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Page settings
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Predict if a person is diabetic using health parameters.")
st.markdown("___")

# Input layout: 2 columns
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0)
    SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0)
    BMI = st.number_input("BMI", min_value=0.0, step=0.1, format="%.1f")

with col2:
    Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0)
    Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
    DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    Age = st.number_input("Age", min_value=1)

st.markdown("___")

# Predict button
if st.button("🔍 Predict"):
    # Prepare the input
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    # Output
    if prediction[0] == 1:
        st.error("🚨 The person is **diabetic**.")
        # Ask for location and suggest doctors
        location = st.text_input("📍 Enter your city to find nearby doctors")

        if location:
            st.markdown(
                f"🔎 [Search for diabetologists in {location}](https://www.google.com/search?q=diabetologist+in+{location})",
                unsafe_allow_html=True
            )

    else:
        st.success("✅ The person is **not diabetic**.")

# Optional info
with st.expander("ℹ️ About this app"):
    st.write("""
        - This app uses an SVM model trained on the PIMA Indians Diabetes Dataset.
        - Inputs are standardized using the same scaler from training.
        - The prediction is a machine learning estimate, not a medical diagnosis.
        - If diabetic, please consult a certified doctor for confirmation and treatment.
    """)
