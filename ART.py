import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('viral_load_model.pkl')

# Title of the web app
st.title("Viral Load Suppression Predictor")

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=120)
sex = st.selectbox("Sex", options=["Male", "Female"])
weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0)
height = st.number_input("Height (m)", min_value=0.0, max_value=3.0)
who_stage = st.selectbox("WHO Stage", options=["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
current_regimen = st.selectbox("Current ART Regimen", options=["Regimen A", "Regimen B", "Regimen C"])

# Button to make prediction
if st.button("Predict"):
    # Convert inputs to the required format
    input_data = np.array([[age, 1 if sex == "Male" else 0, weight, height, 
                             who_stage, current_regimen]])
# Assuming input_data is a DataFrame
input_data = pd.get_dummies(input_data, columns=['stage'], drop_first=True)
  
    # Make prediction
prediction = model.predict(input_data)

    # Display the prediction
if prediction[0] == 1:
        st.success("Viral Load is Suppressed (LDL)")
else:
        st.error("Viral Load is Detectable")
