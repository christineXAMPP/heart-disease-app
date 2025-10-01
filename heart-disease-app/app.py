import streamlit as st
import pandas as pd
import joblib
import json
import os

# Load metadata
with open("metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "KNN": joblib.load("models/knn.pkl")
}

scaler = joblib.load("models/scaler.pkl")

st.title("💓 Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease.")

# Input form
inputs = {}
for feature in FEATURES:
    if feature in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        inputs[feature] = st.selectbox(f"{feature}", [0, 1, 2, 3])
    else:
        inputs[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([inputs])
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    for name, model in models.items():
        prediction = model.predict(input_scaled)[0]
        st.write(f"**{name}:** {'Disease' if prediction == 1 else 'No Disease'}")
