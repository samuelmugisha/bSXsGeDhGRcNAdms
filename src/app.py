
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Import from custom modules (updated to relative imports)
from .config import SAVED_MODEL_PATH, REDUCED_FEATURES
from .predict import load_model, make_prediction

# Load the trained model using the function from predict.py
@st.cache_resource
def get_loaded_model():
    return load_model(SAVED_MODEL_PATH)

loaded_model = get_loaded_model()

st.set_page_config(page_title="ACME Happiness Predictor", layout="centered")

st.title("Customer Happiness Predictor 📈")
st.markdown("This application predicts customer happiness based on survey responses.")
st.markdown("--- ")

st.header("Please provide the customer's survey responses:")

# Input fields for the features (using REDUCED_FEATURES from config)
X1 = st.slider("X1: My order was delivered on time (1-5)", 1, 5, 3)
X2 = st.slider("X2: Contents of my order was as I expected (1-5)", 1, 5, 3)
X3 = st.slider("X3: I ordered everything I wanted to order (1-5)", 1, 5, 3)
X5 = st.slider("X5: I am satisfied with my courier (1-5)", 1, 5, 3)

# Prepare input data as a dictionary matching REDUCED_FEATURES
input_data = {
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X5': X5
}

st.markdown("--- ")

if st.button("Predict Happiness"):
    if loaded_model:
        predicted_happiness, confidence = make_prediction(input_data, loaded_model)

        st.header("Prediction Result:")
        if predicted_happiness == 1:
            st.success("The customer is predicted to be **Happy**! 😊")
            st.write(f"Confidence: {confidence*100:.2f}%")
        elif predicted_happiness == 0:
            st.error("The customer is predicted to be **Unhappy** 😔")
            st.write(f"Confidence: {(1-confidence)*100:.2f}%") # Assuming confidence is for the predicted class
        else:
            st.warning("Could not make a prediction.")

        st.markdown("--- ")
        st.subheader("Input Values:")
        st.write(pd.DataFrame([input_data]))
    else:
        st.error("Model could not be loaded. Please ensure the model file exists and the training pipeline was run.")
