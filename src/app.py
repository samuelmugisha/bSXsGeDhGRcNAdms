
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests # Import the requests library

# Import from custom modules (config and predict are no longer directly used for prediction in app.py)
from config import REDUCED_FEATURES

# Define the backend URL (assuming Flask runs on port 5000 within the same container)
BACKEND_URL = "http://127.0.0.1:5000/predict"

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
    try:
        # Make a POST request to the Flask backend
        response = requests.post(BACKEND_URL, json=input_data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        predicted_happiness = result['predicted_happiness']
        confidence = result['confidence']

        st.header("Prediction Result:")
        if predicted_happiness == 1:
            st.success("The customer is predicted to be **Happy**! 😊")
            st.write(f"Confidence: {confidence*100:.2f}%")
        elif predicted_happiness == 0:
            st.error("The customer is predicted to be **Unhappy** 😣")
            st.write(f"Confidence: {confidence*100:.2f}%")
        else:
            st.warning("Could not make a prediction.")

        st.markdown("--- ")
        st.subheader("Input Values:")
        st.write(pd.DataFrame([input_data]))
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend server. Please ensure the Flask backend is running.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
