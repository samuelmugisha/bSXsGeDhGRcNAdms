
import joblib
import pandas as pd
import numpy as np
from .config import SAVED_MODEL_PATH, REDUCED_FEATURES

def load_model(model_path=SAVED_MODEL_PATH):
    """
    Loads the pre-trained machine learning model.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None

def make_prediction(input_data: dict, model=None):
    """
    Makes a prediction using the loaded model.

    Args:
        input_data (dict): A dictionary containing customer survey responses for REDUCED_FEATURES.
                           Example: {'X1': 4, 'X2': 3, 'X3': 4, 'X5': 5}
        model: The trained model object. If None, the model will be loaded.

    Returns:
        tuple: A tuple containing:
               - int: The predicted happiness (0 for unhappy, 1 for happy).
               - float: The confidence score for the predicted class.
    """
    if model is None:
        model = load_model()
        if model is None:
            return -1, 0.0 # Indicate error

    # Convert input data to DataFrame, ensuring correct feature order
    input_df = pd.DataFrame([input_data], columns=REDUCED_FEATURES)

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    if prediction == 1:
        confidence = prediction_proba[1]
    else:
        confidence = prediction_proba[0]

    return int(prediction), float(confidence)

if __name__ == '__main__':
    # Example usage:
    # Ensure a model is trained and saved before running this directly.
    print(f"Loading model from: {SAVED_MODEL_PATH}")
    loaded_model = load_model()

    if loaded_model:
        # Example customer data
        customer_data = {'X1': 5, 'X2': 4, 'X3': 3, 'X5': 5}
        predicted_happiness, confidence = make_prediction(customer_data, loaded_model)

        if predicted_happiness == 1:
            print(f"\nCustomer is predicted to be Happy with {confidence*100:.2f}% confidence.")
        elif predicted_happiness == 0:
            print(f"\nCustomer is predicted to be Unhappy with {confidence*100:.2f}% confidence.")
        else:
            print("\nAn error occurred during prediction.")

        # Another example
        customer_data_2 = {'X1': 2, 'X2': 1, 'X3': 2, 'X5': 1}
        predicted_happiness_2, confidence_2 = make_prediction(customer_data_2, loaded_model)

        if predicted_happiness_2 == 1:
            print(f"\nCustomer 2 is predicted to be Happy with {confidence_2*100:.2f}% confidence.")
        elif predicted_happiness_2 == 0:
            print(f"\nCustomer 2 is predicted to be Unhappy with {confidence_2*100:.2f}% confidence.")
        else:
            print("\nAn error occurred during prediction for customer 2.")
