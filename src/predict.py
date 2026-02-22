
import joblib
import pandas as pd
import numpy as np

def load_model(path):
    """
    Loads the trained model from the specified path.
    """
    return joblib.load(path)

def make_prediction(input_data, model):
    """
    Makes a prediction using the loaded model.
    input_data: dictionary of feature values
    model: loaded scikit-learn model
    Returns: predicted_class (0 or 1), probability_of_positive_class
    """
    # Ensure input data has the correct feature order and structure
    # For Random Forest, we need a 2D array
    features = [input_data[feature] for feature in ['X1', 'X2', 'X3', 'X5']]
    input_df = pd.DataFrame([features], columns=['X1', 'X2', 'X3', 'X5'])

    # Predict probabilities for the classes
    probabilities = model.predict_proba(input_df)[0]
    predicted_class = model.predict(input_df)[0]

    # Return the predicted class and the probability of the positive class (1)
    return predicted_class, probabilities[1]
