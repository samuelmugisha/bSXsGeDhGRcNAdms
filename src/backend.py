
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from config import SAVED_MODEL_PATH, REDUCED_FEATURES
from predict import load_model, make_prediction

app = Flask(__name__)

# Load the model once when the application starts
model = load_model(SAVED_MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(force=True)

    # Validate input data structure based on REDUCED_FEATURES
    input_data = {}
    for feature in REDUCED_FEATURES:
        if feature not in data:
            return jsonify({'error': f'Missing feature: {feature}'}), 400
        input_data[feature] = data[feature]

    try:
        predicted_happiness, confidence = make_prediction(input_data, model)
        
        response = {
            'predicted_happiness': int(predicted_happiness),
            'confidence': float(confidence)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # This is for local development. For deployment, use a production-ready WSGI server.
    # To run this from Colab, you might need ngrok or a similar tunneling service
    # or deploy it to a platform like Google Cloud Run.
    app.run(host='0.0.0.0', port=5000)
