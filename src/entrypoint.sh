#!/bin/bash

# Start the Flask backend in the background
gunicorn --workers 2 --bind 0.0.0.0:5000 backend:app &

# Wait for a few seconds to allow the backend to start
sleep 5

# Start the Streamlit frontend in the foreground
streamlit run app.py --server.port 7860 --server.enableCORS false --server.enableXsrfProtection false
