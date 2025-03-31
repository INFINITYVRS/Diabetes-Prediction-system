from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load model and scaler
model = joblib.load("trained_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)[0]
    return jsonify({"diabetes": bool(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
