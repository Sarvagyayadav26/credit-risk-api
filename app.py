from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model
with open('credit_risk_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])   # Will only accept POST requests.
def predict():
    try:
        data = request.get_json(force=True)         # read JSON from request
        print("✅ Received:", data)
        features = np.array(data['features']).reshape(1, -1)    #Converts the array from 1D to 2D → required shape for prediction.
        prediction = model.predict(features)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({'error': str(e)})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use dynamic port from Render
    app.run(host="0.0.0.0", port=port)
    
#test