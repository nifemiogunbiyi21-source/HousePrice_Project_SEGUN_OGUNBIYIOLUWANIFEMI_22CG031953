from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# FIXED: Use absolute path for model loading (Scorac.com requirement)
# This prevents path-relative issues during deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'house_price_model.pkl')

# Load the trained model at startup
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    print(f"✓ Model loaded successfully from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"✗ ERROR: Model file not found at {MODEL_PATH}")
    print("Please ensure house_price_model.pkl is in the model/ directory")
    model = None
    scaler = None
    feature_names = None
except Exception as e:
    print(f"✗ ERROR loading model: {str(e)}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    if model is None:
        return "Error: Model not loaded. Please check server logs.", 500
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                             features=feature_names or [],
                             error='Model not loaded. Please contact administrator.')
    
    try:
        # Get input data from form
        features = []
        for feature in feature_names:
            value = float(request.form.get(feature, 0))
            features.append(value)
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Convert prediction to actual price (multiply by 100,000)
        predicted_price = prediction * 100000
        
        return render_template('index.html', 
                             features=feature_names,
                             prediction=f'${predicted_price:,.2f}',
                             input_values=dict(zip(feature_names, features)))
    
    except Exception as e:
        return render_template('index.html', 
                             features=feature_names,
                             error=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Extract features in correct order
        features = []
        for feature in feature_names:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            features.append(float(data[feature]))
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Convert prediction to actual price
        predicted_price = prediction * 100000
        
        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'formatted_price': f'${predicted_price:,.2f}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FIXED: Production-safe configuration (no debug=True)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # debug=False for production deployment (Scorac.com requirement)
    app.run(host='0.0.0.0', port=port, debug=False)