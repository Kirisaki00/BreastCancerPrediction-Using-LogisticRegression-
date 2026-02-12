from flask import Flask, render_template, request
import numpy as np
import pickle

# -----------------------------
# Load trained model and scaler
# -----------------------------
model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Home Route
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -----------------------------
# Prediction Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input
        features = request.form['feature']
        features = features.split(',')

        # Clean input (remove spaces & empty values)
        features = [x.strip() for x in features if x.strip() != ""]

        # Convert to numpy
        np_features = np.array(features, dtype=float).reshape(1, -1)

        # Check correct number of features
        expected_features = model.n_features_in_
        if np_features.shape[1] != expected_features:
            return render_template(
                'index.html',
                message=f"Error: Model expects {expected_features} features, but got {np_features.shape[1]}"
            )

        # Scale input
        scaled_features = scaler.transform(np_features)

        # Predict
        prediction = model.predict(scaled_features)

        # Output message
        if prediction[0] == 1:
            message = "Cancerous"
        else:
            message = "Not Cancerous"

        return render_template('index.html', message=message)

    except Exception as e:
        # Show actual error for debugging
        return render_template('index.html', message=f"Error: {str(e)}")

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
