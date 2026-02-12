from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        features = request.form['feature']
        features = features.split(',')

        features = [x.strip() for x in features if x.strip() != ""]

        np_features = np.array(features, dtype=float).reshape(1, -1)

        expected_features = model.n_features_in_
        if np_features.shape[1] != expected_features:
            return render_template(
                'index.html',
                message=f"Error: Model expects {expected_features} features, but got {np_features.shape[1]}"
            )

        scaled_features = scaler.transform(np_features)

        prediction = model.predict(scaled_features)

        if prediction[0] == 1:
            message = "Cancerous"
        else:
            message = "Not Cancerous"

        return render_template('index.html', message=message)

    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
