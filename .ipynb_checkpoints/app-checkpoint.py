from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

pipeline = joblib.load('model_pipeline.pkl')
FEATURE_ORDER = joblib.load('feature_order.pkl')

@app.route('/')
def home():
    return "Student Disengagement Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    features = [data[f] for f in FEATURE_ORDER]
    x = np.array(features).reshape(1, -1)
    pred = pipeline.predict(x)[0]
    risk = pipeline.predict_proba(x)[0][1]
    return jsonify({'prediction': int(pred), 'risk_score': float(risk)})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    records = request.json
    if not isinstance(records, list):
        return jsonify({'error': 'Input should be a list'}), 400

    results = []
    for record in records:
        missing = [f for f in FEATURE_ORDER if f not in record]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400
        features = [record[f] for f in FEATURE_ORDER]
        x = np.array(features).reshape(1, -1)
        pred = pipeline.predict(x)[0]
        risk = pipeline.predict_proba(x)[0][1]
        results.append({'prediction': int(pred), 'risk_score': float(risk)})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
