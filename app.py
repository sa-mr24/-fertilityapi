from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# تحميل النموذج
model = joblib.load('fertility_model_clean.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = ['N', 'P', 'K', 'ph', 'ec', 'oc', 'S', 'zn', 'fe', 'cu', 'Mn', 'B']
        input_data = [data[feat] for feat in features]
        prediction = model.predict([input_data])[0]
        return jsonify({'fertility': bool(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
