import os
import pickle
import pmdarima
from flask import Flask, request, jsonify


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    with open('../arima_model.pkl', 'rb') as arima_file:
        arima_model = pickle.load(arima_file)

    # a simple page that says hello
    @app.route('/arima')
    def arima_predict():
        try:
            prediction = arima_model.predict(n_periods=4)
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return app