from flask import Flask, jsonify, request
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

models_folder = 'arima_models/'

def load_models(folder):
    models = {}
    for filename in os.listdir(folder):
        if filename.endswith('.pkl') and filename.startswith('arima_model_'):
            track_id = filename[len('arima_model_'):-len('.pkl')]
            model_path = os.path.join(folder, filename)
            with open(model_path, 'rb') as file:
                models[track_id] = pickle.load(file)
    return models

input_features = [
    'id_artist_encoded', 'duration_ms', 'danceability', 'energy', 
    'popularity', 'speechiness', 'acousticness', 'instrumentalness', 'release_year', 'weekly_popularity_score'
]

def get_latest_data_for_lstm(file_path, n_time_steps=4):
    tracks_without_latest_score = pd.read_csv(file_path)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(tracks_without_latest_score[input_features])

    track_weeks_info = []
    X = []
    tracks_grouped = tracks_without_latest_score.groupby('id')

    for track_id, track_data in tracks_grouped:
        if len(track_data) >= n_time_steps:
            track_sequence = track_data.tail(n_time_steps)
            sequence = scaled_features[track_sequence.index]
            X.append(sequence)
            weeks = track_sequence['week'].values
            track_weeks_info.append((track_id, weeks))

    X_latest = np.array(X)
    return X_latest, track_weeks_info

def create_app(test_config=None):
    app = Flask(__name__)
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    arima_models = load_models(models_folder)

    @register_keras_serializable()
    def custom_loss(y_true, y_pred):
        weights = y_true
        error = tf.square(y_true - y_pred)
        weighted_error = error * weights
        return tf.reduce_mean(weighted_error)

    lstm_model = load_model('lstm_model.keras', custom_objects={'custom_loss': custom_loss})
    n_time_steps = 4
    try:
        X_latest, track_weeks_info = get_latest_data_for_lstm("tracks_without_latest_score.csv", n_time_steps)
    except Exception:
        X_latest, track_weeks_info = [], None
    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            # Decyzja o wyborze modelu
            if len(X_latest) > 0:
                # Jeśli dane są wystarczające dla LSTM
                predictions = lstm_model.predict(X_latest)
                track_predictions = []

                for idx, (track_id, _) in enumerate(track_weeks_info):
                    prediction_value = float(predictions[idx][0])
                    track_predictions.append((track_id, prediction_value))

                sorted_predictions = sorted(track_predictions, key=lambda x: x[1], reverse=True)
                top_50_tracks = sorted_predictions[:50]
                top_50_ids = [track_id for track_id, _ in top_50_tracks]
                return jsonify({'model': 'lstm', 'prediction': top_50_ids})

            else:
                # Domyślnie użyj ARIMA
                results = []
                for track_id, model in arima_models.items():
                    forecast = model.predict(n_periods=1).tolist()[-1]
                    results.append({"track_id": track_id, "forecast": forecast})

                sorted_results = sorted(results, key=lambda x: x['forecast'], reverse=True)
                top_50_tracks = sorted_results[:50]
                top_50_ids = [track["track_id"] for track in top_50_tracks]
                return jsonify({'model': 'arima', 'prediction': top_50_ids})

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return app
