import os
import pickle
from flask import Flask, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

models_folder = '../arima_models/'

def load_models(folder):
    models = {}
    for filename in os.listdir(folder):
        if filename.endswith('.pkl'):
            if filename.startswith('arima_model_'):
                track_id = filename[len('arima_model_'):-len('.pkl')]
                model_path = os.path.join(folder, filename)
                with open(model_path, 'rb') as file:
                    models[track_id] = pickle.load(file)
    return models

input_features = [
    'id_artist_encoded', 'duration_ms', 'danceability', 'energy', 
    'popularity', 'speechiness', 'acousticness', 'instrumentalness', 'release_year'
]

def get_latest_data_for_lstm(file_handle):
    output_column = 'weekly_popularity_score'

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(tracks_without_latest_score[input_features])

    n_time_steps = 4  # liczba poprzednich tygodni

    X = []
    y = []

    track_weeks_info = []

    tracks_grouped = tracks_without_latest_score.groupby('id')

    for track_id, track_data in tracks_grouped:
        if len(track_data) >= n_time_steps:
            # wybierz ostatnią sekwencję (ostatnie 4 tygodnie)
            track_sequence = track_data.tail(n_time_steps)
            
            sequence = scaled_features[track_sequence.index]
            X.append(sequence)
            
            y.append(track_data[output_column].iloc[-1])  # Używamy ostatniej wartości 'weekly_popularity_score' dla danego utworu
            
            weeks = track_sequence['week'].values
            track_weeks_info.append((track_id, weeks))

    X_latest = np.array(X)
    y = np.array(y)

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    arima_models = load_models(models_folder)
    @register_keras_serializable()
    def custom_loss(y_true, y_pred):
        weights = y_true
        # weights = tf.square(y_true)
        
        # obliczenie błędu kwadratowego
        error = tf.square(y_true - y_pred)
        
        # pomnożeniu błędu przez wagę - bardziej popularne utwory są ważniejsze
        weighted_error = error * weights
        
        # średnią ważona błędów
        return tf.reduce_mean(weighted_error)

    # Załaduj model
    lstm_model = load_model('../lstm_model.keras', custom_objects={'custom_loss': custom_loss})
    @app.route('/')
    def arima_predict():
        results = []
        try:
            for track_id, model in arima_models.items():
                forecast = model.predict(n_periods=1).tolist()[-1]
                # Zapis wyników
                results.append({
                    "track_id": track_id,
                    "forecast": forecast,
                })
            sorted_results = sorted(
                results,
                key=lambda x: x['forecast'],  # Sortuj po pierwszym elemencie prognozy
                reverse=True  # Od największego do najmniejszego
            )
            
            # Wybór 50 największych
            top_50_results = sorted_results[:50]
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        return jsonify({'prediction': top_50_results})
    
    @app.route("/lstem")
    def lstem_predict():
        try:
            X_latest = np.random.rand(1, 100, 10)
            # sequences = [
            #         [123, 210000, 0.8, 0.75, 50, 0.05, 0.1, 0.0, 2020],
            #         [123, 200000, 0.85, 0.8, 55, 0.06, 0.2, 0.0, 2020],
            #         [123, 190000, 0.78, 0.7, 53, 0.04, 0.15, 0.0, 2020],
            #         [123, 180000, 0.82, 0.72, 52, 0.05, 0.18, 0.0, 2020]
            #     ]
            # scaled_sequences = scaler.transform(sequences)
            # input_data = np.array(scaled_sequences).reshape((1, 4, len(input_features)))

            forecast = lstm_model.predict()
            print(forecast)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        return jsonify({'prediction': len(forecast)})

    return app