import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pmdarima import auto_arima

def avg_popularity_last_8_weeks(group):
    # Posortowanie według tygodnia
    group = group.sort_values(by='week', ascending=False)
    # Wybranie ostatnich 8 tygodni
    last_8_weeks = group.head(8)
    # Obliczenie średniej popularności
    return last_8_weeks['weekly_popularity_score'].mean()

print("Wczytywanie danych...")
sessions = pd.read_json('V1/sessions.jsonl', lines=True)
tracks = pd.read_json('V1/tracks.jsonl', lines=True)

print("Przygotowanie danych do obliczeń tygodniowych...")
relevant_events = sessions[sessions['event_type'].isin(['play', 'skip', 'like'])]

event_counts = relevant_events.groupby(['track_id', 'event_type']).size().unstack(fill_value=0)
event_counts = event_counts.rename(columns={
    'play': 'play_count',
    'skip': 'skip_count',
    'like': 'like_count'
})
tracks_with_counts = tracks.merge(event_counts, how='left', left_on='id', right_index=True)
tracks_with_counts[['play_count', 'skip_count', 'like_count']] = tracks_with_counts[['play_count', 'skip_count', 'like_count']].fillna(0)

w1, w2, w3 = 1.0, 0.8, 0.5
tracks_with_counts['popularity_score'] = (
    w1 * tracks_with_counts['play_count'] +
    w2 * tracks_with_counts['like_count'] -
    w3 * tracks_with_counts['skip_count']
)

print("Grupowanie danych po tygodniach...")
sessions['week'] = pd.to_datetime(sessions['timestamp']).dt.to_period('W')
weekly_scores = sessions.groupby(['track_id', 'week']).apply(
    lambda x: w1 * x['event_type'].eq('play').sum() +
              w2 * x['event_type'].eq('like').sum() -
              w3 * x['event_type'].eq('skip').sum()
).reset_index(name='weekly_popularity_score')

track_popularity = weekly_scores.groupby('track_id').apply(avg_popularity_last_8_weeks).reset_index(name='avg_popularity')
top_tracks = track_popularity.sort_values(by='avg_popularity', ascending=False).head(500)
top_tracks = top_tracks.merge(tracks[['id', 'name']], left_on='track_id', right_on='id', how='left')

print("Rozpoczynanie przetwarzania ARIMA dla każdego utworu...")

results = []
error_results = []
for idx, row in enumerate(top_tracks.itertuples(), 1):
    track_id = row.track_id
    track_name = row.name

    track_data = weekly_scores[weekly_scores['track_id'] == track_id]
    track_data = track_data.set_index('week')['weekly_popularity_score'].sort_index()
    track_data = track_data.asfreq('W')
    # track_data = fill_missing_with_average(track_data)

    train_data = track_data[:-4]
    test_data = track_data[-4:]

    try:
        model = auto_arima(
            train_data,
            seasonal=False,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True)
        forecast = model.predict(n_periods=4)
        # Obliczanie błędów MAE i MSE
        mae = np.mean(np.abs(forecast - test_data.values))
        mse = np.mean((forecast - test_data.values) ** 2)
        error_results.append({
            "track_id": track_id,
            "track_name": track_name,
            "MAE": mae,
            "MSE": mse
        })
        
        # Zapis wyników
        results.append({
            "track_id": track_id,
            "track_name": track_name,
            "forecast": forecast,
            "test_data": test_data
        })
        print(f"Prognoza zakończona dla tracku z id {track_id}.")
        with open(f"arima_models/arima_model_{track_id}.pkl", 'wb') as model_file:
            pickle.dump(model, model_file)
    except Exception as e:
        print(f"Błąd podczas trenowania modelu ARIMA dla tracku {track_id}. Szczegóły: {e}")


print("Wizualizacja wyników dla pierwszych utworów...")
for result in results[:5]:  # Wizualizacja dla 5 pierwszych utworów
    track_name = result['track_name']
    track_id = result['track_id']
    forecast = result['forecast']
    test_data = result['test_data']
    
    # Pobranie danych historycznych
    track_data = weekly_scores[weekly_scores['track_id'] == track_id]
    track_data = track_data.set_index('week')['weekly_popularity_score'].sort_index()
    track_data = track_data.asfreq('W').fillna(0)
    
    track_data.index = track_data.index.to_timestamp()

    # Wizualizacja danych historycznych i prognozy
    plt.figure(figsize=(10, 6))
    plt.plot(track_data.index, track_data, label='Observed')
    plt.plot(test_data.index, test_data, label='Actual Future', linestyle='--', color='orange')
    plt.plot(test_data.index, forecast, label='Forecasted', linestyle='--', color='green')
    plt.title(f"Popularity Forecast for Track {track_name} (ID: {track_id})")
    plt.xlabel("Time")
    plt.ylabel("Popularity")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Zapis błędów do DataFrame
error_df = pd.DataFrame(error_results)

# Histogram błędów MAE
plt.figure(figsize=(10, 6))
plt.hist(error_df['MAE'], bins=10, edgecolor="black", alpha=0.7)
plt.title("Histogram of Mean Absolute Error (MAE)")
plt.xlabel("MAE range")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram błędów MSE
plt.figure(figsize=(10, 6))
plt.hist(error_df['MSE'], bins=10, edgecolor="black", alpha=0.7)
plt.title("Histogram of Mean Squared Error (MSE)")
plt.xlabel("MSE range")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Podsumowanie wyników
print(f"Zakończono przetwarzanie {len(results)} utworów.")
print("Podsumowanie błędów:")
print(error_df.describe())