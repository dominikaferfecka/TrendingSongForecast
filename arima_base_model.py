import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

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

print("Rozpoczynanie przetwarzania ARIMA dla każdego utworu...")
unique_track_ids = weekly_scores['track_id'].unique()

results = []
for idx, track_id in enumerate(unique_track_ids, 1):
    print(f"Przetwarzanie track_id: {track_id} ({idx}/{len(unique_track_ids)})")
    
    track_data = weekly_scores[weekly_scores['track_id'] == track_id].set_index('week')['weekly_popularity_score']
    track_data = track_data.asfreq('W').fillna(0)
    track_data = track_data.sort_index()


    if len(track_data) < 50: 
        print(f"Pomijanie track_id: {track_id}, za mało danych historycznych (tylko {len(track_data)} tygodni).")
        continue
    
    train_data = track_data[-8:-4]
    test_data = track_data[-4:]
    
    try:
        model = ARIMA(train_data, order=(5, 2, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=4)
        results.append((track_id, forecast))
        print(f"Zakończono przetwarzanie dla track_id: {track_id}")
    except Exception as e:
        print(f"Błąd podczas przetwarzania track_id: {track_id}. Szczegóły: {e}")
        continue



if results:
    for single_result in results[:10]:
        print("Wizualizacja wyników dla pierwszego utworu...")
        example_track_id, example_forecast = single_result
        track_data = weekly_scores[weekly_scores['track_id'] == example_track_id].set_index('week')['weekly_popularity_score']
        track_data = track_data.asfreq('W').fillna(0)
        test_data = track_data[-4:]

    # Konwersja indeksu na datetime
        track_data.index = track_data.index.to_timestamp()

        # Wizualizacja danych
        plt.figure(figsize=(10, 6))
        plt.plot(track_data.index, track_data, label='Observed')
        plt.plot(test_data.index.to_timestamp(), test_data, label='Actual Future', linestyle='--')
        plt.plot(test_data.index.to_timestamp(), example_forecast, label='Forecasted', linestyle='--')
        plt.title(f'Popularity Forecast for Track {example_track_id}')
        plt.legend()
        plt.show()

else:
    print("Brak wystarczających danych do wizualizacji wyników.")
# print(results)
# print(test_data)
if results:
    error_results = []
    for result in results:
        mae = np.mean(np.abs(np.array(result[1]) - np.array(train_data[-4])))
        mse = np.mean((np.array(result[1]) - np.array(train_data[-4])) ** 2)
        error_results.append({"track_id": result[0], "MAE": mae, "MSE": mse})

    # Konwersja wyników na DataFrame
    error_df = pd.DataFrame(error_results)

    # Tworzenie histogramu dla MAE
    plt.figure(figsize=(10, 6))
    plt.hist(error_df["MAE"], bins=10, edgecolor="black", alpha=0.7)
    plt.title("Histogram of Mean Absolute Error (MAE)")
    plt.xlabel("MAE range")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Tworzenie histogramu dla MSE
    plt.figure(figsize=(10, 6))
    plt.hist(error_df["MSE"], bins=10, edgecolor="black", alpha=0.7)
    plt.title("Histogram of Mean Squared Error (MSE)")
    plt.xlabel("MSE range")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()