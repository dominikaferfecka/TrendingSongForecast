# Projekt IUM - przewidywanie popularności
## Dominika Ferfecka, Sara Fojt

## Modele
### Model ARIMA
Model bazowy znajduje się w folderze *arima_model/*

Jest to prosty model przewidywania szeregów czasowych na podstawie historycznych danych. Z uwagi na fakt, że nie przyjmuje on żadnych danych wejściowych (oprócz długości predykcji), każdy z utworów musi posiadać swój własny przetrenowany model.

Aby zoptymalizować proces zdecydowałyśmy się na analizę jedynie 500 utworów, które były średnio najpopularniejsze w ciągu ostatnich 8 tygodniu. To pozwala nam na uniknięcie nadmiernej ilości modeli, gdyż możemy założyć, że aktualnie popularne utwory również będą popularne w kolejnym tygodniu. Nie jest to oczywiście rozwiązanie idealne, ale ma najlepszy stosunek optymalizacji czasowej i pojemnościowej do wyniku.

### Model LSTM
Model LSTM i powiązane z nim pliki znajdują się w folderze *lstm_model/*. Opis działania modelu, jak i testy można znaleźć w pliku *model_LSTM.ipynb*. Wytrenowany model został zapisany w pliku *lstm_model.keras*.

## Mikroserwis
Mikroserwis został zaimplementowany przy pomocy narzędzia Flask. Notatnik *microservice.ipynb* w głównym folderze projektu zawiera kod potrzebny do uruchomienia aplikacji. Działa ona na domyślnym porcie o numerze **5000**.

W pliku tym znajdują się przykładowe testy oraz porównanie ich wyników. Domyślnie mikroserwis korzysta z modelu LSTM, o ile jest dostępny plik *tracks_with_latest_score.csv*. Jest to wymagane, ponieważ model ten potrzebuje ostatnich danych wejściowych, do stworzenia predykcji. Jeśli taki plik nie jest dostępny, domyślnie wybierany jest model ARIMA.

Istnieje opcja, aby podać rodzaj modelu w danych zapytania, np. w ten sposób:

```
curl -X POST 
-H "Content-Type: application/json" 
-d '{"model_type": "arima"}' 
http://127.0.0.1:5000/predict

```

Aplikacja działa pod ścieżką `/predict` (metoda POST).

## Rozwój modeli
Przewidywanie popularności utworów jest zjawiskiem bardzo trudnym i złożonym. Zdajemy sobie sprawę, że na ten proces wpływają atrybuty ukryte, takie jak nieoczekiwane trendy, wydarzenia kulturowe, "cancel culture" itd. Dlatego też zalecamy regularne dotrenowywanie modeli do najnowszych danych.

## Ostateczna analiza danych
Dopracowana i uszczegółowiona analiza danych znajduje się w nowej wersji notatnika w głównym folderze pod nazwą *IUM_etap2.ipynb*
