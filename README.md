# Projekt IUM - przewidywanie popularności
## Dominika Ferfecka, Sara Fojt

## Temat projektu
Mamy innowacyjny pomysł - zacznijmy wyprzedzać naszych słuchaczy i tworzyć listę topowych utworów z tydzień przed tym zanim staną się najbardziej popularne! Przegonimy konkurencję i zwiększymy zaangażowanie użytkowników.

## Kontekst
Serwis muzyczny “Pozytywka”, pozwalający użytkownikom na odtwarzanie ulubionych utworów online. Portal pozwala użytkownikom na pomijanie aktualnie odsłuchiwanych utworów oraz na oznaczanie utworów jako polubione.

## Zadanie biznesowe
Tworzenie listy utworów, które w kolejnym tygodniu staną się najbardziej popularne w serwisie. Domyślnie wygenerowana lista będzie zawierać 50 utworów.

## Biznesowe kryterium sukcesu
Piosenki proponowane w stworzonej liście powinny utrzymać się lub stać się jednymi z najpopularniejszych utworów w serwisie w kolejnym tygodniu. Zaproponowana lista powinna zawierać co najmniej 70% takich samych piosenek, co lista rzeczywiście najpopularniejszych piosenek o tej samej liczbie utworów opublikowana w kolejnym tygodniu. Większa popularność utworów oznacza większe zaangażowanie użytkowników co przełoży się na większą ilość odsłuchanych reklam, czyli większy zysk serwisu.

## Zadanie modelowania
Modelowanie szeregów czasowych - przewidywanie popularności piosenek w serwisie w kolejnych tygodniach.

## Dane do modelowania
dane o artystach, zawierające m.in. gatunki tworzonej przez nich muzyki
informacje o sesjach użytkowników oraz jakie akcje wykonywali na danych utworach (pomijanie, polubienie itp.)
informacje o utworach, zawierające liczbowe reprezentacje atrybutów takich jak energiczność, instrumentalność czy popularność (którą będziemy przewidywać)


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
