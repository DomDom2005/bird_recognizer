Cel i pytanie badawcze

Zbudować klasyfikator gatunku ptaka z nagrania i sprawdzić, ile daje modelowanie czasu (CRNN) w porównaniu z modelem, który patrzy na spektrogram bez rekurencji. Taki wątek porządkuje cały projekt i dobrze pasuje do tematu sekwencji.

Etapy

1. Zakres i decyzje

Wybierz 10-15 gatunków, np. polskie ptaki śpiewające o różnej trudności (wróbel, szczygieł, kos, rudzik, sikora, zięba) oraz kilka podobnych do siebie, żeby pomyłki były ciekawe.
Ustal metryki: macro-F1 i dokładność top-3, plus macierz pomyłek.
Przygotuj repozytorium, środowisko (PyTorch, librosa/torchaudio, scikit-learn) i śledzenie eksperymentów (choćby arkusz albo Weights & Biases).

2. Dane

Pobierz nagrania z Xeno-canto przez API (kraj: Polska, wyższa jakość, np. A/B), ok. 50-100 nagrań na gatunek.
Podziel na train/val/test przed cięciem na okna, według nagrań (najlepiej też według autora lub lokalizacji).
Nagraj własny zbiór testowy telefonem, nawet 20-50 krótkich klipów, do sprawdzenia różnicy domen.

3. Preprocessing

Przepróbkowanie (np. 32 kHz), mono, filtr górnoprzepustowy usuwający szum wiatru.
Okna 3-5 s z nakładaniem, mel-spektrogram w skali logarytmicznej (ok. 128 pasm), normalizacja.
Odrzucanie okien bez śpiewu (prosta detekcja po energii), bo inaczej model uczy się na ciszy.
Zapis spektrogramów na dysk (cache), żeby trening był szybki.

4. Baseline

MFCC + statystyki (średnia, odchylenie) → SVM lub Random Forest.
Prosty CNN na spektrogramach z uśrednianiem predykcji po oknach.
To jest punkt odniesienia, z którym porównasz resztę.

5. Modele główne

CNN z przeniesieniem wiedzy: ResNet18 lub EfficientNet-B0 z wagami z ImageNetu, dostrojony.
CRNN: CNN jako ekstraktor cech, potem dwukierunkowy GRU/LSTM po osi czasu, na końcu pooling (średnia lub attention).
Opcjonalnie: embeddingi z gotowego modelu (BirdNET, Perch, PANNs) i własna głowica czasowa (GRU lub mały Transformer).

6. Eksperymenty i ablacje
Każdy tylko z jedną zmienioną rzeczą:

z RNN i bez RNN,
długość okna (3 s vs 5 s vs 10 s),
augmentacje: przesunięcie w czasie, szum, mixup, SpecAugment,
uczenie od zera vs wstępnie wytrenowane wagi,
pooling: średnia vs attention.

Wyniki zbierz w jednej tabeli, bo to będzie rdzeń raportu.

7. Ewaluacja

Wyniki na zbiorze testowym i osobno na własnych nagraniach (spadek jakości pokazuje problem domeny).
Analiza błędów: które gatunki są mylone, czy winne są hałas, nakładające się głosy, czy podobny śpiew.
Agregacja z okien do nagrania (średnia, głosowanie) i jej wpływ na wynik.

8. Demo i raport

Proste demo (Gradio lub Streamlit): wgrywasz nagranie, dostajesz top-3 gatunków z prawdopodobieństwami i wykres spektrogramu.
Raport lub prezentacja: problem, dane, metody, tabela ablacji, wnioski, ograniczenia.
Wersja minimalna i rozszerzenia
Minimum: etapy 1-4 plus CRNN, porównanie z baseline’em, ewaluacja i krótkie demo.
Rozszerzenia: klasyfikacja wieloetykietowa (kilka gatunków naraz), lokalizacja w czasie (kiedy śpiewa który ptak), więcej gatunków.
Główne ryzyka
Wyciek danych przy złym podziale: sprawdź to na samym początku.
Niezbalansowane klasy: ważenie strat lub oversampling, ocena macro-F1.
Czas treningu: ogranicz się do małych modeli i cache’owanych spektrogramów, a w razie potrzeby użyj darmowego GPU (Colab lub Kaggle).