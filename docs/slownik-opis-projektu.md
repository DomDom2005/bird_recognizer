# Słownik terminów z `opis-projektu.md`

> Ten dokument wyjaśnia **każdy termin techniczny** użyty w [`opis-projektu.md`](opis-projektu.md).
> Układ: krótka definicja, a po niej „W projekcie:” — gdzie i po co ten termin występuje.
> Dla wygody na końcu są wzory metryk i tabela skrótów.

---

## 1. Jak czytać ten słownik

- **Termin** — nazwa, która pojawia się w opisie projektu.
- **Definicja** — co to jest, bez żargonu.
- **W projekcie** — konkretna wartość, plik albo rola w `bird_recognizer`.
- Terminy techniczne są pogrubione; wartości liczbowe pochodzą z konfiguracji i wyników
  (stan 2026-11-24).

---

## 2. Dane i API

**API (Application Programming Interface)**
Zestaw reguł, przez który program rozmawia z innym systemem. W projekcie: xeno-canto API v3,
adres `https://xeno-canto.org/api/3/recordings`.
*W projekcie:* klient `src/bird_recognizer/xc/client.py`; klucz API w `.env`.

**xeno-canto**
Publiczna baza nagrań ptaków z całego świata, udostępniająca dane przez API.
*W projekcie:* źródło wszystkich 2917 nagrań.

**Tag zapytania**
Fragment filtra w zapytaniu API, np. `sp:"Parus major"`. W v3 **tylko tagi** są akceptowane;
goła nazwa gatunku zwraca błąd.
*W projekcie:* `sp` (gatunek), `cnt` (kraj), `q` (jakość), `type` (rodzaj głosu), `status`.

**`sp`, `cnt`, `q`**
Skróty tagów: `sp` = species (gatunek łaciński), `cnt` = country (kraj), `q` = quality (jakość
nagrania w skali A–E).
*W projekcie:* filtr `sp:"<gatunek>" cnt:poland q:">C"` (czyli jakość A i B).

**`q:">C"`**
Zapis „jakość lepsza niż C”, czyli A lub B. Cudzysłowy są częścią składni tagu.
*W projekcie:* 2917 nagrań, głównie wysokiej jakości.

**`status:identified`**
Tag „nagranie zidentyfikowane przez eksperta”. Dokumentacja go opisuje, ale **żywe API zwraca
na niego HTTP 400**, dlatego jest wyłączony (`api.status: ""`).
*W projekcie:* brak filtra statusu → możliwy szum etykiet; opisane w ograniczeniach.

**HTTP 400 / 401 / 403 / 429 / 5xx**
Kody odpowiedzi serwera: 400 = złe zapytanie, 401/403 = brak/nieprawidłowy klucz,
429 = limit zapytań, 5xx = błąd serwera.
*W projekcie:* klient mapuje je na wyjątki i ponawia tylko 429/5xx.

**Redacted (zredagowane)**
Nagranie gatunku chronionego, dla którego API ukrywa lokalizację i link do pliku
(`_meta.redacted_fields`).
*W projekcie:* 0 takich wpisów; kod i tak je obsługuje (pomija, nie liczy jako błąd).

**`also`**
Lista gatunków słyszanych w tle nagrania.
*W projekcie:* używana w analizie błędów jako podejrzenie szumu etykiet (3/10 najgorszych nagrań).

**`rmk`**
Uwagi nagrywającego (np. o filtrze high-pass).
*W projekcie:* metadana doklejana do analizy błędów.

**Katalog (`catalog.jsonl`)**
Spis nagrań z metadanymi: `id`, gatunek, nagrywający, jakość, licencja, `sha256_audio`.
*W projekcie:* `data/catalog.jsonl`, 2917 rekordów.

**JSON / JSONL**
Tekstowy format danych: JSON = obiekt `{...}`; JSONL = **jeden JSON na linię** (wygodny do
dopisywania i strumieniowania).
*W projekcie:* katalog i raport pobierania.

**`sha256`**
Kryptograficzny „odcisk palca” pliku; ten sam plik = ten sam hash.
*W projekcie:* wykrywanie duplikatów i kontrola wycieku (hash nie może być w dwóch podziałach).

**Duplikat**
Dwa wpisy o tej samej treści (ten sam `sha256`).
*W projekcie:* 1 para (`1052072` = `1052075`), oba w train — nie wyciek, tylko to samo nagranie
wgrane dwa razy.

---

## 3. Podział danych

**Split (podział)**
Podział nagrań na trzy części: **train** (nauka), **val** (podgląd podczas treningu, wybór modelu),
**test** (egzamin, używany raz).
*W projekcie:* 2203 / 407 / 307 nagrań.

**Wyciek danych (data leakage)**
Sytuacja, w której ten sam materiał (nagranie, nagrywający, plik) jest w train i w test — model
„zna odpowiedzi”, a wynik jest nie do obrony.
*W projekcie:* `scripts/check_leakage.py` sprawdza `id`, `rec` i `sha256`; wynik: brak wycieku.

**`rec` (recordist)**
Nagrywający. Grupa nagrań jednej osoby ma podobny sprzęt i miejsce, więc dzielimy **grupami
po `rec`**, nie pojedynczymi plikami.
*W projekcie:* train ma 7 nagrywających (kilku bardzo płodnych), val/test po 43.

**Stratyfikacja**
Zapewnienie, że każdy gatunek jest obecny w każdym podziale.
*W projekcie:* 12 gatunków w train, val i test.

**`splits.json`**
Plik z listami nagrań w każdym podziale i wersją podziału.
*W projekcie:* `data/splits.json`; nie zmienia się w trakcie porównywania runów.

---

## 4. Audio i spektrogram

**Częstotliwość próbkowania (sample rate)**
Ile razy na sekundę zapisano amplitudę dźwięku. 32 000 Hz = 32 tys. próbek/s.
*W projekcie:* wszystko resample'owane do 32 kHz (wtedy Nyquist = 16 kHz, a ptaki śpiewają
wysoko, więc to wystarcza).

**Mono**
Jeden kanał dźwięku (zamiast stereo).
*W projekcie:* każdy plik sprowadzany do mono.

**Filtr górnoprzepustowy (high-pass)**
Filtr usuwający najniższe częstotliwości (tu < 100 Hz: szum wiatru, stuki, przydźwięk).
*W projekcie:* Butterworth rząd 4, `filtfilt` (filtr niephazowy, dwukierunkowy), `highpass_hz=100`.

**Okno (window)**
Fragment nagrania, na który patrzy model w jednym kroku; tutaj 3 s (ablacja: 5 i 10 s).
*W projekcie:* `audio.window_s=3.0`.

**Hop (przeskok)**
O ile przesuwamy okno do następnego fragmentu; 1,5 s = 50% zakładki.
*W projekcie:* `audio.hop_s=1.5`.

**Dopełnianie zerami (zero padding)**
Krótsze ostatnie okno uzupełniamy zerami do pełnej długości, żeby wszystkie okna miały ten sam
kształt.
*W projekcie:* prawdziwą długość pamięta `valid_frames` i maska; krótkie okno nie jest „udawane”.

**`valid_frames`**
Liczba ramek spektrogramu, które pochodzą z prawdziwego dźwięku (bez dopełnienia).
*W projekcie:* używane przez CRNN do maskowania poolingu.

**RMS (Root Mean Square)**
Miara głośności fragmentu: pierwiastek ze średniej kwadratów amplitud.
*W projekcie:* podstawa bramki ciszy.

**`p90` (percentyl 90)**
Wartość, poniżej której leży 90% obserwacji; tu: głośność „typowego głośnego” okna nagrania.
*W projekcie:* próg względny = 0,1 × p90 RMS.

**Bramka ciszy (silence gating)**
Odrzucanie okien, w których nie ma śpiewu: `RMS < max(1e-3, 0,1 × p90 RMS)`.
*W projekcie:* odrzucono 12,7% okien; 19 nagrań odpadło w całości.

**`empty_after_gating`**
Nagranie, w którym **wszystkie** okna odrzuciła bramka (bardzo ciche).
*W projekcie:* 19 nagrań; nie są błędem pobierania.

**Spektrogram**
Obraz widma dźwięku w czasie: oś X = czas, oś Y = częstotliwość, jasność/kolor = energia.
*W projekcie:* wejście modeli, zapisywane w cache jako `mel.npy`.

**Mel / skala mel**
Skala częstotliwości przybliżająca ludzkie (i ptasie) postrzeganie wysokości — więcej rozdzielczości
w niskich tonach.
*W projekcie:* 128 pasm mel (`n_mels=128`).

**log-mel**
Spektrogram mel w skali logarytmicznej (decybele), bo energia dźwięku zmienia się w ogromnym
zakresie.
*W projekcie:* `mel → AmplitudeToDB` z `top_db=80`.

**`n_fft` / `win_length`**
Rozmiar okna FFT (analizy widma) / długość okna analizy. 1024 próbki przy 32 kHz ≈ 32 ms.
*W projekcie:* 1024 / 1024.

**`hop_length`**
Przeskok między kolejnymi ramkami FFT w próbkach; 512 próbek ≈ 16 ms → ~62,5 ramki/s.
*W projekcie:* 512 (stąd 3 s ≈ 188 ramek).

**`n_mels`**
Liczba pasm mel (wymiar „wysokości” spektrogramu).
*W projekcie:* 128.

**`fmin` / `fmax`**
Zakres częstotliwości, który zostaje na spektrogramie: 150–15 000 Hz (poniżej Nyquista 16 kHz).
*W projekcie:* odcięcie nieistotnego dołu i góry.

**`power`**
Wykładnik przy liczeniu widma mocy (`|STFT|^power`); 2.0 = widmo mocy.
*W projekcie:* 2.0.

**`top_db`**
Dynamiczny zakres w dB: wszystko cichsze niż `max − 80 dB` jest przycinane.
*W projekcie:* 80 dB.

**Normalizacja per okno**
Dla każdego okna: `(x − średnia) / odchylenie standardowe`, żeby poziom głośności nie dominował.
*W projekcie:* zapisywana już znormalizowana; dlatego augmentacja `gain` (stałe wzmocnienie) jest
bezużyteczna i pominięta.

**Cache**
Przeliczone raz dane (tu: spektrogramy), żeby nie liczyć ich przy każdym treningu.
*W projekcie:* `data/cache/<id>/mel.npy` + `meta.json`, 6,06 GB, 2897 nagrań.

**`float16` (half precision)**
Format liczby zmiennoprzecinkowej 16-bitowy — dwa razy mniejszy niż `float32`, wystarczający
dla zapisu spektrogramu.
*W projekcie:* cache w float16, trening w float32.

**`config_hash`**
Skrót parametrów spektrogramu/audio; jeśli się zmienią, cache jest nieaktualny.
*W projekcie:* `prep cache --verify` porównuje `config_hash` z `meta.json`.

**`memmap` (memory-mapped file)**
Czytanie pliku „na żądanie” bez wczytywania całości do RAM.
*W projekcie:* przyspieszyło epokę CNN z 757 s do 130 s.

---

## 5. Modele i uczenie

**Model**
Funkcja z parametrami (wagami), która z wejścia (spektrogram) liczy prawdopodobieństwa klas.
*W projekcie:* cztery rodziny modeli, rejestr w `models_registry.py`.

**Parametry / wagi**
Liczby, których model się uczy; „rozmiar modelu” = liczba parametrów.
*W projekcie:* CNN 391 404, CRNN 1 212 396, ResNet18 11 176 396.

**Klasa / etykieta**
Gatunek (klasa) i jego numer (etykieta 0–11, kolejność z `configs/species.yaml`).
*W projekcie:* zmiana kolejności unieważnia zapisane modele.

**Logits**
Surowe wyjście modelu (bez normalizacji); softmax zamienia je na prawdopodobieństwa.

**Softmax**
Funkcja zamieniająca logits na prawdopodobieństwa sumujące się do 1.

**Prawdopodobieństwo klasy**
Liczba 0–1 mówiąca, jak bardzo model „stawia” na dany gatunek.

**CNN (konwolucyjna sieć neuronowa)**
Sieć, która przesuwa małe filtry po obrazie (spektrogramie) i wykrywa lokalne wzory: kreski,
harmoniczne, modulacje.
*W projekcie:* mały CNN `models/cnn.py` (bez czasu) i backbone CRNN.

**Konwolucja**
Operacja „filtra przesuwanego po obrazie”: iloczyn skalarny fragmentu obrazu i filtra.

**Kernel / stride / padding**
Rozmiar filtra / o ile się przesuwa / ile zer dookoła obrazu, żeby nie tracić brzegów.
*W projekcie:* 3×3, stride 1, padding 1 w blokach CNN.

**Blok konwolucyjny**
Zestaw: konwolucja → normalizacja → aktywacja → pooling.
*W projekcie:* 4 bloki z 32/64/128/256 kanałami.

**BatchNorm (BN)**
Normalizacja aktywacji w batchu — stabilizuje i przyspiesza trening.

**ReLU**
Prosta aktywacja: `max(0, x)`; wprowadza nieliniowość.

**Max-pooling**
Zmniejszanie obrazu: z każdego 2×2 bierzemy maksimum (2×2 → 1).
*W projekcie:* 4 poolingi → długość czasu dzielona przez 16.

**GAP (Global Average Pooling)**
Uśrednienie map cech do jednego wektora na kanał; eliminuje wymóg stałego rozmiaru wejścia.

**Dropout**
Losowe wyłączanie neuronów podczas treningu (tu 0,3) — zapobiega „wkuwaniu” na pamięć.

**Głowica (head)**
Ostatnia warstwa liniowa mapująca cechy na `n_classes` (tu 12).
*W projekcie:* nowa głowica w transferze.

**MFCC (Mel-Frequency Cepstral Coefficients)**
Kompaktowy opis barwy dźwięku (40 liczb na ramkę); klasyczna cecha w rozpoznawaniu mowy.
*W projekcie:* 40 MFCC → średnia + odchylenie = 80 cech dla SVM.

**SVM (Support Vector Machine)**
Klasyczny klasyfikator szukający granicy między klasami; z jądrem RBF radzi sobie z nieliniowością.

**RBF (jądro)**
Funkcja podobieństwa oparta na odległości („dzwon”); pozwala SVM tworzyć zakrzywione granice.

**Random Forest (RF)**
Las drzew decyzyjnych; głosowanie drzew daje klasę.

**`StandardScaler`**
Skalowanie cech do średniej 0 i odchylenia 1; fit **tylko na train**, żeby nie podejrzeć val/test.

**`class_weight=balanced`**
Wagi klas odwrotnie proporcjonalne do liczności — rzadkie klasy ważą więcej.

**Transfer learning**
Wykorzystanie modelu nauczonego na innym zadaniu (tu: ImageNet) i dostrojenie go do ptaków.

**ImageNet**
Ogromny zbiór zwykłych zdjęć (1000 klas); wagi z niego są punktem startowym transferu.

**`pretrained` / `scratch`**
Z wstępnie nauczonymi wagami / od zera (ta sama architektura, losowa inicjalizacja).
*W projekcie:* grupa ablacji `pretrained`.

**ResNet18**
Konwolucyjna sieć z połączeniami rezydualnymi („skrótami”), 18 warstw.

**EfficientNet-B0**
Lżejsza, „skalowana” architektura CNN; w projekcie zaimplementowana i przetestowana, ale macierz
liczona na ResNet18.

**Wejście 1-kanałowe**
Spektrogram ma jeden kanał; pierwsza konwolucja przyjmuje 1 kanał (przy transferze wagi RGB są
sumowane).
*W projekcie:* `_adapt_input` w `models/transfer.py`.

**RNN (rekurencyjna sieć neuronowa)**
Sieć przetwarzająca sekwencję krok po kroku, pamiętając poprzednie kroki.

**GRU / LSTM**
Rodzaje komórek RNN z bramkami kontrolującymi pamięć; LSTM ma więcej bramek niż GRU.

**BiGRU / BiLSTM**
RNN czytający sekwencję w obie strony (od początku i od końca); suma informacji z obu kierunków.

**Sekwencja / ramka**
Sekwencja = kolejne ramki spektrogramu w czasie; ramka ≈ 16 ms dźwięku.

**Hidden size**
Rozmiar ukrytego stanu RNN (tu 128); większy = więcej pamięci i parametrów.

**Projekcja (projection)**
Warstwa liniowa zmniejszająca wektor cech ramki (tu do 256) przed RNN.

**CRNN**
CNN wyciąga cechy z każdej ramki, RNN czyta je w czasie, pooling redukuje sekwencję do wektora.

**Pooling mean**
Uśrednienie wektorów po czasie.

**Pooling attention**
Model uczy się wag („na co patrzeć”) i liczy ważoną sumę po czasie.

**Maska długości**
Wektor 0/1 mówiący, które ramki są prawdziwe (bez dopełnienia); attention/mean pomijają resztę.

**`accepts_lengths`**
Flaga modelu: „przyjmuję długości okien” (CRNN tak, CNN nie).

---

## 6. Trening

**Epoka (epoch)**
Jeden pełny przegląd danych treningowych.
*W projekcie:* macierz na 15 epokach (max 50), z early stopping.

**Batch**
Paczka przykładów przetwarzana razem (tu 32).

**Learning rate (lr)**
Tempo uczenia: jak duży krok robi optymalizator. Za duży = skacze, za mały = wolno.
*W projekcie:* 1e-3, fine-tune transferu 3e-4.

**AdamW**
Optymalizator łączący momentum i adaptacyjne tempo, z odseparowanym weight decay.

**Weight decay**
Kara za duże wagi (tu 1e-4) — regularyzacja, mniej przeuczenia.

**Scheduler cosine**
Harmonogram zmniejszający lr po cosinusie do końca treningu.

**Warmup**
Pierwsze epoki z rosnącym lr, żeby trening startował stabilnie (tu 1 epoka).

**Gradient clipping**
Ograniczenie normy gradientu (tu 5,0), żeby jeden „wybuch” nie zepsuł wag.

**Early stopping**
Przerwanie treningu, gdy od `patience` epok nie ma poprawy na val.
*W projekcie:* patience 8, `min_delta=1e-3`.

**`patience` / `min_delta`**
Liczba epok bez poprawy / minimalna poprawa uznawana za realną.

**Label smoothing**
„Rozmycie” etykiet (np. 0,9 dla prawdy); tu 0,0 (wyłączone).

**Augmentacja**
Sztuczne modyfikacje danych treningowych, żeby model nie uczył się na pamięć.
*W projekcie:* tylko train; val/test zawsze czyste.

**Shift (przesunięcie w czasie)**
Przesunięcie okna o ±1 s (z zerami na brzegu).

**Szum / SNR**
Dodawanie szumu; SNR (signal-to-noise ratio) w dB mówi, ile razy sygnał jest głośniejszy od szumu
(tu 5–20 dB).

**Mixup**
Sklejanie dwóch przykładów: `x = λx₁ + (1−λ)x₂`, strata kombinowana; α=0,4.

**SpecAugment**
Zaklejanie fragmentów spektrogramu (2 maski częstotliwości ≤16 pasm, 2 maski czasu ≤20 ramek),
żeby model nie polegał na jednym fragmencie.

**Gain (wzmocnienie)**
Zmiana poziomu głośności; **pominięta**, bo normalizacja per okno ją znosi.

**Seed**
Liczba startowa losowości; ten sam seed + ten sam kod = ten sam wynik (na CPU).
*W projekcie:* 42, ustawiany **przed** utworzeniem modelu.

**Determinizm**
Powtarzalność wyniku; na GPU bywa ograniczona, dlatego raportujemy urządzenie.

**Device**
CPU lub GPU; `--device cpu` wymusza CPU.

**GPU / CUDA / VRAM**
Karta graficzna / środowisko obliczeń na GPU / pamięć karty. Tu RTX 5070, 12 GB, CUDA 13.2.

**Torch / PyTorch**
Biblioteka do sieci neuronowych; `torch 2.15.0+cu132` (build CUDA).

**Checkpoint / `best.pt`**
Zapisane wagi najlepszego modelu (po val macro-F1).
*W projekcie:* `runs/<run_id>/best.pt`.

---

## 7. Eksperymenty

**Ablacja**
Eksperyment, w którym zmieniamy **dokładnie jeden** czynnik, żeby zmierzyć jego wpływ.

**Grupa ablacji**
Zestaw wariantów jednego czynnika (np. `window`: 3/5/10 s).
*W projekcie:* 5 grup (`rnn`, `window`, `aug`, `pretrained`, `pool`).

**Eksperyment jednoczynnikowy**
Wariant, w którym względem referencji zmieniono tylko jedną rzecz.
*W projekcie:* `runs/ablation_<grupa>.json` zapisuje tę różnicę.

**Run (przebieg)**
Jeden trening z pełnym zapisem: config, seed, model, metryki, `git_sha`, urządzenie.
*W projekcie:* `runs/<run_id>/` + wiersz w `runs/index.csv`.

**Run referencyjny**
Punkt odniesienia w grupie ablacji (pierwszy wariant), względem którego liczone są różnice.

**Budżet epok**
Wspólna maksymalna liczba epok dla wszystkich wariantów (tu 15), żeby porównanie było uczciwe.

**`--smoke`**
Szybki przebieg techniczny (2 nagrania/gatunek, 1 epoka) sprawdzający, że kod działa.
*W projekcie:* **nie jest wynikiem badania**; w tabelach oznaczony.

**Seed 42 / jeden seed**
Cała macierz policzona na jednym seedzie; różnice ±0,01 wymagają powtórzeń na 3 seedach.

---

## 8. Metryki i ewaluacja

**Ewaluacja**
Policzenie metryk na zbiorze, którego model nie widział podczas treningu (val/test/own).

**Accuracy (dokładność)**
Odsetek poprawnych predykcji. Myląca przy nierównych klasach.
*W projekcie:* top-1 0,772 dla najlepszego modelu.

**Precision (precyzja)**
Z nagrań uznanych za gatunek X, jaka część naprawdę jest X („czy nie krzyczysz za często X”).

**Recall (czułość)**
Z nagrań, które naprawdę są X, jaka część została znaleziona („czy nie przegapiasz X”).

**F1**
Średnia harmoniczna precision i recall: `2PR/(P+R)`; kara za zarówno fałszywe alarmy,
jak i pominięcia.

**Macro-F1**
F1 liczone osobno dla każdej klasy, potem **zwykła średnia po klasach** — każdy gatunek waży tyle
samo, niezależnie od liczby nagrań.
*W projekcie:* metryka główna; najlepszy model 0,736.

**Micro-F1 / accuracy**
F1 liczony po wszystkich próbkach łącznie — zdominowany przez liczne klasy; w projekcie nieużywany
jako główny.

**Balanced accuracy**
Średni recall po klasach; odporna na nierównowagę klas.

**Top-1 / top-3 accuracy**
Czy prawdziwy gatunek jest typem nr 1 / w pierwszej trójce.
*W projekcie:* top-3 = 0,902 (najlepszy model).

**Agregacja mean**
Uśrednienie prawdopodobieństw okien jednego nagrania → jedna predykcja nagrania.

**Agregacja vote**
Głosowanie etykiet okien; remis rozstrzyga suma prawdopodobieństw.

**Coverage (pokrycie)**
Jaka część nagrań zostaje po odrzuceniu tych poniżej progu pewności.

**Próg pewności (threshold)**
Minimalne prawdopodobieństwo, żeby uznać predykcję; poniżej → „nie wiem”.
*W projekcie:* planowany w aplikacji A+B, kalibrowany na `predictions.csv`.

**Macierz pomyłek (confusion matrix)**
Tabela prawda × predykcja: na przekątnej trafienia, poza nią — kto z kim mylony.

**Per-class P/R/F1**
Precision, recall i F1 dla każdego gatunku osobno (z `n` = liczbą nagrań).

**Pary mylące**
Najczęstsze pomyłki w postaci „prawda → predykcja”, np. bogatka → modra.

**`n` / support**
Liczba przykładów danej klasy w zbiorze; bez `n` metryka jest bezużyteczna.

**Szum etykiet (label noise)**
Błędne lub niejednoznaczne etykiety (np. w nagraniu słychać też inny gatunek — `also`).

**`also` w analizie błędów**
Jeśli najgorsze nagranie ma niepuste `also`, możliwe, że „błąd” to problem etykiety, nie modelu.

---

## 9. Artefakty i reprodukcja

**`run_id`**
Nazwa katalogu runu: `{data-godzina}-{model}-{tag}-s{seed}`.
*W projekcie:* np. `20261121-225300-cnn-window-10s-s42`.

**`config.yaml` (w runie)**
Snapshot całej konfiguracji użytej w treningu (geometria, model, opcje, trening).

**`env.txt`**
Wersje bibliotek, `git_sha`, liczba parametrów, urządzenie, platforma.

**`metrics.json` / `metrics.jsonl`**
Metryki końcowe (JSON) i historia epok (JSONL — jedna epoka na linię).

**`runs/index.csv`**
Jedna linia na run: model, okno, aug, pooling, seed, metryki, `duration`, `device`, `git_sha`,
`notes`. Źródło tabeli w `docs/experiments.md`.

**`git_sha`**
Hash commita kodu, na którym powstał run — po nim wiadomo, którą wersję kodu uruchomić.

**Snapshot konfiguracji**
Kopia konfiguracji zapisana przy runie; zmiana pliku `configs/` nie zmienia historii runów.

**Odtwarzalność**
Możliwość powtórzenia wyniku z tych samych danych, kodu, configu i seeda.

**`eval_<split>/`**
Katalog z wynikami ewaluacji runu: metryki, per-class, macierz CSV+PNG, predykcje.

**`analysis/`**
Wyniki analizy błędów: `errors.md`, `pairs.csv`, `per_species.csv`, `worst_recordings.csv`.

---

## 10. Narzędzia i inżynieria

**CLI (Command Line Interface)**
Sterowanie programem komendami w terminalu: `python -m bird_recognizer.cli ...`.

**`python -m`**
Uruchomienie modułu Pythona (tu CLI) bez polegania na wpisie w PATH.

**pytest**
Framework testów; tu 259 testów offline.

**Test offline**
Test, który nie łączy się z siecią; `conftest.py` blokuje gniazda.

**Pokrycie (coverage)**
Procent linii kodu wykonanych przez testy; tu ~91%.

**ruff**
Linter Pythona: styl, nieużywane importy, proste błędy.

**Gradio**
Biblioteka do prostego UI w przeglądarce; tu demo (mikrofon/plik → top-3 + spektrogram).

**Telemetria**
Automatyczne „raportowanie” biblioteki do serwera; tu wyłączone (`GRADIO_ANALYTICS_ENABLED=False`),
żeby demo było offline.

**`soundfile`**
Biblioteka czytająca pliki audio (wav/mp3/flac/ogg) bez TorchCodec.

**TorchCodec**
Dodatek wymagany przez nowsze `torchaudio.load`; w projekcie nieużywany (problem rozwiązany
przez `soundfile`).

**`torchaudio`**
Biblioteka audio dla PyTorcha; tu tylko do resamplingu.

**`numpy` / `pandas` / `scikit-learn` / `matplotlib`**
Obliczenia macierzowe / tabele / klasyczne ML i metryki / wykresy (spektrogram, macierz pomyłek).

**Kody wyjścia**
`0` sukces, `1` błąd użytkownika/konfiguracji, `2` API/klucz, `3` dane/audio, `4` model/OOM.

**Redakcja logów**
Maskowanie sekretów (`key=...`, tokeny) w logach.

**`.env`**
Plik z sekretami poza repo (klucz API); wzór w `.env.example`.

**`cuBLASLt`**
Biblioteka NVIDIA do operacji macierzowych na GPU; w CUDA 13.2 na Windowsie potrafiła przepełnić
stos — rozwiązane uruchomieniem w wątku z większym stosem.

**`venv` / `pip`**
Wirtualne środowisko Pythona / instalator pakietów.

---

## 11. Skróty

| Skrót | Rozwinięcie | Znaczenie |
|---|---|---|
| API | Application Programming Interface | interfejs do xeno-canto |
| CLI | Command Line Interface | komendy w terminalu |
| ML | Machine Learning | uczenie maszynowe |
| CNN | Convolutional Neural Network | sieć konwolucyjna |
| RNN | Recurrent Neural Network | sieć rekurencyjna |
| CRNN | CNN + RNN | model z czasem |
| GRU / LSTM | Gated Recurrent Unit / Long Short-Term Memory | komórki RNN |
| MFCC | Mel-Frequency Cepstral Coefficients | klasyczne cechy audio |
| SVM / RF | Support Vector Machine / Random Forest | klasyczne klasyfikatory |
| RBF | Radial Basis Function | jądro SVM |
| GAP | Global Average Pooling | uśrednianie map cech |
| BN | Batch Normalization | normalizacja w batchu |
| HPF | High-Pass Filter | filtr górnoprzepustowy |
| FFT | Fast Fourier Transform | szybka transformata Fouriera |
| SNR | Signal-to-Noise Ratio | stosunek sygnału do szumu |
| GPU / VRAM | Graphics Processing Unit / Video RAM | karta graficzna / pamięć karty |
| CUDA | Compute Unified Device Architecture | środowisko GPU NVIDII |
| F1 | miara łącząca precision i recall | metryka jakości |
| JSON / JSONL | JavaScript Object Notation / JSON Lines | format danych |
| YAML | YAML Ain't Markup Language | format konfiguracji |
| CSV | Comma-Separated Values | tabela tekstowa |
| PNG | Portable Network Graphics | format obrazu |
| SHA-256 | Secure Hash Algorithm 256 | odcisk palca pliku |

---

## 12. Wzory i przykłady

Dla klasy X na zbiorze nagrań:

```
TP = prawda X, model mówi X
FP = nie X, model mówi X
FN = prawda X, model mówi nie X

precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

**Macro-F1** = `(1/K) * Σ F1_k` po `K` klasach (tu K = 12).

**Przykład ręczny (3 klasy):** macierz pomyłek

| prawda \ pred | A | B | C |
|---|---|---|---|
| A | 2 | 1 | 0 |
| B | 0 | 3 | 1 |
| C | 1 | 0 | 2 |

F1(A)=2/3, F1(B)=3/4, F1(C)=2/3 → macro-F1 = (2/3 + 3/4 + 2/3) / 3 = **25/36 ≈ 0,694**.
Ten sam wynik jest w teście `tests/test_metrics.py`, więc kod i definicja są zgodne.

**Top-k:** top-1 = typ o najwyższym prawdopodobieństwie jest poprawny; top-3 = poprawny typ
jest w trójce najwyższych.

**Coverage vs accuracy:** przy progu 0,5 zostaje tylko część nagrań (coverage), ale wśród nich
trafność jest wyższa — tabela `coverage.csv`.

---

## 13. Gdzie szukać

| Temat | Plik |
|---|---|
| Opis projektu (dla wykładowcy) | [`opis-projektu.md`](opis-projektu.md) |
| Przewodnik po projekcie | [`przewodnik.md`](przewodnik.md) |
| Specyfikacja | [`project info/SPEC.md`](../project%20info/SPEC.md) |
| Dziennik AI | [`ai-journal.md`](ai-journal.md) |
| Wyniki | [`experiments.md`](experiments.md) |
| Fakty o API | [`api-notes.md`](api-notes.md) |
