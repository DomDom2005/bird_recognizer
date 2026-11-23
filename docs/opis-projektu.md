# Opis projektu — bird_recognizer

> **Dla wykładowcy:** dokument zbiera wszystkie szczegóły techniczne (dane, preprocessing, modele,
> trening, ablacje, metryki, wyniki, reprodukcja, ograniczenia).
> **Dla autora:** każda sekcja ma ramkę **„Prościej”**, która wyjaśnia ten sam fragment bez żargonu.
>
> Wersja: 2026-11-24 · repo: `github.com/DomDom2005/bird_recognizer`, branch `main` ·
> specyfikacja: [`project info/SPEC.md`](../project%20info/SPEC.md) · wyniki:
> [`docs/experiments.md`](experiments.md).

---

## 1. Streszczenie

Projekt bada, **czy modelowanie czasu w klasyfikatorze dźwięku daje przewagę nad modelem, który
patrzy tylko na obraz spektrogramu**. Zbudowano kompletny, odtwarzalny pipeline: pobranie nagrań
12 gatunków polskich ptaków z xeno-canto → podział bez wycieku → okienkowanie i cache
log-mel → trening czterech rodzin modeli → macierz ablacji jednoczynnikowych → ewaluacja
i analiza błędów → demo.

Najlepszy wynik: **CNN z oknem 10 s — test macro-F1 0,736** (top-1 0,772, top-3 0,902) na
307 nagraniach testowych. Rekurencja (CRNN-GRU) nie dała przewagi nad CNN (+0,003, w granicach
szumu jednego seeda), dłuższe okno i transfer z ImageNet dały mierzalny zysk, a „pełna”
augmentacja przy budżecie 15 epok wyszła gorzej niż podstawowa. Wszystko jest odtwarzalne:
`runs/index.csv` + snapshoty konfiguracji + seed + `git_sha` + urządzenie.

> **Prościej:** mamy 12 ptaków i nagrania z internetu. Program zamienia dźwięk na „obrazek”
> (spektrogram), uczy kilka modeli rozpoznawania i sprawdza, który jest najlepszy. Dodatkowo
> testujemy, czy pomaga patrzenie na **kolejność** dźwięków w czasie — nie pomaga bardziej niż
> szum losowości, ale pomaga dłuższe okno i wstępne uczenie na obrazach.

---

## 2. Pytanie badawcze i zakres

- **Pytanie główne:** ile daje modelowanie czasu (CRNN: CNN → BiGRU/BiLSTM) w porównaniu
  z modelem bez rekurencji (CNN) przy **tym samym zbiorze, podziale, seedzie i budżecie**?
- **Pytania dodatkowe (ablacje):** znaczenie długości okna (3/5/10 s), augmentacji
  (none/basic/full), transferu z ImageNet (scratch vs pretrained) i sposobu agregacji
  (mean vs attention).
- **W zakresie:** 12 gatunków (w tym 4 pary trudne do rozróżnienia), Polska, nagrania
  xeno-canto jakości A/B, praca na CPU/GPU, demo offline.
- **Poza zakresem:** klasyfikacja wieloetykietowa, gatunki spoza 12, detekcja w czasie,
  aplikacja mobilna, użycie komercyjne.

> **Prościej:** pytanie brzmi „czy RNN bije CNN?”. Żeby odpowiedź była uczciwa, wszystkie modele
> muszą dostać dokładnie te same dane i te same warunki — inaczej porównanie nic nie znaczy.

---

## 3. Dane

| Element | Wartość |
|---|---|
| Źródło | xeno-canto API v3, `https://xeno-canto.org/api/3/recordings` |
| Filtr | tagi `sp:"<gatunek>"`, `cnt:poland`, `q:">C"` (jakość A/B); tag `status` wyłączony, bo API zwraca na niego HTTP 400 |
| Gatunki | 12, w tym 4 pary mylące: wróbel/mazurek, bogatka/modra, kos/kwiczoł, pierwiosnek/piecuszek |
| Katalog | 2917 nagrań (`data/catalog.jsonl`), 0 zredagowanych |
| Audio | 2917 plików, 8,2 GB, 3 756 min (~62,6 h) |
| Cache | 2897 nagrań, 131 900 okien, 6,06 GB; 12,7% okien odrzuconych przez bramkę ciszy |
| Braki | 20 nagrań bez cache: 19 cisza (`empty_after_gating`), 1 błąd dekodowania; **0 w zbiorze testowym** |
| Duplikaty | 1 para tego samego pliku (ten sam `sha256`), oba w train — nie wyciek |
| Podział | train 2203 / val 407 / test 307 nagrań; okna: 95 898 / 24 982 / 11 020 |
| Zasada podziału | grupowanie po nagrywającym (`rec`); zero wspólnych `id`, `rec` i `sha256` między podziałami (`scripts/check_leakage.py`) |
| Licencje | głównie CC BY-NC-SA/ND (1940 + 521 + 223 + 125 + 106 + 1 + 1), użycie niekomercyjne |

> **Prościej:** pobieramy nagrania z publicznej bazy xeno-canto. Dzielimy je na „naukę”,
> „podgląd” i „egzamin”. Dzielimy **całymi nagraniami i nagrywającymi**, żeby ten sam ptak
> ani ten sam mikrofon nie trafił do dwóch części — to najczęstszy błąd w takich projektach.

---

## 4. Preprocessing (audio → spektrogram)

| Parametr | Wartość |
|---|---|
| Częstotliwość | 32 000 Hz, mono |
| Filtr górnoprzepustowy | 100 Hz, Butterworth rząd 4, filtfilt |
| Okno / hop | 3,0 s / 1,5 s (50% zakładki); krótkie ostatnie okno dopełniane zerami i maskowane |
| Bramka ciszy | okno odrzucone, gdy RMS < max(1e-3, 0,1 × p90 RMS nagrania) |
| log-mel | `n_fft=1024`, `win_length=1024`, `hop_length=512`, 128 pasm mel, `fmin=150`, `fmax=15000`, `power=2`, `top_db=80` |
| Normalizacja | per okno: `(x − mean) / std` (zapisywane już znormalizowane) |
| Cache | `data/cache/<id>/mel.npy` float16 + `meta.json` (`kept`, `valid_frames`, `config_hash`, `sha256_audio`) |

Cache jest w pełni odtwarzalny z `data/raw`; zmiana parametrów spektrogramu zmienia `config_hash`
i unieważnia cache. Odczyt idzie przez `numpy.memmap`, więc trening nie wczytuje całych nagrań.

> **Prościej:** spektrogram to „obrazek dźwięku”: oś pozioma = czas, pionowa = wysokość tonu,
> kolor = głośność. Tniemy nagranie na 3-sekundowe kawałki, wyrzucamy ciszę i każdy kawałek
> zapisujemy jako mały plik, żeby nie liczyć go w kółko.

---

## 5. Modele

| Model | Implementacja | Parametry | Rola |
|---|---|---|---|
| MFCC + SVM/RF | 40 MFCC → mean+std = 80 cech, `StandardScaler` fit na train, SVM RBF `class_weight=balanced` lub Random Forest | — | baseline klasyczny (tani) |
| Mały CNN | 4 bloki konwolucyjne 32/64/128/256, BN, ReLU, max-pool 2×2, GAP, dropout 0,3 | 391 404 | model „bez czasu” (rdzeń porównania) |
| Transfer | ResNet18 / EfficientNet-B0, wejście 1-kanałowe (wagi RGB sumowane), nowa głowica 12 klas, `--pretrained`/`--scratch` | 11 176 396 (ResNet18) | czy pomaga wiedza z ImageNet |
| CRNN | backbone CNN → rzut ramki do 256 → BiGRU/BiLSTM(hidden 128) → pooling `mean`/`attention`, maska długości okna | 1 212 396 | model „z czasem” (pytanie główne) |

Rejestr modeli (`models_registry`) mapuje nazwę + opcje (`arch`, `pretrained`, `pool`, `rnn`)
na fabrykę; CLI nie zna konkretnych klas. Trening i ewaluacja obsługują zmienną długość okna
(T=188 dla 3 s, T=313 dla 5 s, T=625 dla 10 s).

> **Prościej:** CNN to „wzrokowanie” po obrazku spektrogramu. RNN czyta obrazek linijka po linijce
> i pamięta poprzednie linijki — teoretycznie lepiej rozumie ptasi śpiew. Transfer to model,
> który najpierw nauczył się rozpoznawać zwykłe zdjęcia, a potem dostrajamy go do ptaków.

---

## 6. Trening

| Parametr | Wartość |
|---|---|
| Optymalizator | AdamW, `lr=1e-3` (transfer fine-tune `3e-4`), `weight_decay=1e-4` |
| Batch | 32 |
| Epoki | max 50 (transfer 20); w macierzy ablacji wspólny budżet **15 epok** |
| Early stopping | patience 8 po val macro-F1, `min_delta=1e-3` |
| Stabilizacja | gradient clipping 5,0, scheduler cosine, warmup 1 epoka |
| Klasy | ważenie `balanced` (odwrotność liczności), `label_smoothing=0` |
| Augmentacje | `basic`: przesunięcie ±1 s + szum SNR 5–20 dB; `full`: + mixup α=0,4 + SpecAugment (2 maski częstotliwości ≤16 pasm, 2 maski czasu ≤20 ramek); **gain pominięty** (normalizacja per okno znosi stałą zmianę poziomu) |
| Walidacja/test | zawsze czyste (bez augmentacji) |
| Determinizm | `set_seed(seed)` **przed** utworzeniem modelu; na CPU w pełni powtarzalne, na GPU raportujemy urządzenie |
| Sprzęt | RTX 5070 12 GB, `torch 2.15.0+cu132`, CUDA 13.2, Windows; pełna epoka CNN ≈ 130 s |

> **Prościej:** model uczy się na paczkach 32 kawałków; po każdej epoce sprawdzamy wynik na
> „podglądzie” i jeśli od 8 epok nie ma poprawy, przestajemy. Augmentacje to sztuczne utrudnianie
> danych, żeby model nie uczył się na pamięć.

---

## 7. Eksperymenty (macierz ablacji)

Zasada: w każdej grupie zmieniamy **dokładnie jeden** czynnik względem runu referencyjnego;
wszystkie runy dzielą split, seed 42 i budżet 15 epok. Każdy run zapisuje `config.yaml`, seed,
`device` i `git_sha`; definicja grupy trafia do `runs/ablation_<grupa>.json`.

| Grupa | Warianty | Pytanie |
|---|---|---|
| `rnn` | CNN / CRNN-GRU / CRNN-LSTM | czy rekurencja daje przewagę |
| `window` | 3 s / 5 s / 10 s | ile kontekstu potrzeba |
| `aug` | none / basic / full | co daje augmentacja |
| `pretrained` | scratch / ImageNet | ile daje transfer |
| `pool` | mean / attention | czy attention nad sekwencją pomaga |

Razem **13 runów na pełnym zbiorze** (grupy `window` mają mniej okien, bo dłuższe okno = mniej
kawałków). Porównania wewnątrz grupy są ważne; między grupami traktujemy je jako opis, nie
rywalizację (każda grupa ma własny punkt odniesienia).

> **Prościej:** to test „zmieniamy jedną rzecz naraz”. Dzięki temu wiadomo, czemu wynik się
> poprawił albo pogorszył — a nie tylko, że „model B wyszedł lepszy”.

---

## 8. Ewaluacja i metryki

| Metryka | Definicja | Rola |
|---|---|---|
| **macro-F1 (nagrania)** | F1 liczony per klasa, potem średnia **równo po klasach**, agregacja po oknach do nagrania | metryka główna (klasy nierówne) |
| top-1 / top-3 | czy prawda jest typem nr 1 / w pierwszej trójce | praktyczna użyteczność |
| balanced accuracy | średni recall po klasach | kontrola nierównowagi |
| agregacja `mean` | średnia prawdopodobieństw okien | główna predykcja nagrania |
| agregacja `vote` | głosowanie etykiet okien (remis rozstrzyga suma prawd.) | raportowana obok |
| coverage vs accuracy | przy progu pewności: ile nagrań zostaje i jaka jest ich trafność | kalibracja progu „nie wiem” |
| macierz pomyłek + per-class P/R/F1 | pełny obraz błędów | analiza błędów |
| pary mylące | 10 najczęstszych par prawda→predykcja | diagnostyka podobnych gatunków |

Ewaluacja (`eval`) odtwarza model **z konfiguracji runu** (geometria okna, model, opcje),
a nie z bieżącego `configs/`, i zapisuje `runs/<run>/eval_<split>/` (metryki, per-class,
macierz CSV+PNG, predykcje). `error-analysis` dokłada pary, recall per gatunek i najgorsze
nagrania z metadanymi (`q`, `type`, `also`, `rmk`, liczba okien, udział odrzuconych).

> **Prościej:** nie patrzymy na samą „skuteczność”, bo klas jest nierówno — liczymy średnią
> jakość po gatunkach. Sprawdzamy też, czy błąd to mylenie dwóch podobnych ptaków, czy coś
> dziwnego (np. w nagraniu w tle śpiewa inny gatunek).

---

## 9. Wyniki

**Test macro-F1 na poziomie nagrania** (seed 42, pełny zbiór, 15 epok z early stopping):

| Grupa | Porównanie | Wynik |
|---|---|---|
| `rnn` | CNN / CRNN-GRU / CRNN-LSTM | **0,704 / 0,707 / 0,668** |
| `window` | 3 s / 5 s / 10 s | **0,704 / 0,712 / 0,736** |
| `aug` | none / basic / full | **0,685 / 0,704 / 0,661** |
| `pretrained` | scratch / ImageNet | **0,644 / 0,698** |
| `pool` | mean / attention | **0,707 / 0,716** |

- **Najlepszy model całości:** CNN + okno 10 s — macro-F1 **0,736**, top-1 **0,772**,
  top-3 **0,902**, val 0,721.
- **Rekurencja:** GRU +0,003 nad CNN (szum jednego seeda), LSTM −0,036 — brak podstaw do
  ogłoszenia przewagi RNN.
- **Kontekst:** 10 s vs 3 s = +0,032; **transfer:** +0,054; **attention:** +0,009 (szum).
- **Augmentacja:** `basic` > `none` (+0,019), ale `full` < `basic` przy 15 epokach (−0,043) —
  mixup i SpecAugment potrzebują dłuższego treningu, to obserwacja o budżecie.
- **Powtarzalność:** te same konfiguracje w różnych grupach dały identyczne liczby (0,704; 0,707)
  — niezależne runy są powtarzalne na GPU.
- **Najsłabsze klasy** (najlepszy run): wróbel F1 0,400 (n=4 w teście), szczygieł 0,476 (n=6);
  najmocniejsze: piecuszek 0,938, dzwoniec 0,867, zięba 0,866. Macro-F1 zaniżają małe klasy.
- **Pary mylące (błędy):** bogatka→modra, rudzik→modra, pierwiosnek→bogatka/szczygieł/rudzik;
  3/10 najgorszych nagrań ma niepuste `also` (podejrzenie szumu etykiet).

> **Prościej:** najlepiej wypada model bez RNN, ale z **dłuższym oknem** (10 s) i z wstępnym
> uczeniem na ImageNet. RNN nie zaszkodził, ale nie pomógł bardziej niż losowość. Wynik 0,74
> oznacza „średnio dobra klasyfikacja 12 gatunków”; top-3 0,90 oznacza, że prawdziwy ptak
> prawie zawsze jest w pierwszej trójce.

---

## 10. Reprodukcja i artefakty

```bash
# dane (raz): katalog -> audio -> podział -> cache
python -m bird_recognizer.cli data catalog && python -m bird_recognizer.cli data download
python -m bird_recognizer.cli data split && python scripts/check_leakage.py --data-dir data
python -m bird_recognizer.cli prep cache && python -m bird_recognizer.cli prep cache --verify

# trening / ablacje / raport
python -m bird_recognizer.cli train cnn --smoke
python -m bird_recognizer.cli ablate --group rnn --epochs 15
python -m bird_recognizer.cli report --runs <id...> --out docs/experiments.md

# ewaluacja i analiza
python -m bird_recognizer.cli eval --run <id> --split test
python -m bird_recognizer.cli aggregate --run <id> --method vote
python -m bird_recognizer.cli error-analysis --run <id>

# demo (mikrofon lub plik)
python -m bird_recognizer.cli demo --run <id>
```

Artefakty runu: `runs/<run_id>/{config.yaml, env.txt, metrics.json, metrics.jsonl, best.pt}`;
jeden wiersz w `runs/index.csv`; `env.txt` ma wersje bibliotek, `git_sha`, liczbę parametrów
i urządzenie. Dane i wyniki są **poza repo** (rozmiar + licencje); repo zawiera kod, konfigurację,
dokumentację i demo.

> **Prościej:** żeby powtórzyć wynik, potrzebujesz tego samego configu, seeda, kodu (commit)
> i danych. Wszystko to jest zapisane w katalogu runu — bez tego liczba w tabeli byłaby
> nie do obrony.

---

## 11. Jakość inżynierska

- **Testy:** 259 testów `pytest`, offline (blokada gniazd, generowane WAV-y), pokrycie ~91%
  (m.in. `eval/` 96%, `train/` 93–100%, `models/` 83–100%).
- **Jakość kodu:** `ruff check` bez błędów, `from __future__ import annotations`, typy.
- **Błędy i kody wyjścia** (`0` ok, `1` użytkownik, `2` API/klucz, `3` dane/audio, `4` model).
- **Bezpieczeństwo:** klucz API tylko w `.env`; logi maskują `key=`; audio nie w repo;
  demo nie zapisuje nagrań i nie wysyła ich nigdzie (Gradio bez telemetrii).
- **Obsłużone awarie:** przepełnienie stosu cuBLASLt na CUDA 13.2 (rozwiązane wątkiem z większym
  stosem), `torch` CPU vs CUDA, dekodowanie mp3 przez `soundfile` (torchaudio wymagał TorchCodec).

> **Prościej:** to nie jest zbiór skryptów „na zaliczenie” — każda liczba ma run, testy pilnują
> powtarzalności, a kod jest sprawdzany linterem.

---

## 12. Ograniczenia (uczciwie)

1. **Jeden seed** — różnice ±0,01 (RNN vs CNN, mean vs attention) nie są istotne; potrzebne 3 seeda.
2. **Małe klasy** — wróbel 4 nagrania testowe, szczygieł 6; macro-F1 jest przez to niestabilny.
3. **Szum etykiet** — brak filtra `status:identified` (API go odrzuca), `also` w tle, tylko jakość A/B.
4. **Domena** — model uczył się na mikrofonach kierunkowych; **brak nagrań z telefonu**, więc
   nie wiadomo, jak działa w terenie.
5. **12 gatunków, jeden kraj** — każde nagranie spoza listy zostanie przypisane do jednej z 12 klas
   (próg pewności „nie wiem” jest dopiero planowany w aplikacji).
6. **Budżet** — macierz liczona na 15 epokach; niektóre efekty (np. `aug full`) mogą wymagać więcej.
7. **GPU** — mimo seeda może być niedeterministyczne; raportujemy urządzenie przy każdym runie.

> **Prościej:** wynik jest dobry jak na projekt studencki, ale nie publikacyjny — za mało
> powtórzeń, zbyt małe klasy i brak danych z prawdziwego telefonu. Na obronie trzeba to powiedzieć
> wprost, a nie tylko pokazać najlepszą liczbę.

---

## 13. Wnioski (fakty, nie interpretacja)

- Modelowanie czasu **nie dało** mierzalnej przewagi przy tym budżecie: CRNN-GRU 0,707 vs CNN 0,704.
- **Największy zysk** dały: dłuższe okno (10 s: +0,032) i transfer z ImageNet (+0,054).
- **Agregacja attention** minimalnie lepsza od mean (+0,009), w granicach szumu.
- **`aug full`** przy 15 epokach nie zdążyła pomóc — potrzebny dłuższy trening.
- Pipeline jest **odtwarzalny** (identyczne wyniki dla tych samych konfiguracji) i **kompletny**
  (dane → trening → ablacje → ewaluacja → demo).

> **Prościej:** odpowiedź na pytanie badawcze brzmi „przy tym budżecie RNN nic nie daje”.
> To też jest wynik — i uczciwszy niż dopasowywanie eksperymentu pod tezę.

---

## 14. Słowniczek (dla autora)

| Termin | Prościej |
|---|---|
| **log-mel / spektrogram** | obrazek dźwięku: czas × wysokość tonu × głośność |
| **okno / hop** | kawałek nagrania, na który patrzy model / o ile przesuwamy kolejny kawałek |
| **bramka ciszy** | odrzucanie kawałków bez śpiewu (cisza uczy modelu ciszy) |
| **CNN** | sieć szukająca wzorów na obrazku (kształtów na spektrogramie) |
| **RNN / GRU / LSTM** | sieć czytająca sekwencję i pamiętająca poprzednie kroki |
| **CRNN** | CNN + RNN: cechy z obrazka czytane w czasie |
| **transfer / ImageNet** | model najpierw uczył się zwykłych zdjęć, potem dostrajamy do ptaków |
| **macro-F1** | średnia jakość po gatunkach (każdy równo), odporna na nierówne klasy |
| **top-3** | czy prawda jest w pierwszej trójce typów |
| **agregacja mean/vote** | jak połączyć typy z kawałków w typ całego nagrania |
| **ablacja** | zmieniamy jedną rzecz i patrzymy, czy pomaga |
| **run** | jeden trening z pełnym zapisem (config, seed, model, metryki, `git_sha`) |
| **seed** | liczba-startowa losowości; ten sam seed = powtarzalny eksperyment |
| **cache** | przeliczone raz dane, żeby nie liczyć ich w kółko |
| **smoke** | szybki przebieg techniczny (2 nagrania/gatunek); **nie jest wynikiem badania** |

---

## 15. Gdzie szukać szczegółów

| Temat | Plik |
|---|---|
| Wymagania, decyzje, etapy, bramki | [`project info/SPEC.md`](../project%20info/SPEC.md) |
| Plan projektu | [`project info/PLAN.md`](../project%20info/PLAN.md) |
| Dziennik pracy AI (w tym pomyłki) | [`docs/ai-journal.md`](ai-journal.md) |
| Wygenerowana tabela wyników | [`docs/experiments.md`](experiments.md) |
| Fakty o API xeno-canto | [`docs/api-notes.md`](api-notes.md) |
| Przewodnik po projekcie bez żargonu | [`docs/przewodnik.md`](przewodnik.md) |
| Specyfikacja aplikacji (wariant A+B) | [`project info/APP_SPEC.md`](../project%20info/APP_SPEC.md) |
