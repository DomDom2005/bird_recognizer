# bird_recognizer

Klasyfikacja gatunku ptaka z nagrania: mel-spektrogramy, CNN, transfer wiedzy i CRNN.
Projekt badawczy (studia AI), pytanie: **ile daje modelowanie czasu w porównaniu z modelem,
który patrzy na spektrogram okna bez rekurencji?**

> **Status: etap 4 z 8** (patrz `project info/SPEC.md` §15). Dane, cache, baseline MFCC+SVM
> i mały CNN są gotowe; modele główne (transfer, CRNN) i ablacje w etapach 5–6, demo w etapie 8.

## 1. Cel i pytanie badawcze

Model dostaje 3-sekundowe okno nagrania i zwraca gatunek. Porównujemy cztery rodziny modeli
na tym samym zbiorze, podziale i augmentacjach:

- baseline klasyczny: MFCC + statystyki → SVM / Random Forest,
- mały CNN bez rekurencji (predykcja nagrania = średnia po oknach),
- ResNet18 / EfficientNet-B0 z transferem wiedzy,
- CRNN: CNN → BiGRU → pooling (mean / attention) po osi czasu.

Rdzeń raportu to tabela ablacji jednoczynnikowych: z RNN i bez RNN, długość okna, augmentacje,
od zera vs. wstępnie wytrenowane, pooling. Szczegóły: `project info/SPEC.md` §10–§11.

## 2. Dane

- Źródło: **xeno-canto API v3** (`https://xeno-canto.org/api/3/recordings`).
- Filtr domyślny: `sp:"<gatunek>" cnt:poland q:">C"` (Polska, jakość A/B). Wartości w
  `configs/default.yaml`.
- 12 gatunków, w tym cztery pary mylące się (`configs/species.yaml`, decyzja D1).
- Dostępność sprawdzona 2026-11-02: **81–606 nagrań na gatunek** (najmniej szczygieł 81,
  najwięcej bogatka 606) — wszystkie powyżej progu 50. Szczegóły: `docs/api-notes.md` §4.
- Katalog zbudowany 2026-11-18: **2917 nagrań**, 0 zredagowanych (wszystkie z tej listy mają plik).
- Podział train/val/test **według nagrań i grup nagrywających**, z kontrolą wycieku
  (`scripts/check_leakage.py`): train 2203 (7 nagrywających) / val 407 (43) / test 307 (43),
  wszystkie 12 gatunków w każdej części, 0 ostrzeżeń.
- Licencje nagrań są różne (Creative Commons); **audio nie trafia do repo**.

## 3. Przepływ danych

Diagram Mermaid i opis katalogów: [`docs/data-flow.md`](docs/data-flow.md).

## 4. Instrukcja uruchomienia

Wymagania: Python 3.11+ (testowane na 3.14.6), pip, ffmpeg (dekodowanie mp3).

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

Klucz API: skopiuj `.env.example` do `.env` i wpisz klucz xeno-canto. Kod czyta
`BIRD_XC_API_KEY`, a także starszy alias `xc_api_key` (patrz `docs/ai-journal.md`).

Sprawdzenie konfiguracji bez sieci:

```bash
python scripts/probe_api.py --dry-run
python scripts/probe_api.py --live      # jeden request na punkt; wymaga klucza
```

Pobranie i przygotowanie danych (etapy 2–3):

```bash
bird_recognizer data catalog            # metadane wszystkich gatunków -> data/catalog.jsonl
bird_recognizer data download --limit 10   # audio -> data/raw, wznawialne
bird_recognizer data split              # -> data/splits.json
python scripts/check_leakage.py --data-dir data
bird_recognizer prep cache --limit 24   # okna + log-mel -> data/cache (float16)
bird_recognizer prep cache --verify     # kontrola config_hash, kształtów i dtype
```

Trening baseline'ów (etap 4):

```bash
bird_recognizer train mfcc --clf svm    # albo --clf rf
bird_recognizer train cnn               # mały CNN bez rekurencji
bird_recognizer --smoke train cnn       # szybki podzbiór (2 nagrania/gatunek, 1 epoka)
```

Każdy przebieg zapisuje `runs/<run_id>/{config.yaml, env.txt, metrics.json, metrics.jsonl,
best.pt | model.joblib}` i dopisuje wiersz do `runs/index.csv`.

Jeśli komenda `bird_recognizer` nie jest w PATH (tak jest na części instalacji Windows,
gdzie `pip` instaluje skrypty do katalogu spoza PATH), użyj równoważnej formy
`python -m bird_recognizer.cli …`.

Kolejne kroki (etapy 5–8) opisuje `project info/SPEC.md` §15: karta etapu = zadania + bramka.

## 5. Tryb bez internetu

- `python scripts/probe_api.py --dry-run` — waliduje config i wypisuje zapytania, zero requestów.
- Testy generują własne krótkie WAV-y w `tmp_dir` (`tests/support.py`) — zero sieci i zero
  prawdziwych nagrań w testach. Nie ma osobnego „zbioru syntetycznego”; `--smoke` korzysta
  z realnego katalogu z `--limit`.

## 6. Wyniki

Dla etapu 4: brak wyników badawczych — smoke runy służą tylko sprawdzeniu pipeline'u
(CNN: 391 404 parametry, 1 epoka ≈ 23 s CPU; `--epochs 2` daje spadek train loss 1,74 → 0,81).
Tabela ablacji powstanie w `docs/experiments.md` generowanym z `runs/index.csv` (etap 6).
Wnioski badawcze dopisuje człowiek.

## 7. Testy

```bash
pytest                 # całość, offline (gniazda zablokowane)
ruff check .
pytest --cov=bird_recognizer.models --cov=bird_recognizer.train   # pokrycie etapu 4
```

Stan: **170 testów**; pokrycie `models/` 83–100%, `train/` 93–100% (razem 92%), `data/`
88–100%, `xc/` 90–94%. Zakres: konfiguracja, redakcja sekretów, blokada sieci, klient API,
katalog, pobieranie, podział i wyciek, audio, okna, log-mel, cache, modele, augmentacje,
pętla treningu, metryki, runner i CLI.

## 8. Struktura repo

```
bird_recognizer/
├── project info/          # SPEC.md, PLAN.md, AGENTS.md
├── configs/               # default.yaml (progi), species.yaml (12 gatunków)
├── src/bird_recognizer/   # config, errors, logging_setup, cli, models_registry
│   ├── xc/                # client, catalog, download
│   ├── data/              # splits, audio, windows, spectrogram, dataset (cache)
│   ├── models/            # cnn, mfcc_baseline
│   ├── train/             # augment, loop, runner
│   └── eval/              # metrics (+ etap 5: aggregate, error_analysis)
├── scripts/               # probe_api.py, check_leakage.py
├── tests/                 # + fixtures/api, fixtures/synthetic_data, support.py (WAV-y)
├── docs/                  # api-notes, ai-journal, data-flow, manual-checklist
└── README.md
```

Pełne drzewo: `project info/SPEC.md` §5. Dane (`data/`) i wyniki (`runs/`) są poza repo.

## 9. Ograniczenia

- Jeden kraj (Polska) i jedna „scena” nagraniowa: mikrofony kierunkowe, śpiew z bliska.
- **Brak filtra jakości identyfikacji**: tag `status:identified` z dokumentacji jest odrzucany
  przez API (HTTP 400), więc szum etykiet jest w danych i raportowany w analizie błędów.
- Nagrania wielogatunkowe (pole `also`) to szum etykiet.
- Gatunki chronione są zredagowane w API — nie mają plików audio.
- Własne nagrania z telefonu to inna domena (tło miasta, odległość) — spadek jakości jest
  oczekiwany i raportowany osobno.
- Trening na GPU bywa niedeterministyczny mimo seeda; raportujemy urządzenie przy każdym runie.
- Grupy nagrywających są bardzo nierówne (jeden nagrywający = setki plików), więc train zbiera
  kilku najpłodniejszych, a val/test — dziesiątki mniejszych; podział jest deterministyczny
  i bez wycieku, ale nie jest losowaniem pojedynczych nagrań.
- Liczby w README to snapshoty z podaną datą; katalog xeno-canto rośnie, więc przed kolejnym
  etapem odśwież `data catalog` i zapisz nowe wartości.
- Cache spektrogramów zajmuje **~1,5 MB na minutę audio** (~5,6 GB dla całego zbioru), więc
  `data/cache` trzymaj poza repo i usuwaj, gdy brakuje miejsca — jest w pełni odtwarzalny.
- Bramka ciszy (`abs_floor` ≈ −60 dBFS RMS) odrzuca bardzo ciche nagrania w całości
  (`empty_after_gating`); na próbce 24 nagrań dotyczyło to 2 (peak −42 i −34 dBFS).
- `TODO(verify)` z `project info/SPEC.md` §4.3 są otwarte do czasu uruchomienia `probe_api.py --live`.

## 10. Bezpieczeństwo i licencje

- Klucz API tylko w `.env` (poza repo). Klucz jedzie w query stringu, więc logi maskują `key=...`.
- Nagrania i wytrenowane modele nie są commitowane; repo zawiera kod, konfigurację i dokumentację.
- Atrybucja nagrań (gatunek, nagrywający, numer XC, licencja, link) trafi do raportu i demo
  w etapie 8.

## 11. Dziennik AI

Zapis pracy z AI, w tym pomyłki i odstępstwa od SPEC: [`docs/ai-journal.md`](docs/ai-journal.md).
