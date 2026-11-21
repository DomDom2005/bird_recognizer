# bird_recognizer

Klasyfikacja gatunku ptaka z nagrania: mel-spektrogramy, CNN, transfer wiedzy i CRNN.
Projekt badawczy (studia AI), pytanie: **ile daje modelowanie czasu w porównaniu z modelem,
który patrzy na spektrogram okna bez rekurencji?**

> **Status: etap 7 z 8** (patrz `project info/SPEC.md` §15). Dane, cache, modele, ablacje oraz
> ewaluacja i analiza błędów są gotowe; pełne runy macierzy dochodzą po etapie 7, a demo,
> raport końcowy i domknięcie to etap 8.

> **Nowy w projekcie?** Zacznij od [`docs/przewodnik.md`](docs/przewodnik.md) — prosty słownik
> terminów i wyjaśnienie, jak wszystko się łączy, bez żargonu.

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

Wymagania: Python 3.11+ (testowane na 3.14.6), pip, ffmpeg (dekodowanie mp3), GPU NVIDIA do
pełnych runów (testowane na RTX 5070 12 GB, CUDA 13.2; bez GPU wszystko działa na CPU, tylko wolniej).

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cu132
pip install -e .
```

`requirements.txt` przypina `torch==2.15.0+cu132` (build CUDA); na maszynie bez GPU zainstaluj
`torch==2.15.0` z PyPI albo pomiń `--extra-index-url` i popraw pin.

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

Trening modeli (etapy 4–5):

```bash
bird_recognizer train mfcc --clf svm    # albo --clf rf
bird_recognizer train cnn               # mały CNN bez rekurencji
bird_recognizer train crnn --pool attention   # CRNN: CNN + BiGRU + pooling
bird_recognizer train transfer --arch resnet18 --scratch   # ablacja: bez wag ImageNet
bird_recognizer --smoke train cnn       # szybki podzbiór (2 nagrania/gatunek, 1 epoka)
```

`train transfer` bez `--scratch` używa wag ImageNet (dociągane raz do `~/.cache/torch`;
gdy ten katalog jest niezapisywalny, wskaż inny przez `TORCH_HOME`).

Ablacje i tabela wyników (etap 6):

```bash
bird_recognizer ablate --group rnn --smoke     # macierz jednoczynnikowa (5 grup)
bird_recognizer report --runs RUN1 RUN2 RUN3 --out docs/experiments.md
```

Każdy run macierzy to osobny katalog w `runs/`; definicja grupy (run referencyjny i różnice)
ląduje w `runs/ablation_<grupa>.json`. `report` generuje `docs/experiments.md` z `runs/index.csv`,
a sekcję „Wnioski” uzupełnia człowiek.

Ewaluacja i analiza błędów (etap 7):

```bash
bird_recognizer eval --run RUN --split test    # metryki okna i nagrania, macierz pomyłek
bird_recognizer aggregate --run RUN --method vote   # mean vs vote + coverage
bird_recognizer error-analysis --run RUN       # pary mylące, najgorsze nagrania
bird_recognizer eval --run RUN --split own     # zbiór własny: data/own/<gatunek>/*.wav
```

`eval` zapisuje `runs/<run>/eval_<split>/` (`metrics.json`, `per_class.csv`,
`confusion_matrix.csv`, `confusion_matrix.png`, `predictions.csv`), `aggregate` dopisuje
`aggregation.csv` i `coverage.csv`, a `error-analysis` — `runs/<run>/analysis/errors.md` + CSV.
Zbiór własny (`--split own`) czytamy z `data/own/<gatunek>/`; brak katalogu oznacza kod 3
i jawne „brak nagrań własnych” w raporcie — nic nie jest podstawiane w zamian.

Demo (etap 8):

```bash
bird_recognizer demo --run RUN             # Gradio: plik → top-3 + spektrogram z oknami
bird_recognizer demo --run RUN --smoke     # bez serwera: sprawdza ścieżkę na tonie syntetycznym
```

Demo działa na CPU, bez sieci; brak `best.pt` daje czytelny komunikat (kod 4), nie traceback.

Każdy przebieg zapisuje `runs/<run_id>/{config.yaml, env.txt, metrics.json, metrics.jsonl,
best.pt | model.joblib}` i dopisuje wiersz do `runs/index.csv`.

Jeśli komenda `bird_recognizer` nie jest w PATH (tak jest na części instalacji Windows,
gdzie `pip` instaluje skrypty do katalogu spoza PATH), użyj równoważnej formy
`python -m bird_recognizer.cli …`.

Kolejny krok (etap 8) opisuje `project info/SPEC.md` §15: karta etapu = zadania + bramka.

## 5. Tryb bez internetu

- `python scripts/probe_api.py --dry-run` — waliduje config i wypisuje zapytania, zero requestów.
- Testy generują własne krótkie WAV-y w `tmp_dir` (`tests/support.py`) — zero sieci i zero
  prawdziwych nagrań w testach. Nie ma osobnego „zbioru syntetycznego”; `--smoke` korzysta
  z realnego katalogu z `--limit`.
- `--pretrained` w transferze to jedyne miejsce, które potrzebuje sieci (pobranie wag ImageNet,
  raz); testy podstawiają backbone bez wag, a `--scratch` działa w pełni offline.

## 6. Wyniki

Dla etapu 7: `docs/experiments.md` jest generowane z `runs/index.csv` i wzbogacane o sekcję par
mylących z `eval_test/per_class.csv` oraz o tabelę test vs zbiór własny (`eval_own`). Tabela
pochodzi jednak z przebiegów `--smoke` (2 nagrania/gatunek) — wiersze mają `smoke` w uwagach,
więc **nie są wynikami badania**. Pełne runy macierzy (5 grup, GPU RTX 5070, ~15 epok) są dokładane
po etapie 7; każdy run zapisuje `config.yaml`, seed, urządzenie i `git_sha`. Wnioski badawcze
dopisuje człowiek w sekcji 5 pliku. Zbiór własny (`data/own/<gatunek>/`) jest zadaniem człowieka —
bez niego raport jawnie pokazuje brak danych.

## 7. Testy

```bash
pytest                 # całość, offline (gniazda zablokowane)
ruff check .
pytest --cov=bird_recognizer.eval --cov-report=term-missing   # bramka pokrycia etapu 7
```

Stan: **253 testy**; pokrycie całości ~91% (`eval/` 96%, `models/` 83–100%, `train/` 93–100%,
`ablation.py` 90%, `data/` 88–100%, `xc/` 90–94%). Zakres: konfiguracja, redakcja sekretów,
blokada sieci, klient API, katalog, pobieranie, podział i wyciek, audio, okna, log-mel, cache,
modele (CNN, CRNN, transfer), augmentacje (basic i full), pętla treningu, metryki (macro-F1,
balanced accuracy, per-class, pary pomyłek), agregacja mean/vote z coverage, analiza błędów
(MD + CSV), ewaluacja runu z jego configu, macierz ablacji i generator raportu.

## 8. Struktura repo

```
bird_recognizer/
├── project info/          # SPEC.md, PLAN.md, AGENTS.md
├── configs/               # default.yaml (progi), species.yaml (12 gatunków)
├── src/bird_recognizer/   # config, errors, logging_setup, cli, models_registry, ablation
│   ├── xc/                # client, catalog, download
│   ├── data/              # splits, audio, windows, spectrogram, dataset (cache)
│   ├── models/            # cnn, crnn, transfer, mfcc_baseline
│   ├── train/             # augment (basic/full), loop, runner
│   └── eval/              # metrics, aggregate (mean/vote+coverage), error_analysis, evaluate, report
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
- Cache spektrogramów zajmuje **~1,6 MB na minutę audio** (6,06 GB dla 2897 nagrań / 3756 min),
  więc `data/cache` trzymaj poza repo i usuwaj, gdy brakuje miejsca — jest w pełni odtwarzalny.
- Bramka ciszy (`abs_floor` ≈ −60 dBFS RMS) odrzuca bardzo ciche nagrania w całości
  (`empty_after_gating`); na pełnym zbiorze dotyczyło to 19 nagrań, a 1 nie dało się zdekodować
  (razem 20/2917 bez cache; 0 w zbiorze testowym).
- Wszystkie punkty SPEC §4.3 zostały sprawdzone przez `probe_api.py --live` (wyniki:
  `docs/api-notes.md`); otwarte pozostaje tylko `ODŁOŻONE (D12)` — dostępność i licencje wag
  BirdNET / Perch / PANNs, potrzebne wyłącznie przy rozszerzeniu D12.
- Repozytorium docelowe `https://github.com/DomDom2005/bird_recognizer` **nie jest jeszcze
  utworzone na GitHubie** (potwierdzone 2026-11-21); branch docelowy `main`.

## 10. Bezpieczeństwo i licencje

- Klucz API tylko w `.env` (poza repo). Klucz jedzie w query stringu, więc logi maskują `key=...`.
- Nagrania i wytrenowane modele nie są commitowane; repo zawiera kod, konfigurację i dokumentację.

**Atrybucja nagrań.** Pełna lista (`id`, `rec`, `lic`, link `https://xeno-canto.org/<id>`) jest
w `data/catalog.jsonl`; poniżej rozkład licencji dla 2917 nagrań z katalogu (stan 2026-11-21):

| Licencja | Nagrania |
|---|---|
| CC BY-NC-SA 4.0 | 1940 |
| CC BY-NC-ND 4.0 | 521 |
| CC BY-NC-SA 3.0 | 223 |
| CC BY-NC-ND 2.5 | 125 |
| CC BY-NC-ND 3.0 | 106 |
| CC BY-NC 4.0 | 1 |
| CC BY-SA 4.0 | 1 |

Nagrania są używane wyłącznie do celów badawczych/edukacyjnych (większość licencji zabrania
użycia komercyjnego); pliki audio nie trafiają do repozytorium.

## 11. Dziennik AI

Zapis pracy z AI, w tym pomyłki i odstępstwa od SPEC: [`docs/ai-journal.md`](docs/ai-journal.md).
