# bird_recognizer

Klasyfikacja gatunku ptaka z nagrania: mel-spektrogramy, CNN, transfer wiedzy i CRNN.
Projekt badawczy (studia AI), pytanie: **ile daje modelowanie czasu w porównaniu z modelem,
który patrzy na spektrogram okna bez rekurencji?**

> **Status: etap 1 z 8** (patrz `project info/SPEC.md` §15). Ten README opisuje stan faktyczny:
> sekcje wyników są puste do etapu 6, a demo powstaje w etapie 8.

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
- Podział train/val/test **według nagrań i grup nagrywających**, z kontrolą wycieku
  (`scripts/check_leakage.py`).
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

Kolejne kroki (etapy 2–8) opisuje `project info/SPEC.md` §15: karta etapu = zadania + bramka.

## 5. Tryb bez internetu

- `python scripts/probe_api.py --dry-run` — waliduje config i wypisuje zapytania, zero requestów.
- `python scripts/make_synthetic_demo.py` (etap 3) — mały syntetyczny zbiór do testów end-to-end.
- Testy nigdy nie dotykają sieci: `tests/conftest.py` blokuje gniazda na całą sesję.

## 6. Wyniki

Dla etapu 1: brak — modeli jeszcze nie ma. Tabela ablacji powstanie w `docs/experiments.md`
generowanym z `runs/index.csv`. Wnioski badawcze dopisuje człowiek.

## 7. Testy

```bash
pytest                 # całość, offline (gniazda zablokowane)
ruff check .
```

Zakres: konfiguracja i jej walidacja, redakcja sekretów w logach, blokada sieci w testach.
Kolejne etapy dodają testy klienta API, podziału danych, okien, modeli i metryk.

## 8. Struktura repo

```
bird_recognizer/
├── project info/          # SPEC.md, PLAN.md, AGENTS.md
├── configs/               # default.yaml (progi), species.yaml (12 gatunków)
├── src/bird_recognizer/   # config, errors, logging_setup (+ pakiety etapów 2–7)
├── scripts/               # probe_api.py, (etap 2–3: check_leakage, synthetic demo)
├── tests/
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
- `TODO(verify)` z `project info/SPEC.md` §4.3 są otwarte do czasu uruchomienia `probe_api.py --live`.

## 10. Bezpieczeństwo i licencje

- Klucz API tylko w `.env` (poza repo). Klucz jedzie w query stringu, więc logi maskują `key=...`.
- Nagrania i wytrenowane modele nie są commitowane; repo zawiera kod, konfigurację i dokumentację.
- Atrybucja nagrań (gatunek, nagrywający, numer XC, licencja, link) trafi do raportu i demo
  w etapie 8.

## 11. Dziennik AI

Zapis pracy z AI, w tym pomyłki i odstępstwa od SPEC: [`docs/ai-journal.md`](docs/ai-journal.md).
