# APP_SPEC.md — aplikacja „nagraj albo wgraj” (wariant A + B)

> Wersja 1.0 · 2026-11-21 · Specyfikacja aplikacji dla agenta AI. Język kodu, komentarzy
> i commitów: **angielski**. Dokumentacja, komunikaty UI i raporty: **polski**.
>
> Ten plik opisuje **aplikację** zbudowaną na projekcie `bird_recognizer` (specyfikacja główna:
> `project info/SPEC.md`, reguły pracy: `project info/AGENTS.md`, przewodnik dla człowieka:
> `docs/przewodnik.md`). Aplikacja **nie zmienia** istniejącego modelu ani pipeline'u — używa
> wytrenowanych runów (`runs/<run_id>/best.pt` + `config.yaml` + `metrics.json`).

---

## 0. Zasady pracy agenta (przeczytaj najpierw)

1. **Zależność od planu głównego.** Aplikacja startuje dopiero po zamknięciu **etapu 8**
   głównego SPEC (mamy checkpointy, `eval_test`, tabelę wyników i — jeśli człowiek dostarczył —
   zbiór własny). Bez tego nie ma czego oceniać jakości.
2. **Jedna zmiana = jeden eksperyment.** Próg pewności, wybór runu i limit długości nagrania są
   zapisywane w konfiguracji, nie w kodzie. Zmiana progu = nowy zapis kalibracji, nie edycja wyników.
3. **Testy offline.** Żadnych prawdziwych nagrań ani sieci w testach; nagrania generowane przez
   `tests/support.py` (tone/noise), blokada gniazd z `conftest.py`.
4. **Zero sekretów.** Aplikacja nie potrzebuje klucza API; logi nie mogą zawierać ścieżek
   do przesłanych plików ani danych osobowych.
5. **Uczciwość wyniku.** UI zawsze pokazuje, że to wynik **eksperymentalny**, ograniczony do
   12 gatunków z xeno-canto (Polska). Nie udajemy pewności tam, gdzie jej nie ma (wariant B).
6. **Praca etapami** (sekcja 10): status w tabeli, każdy etap ma bramkę, commit, dziennik.

---

## 1. Cel i kontekst

**Cel:** dać człowiekowi prostą aplikację, w której może **nagrać głos ptaka mikrofonem** albo
**wgrać plik audio**, a dostanie **top-3 gatunki** z prawdopodobieństwami, spektrogram z oknami
i czytelne zastrzeżenie. W wariancie B aplikacja potrafi powiedzieć **„nie wiem / prawdopodobnie
to nie jeden z 12 gatunków”**, zamiast na siłę wybierać jeden z nich.

**Dla kogo:** właściciel projektu (do zaprezentowania na obronie), potem ewentualnie znajomi
do testów. **Nie** jest to aplikacja produkcyjna ani narzędzie do identyfikacji terenowej.

**Nie-cele (poza zakresem):** wieloetykietowość (kilka ptaków naraz), lokalizacja dźwięku
w czasie, detekcja gatunków spoza 12, wersja mobilna, hosting publiczny, użycie komercyjne.

**Szacunek pracy:** wariant A+B to ok. **1 tydzień** pracy agenta (etapy APP-1..APP-3 i APP-5);
APP-4 (pakowanie) jest opcjonalne i dokłada ok. 2–3 dni.

---

## 2. Wymagania funkcjonalne

### Wariant A — nagrywanie i wgrywanie

- **FR-A1 Mikrofon.** UI pozwala nagrać dźwięk bezpośrednio w przeglądarce
  (`gr.Audio(sources=["microphone", "upload"], type="filepath")`).
- **FR-A2 Plik.** UI przyjmuje plik audio: `wav`, `mp3`, `ogg`, `flac`, `m4a`.
- **FR-A3 Walidacja audio.** Nieczytelny plik albo nagranie bez żadnego okna (cisza) daje
  czytelny komunikat, nie traceback.
- **FR-A4 Predykcja.** Top-3 gatunki z prawdopodobieństwami (softmax), liczone dokładnie tak
  jak w `eval` (okna → model → średnia po oknach).
- **FR-A5 Spektrogram.** Wykres log-mel z zaznaczonymi oknami, które przeszły bramkowanie ciszy.
- **FR-A6 Karta modelu.** UI pokazuje: nazwę modelu, opcje (`pool`, `arch`, `pretrained`),
  run id, liczbę gatunków, okno, urządzenie i zastrzeżenie eksperymentalne.
- **FR-A7 Tryb offline/CPU.** Całość działa na CPU, bez sieci (poza pierwszym pobraniem wag
  ImageNet, jeśli run jest pretrained i wag nie ma w cache).
- **FR-A8 Wybór runu.** Użytkownik może wskazać run (`--run RUN` albo lista z `runs/index.csv`),
  a nie tylko domyślny.
- **FR-A9 Start jednym poleceniem.** `bird_recognizer app` uruchamia aplikację lokalnie
  (Gradio, domyślnie `http://127.0.0.1:7860`).
- **FR-A10 Smoke.** `bird_recognizer app --smoke` sprawdza całą ścieżkę na syntetycznym tonie
  bez uruchamiania serwera.

### Wariant B — open-set („nie wiem")

- **FR-B1 Próg pewności.** Predykcja ma próg (`threshold`); jeżeli maksymalne prawdopodobieństwo
  jest poniżej progu, aplikacja odpowiada **„nie rozpoznano / prawdopodobnie to nie jeden
  z 12 gatunków”**, a top-3 pokazuje tylko jako podpowiedź („najbliższe, ale niepewne”).
- **FR-B2 Kalibracja progu.** Próg nie jest zgadywany: powstaje z `runs/<run>/eval_test/predictions.csv`
  (pewność vs poprawność) i jest zapisywany w `runs/<run>/calibration.json` razem z krzywą
  `coverage vs accuracy` (ta sama krzywa co w `aggregate`).
- **FR-B3 Widoczna niepewność.** UI pokazuje pewność i, przy odrzuceniu, wyjaśnia **dlaczego**
  („model jest uczony na 12 gatunkach; poza nimi wynik nie jest wiarygodny”).
- **FR-B4 Świadome ograniczenie.** Komunikat stały: 12 gatunków, xeno-canto, Polska, mikrofony
  kierunkowe, nagrania z telefonu to inna domena.
- **FR-B5 Oznakowanie.** Wynik nie może wyglądać jak rozstrzygnięcie — zawsze z etykietą
  „eksperymentalnie” i bez obietnic skuteczności.
- **FR-B6 (opcjonalne, decyzja D-app-2)** klasa 13. „inne”: trening z negatywami spoza 12
  gatunków; **nie wchodzi do MVP A+B** — tylko jeśli człowiek zdecyduje i dostarczy dane.

---

## 3. Wymagania niefunkcjonalne

- **NFR-A1 Szybkość.** Predykcja pojedynczego nagrania ≤ ~2 s na CPU (RTX/CPU); plik do 60 s
  jest przycinany/ograniczany, żeby UI się nie zawieszał.
- **NFR-A2 Pamięć.** Aplikacja + model ≤ ~2 GB RAM (checkpointy CNN/CRNN są małe; transfer ~45 MB).
- **NFR-A3 Bez sieci w runtime.** Tylko `--pretrained` może raz pobrać wagi; reszta offline.
- **NFR-A4 Deterministyczne testy.** Testy nie zależą od GPU; CLI testów wymusza `--device cpu`.
- **NFR-A5 Jakość kodu.** `ruff check .` czysty; `from __future__ import annotations`; typy.
- **NFR-A6 Kody wyjścia:** `0` ok, `1` błąd użytkownika/configu, `3` błąd audio, `4` brak modelu/checkpointu.
- **NFR-A7 Windows first.** Wspieramy Windows (testowane); Linux/macOS powinny działać, ale bez obietnic.
- **NFR-A8 Bez logów wrażliwych.** Nie logujemy ścieżek przesłanych plików ani treści audio.

---

## 4. Architektura i komponenty

```
                    ┌─────────────────────────────────────────────┐
przeglądarka (mic/plik) ──► Gradio UI ──► app.predict ──► demo/eval
                    │        │                │            │
                    │   configs/app.yaml   top-3 + pewność  load_run(run)
                    │        │                │            │
                    │   próg z calibration.json  │       model (best.pt)
                    │        └────► UI: wynik albo „nie wiem”
                    └─────────────────────────────────────────────┘
```

Reużywamy istniejących klocków, **nie kopiujemy ich**:

- `eval/evaluate.py::load_run` — wczytanie runu (config + `best.pt` + opcje).
- `eval/evaluate.py::model_probs` — predykcja na oknach.
- `demo.py::predict_file`, `render_spectrogram`, `model_card`, `ranking` — rdzeń UI.
- `eval/aggregate.py::coverage_curve` — ta sama krzywa, która kalibruje próg.

Nowe pliki (tworzy aplikacja):

| Plik | Odpowiedzialność |
|---|---|
| `configs/app.yaml` | konfiguracja aplikacji (run, próg, top-k, limit sekund, zastrzeżenie) |
| `src/bird_recognizer/app/__init__.py` | pakiet |
| `src/bird_recognizer/app/config.py` | `AppConfig` (zamrożony dataclass) + walidacja YAML |
| `src/bird_recognizer/app/calibrate.py` | budowa `runs/<run>/calibration.json` z `predictions.csv` |
| `src/bird_recognizer/app/predict.py` | `classify()` → top-3 + pewność + flaga odrzucenia |
| `src/bird_recognizer/app/ui.py` | Gradio Blocks (mikrofon + plik, wynik/albo „nie wiem") |
| `cli.py` | komenda `app` (`--run`, `--threshold`, `--port`, `--smoke`) |
| `tests/test_app.py`, `tests/test_app_calibration.py` | testy |

**Kalibracja** (`app calibrate --run RUN --split test`): czyta
`runs/<run>/eval_test/predictions.csv` (kolumny `confidence`, `truth`, `predicted`), liczy
krzywą `coverage vs accuracy` i zapisuje `runs/<run>/calibration.json`:
`{"thresholds": [...], "coverage": [...], "accuracy": [...], "default_threshold": X}`.
`default_threshold` = najwyższy próg, przy którym accuracy na pokrytej części ≥ `target_accuracy`
(domyślnie 0.9) **i** pokrycie ≥ `min_coverage` (domyślnie 0.5) — wartości w `configs/app.yaml`.

---

## 5. Konfiguracja `configs/app.yaml`

```yaml
run: ""                 # domyślny run; puste = najlepszy z runs/index.csv (bez smoke)
threshold: null         # null = wczytaj default_threshold z calibration.json
top_k: 3
max_seconds: 60         # dłuższe pliki są przycinane do pierwszych N sekund
min_seconds: 1          # krótsze są odrzucane z komunikatem
store_audio: false      # czy zapisywać przesłane audio lokalnie (decyzja D-app-3)
target_accuracy: 0.9    # do kalibracji progu
min_coverage: 0.5       # do kalibracji progu
disclaimer: "Wynik eksperymentalny: 12 gatunków, xeno-canto, Polska."
port: 7860
```

`AppConfig` jest walidowany (nieznane klucze = błąd; `0 < threshold < 1`; `top_k ≤ liczba klas`).

---

## 6. Zachowanie i błędy

| Sytuacja | Zachowanie | Kod |
|---|---|---|
| brak `best.pt` / niekompletny run | czytelny komunikat, bez tracebacka | 4 |
| nieczytelny plik audio | komunikat „nie udało się odczytać pliku” | 3 |
| całe nagranie = cisza | komunikat „brak okien po bramkowaniu ciszy” | 3 |
| plik dłuższy niż `max_seconds` | przycinamy z informacją w UI | 0 |
| pewność < próg | „nie rozpoznano / możliwie spoza 12 gatunków” + top-3 jako podpowiedź | 0 |
| brak `calibration.json` i `threshold: null` | ostrzeżenie + próg domyślny `0.0` (pokazuj top-3, oznacz niepewność) | 0 |

---

## 7. Bezpieczeństwo, prywatność i licencje

- **Audio nie opuszcza komputera.** Aplikacja działa lokalnie; nic nie jest wysyłane.
- **Nie zapisujemy audio domyślnie** (`store_audio: false`). Włączenie zapisu to decyzja D-app-3
  i musi być jawnie widoczne w UI.
- **Licencje danych:** nagrania xeno-canto to głównie CC BY-NC (zakaz użycia komercyjnego).
  Aplikacja jest **niekomercyjna/edukacyjna**; dystrybucja komercyjna wymaga analizy licencji
  i osobnej decyzji.
- **Odpowiedzialność:** aplikacja nie zastępuje eksperta; zastrzeżenie jest obowiązkowe w UI.

---

## 8. Testy

`pytest` offline; nagrania z `tests/support.py` (tone/noise), gniazda zablokowane. Zakres:

| Obszar | Przypadek |
|---|---|
| app/config | `app.yaml` poprawny; nieznany klucz → błąd; zły `threshold` → błąd |
| app/calibrate | z `predictions.csv` o znanych wartościach → krzywa i próg zgodne z ręcznym liczeniem; `default_threshold` respektuje `target_accuracy`/`min_coverage` |
| app/predict | ton syntetyczny → 3 gatunki, malejące prawdopodobieństwa; cisza → błąd 3; próg działa: pewność poniżej progu → flaga odrzucenia |
| app/ui | `build_app()` zwraca Blocks; callback z `None` → prośba o plik; zepsuty plik → komunikat |
| CLI | `app --smoke` exit 0 na realnym runie; brak runu → 4; `app calibrate` tworzy `calibration.json` |

Pokrycie `src/bird_recognizer/app/` ≥ 80% (sprawdzane w bramce).

---

## 9. Decyzje do podjęcia przez człowieka (D-app)

| Id | Decyzja | Domyślnie / rekomendacja |
|---|---|---|
| D-app-1 | Domyślny próg pewności: auto (z kalibracji) czy stały | **auto z kalibracji** |
| D-app-2 | Strategia open-set: próg vs 13. klasa „inne" | **próg** (MVP); 13. klasa tylko z danymi negatywnymi |
| D-app-3 | Czy zapisywać przesłane audio lokalnie | **nie** (opt-in w configu) |
| D-app-4 | Nazwa i opis aplikacji w UI/README | do ustalenia (np. „Rozpoznawanie ptaków — demo") |
| D-app-5 | Dystrybucja: lokalny Gradio vs pakowany EXE (APP-4) | **lokalny Gradio** |
| D-app-6 | Język UI | **polski** |

---

## 10. Etapy pracy nad aplikacją

Kolejność: APP-1 → APP-2 → APP-3 → APP-5 → (APP-4 opcjonalne). Bez zielonej bramki nie ma
następnego etapu. Status trzymamy tutaj (tabela 10.1); każdy etap kończy się commitem
(Conventional Commits, po angielsku) i wpisem w `docs/ai-journal.md`.

### 10.1 Tabela statusu

| Etap | Nazwa | Zależy od | Status |
|---|---|---|---|
| APP-1 | Szkielet i UI: mikrofon + plik, top-3, spektrogram | etap 8 głównego planu | `[ ]` |
| APP-2 | Open-set: próg pewności i kalibracja | APP-1 | `[ ]` |
| APP-3 | UX, walidacja, wybór runu, bezpieczeństwo | APP-2 | `[ ]` |
| APP-4 | (opcjonalne) lokalne pakowanie/launcher | APP-3, D-app-5 | `[-]` |
| APP-5 | Odbiór: README-app, checklista, DoD | APP-3 | `[ ]` |

### 10.2 Protokół jednego etapu

0. Weź pierwszy `[ ]`, którego poprzednik ma `[x]`.
1. Przeczytaj kartę etapu, ostatnie wpisy `docs/ai-journal.md`, `git log --oneline -5`.
2. Zapisz plan w 3–6 punktach w treści pierwszego commita etapu.
3. Wykonuj zadania po kolei, odhaczając `[x]`; testy pisz razem z kodem.
4. Uruchom bramkę dokładnie w podanej postaci; czerwona bramka = etap się nie wydarzył.
5. Commit (angielski), wpis w dzienniku, aktualizacja tabeli 10.1 w tym samym commicie.
6. Raport do człowieka ≤ 10 linii.

### 10.3 APP-1 — Szkielet i UI (mikrofon + plik, top-3, spektrogram)

**Cel:** lokalna aplikacja, która wczytuje run i robi to, co dzisiejsze `demo`, ale z mikrofonem.
**Tworzy/zmienia:** `configs/app.yaml`, `app/{__init__,config}.py`, `app/predict.py`, `app/ui.py`,
`cli.py` (`app`), `tests/test_app.py`.

**Zadania**
- [ ] `AppConfig` + `configs/app.yaml` (bez progu — próg wchodzi w APP-2).
- [ ] `app predict`: wczytaj run (`load_run`), wytnij okna, policz top-3 i spektrogram (reuse `demo`).
- [ ] `app ui`: `gr.Audio(sources=["microphone","upload"], type="filepath")` + `gr.Label(num_top_classes=3)` + `gr.Plot` + karta modelu.
- [ ] CLI `app [--run] [--port] [--smoke]`; `--smoke` bez serwera.
- [ ] testy: config, predykcja tonu, UI callbacki, CLI smoke.

**Bramka**
```bash
ruff check .
pytest -q
python -m bird_recognizer.cli app --run <RUN> --smoke
```

**Commit:** `feat(app): microphone and upload UI with top-3` · **Dziennik:** czy Gradio 6 obsługuje
`sources=["microphone","upload"]` zgodnie z oczekiwaniem (sprawdzono: tak, wersja 6.31).

### 10.4 APP-2 — Open-set: próg pewności i kalibracja

**Cel:** aplikacja umie powiedzieć „nie wiem" zamiast zgadywać.
**Tworzy/zmienia:** `app/calibrate.py`, `app/predict.py` (próg), `app/ui.py` (komunikat),
`configs/app.yaml` (`threshold`, `target_accuracy`, `min_coverage`), `cli.py` (`app calibrate`),
`tests/test_app_calibration.py`.

**Zadania**
- [ ] `app calibrate --run RUN --split test` → `runs/<run>/calibration.json` (krzywa + `default_threshold`).
- [ ] `predict.classify()` zwraca flagę odrzucenia i najbliższe top-3 z etykietą niepewności.
- [ ] UI: wynik albo „nie rozpoznano / możliwie spoza 12 gatunków".
- [ ] testy: kalibracja na znanych liczbach; próg odrzuca niską pewność; `--threshold` nadpisuje.

**Bramka**
```bash
pytest -q --cov=bird_recognizer.app --cov-report=term-missing
python -m bird_recognizer.cli app calibrate --run <RUN> --split test
python -m bird_recognizer.cli app --run <RUN> --smoke
```

- Pokrycie `app/` ≥ 80%; `default_threshold` zgadza się z ręcznym rachunkiem na znanej krzywej.

**Commit:** `feat(app): confidence threshold and calibration` · **Dziennik:** jaki próg wyszedł
z kalibracji i jaką ma coverage/accuracy.

### 10.5 APP-3 — UX, walidacja, wybór runu, bezpieczeństwo

**Cel:** aplikacja jest odporna i uczciwa: walidacja audio, wybór runu, zastrzeżenia, bez logów.
**Tworzy/zmienia:** `app/ui.py`, `app/predict.py`, `cli.py`, `tests/test_app.py`.

**Zadania**
- [ ] walidacja: `min_seconds`, `max_seconds`, nieczytelny plik, cisza → komunikaty.
- [ ] wybór runu z `runs/index.csv` (bez `smoke` domyślnie), fallback, czytelny błąd przy braku.
- [ ] karta modelu + zastrzeżenie + licencja; `store_audio: false` jawnie opisane w UI.
- [ ] testy błędów i wyboru runu.

**Bramka**
```bash
ruff check .
pytest -q
python -m bird_recognizer.cli app --smoke
```

**Commit:** `feat(app): validation, run selection and safe defaults` · **Dziennik:** co sprawdzono
ręcznie (długi plik, cisza, zły format).

### 10.6 APP-4 — (opcjonalne) lokalne pakowanie/launcher

Tylko po D-app-5 = „pakowany EXE". Zakres: skrypt/instrukcja instalacji (venv + `pip install`
+ wskaźnik na `data/`), ewentualnie PyInstaller/Tauri z bundlowanym `torch` i ffmpeg.
**Bramka:** aplikacja startuje na czystej maszynie według instrukcji.

### 10.7 APP-5 — Odbiór i domknięcie

**Cel:** człowiek może uruchomić i ocenić aplikację; wszystko udokumentowane.
**Tworzy/zmienia:** `docs/app-guide.md` (instrukcja użytkownika), `README.md` (sekcja „Aplikacja"),
`docs/manual-checklist.md` (kroki D-app), `docs/ai-journal.md` (domknięcie).

**Zadania**
- [ ] `docs/app-guide.md`: jak uruchomić, co pokazuje UI, jak interpretować „nie wiem".
- [ ] README: sekcja aplikacji (komendy, limity, licencje, zastrzeżenie).
- [ ] checklista D-app-1..D-app-6 odhaczona.
- [ ] przegląd DoD (sekcja 11) i domknięcie dziennika.

**Bramka**
```bash
ruff check .
pytest -q
python -m bird_recognizer.cli app --run <BEST> --smoke
git ls-files | Select-String -Pattern '\.(mp3|wav|flac|ogg)$'
```

**Commit:** `docs(app): user guide and acceptance` · **Dziennik:** co zostało zweryfikowane na
żywo, czego nie udało się sprawdzić bez nagrań własnych.

---

## 11. Definition of Done (aplikacja)

- [ ] `bird_recognizer app --run <BEST>` startuje lokalnie; mikrofon i upload działają.
- [ ] Top-3 + spektrogram + karta modelu + zastrzeżenie są widoczne.
- [ ] Próg pewności działa: niska pewność → „nie rozpoznano", nie losowy gatunek.
- [ ] `runs/<BEST>/calibration.json` istnieje i jest użyty (albo `--threshold` jawnie podany).
- [ ] `ruff check .` czysty, `pytest -q` zielone, pokrycie `app/` ≥ 80%.
- [ ] `app --smoke` przechodzi; testy nie zależą od sieci ani GPU.
- [ ] `docs/app-guide.md` i sekcja aplikacji w README napisane po polsku.
- [ ] Brak audio i sekretów w repo; audio użytkownika nie jest zapisywane bez zgody.
- [ ] D-app-1..D-app-6 odhaczone w `docs/manual-checklist.md`.

---

## 12. Ryzyka i otwarte pytania

1. **Jakość na nagraniach z telefonu.** Model uczył się na mikrofony kierunkowe; bez `data/own/`
   (etap 7) nie wiemy, jak spada jakość. Aplikacja to pokaże — musimy to uczciwie opisać.
2. **Pułapka 12 klas.** Każdy dźwięk zostanie przypisany do jednej z 12 klas; próg (B) łagodzi,
   ale nie rozwiązuje do końca. Prawdziwe rozwiązanie to 13. klasa (D-app-2, poza MVP).
3. **Przesadna pewność.** Softmax bywa zbyt pewny siebie; kalibracja progu na `predictions.csv`
   to minimum, ale nie zastępuje kalibracji probabilistycznej (temperature scaling) — opcjonalnie.
4. **Licencje CC BY-NC.** Aplikacja musi pozostać niekomercyjna, dopóki licencje danych na to
   nie pozwolą inaczej.
5. **Rozmiar przy pakowaniu.** Jeśli APP-4 (EXE), trzeba związać `torch` (~2–3 GB) i ffmpeg.
6. **Mikrofon w przeglądarce.** Wymaga `localhost`/HTTPS; przy starcie lokalnym `127.0.0.1` jest OK.
