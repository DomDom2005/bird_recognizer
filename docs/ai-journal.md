# Dziennik AI — bird_recognizer

> Zasada: uczciwie i na bieżąco. Zapisujemy, co zrobiło AI, **jak to sprawdzono** i co się nie udało.
> Niczego nie upiększamy i nie dopisujemy na zapas (SPEC §0 pkt 6, §14).

| Data | Etap | Co zrobiło AI | Jak zweryfikowano | Wynik |
|---|---|---|---|---|
| 2026-11-02 | przygotowanie | `SPEC.md` i oba `AGENTS.md` na podstawie `PLAN.md`, w schemacie z `view_tracker_ai_1`; fakty o API v3 sprawdzone w dokumentacji | przegląd właściciela; D1 zatwierdzone | trafne |
| 2026-11-02 | przygotowanie | SPEC przepisany na wykonalny etapowo: tabela statusu, protokół, karty etapów z bramkami | przegląd właściciela | trafne |
| 2026-11-02 | 1 | szkielet repo: `config.py` (walidacja YAML + alias klucza), `logging_setup.py` z redakcją, `errors.py`, `configs/*.yaml`, `tests/conftest.py` z blokadą gniazd, `scripts/probe_api.py` (dry-run/live), szkielet `docs/` i README | bramka etapu 1: `ruff check .` czysty, `pytest -q` → 26 passed, `load_config()` ok, `probe_api.py --dry-run` exit 0 | trafne po korektach (patrz niżej) |
| 2026-11-02 | 1 | `probe_api.py --live` na prawdziwym kluczu: 7 sprawdzeń, tempo 1 req/s, klucz nigdy nie wypisany | wyniki w `docs/api-notes.md` i SPEC §4.3 | trafne, ale ujawniło rozbieżność dokumentacji z API (status tag) |
| 2026-11-18 | 2 | `xc/client.py` (tagi, paginacja, throttle, backoff `429`/`503`, mapowanie błędów), `xc/catalog.py` (scalanie po `id`, `_meta.redacted_fields`, zapis atomowy), `xc/download.py` (wznawianie, sha256, `--retry-failed`, raport), `data/splits.py` (grupy po `rec`, naprawa pokrycia, `check_leakage`), `cli.py` + `scripts/check_leakage.py`, testy na wstrzykiwanym transporcie | bramka etapu 2: `ruff check .` czysty, `pytest -q --cov=…` → 99 passed, pokrycie xc 90–94% i `data/splits.py` 96%, `check_leakage.py` exit 0 na fixtures | trafne po dwóch korektach algorytmu podziału |
| 2026-11-18 | 2 | uruchomienie na żywo: `data catalog` (2917 nagrań), `data split` (2203/407/307), `data download --limit 2` (3,3 MB) i ponowne uruchomienie (wznawianie: 2 pominięte) | `check_leakage.py --data-dir data` → brak wycieku; raport pobierania i `sha256_audio` w katalogu | trafne; live run wykrył błąd podziału, którego testy nie łapały |
| 2026-11-20 | 3 | `data/audio.py` (soundfile — czyta mp3, torchaudio tylko do resamplingu, HPF Butterwortha), `data/windows.py` (3 s / 1,5 s, dopełnianie ogona, bramka RMS), `data/spectrogram.py` (log-mel 128 pasm, normalizacja per okno, `config_hash`), `data/dataset.py` (cache float16 + meta, `--verify`, etykiety, maska), CLI `prep cache`, generator WAV w testach zamiast zbioru syntetycznego | bramka etapu 3: `ruff check .` czysty, 136 testów, pokrycie data 88–100%, `prep cache --limit 24` + `--verify` exit 0 | trafne po korektach (odstępstwa 19–22) |
| 2026-11-20 | 3 | pomiar na żywych nagraniach: 22/24 przetworzone, 2 odrzucone przez `abs_floor`, 521/640 okien (18,6% odpadło), 15,8 min audio, cache 24 MB | `--verify` OK, dwa przebiegi dają identyczne `mel.npy` | trafne; rozmiar cache okazał się ~3× większy niż szacunek SPEC |
| 2026-11-20 | 4 | `models/cnn.py` (4 bloki, 391 404 param.), `models_registry.py`, `models/mfcc_baseline.py` (MFCC+SVM/RF, cechy z tych samych okien co cache), `train/{augment,loop,runner}.py`, `eval/metrics.py` (potrzebne do early stoppingu), CLI `train mfcc/cnn`, `--smoke`, `--runs-dir`, `--epochs`, cache rozszerzony do 801 nagrań | bramka etapu 4: `ruff check .` czysty, 170 testów, pokrycie `models/` 83–100% i `train/` 93–100%, oba `--smoke` exit 0, dwa przebiegi CNN z tym samym seedem identyczne | trafne po naprawie determinizmu (odstępstwo 24) |
| 2026-11-20 | 4 | pomiar na realnym podzbiorze (2 nagrania/gatunek): CNN 1 epoka ≈ 23 s CPU, `--epochs 2` → train loss 1,74 → 0,81; MFCC/SVM < 1 s; runs mają `config.yaml`, `env.txt`, `best.pt`/`model.joblib`, `metrics.jsonl`, wpis w `index.csv` | ponowne uruchomienie identyczne; `runs/index.csv` z 8 przebiegami | trafne |
| 2026-11-20 | 5 | `models/transfer.py` (ResNet18/EfficientNet-B0, 1-kanałowa pierwsza konwolucja, nowa głowica, `--pretrained|--scratch`), `models/crnn.py` (backbone z etapu 4 → BiGRU → pooling `mean`/`attention` z maską długości), `train/augment.py` wariant `full` (SpecAugment + mixup), rejestr z opcjami modelu, CLI `train crnn`/`train transfer`, `WindowDataset(with_lengths=True)`, pętla treningu z maską i mixupem | bramka etapu 5: `ruff check .` czysty, **203 testy** (14,9 s), pokrycie `models/` 83–100% (crnn 97%, transfer 91%), `train/` 93–100% (augment 100%), `train crnn --smoke` i `train transfer --arch resnet18 --scratch --smoke` exit 0 | trafne po korekcie konstrukcji poolingu (patrz odstępstwo 29) |
| 2026-11-20 | 5 | pomiar na realnym podzbiorze: CRNN **1 212 396** parametrów, 1 epoka ≈ 21 s CPU; ResNet18 **11 176 396** parametrów, ≈ 28 s; ponowny `train crnn --smoke` dał identyczne metryki (determinizm); `train cnn --smoke` nadal 0,190 (brak regresji etapu 4) | `runs/index.csv` z nowymi wierszami: `crnn-mean`, `transfer-resnet18-scratch`, `transfer-resnet18-pretrained`; `env.txt` ma `model_parameters` | trafne |
| 2026-11-20 | 5 | dokończone pobieranie całego zbioru (2917/2917, 0 błędów, 0 zredagowanych, 969 pominiętych przy wznowieniu w dwóch przebiegach) i przebudowa cache: 2897 nagrań, 131 900/151 115 okien (12,7% odrzuconych), 3755,7 min audio, 6,06 GB; 1 duplikat audio `1052072`=`1052075` (oba w train) | `prep cache --verify` → „Cache OK (2897 nagrań)”; `check_leakage.py` exit 0 (brak wspólnych id/rec/sha256); 20 nagrań bez cache: 19 `empty_after_gating` + 1 błąd dekodowania, 0 w test | trafne; pomiar trafił do SPEC §7.4 i §10.2 |
| 2026-11-21 | 6 | `ablation.py` (5 grup, jedna dźwignia na grupę, `runs/ablation_<grupa>.json`), `eval/report.py` (tabela z `index.csv`, sekcje etapu 7 jako placeholdery), CLI `ablate`/`report`, refaktor `run_neural` (wspólny dla `train` i `ablate`), cache okien 5 s/10 s na twardych linkach, instalacja `torch 2.15.0+cu132` | bramka etapu 6: `ruff check .` czysty, **226 testów**, `ablate --group rnn --smoke` → 3/3 runy, `report` → `docs/experiments.md` z 3 runów, test „jedna różnica w configu” zielony dla 5 grup | trafne po naprawie urządzenia modelu (odstępstwa 35–36) |
| 2026-11-21 | 6 | pomiar GPU: CRNN `--smoke` 1 epoka 1,6 s (cuda) vs 12 s (cpu); testy CLI wymuszają `--device cpu`, żeby nie zależeć od GPU; cache 5 s i 10 s budowany w tle pod grupę `window` | `train crnn --smoke` → `device=cuda`; wiersze `docs/experiments.md` mają `smoke` w uwagach; pełne runy macierzy dochodzą po etapie 6 (dziennik niżej) | trafne |
| 2026-11-21 | 6 | **pomiar pełnego epoch** CNN na całym train (95 898 okien, GPU): 757 s przy 38% wykorzystania GPU → wąskie gardło to `WindowDataset` czytający cały `mel.npy` (~4 MB) dla każdego okna przy tasowanym batchu | profil po zmianie: `np.load(mmap_mode="r")` + konwersja jednego okna → **129,7 s/epoch** (5,8× szybciej), identyczne metryki (val macro-F1 0,534, test 0,506); cache okien 5 s/10 s zweryfikowany (2889/2881 nagrań, 0 problemów) | trafne; poprawka w osobnym commicie `perf(data)` |

## Odstępstwa od SPEC

Każdy wpis: co SPEC zakładał, co jest naprawdę, gdzie poprawiono.

1. **Zmienna klucza API (§13).** SPEC mówi `BIRD_XC_API_KEY`, a `.env` właściciela używa
   `xc_api_key`. Kod akceptuje obie nazwy (`config.API_KEY_ENV_NAMES`), pierwszeństwo ma nazwa
   dokumentowana. SPEC §13 zaktualizowany.
2. **Tag `status:identified` (§4.1, FR-1).** Dokumentacja podaje tag, **żywe API zwraca 400**
   na każdy wariant (`status:identified`, `status:"identified"`, `status:"=identified"`,
   także z `sp:`/`cnt:`). `id:identified` jest akceptowane, ale zwraca 1,15 mln nagrań — to nie
   filtr identyfikacji. Decyzja: tag domyślnie wyłączony (`api.status: ""`), klucz w configu
   zostaje. Skutek: brak filtra jakości identyfikacji → szum etykiet trafia do danych i musi być
   widoczny w analizie błędów (`also`, `rmk`). SPEC §4.1/§4.3/FR-1 zaktualizowane.
3. **Kształt błędu (§4.1, §9).** Dokumentacja: `{"error": {"code", "message"}}`. Live:
   `{"error": "client_error", "message": ...}`. Parser akceptuje oba kształty.
4. **Rozszerzenie pliku (FR-2).** `file` wskazał `audio/wav`, nie mp3. Rozszerzenie bierzemy
   z `file-name`, a gdy brak — z `Content-Type`.
5. **`page` poza zakresem (§4.3 pkt 6).** HTTP 400 `client_error`, nie pusta lista — kod musi
   traktować to jako błąd parametru, nie jako koniec paginacji.
6. **`pip` w sandboxie DSH (środowisko agenta).** `python -m venv` i `pip install` padały na
   `PermissionError`: `ensurepip` i pip nie mogą pisać do katalogów tworzonych przez
   `tempfile`/`TemporaryDirectory` (dotyczy zarówno `%TEMP%\dsh-*`, jak i `.tmp/` w repo).
   Obejście: uruchomienie instalacji z szerszym dostępem (`danger-full-access`) — przeszło;
   `requirements.txt` i `requirements-dev.txt` przypięte do zainstalowanych wersji.
   W zwykłym terminalu właściciela instalacja działa bez zmian.
7. **`pytest` a katalogi tymczasowe.** `tmp_path` (i `--basetemp`) kończyły się
   `PermissionError` przy `os.scandir`/`chmod`, bo `tempfile.mkdtemp` tworzy katalog z prawami,
   których sandbox pilnuje. Obejście: fixture `tmp_dir` w `tests/conftest.py` tworzy katalog
   przez `Path.mkdir` z losowym sufiksem w `.tmp/` (gitignored). Dodatkowo
   `addopts = "-ra -p no:cacheprovider"`, bo `.pytest_cache` też wypadał na tych samych prawach.
8. **Kodowanie konsoli (Windows).** `probe_api.py` wypisywał polskie znaki w kodowaniu cp1250
   i przewracał się na `→`. Skrypt przełącza stdout/stderr na UTF-8 (`_configure_stdout`).
9. **Bramka etapu 1 (§15.6).** Dopisane `python scripts/probe_api.py --dry-run`, żeby bramka
   sprawdzała też FR-16 offline. SPEC podbite do 1.1.
10. **`responses` usunięte (§12).** Postanowienie z etapu 1 zrealizowane: klient siedzi na
    `urllib`, a testy wstrzykują własny `FakeTransport` (kolejka odpowiedzi). `responses`
    zniknęło z `requirements-dev.txt`, SPEC §12 i drzewo repo zaktualizowane.
11. **Błąd własny (naprawiony):** przepisałem `tests/test_config.py` przez PowerShell
    (`Get-Content -Raw ... | Set-Content`) i skorzystałem z domyślnego kodowania konsoli —
    polskie znaki w pliku zamieniły się w mojibake („SPEC Â§12”), przez co test na etykiety
    gatunków zaczął padać z `KeyError`. Naprawa: plik napisany od nowa w UTF-8 narzędziem
    do zapisu plików, a nie przez PowerShell; testy wróciły do 26 passed. Wniosek: pliki
    z polskim tekstem edytuję tylko przez `write`/`edit`, nie przez `Set-Content`.
12. **Błąd podziału na żywych danych (naprawiony).** Pierwsza wersja `make_split` wybierała
    podział minimalizujący `|przypisane + rozmiar − cel|` — na 2917 nagraniach dało to
    **train = 0**, bo grupy nagrywających są bardzo nierówne (jeden nagrywający = setki plików).
    Testy na syntetycznych, równych grupach tego nie łapały. Naprawa: val/test wypełniane
    najpierw małymi grupami (różnorodność nagrywających), duże grupy → train, potem naprawa
    pokrycia gatunków. Dodane testy regresyjne: `test_train_keeps_the_majority_share`,
    `test_uneven_recordist_sizes_do_not_starve_train`.
13. **`bird_recognizer` nie jest w PATH.** Editable install zarejestrował `bird_recognizer.exe`
    w `…\Python\…\Scripts`, ale ten katalog nie jest w PATH tej sesji. Do czasu dodania PATH
    komendy uruchamiam jako `python -m bird_recognizer.cli …`; README i AGENTS mają fallback.
14. **`counts` w `splits.json` (§7.2).** SPEC pokazywał `windows`; przy podziale nie znamy
    jeszcze liczby okien (powstaje w etapie 3). Zapisujemy `recordings`, `species`, `groups`,
    a `windows` dojdzie po cache — świadome odstępstwo, nie brak.
15. **Wspólny `configure_console()`.** Poprawka kodowania z etapu 1 (`_configure_stdout`
    w `probe_api.py`) przeniesiona do `logging_setup.py`, bo ten sam problem dotyczył CLI
    i `check_leakage.py`; `probe_api.py` korzysta teraz z funkcji wspólnej.
16. **`.gitignore` wycinał pakiet (przyłapane przy commicie etapu 2).** Wzorzec `data/`
    bez wiodącego `/` ignoruje **każdy** katalog o nazwie `data`, w tym `src/bird_recognizer/data/`.
    Skutek: `data/__init__.py` nie wszedł do repo już w etapie 1, a `data/splits.py` nie wszedł
    w etapie 2 — testy przechodziły lokalnie tylko dlatego, że pliki leżą na dysku. Naprawa:
    wzorce zakotwiczone (`/data/`, `/runs/`, `/reports/`, `/.tmp/`, `/pytest-cache-files-*/`,
    `/pytest-of-*/`) i osobny commit `fix(repo): …` z pakietem. Ten sam błąd wyłączał pakiet
    z `ruff check` (ruff respektuje `.gitignore`) — po naprawie wyszedł jeden `E501` w
    `splits.py`, poprawiony. Wniosek: po `git add` porównuję `git diff --cached --name-only`
    z listą plików, które miały wejść; „zielone testy lokalnie” nie dowodzą, że repo jest kompletne.
17. **Błędna liczba zredagowanych nagrań.** W podsumowaniu etapu 2 napisałem „2 zredagowane”
    bez sprawdzenia artefaktu; katalog ma **0** zredagowanych wpisów (`catalog.meta.json`:
    `redacted: 0`, wszystkie 2917 ma `file`). Poprawione w SPEC §15.7 i README. Wniosek:
    liczby w dokumentacji biorę z pliku wynikowego, nie z pamięci.
18. **Rezygnacja z syntetycznego „zbioru demo” (decyzja z 2026-11-18).** Właściciel zaproponował
    pobranie całego realnego audio od razu i trzymanie `data/` w `.gitignore` — to wystarcza,
    bo syntetyk nigdy nie był danymi treningowymi, tylko rusztowaniem offline. Uproszczenie:
    `make_synthetic_demo.py` i FR-15 znikają; testy generują kilkusekundowe WAV-y w `tmp_dir`,
    a przebiegi `--smoke` w etapach 4–6 korzystają z realnego katalogu z `--limit` na gatunek.
    SPEC podbite do 1.3 przy wejściu w etap 3 (FR-15 przedefiniowane, bramki etapów 3–5 bez
    `data/synthetic`).
19. **`torchaudio.load` wymaga TorchCodec (etap 3).** W zainstalowanej wersji torchaudio 2.11.1
    `torchaudio.load` rzuca `ImportError: TorchCodec is required`, a `torchaudio.list_audio_backends`
    nie istnieje. `soundfile` (libsndfile) czyta mp3 bez problemu (48 kHz, mono). Decyzja:
    dekodowanie przez `soundfile`, resampling przez `torchaudio.functional.resample`; SPEC §10.1
    dopisane. Wniosek: wersję biblioteki sprawdzaj funkcją, nie założeniem z dokumentacji.
20. **Rozmiar cache okazał się ~3× większy niż szacunek (§7.4).** SPEC szacował ~2 GB dla
    1000 min audio; pomiar: 24 MB na 15,8 min = **~1,5 MB/min**, czyli ~5,6 GB dla pełnego
    zbioru (3738 min). Powód: realne nagrania są dłuższe niż zakładane 1000 min. Dysk: 900+ GB
    wolne, więc nie blokuje; §7.4 zaktualizowane liczbą zmierzoną.
21. **Ogon okna dopełniany zerami.** Krótkie ostatnie okno jest zachowane, ale dopełnione zerami
    do pełnej długości — inaczej `np.stack` nie zbuduje jednolitego tensora. Prawdziwą długość
    trzyma `meta.valid_frames`, z której `load_recording` buduje maskę. SPEC §15.8 opisuje to
    jawnie.
22. **Dwa nagrania odrzucone w całości przez bramkę (2/24).** `1054634` (peak −42 dBFS) i
    `1058630` (peak −34 dBFS) mają RMS poniżej `abs_floor = 1e-3`; dla porównania normalne
    nagranie ma peak 0,70 i RMS 0,039. Progi zostają (to nagrania poniżej progu użyteczności),
    a przypadki są raportowane jako `empty_after_gating`; nota w SPEC §10.2 i README.
23. **Gain (wzmocnienie) pominięty w augmentacjach.** Cache trzyma spektrogramy znormalizowane
    per okno, więc dodanie stałej w dB znosi się przy normalizacji — augmentacja byłaby
    matematycznie identycznością. Zaimplementowane: przesunięcie w czasie i szum. Powrót gain
    wymaga przeniesienia normalizacji do ładowania (etap 5) albo usunięcia go z listy; SPEC
    §10.3 opisuje stan.
24. **Model tworzony przed `set_seed` (błąd determinizmu, złapany przez bramkę).** Dwa przebiegi
    `train cnn --smoke` dały różne metryki (0,822 vs 0,190), bo wagi powstawały przed ustawieniem
    seeda, a torch startuje z losowym stanem procesu. Naprawa: `set_seed(config.seed)` w CLI
    **przed** `build_model`, docstring w `train_cnn` oraz test determinizmu, który seeduje przed
    utworzeniem modelu. Po naprawie dwa osobne uruchomienia dają identyczne metryki.
25. **`eval/metrics.py` powstał w etapie 4.** Early stopping potrzebuje macro-F1, a stage 5 ma
    dopiero `eval/*`. Wyciągnąłem czyste funkcje metryk wcześniej; etap 5 rozbuduje je o
    agregację, macierz pomyłek i analizę błędów (bez duplikacji).
26. **`top_k_accuracy` zwraca `None`, gdy k ≥ liczba klas.** W smoke runach obecne są 3 klasy,
    więc top-3 byłby trywialnie 1,0 i mylący; CLI pokazuje `n/d`, a `index.csv` zostawia puste
    pole. Na pełnych 12 klasach top-3 liczy się normalnie.
27. **Dodatkowe flagi CLI:** `--runs-dir` (był w SPEC §8, brakowało go w parserze) i `--epochs`
    w `train cnn` (dev, do sprawdzenia spadku lossu na podzbiorze; SPEC §8 zaktualizowane).
28. **Transfer: wybór wejścia 1-kanałowego.** SPEC FR-8 dopuszczał replikację mel do 3 kanałów
    albo pierwszą konwolucję z 1 kanałem i kazał zapisać wybór w configu. Wybrano **1-kanałową
    pierwszą konwolucję**; przy `--pretrained` wagi RGB są **sumowane** (zachowuje energię), nie
    uśredniane. Nie ma normalizacji ImageNet (mean/std) — cache jest już znormalizowany per okno,
    a druga normalizacja zmieniłaby statystyki wejścia obiecane w §10.1. Wybór jest w docstringu
    `transfer.py`, w `metrics.json` (`options`) i w kolumnach `index.csv` (`pretrained`), ale
    **nie w `config.yaml` runu** — pełny zapis wymaga sekcji `model` w configu; to zadanie dla
    etapu 6 razem z ablacjami. SPEC §15.10 zaktualizowane.
29. **Konstrukcja poolingu (błąd złapany przez testy).** `POOLINGS[pool](hidden*2)` przekazywało
    rozmiar także do `MeanPooling`, który nie ma parametrów → `TypeError` w 6 testach. Naprawa:
    `MeanPooling()` bez argumentów, `AttentionPooling(hidden*2)`, walidacja nazw w konstruktorze.
    Testy pokrywają T=188 i T=50, oba poolingi i maskę długości.
30. **Maska długości trafiła też do `data/dataset.py`.** Karta etapu 5 nie wymienia tego pliku,
    ale FR-7 wymaga maskowania krótkich okien, a `WindowDataset` zwracał tylko `(mel, label)`.
    Dodane `WindowRef.valid_frames` (z `meta.json`) i `WindowDataset(with_lengths=True)` →
    `(mel, label, valid_frames)`; domyślnie nadal 2-krotka, więc starsze wywołania są bez zmian.
    Pętla treningu rozpoznaje oba kształty batcha.
31. **„Maskowanie w loss” nie ma zastosowania per ramka.** Klasyfikacja jest per okno (jedna
    etykieta), więc nie istnieje strata per ramka do maskowania. Maska działa w poolingu `mean`
    i `attention`: dopełnione ramki nie wchodzą do średniej ani do softmax. SPEC FR-7 mówi
    o „loss i poolingu” — zapisuję różnicę, żeby nie udawać, że zrobiono coś, czego nie ma.
32. **Wagi ImageNet w sandboxie DSH.** `~/.cache/torch` leży poza workspace, a zapis jest
    odrzucany (`WinError 5`), więc pierwsze `--pretrained` padło z czytelnym `ModelError`.
    Obejście: `TORCH_HOME=data/torch` (dane są poza repo); wagi (44,7 MB) pobrane raz, przebieg
    `train transfer --arch resnet18 --pretrained --smoke` przeszedł. W zwykłym terminalu
    właściciela domyślna ścieżka działa bez zmian; pułapka dopisana do AGENTS.
33. **`--aug` dodane, `--window` odłożone.** SPEC §8 obiecywał `train cnn [--window 3] [--aug …]`.
    `--aug {none,basic,full}` zaimplementowano teraz dla `cnn`/`crnn`/`transfer` (etap 6 go
    potrzebuje), z zapisem w `config.yaml` runu i w kolumnie `aug`. `--window` wymaga cache
    zbudowanego dla innej geometrii okna, więc dochodzi razem z `ablate` w etapie 6; SPEC §8
    zaktualizowane.
34. **`train_cnn` → `train_model`.** Pętla obsługuje trzy rodziny modeli, więc nazwa została
    zmieniona; alias `train_cnn` zostaje dla zgodności. CLI i testy używają nowej nazwy.
35. **Środowisko GPU (etap 6).** W maszynie jest RTX 5070 12 GB (driver 610.60, CUDA 13.2/13.3),
    ale zainstalowany `torch` był buildem CPU-only. Kół `2.15.0` **nie ma** na indeksach cu128
    i cu130 (maks. 2.11.0 / 2.14.1); jest na cu132. Instalacja: `pip install --no-deps
    "torch==2.15.0+cu132" --index-url https://download.pytorch.org/whl/cu132`. `torchvision`
    i `torchaudio` zostały buildami CPU (wystarczają: definiują modele i robią resampling).
    `requirements.txt` przypina teraz `torch==2.15.0+cu132` i podaje `--extra-index-url`;
    testy CLI wymuszają `--device cpu`, żeby nie zależeć od GPU. Pomiar: CRNN `--smoke`
    1 epoka 1,6 s na CUDA vs 12 s na CPU.
36. **Model nie był przenoszony na urządzenie (błąd złapany po instalacji CUDA).** `_resolve_device`
    zwracało `cuda`, a `run_neural` budował model na CPU → `RuntimeError: Input type
    (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)`. Naprawa: `.to(device)`
    w `run_neural` oraz przenoszenie `lengths`/maski na urządzenie sekwencji w CRNN. Wniosek:
    zmiana środowiska (CPU → GPU) musi przejść **całą** bramkę, bo ujawnia błędy niewidoczne na CPU.
37. **Ablacja okna wymaga osobnego cache.** `data/window5` i `data/window10` powstają przez
    twarde linki do `data/raw` (`os.link`, bez kopiowania 8 GB) + kopię `catalog.jsonl`
    i `splits.json`; `ablate` buduje cache na żądanie (`ensure_window_cache`), więc grupa `window`
    nie jest pomijana. Katalogi są poza repo (`/data/` w `.gitignore`).
38. **Ablacja `pretrained`: ten sam `lr` w obu gałęziach.** SPEC §10.5 sugeruje 3e-4 dla
    fine-tune, ale w ablacji zmiana `lr` byłaby **drugą** różnicą względem „od zera”. Obie
    gałęzie uczą się z `lr=1e-3`; zapisane w `note` w `runs/ablation_pretrained.json`.
39. **`report` zostawia sekcje 3–5 jako placeholdery.** Pary mylące (per-class F1), zbiór własny
    i „Wnioski” należą do etapu 7 i do człowieka; nagłówek mówi wprost, że tabela jest generowana
    i że wniosków nie wymyśla agent.
40. **Smoke w ablacji ma nierówne `n` między podziałami.** `--limit 2` działa per gatunek
    i per podział, a nagrania mają różne długości, więc wiersze mają np. 762/910/825 okien;
    są oznaczone `smoke` w uwagach i nie są wynikami badania.
41. **Dataset czytał całe nagrania dla każdego okna (zmierzone, naprawione).** Pełny epoch CNN
    na 95 898 oknach trwał **757 s** przy 38% wykorzystania GPU. Przyczyna: `WindowDataset`
    robił `np.load(...).astype(float32)` całego `mel.npy` (~4 MB) przy każdym nowym nagraniu,
    a tasowane batche dotykają kilkudziesięciu nagrań na raz; LRU 16 nie wystarczał. Naprawa:
    `np.load(mmap_mode="r")` i konwersja tylko wybranego okna do float32. Ten sam epoch:
    **129,7 s** (5,8× szybciej), a metryki **identyczne** (val macro-F1 0,534, test 0,506) —
    dowód, że zmiana jest wyłącznie wydajnościowa. Dodany test dtype i zgodności z pełnym odczytem.

## Wznowienie po przerwie (2026-11-18)

Praca przerwana na życzenie właściciela; wszystko jest w spójnym stanie i nie ma
nic niezacommitowanego (poza `data/`, które jest poza repo).

- **Zrobione:** etap 1 `[x]`, etap 2 `[x]` (commity `e715514`, `1d77624`, `a5f1a11`).
- **Nierozpoczęte:** etap 3 (preprocessing i cache) — karta w SPEC §15.8.
- **Pobieranie audio przerwane na 169/2917 plikach** (`data/raw`). Wznowienie:
  `python -m bird_recognizer.cli data download` — pomija istniejące pliki; jeśli w raporcie
  są błędy, `--retry-failed`. Plików tymczasowych `.*.tmp` nie ma.
- **Decyzja do wprowadzenia w SPEC (1.3):** usunąć FR-15 i `make_synthetic_demo.py` oraz
  odwołania w §15.8–15.10; testy generują krótkie WAV-y w `tmp_dir`, a `--smoke` korzysta
  z realnego katalogu z `--limit` (patrz odstępstwo 18).
- **Pierwszy krok nowej sesji:** §15.5 — `git log` → tabela 15.1 → karta etapu 3 → ten wpis.

## Refleksja

Sekcja dla człowieka — uzupełniana po zakończeniu projektu (SPEC §16 pkt 8).
