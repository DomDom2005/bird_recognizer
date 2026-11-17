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
    SPEC do podbicia na 1.3 przy wejściu w etap 3.

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
