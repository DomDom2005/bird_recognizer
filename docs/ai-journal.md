# Dziennik AI — bird_recognizer

> Zasada: uczciwie i na bieżąco. Zapisujemy, co zrobiło AI, **jak to sprawdzono** i co się nie udało.
> Niczego nie upiększamy i nie dopisujemy na zapas (SPEC §0 pkt 6, §14).

| Data | Etap | Co zrobiło AI | Jak zweryfikowano | Wynik |
|---|---|---|---|---|
| 2026-11-02 | przygotowanie | `SPEC.md` i oba `AGENTS.md` na podstawie `PLAN.md`, w schemacie z `view_tracker_ai_1`; fakty o API v3 sprawdzone w dokumentacji | przegląd właściciela; D1 zatwierdzone | trafne |
| 2026-11-02 | przygotowanie | SPEC przepisany na wykonalny etapowo: tabela statusu, protokół, karty etapów z bramkami | przegląd właściciela | trafne |
| 2026-11-02 | 1 | szkielet repo: `config.py` (walidacja YAML + alias klucza), `logging_setup.py` z redakcją, `errors.py`, `configs/*.yaml`, `tests/conftest.py` z blokadą gniazd, `scripts/probe_api.py` (dry-run/live), szkielet `docs/` i README | bramka etapu 1: `ruff check .` czysty, `pytest -q` → 26 passed, `load_config()` ok, `probe_api.py --dry-run` exit 0 | trafne po korektach (patrz niżej) |
| 2026-11-02 | 1 | `probe_api.py --live` na prawdziwym kluczu: 7 sprawdzeń, tempo 1 req/s, klucz nigdy nie wypisany | wyniki w `docs/api-notes.md` i SPEC §4.3 | trafne, ale ujawniło rozbieżność dokumentacji z API (status tag) |

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
10. **Otwarte na etap 2:** `requirements-dev.txt` zawiera `responses`, które mockuje `requests`.
    Nasz klient będzie na `urllib` (stdlib), więc mockowanie zrobimy na granicy modułu albo
    świadomie dodamy `requests`. Decyzja i ewentualna poprawka SPEC §12 — w etapie 2.
11. **Błąd własny (naprawiony):** przepisałem `tests/test_config.py` przez PowerShell
    (`Get-Content -Raw ... | Set-Content`) i skorzystałem z domyślnego kodowania konsoli —
    polskie znaki w pliku zamieniły się w mojibake („SPEC Â§12”), przez co test na etykiety
    gatunków zaczął padać z `KeyError`. Naprawa: plik napisany od nowa w UTF-8 narzędziem
    do zapisu plików, a nie przez PowerShell; testy wróciły do 26 passed. Wniosek: pliki
    z polskim tekstem edytuję tylko przez `write`/`edit`, nie przez `Set-Content`.

## Refleksja

Sekcja dla człowieka — uzupełniana po zakończeniu projektu (SPEC §16 pkt 8).
