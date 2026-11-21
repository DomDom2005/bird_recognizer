# AGENTS.md

bird_recognizer: projekt badawczy w Pythonie 3.11+, klasyfikuje gatunek ptaka z nagrania
(mel-spektrogramy, CNN / transfer / CRNN) i porównuje model z rekurencją i bez niej.
Specyfikacja: `project info/SPEC.md`. Plan etapów: `project info/PLAN.md`.
Dokumenty projektu mieszkają w `project info/`; kod aplikacji leży w katalogu głównym repo.
Dane (`data/`) i wyniki (`runs/`) są poza repo.

## Komendy

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .                          # pakiet z src/ → komenda bird_recognizer
ruff check . && pytest                    # przed zakończeniem każdego zadania
pytest tests/test_splits.py -k recordist  # pojedynczy plik / wzorzec

python scripts/probe_api.py --dry-run     # składa query bez sieci (SPEC §4.3)
python scripts/probe_api.py --live        # jeden request na punkt 4.3 — wymaga klucza
bird_recognizer data catalog && bird_recognizer data download && bird_recognizer data split
python scripts/check_leakage.py --data-dir data   # ZAWSZE przed treningiem
bird_recognizer prep cache --limit 24 && bird_recognizer prep cache --limit 24 --verify
bird_recognizer train mfcc --smoke && bird_recognizer train cnn --smoke
bird_recognizer train crnn --pool attention --smoke
bird_recognizer train transfer --arch resnet18 --scratch --smoke   # bez sieci
bird_recognizer train transfer --arch resnet18 --smoke            # wagi ImageNet (raz, sieć)
bird_recognizer ablate --group rnn --smoke
bird_recognizer report --runs RUN1 RUN2 RUN3 --out docs/experiments.md
bird_recognizer eval --run RUN --split test
bird_recognizer aggregate --run RUN --method vote
bird_recognizer error-analysis --run RUN
bird_recognizer eval --run RUN --split own     # data/own/<gatunek>/* (zadanie człowieka)
bird_recognizer demo --run RUN
```

Jeśli `bird_recognizer` nie jest w PATH (zdarza się po `pip install -e .` na Windows),
używaj równoważnej formy `python -m bird_recognizer.cli <komenda>`.

## Zasady

- Testy działają offline: HTTP jest mockowany wstrzykiwanym transportem (`tests/support.py`),
  a `conftest.py` blokuje gniazda. Żadnych prawdziwych żądań do xeno-canto ani pobierania audio w testach.
- Klucz API tylko w `.env` jako `BIRD_XC_API_KEY`, poza repo. Klucz jedzie w query stringu,
  więc **nie loguj pełnych URL-i** — filtr redakcji maskuje `key=`.
- Audio z xeno-canto **nie wchodzi do repo** (licencje CC, część zabrania użycia komercyjnego).
  Commitujemy kod, metadane i wyniki, nie pliki dźwiękowe.
- Podział tylko według nagrań i grup nagrywających; `check_leakage.py` przed każdym treningiem.
  Nie zmieniaj `splits.json` w trakcie porównywania runów (bez `--force` i bez wpisu w dzienniku).
- Jedna zmiana = jeden run = jeden katalog w `runs/`. Run bez configu, seeda i `git_sha` nie istnieje.
- Hiperparametry i progi wyłącznie w `configs/default.yaml`; żadnych liczb w kodzie modeli.
- Augmentacje tylko na zbiorze treningowym; walidacja i test zawsze czyste.
- Przed użyciem pola lub parametru API sprawdź https://xeno-canto.org/explore/api.
  Jeśli dokumentacja przeczy `SPEC.md`, wygrywa dokumentacja — popraw SPEC i dziennik.
- Kod, komentarze i commity po angielsku (Conventional Commits); README, komunikaty CLI,
  tabele wyników i raport po polsku.

## Pułapki

- **Wyciek danych to ryzyko numer jeden.** Ten sam nagrywający albo ten sam plik w dwóch
  podziałach daje wyniki, których nie da się obronić. Jeśli macro-F1 wygląda podejrzanie dobrze,
  najpierw sprawdź podział, nie model.
- `query` w v3 przyjmuje **tylko tagi**; gołe nazwy gatunków zwracają błąd. Tagi z operatorem
  i wielowyrazowe muszą być w cudzysłowie: `sp:"Parus major"`, `q:">C"`, `cnt:poland`.
- Gatunki chronione: odpowiedź ma `_meta.redacted_fields` i **brak `file`** — takie nagranie
  pomijamy i liczymy jako `redacted`, to nie jest błąd pobierania.
- `length` to łańcuch typu `"4:08"`, nie liczba sekund. Parsuj defensywnie.
- `also` (gatunki w tle) to szum etykiet, a nie ciekawostka — używaj go w analizie błędów.
- Okna ciszy muszą odpadać (bramkowanie po energii), inaczej model uczy się ciszy.
- CRNN widzi zmienną długość sekwencji (krótsze ostatnie okno) — maskuj w loss i poolingu.
- Metryka główna to macro-F1 na poziomie nagrania; zawsze podawaj `n` i udział odrzuconych okien.
- Rozkład klas jest niezbalansowany: nie raportuj samej accuracy.
- GPU bywa niedeterministyczne mimo seeda — raportuj urządzenie i nie obiecuj powtarzalności.
- **Determinizm zaczyna się od seeda:** `set_seed(config.seed)` musi być **przed** utworzeniem
  modelu (torch startuje z losowym stanem procesu). Ten błąd dał dwa różne wyniki z tym samym
  configiem — patrz dziennik, odstępstwo 24.
- `--pretrained` pobiera wagi ImageNet **raz** do `~/.cache/torch`. W sandboxie DSH ten katalog
  bywa niezapisywalny (`WinError 5`) — ustaw wtedy `TORCH_HOME` na katalog w `data/`
  (dane są poza repo). `--scratch` działa bez sieci i bez cache.
- Augmentacja `full` = przesunięcie + szum + SpecAugment (per okno) + mixup (batch treningowy).
  Na znormalizowanym mel zero w masce to średnia pasma, a mixup miesza **tylko** batch treningowy;
  walidacja i test zostają czyste — to warunek porównywalności runów.
- **Ablacja to jedna dźwignia na grupę** (`architecture` | `window_s` | `aug`). W grupie `pretrained`
  obie gałęzie uczą się z tym samym `lr`, bo zmiana `lr` byłaby drugą zmianą; w grupie `window`
  `n` okien różni się z definicji — porównuj F1, nie `n`. Definicję grupy trzymaj w `ablation.py`,
  a wynikowy `runs/ablation_<grupa>.json` jest dowodem różnicy.
- Grupa `window` potrzebuje cache dla innej geometrii okna. `ablate` sam tworzy `data/window5`
  i `data/window10` (twarde linki do `data/raw`, kopia `catalog.jsonl` i `splits.json`) i buduje
  cache — nie rób tego ręcznie i nie commituj tych katalogów.
- **GPU:** `requirements.txt` pinuje `torch==2.15.0+cu132`; urządzenie wybiera się automatycznie,
  `--device cpu` wymusza CPU (testy CLI używają CPU, żeby były szybkie i powtarzalne). Wyniki
  GPU mogą różnić się od CPU mimo seeda — raportuj `device`, nie obiecuj powtarzalności.
- **CUDA 13.2 + cuBLASLt a stos wątku (Windows):** ResNet18 na GPU potrafi paść z
  `0xC0000409`/`0xc00000fd` (przepełnienie stosu w `cublasLt64_13.dll`) na domyślnym 1 MB stosie
  wątku głównego. CLI uruchamia komendy w wątku z 64 MB stosu (`_call_with_big_stack`), a `demo`
  działa na CPU. Jeśli dodajesz nowy kod wołający duże GEMM-y na CUDA, uruchamiaj go przez CLI
  albo w takim wątku — nie „naprawiaj" tego zmianą modelu.
- **`eval` czyta geometrię z `config.yaml` runu**, nie z bieżącego `configs/`: run 5 s/10 s
  musi być liczony na cache `data/windowN`. Wyniki idą do `runs/<run>/eval_<split>/` i
  `runs/<run>/analysis/`, nie nadpisują `metrics.json` treningu.
- **Zbiór własny:** `data/own/<gatunek>/` (`*.wav|mp3|flac|ogg`). Brak danych → kod 3 i jawne
  „brak nagrań własnych” w raporcie; nie podstawiaj syntetyku ani cudzych nagrań.
- `balanced_accuracy` = macro recall po **wszystkich** klasach; klasa nieobecna w podziale ma
  recall 0 (nie znika z metryki). Pary mylące liczone tylko z realnych pomyłek (prawda ≠ predykcja).
- `data/cache` jest w pełni odtwarzalny z `data/raw`; usunięcie cache nie traci danych.
- Windows: `DataLoader` z `num_workers>0` wymaga `if __name__ == "__main__"` w skryptach
  startowych; przy problemach zacznij od `num_workers=0`.
- Nie mieszaj `librosa` i `torchaudio` w jednej ścieżce przetwarzania — mel-spektrogramy
  różnią się normalizacją i skalami, a porównania runów przestają mieć sens.

## Etapy

Pracujesz etapami 1–8 z `SPEC.md` §15. Status trzymasz **w tabeli 15.1 w SPEC**, nie w osobnym pliku.
Zaczynasz pierwszy etap ze statusem `[ ]`, którego poprzednik ma `[x]`. Bez zielonej bramki nie ma następnego etapu.
Jeden etap = karta w SPEC (zadania, bramka, commit, dziennik) = jeden domykający commit = jeden wpis w dzienniku.
Nowa sesja zaczyna od §15.5: `git log` → tabela 15.1 → karta etapu → ostatnie wpisy dziennika → bramka poprzednika.

## Dziennik AI

Po każdej zmianie, w której coś Cię zaskoczyło, poprawiłeś założenie albo dane okazały się
inne niż w SPEC, dopisz wpis w `docs/ai-journal.md` (tabela + sekcja „Odstępstwa od SPEC”).
Wnioski badawcze w `docs/experiments.md` pisze człowiek — Ty dostarczasz tabele i liczby.
