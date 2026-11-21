# manual-checklist.md — kroki dla człowieka

Lista z `SPEC.md` §16. Agent nie ma dostępu do klucza, telefonu ani sali — te kroki wykonuje
właściciel projektu. Odhaczaj i dopisuj datę.

| # | Krok | Stan | Data / uwagi |
|---|---|---|---|
| 1 | Konto xeno-canto, weryfikacja e-mail, klucz API w `.env` | `[x]` | 2026-11-02 — klucz obecny w `.env` (nazwa `xc_api_key`) |
| 2 | Potwierdzenie listy 12 gatunków (D1) | `[x]` | 2026-11-02 — zatwierdzone bez zmian |
| 3 | `python scripts/probe_api.py --live` i zatwierdzenie ustaleń z SPEC §4.3 | `[x]` | 2026-11-02 — 7 sprawdzeń; wyniki w `docs/api-notes.md` §3–§4 i SPEC §4.3 |
| 4 | Realna liczebność nagrań per gatunek; decyzja, jeśli < 50 | `[x]` | 2026-11-02 — 81–606 na gatunek (min. szczygieł 81), próg spełniony |
| 5 | Nagrania własne telefonem: 20–50 klipów 5–30 s + arkusz etykiet | `[ ]` | pliki w `data/own/<gatunek>/*.wav|mp3|flac|ogg` (poza repo); bez nich `eval --split own` kończy się kodem 3, a raport jawnie pokazuje brak |
| 6 | Decyzja o dostępie do GPU (Colab/Kaggle) na pełne ablacje | `[x]` | 2026-11-21 — lokalna RTX 5070 12 GB; zainstalowany `torch 2.15.0+cu132` (decyzja właściciela), pełne runy macierzy na GPU |
| 7 | Odsłuch próbki pobranych nagrań (błędy etykiet, gatunki w tle) | `[ ]` | człowiek; `data/catalog.jsonl` ma `q`, `type`, `also`, `rmk` do przeglądu |
| 8 | Wnioski w `docs/experiments.md` i „Refleksja” w dzienniku | `[ ]` | tabela gotowa (pełne wyniki `rnn`: 0,704/0,707/0,668 test macro-F1); wnioski i refleksję uzupełnia człowiek |
| 9 | Decyzja o rozszerzeniach (D12, wieloetykietowość) | `[ ]` | po pierwszych pełnych wynikach |

## Uwagi środowiskowe

- Instalacja: `pip install -r requirements.txt -r requirements-dev.txt --extra-index-url
  https://download.pytorch.org/whl/cu132` (build CUDA); na maszynie bez GPU zainstaluj
  `torch==2.15.0` z PyPI.
- Waga zależności runtime (torch, gradio) jest duża; pierwsza instalacja może potrwać kilka minut.
- Pełny epoch CNN na całym zbiorze: ~130 s na RTX 5070 (po poprawce I/O; wcześniej 757 s).
