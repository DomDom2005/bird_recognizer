# manual-checklist.md — kroki dla człowieka

Lista z `SPEC.md` §16. Agent nie ma dostępu do klucza, telefonu ani sali — te kroki wykonuje
właściciel projektu. Odhaczaj i dopisuj datę.

| # | Krok | Stan | Data / uwagi |
|---|---|---|---|
| 1 | Konto xeno-canto, weryfikacja e-mail, klucz API w `.env` | `[x]` | 2026-11-02 — klucz obecny w `.env` (nazwa `xc_api_key`) |
| 2 | Potwierdzenie listy 12 gatunków (D1) | `[x]` | 2026-11-02 — zatwierdzone bez zmian |
| 3 | `python scripts/probe_api.py --live` i zatwierdzenie ustaleń z SPEC §4.3 | `[x]` | 2026-11-02 — 7 sprawdzeń; wyniki w `docs/api-notes.md` §3–§4 i SPEC §4.3 |
| 4 | Realna liczebność nagrań per gatunek; decyzja, jeśli < 50 | `[x]` | 2026-11-02 — 81–606 na gatunek (min. szczygieł 81), próg spełniony |
| 5 | Nagrania własne telefonem: 20–50 klipów 5–30 s + arkusz etykiet | `[ ]` | pliki w `data/own/` (poza repo) |
| 6 | Decyzja o dostępie do GPU (Colab/Kaggle) na pełne ablacje | `[ ]` | bez GPU: grupy `rnn` i `pretrained` |
| 7 | Odsłuch próbki pobranych nagrań (błędy etykiet, gatunki w tle) | `[ ]` | po pobraniu danych (etap 2) |
| 8 | Wnioski w `docs/experiments.md` i „Refleksja” w dzienniku | `[ ]` | etap 8 |
| 9 | Decyzja o rozszerzeniach (D12, wieloetykietowość) | `[ ]` | po pierwszych wynikach |

## Uwagi środowiskowe

- `pip install -r requirements.txt -r requirements-dev.txt` uruchamiaj w zwykłym terminalu
  (poza sandboxem agenta) — patrz „Odstępstwa od SPEC” w `docs/ai-journal.md`.
- Waga zależności runtime (torch, gradio) jest duża; pierwsza instalacja może potrwać kilka minut.
