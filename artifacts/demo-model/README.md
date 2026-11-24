# Wbudowane modele demo

Ten katalog zawiera **mały checkpoint CNN** (1,6 MB); obok leży drugi model, **CRNN-GRU**
(`../demo-model-crnn`, 4,9 MB). Razem pozwalają uruchomić demo po sklonowaniu repo — bez
trenowania, bez pobierania danych i bez `.env`:

```bash
python -m bird_recognizer.cli demo            # oba modele, wybór w UI
python -m bird_recognizer.cli demo --smoke    # test ścieżki na tonie syntetycznym (CNN)
```

W UI modele są opisane jako „CNN, okno 10 s — test macro-F1 0,736” i
„CRNN-GRU, okno 3 s — test macro-F1 0,707”; domyślnie wybrany jest CNN.

## Skąd to jest

| Pole | Wartość |
|---|---|
| Run źródłowy | `20261121-225300-cnn-window-10s-s42` |
| Model | mały CNN (391 404 parametry), okno **10 s** |
| Test macro-F1 | 0,736 (top-1 0,772, top-3 0,902; 307 nagrań testowych) |
| Seed | 42 |
| Pliki | `best.pt` (wagi), `config.yaml` (geometria 10 s), `metrics.json` (metryki) |

Run powstał w pełnej macierzy ablacji (`docs/experiments.md`); to najlepszy wynik całości.
Drugi model (CRNN-GRU) opisany jest w [`../demo-model-crnn/README.md`](../demo-model-crnn/README.md).

## Licencja i uczciwe zastrzeżenie

- Oba modele wytrenowano na nagraniach z **xeno-canto** (głównie CC BY-NC-SA/ND; użycie
  **niekomercyjne**). W repo **nie ma audio** — to wyłącznie wagi i konfiguracja.
- Jeśli korzystasz z modeli publicznie, podaj źródło danych (xeno-canto + nagrywający)
  i nie używaj ich komercyjnie.
- To **artefakt badawczy**, nie produkt: 12 gatunków, Polska, mikrofony kierunkowe;
  dla nagrań z telefonu jakość nie została zmierzona. Brak gwarancji i odpowiedzialności.
- Pliki `best.pt` są wyjątkiem od reguły „modele poza repo” (świadoma decyzja: 1,6 MB + 4,9 MB,
  żeby demo działało po klonie). Duże checkpointy (ResNet18 itd.) nadal zostają w `runs/`.
