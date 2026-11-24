# Wbudowany model demo

Ten katalog zawiera **jeden mały checkpoint** (1,6 MB), żeby po sklonowaniu repo można było
uruchomić demo bez trenowania i bez pobierania danych:

```bash
python -m bird_recognizer.cli demo            # używa artifacts/demo-model
python -m bird_recognizer.cli demo --smoke    # test ścieżki na tonie syntetycznym
```

## Skąd to jest

| Pole | Wartość |
|---|---|
| Run źródłowy | `20261121-225300-cnn-window-10s-s42` |
| Model | mały CNN (391 404 parametry), okno **10 s** |
| Test macro-F1 | 0,736 (top-1 0,772, top-3 0,902; 307 nagrań testowych) |
| Seed | 42 |
| Pliki | `best.pt` (wagi), `config.yaml` (geometria 10 s), `metrics.json` (metryki) |

Run powstał w pełnej macierzy ablacji (`docs/experiments.md`); to najlepszy wynik całości.

## Licencja i uczciwe zastrzeżenie

- Model wytrenowano na nagraniach z **xeno-canto** (głównie CC BY-NC-SA/ND; użycie
  **niekomercyjne**). W repo **nie ma audio** — to wyłącznie wagi i konfiguracja.
- Jeśli korzystasz z modelu publicznie, podaj źródło danych (xeno-canto + nagrywający)
  i nie używaj go komercyjnie.
- To **artefakt badawczy**, nie produkt: 12 gatunków, Polska, mikrofony kierunkowe;
  dla nagrań z telefonu jakość nie została zmierzona. Brak gwarancji i odpowiedzialności.
- Plik `best.pt` jest wyjątkiem od reguły „modele poza repo” (świadoma decyzja: 1,6 MB,
  żeby demo działało po klonie). Duże checkpointy (ResNet18 itd.) nadal zostają w `runs/`.
