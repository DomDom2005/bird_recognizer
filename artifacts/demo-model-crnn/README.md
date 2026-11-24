# Wbudowany model demo — CRNN-GRU

Drugi checkpoint demo (4,9 MB), żeby w UI można było porównać model „z czasem” z CNN:

```bash
python -m bird_recognizer.cli demo      # wybór modelu w UI
```

## Skąd to jest

| Pole | Wartość |
|---|---|
| Run źródłowy | `20261121-180015-crnn-rnn-crnn-gru-s42` |
| Model | CRNN: CNN + BiGRU, pooling mean (1 212 396 parametrów), okno **3 s** |
| Test macro-F1 | 0,707 (top-1 0,744, top-3 0,899; 307 nagrań testowych) |
| Seed | 42 |
| Pliki | `best.pt` (wagi), `config.yaml` (geometria 3 s), `metrics.json` (metryki) |

## Licencja i zastrzeżenia

Jak [`../demo-model/README.md`](../demo-model/README.md): dane z xeno-canto (głównie
CC BY-NC-SA/ND), użycie **niekomercyjne**, brak gwarancji, 12 gatunków, Polska, mikrofony
kierunkowe. W repo jest tylko wagi + konfiguracja, bez audio.
