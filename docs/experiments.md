# Eksperymenty — bird_recognizer

> **Plik generowany przez `bird_recognizer report` — nie edytuj tabeli ręcznie.** Sekcje 1–4 pochodzą z `runs/index.csv`; sekcję „Wnioski” uzupełnia człowiek.

## 1. Tabela główna

| run | model | okno [s] | aug | pretrained | pooling | okna train | okna val | okna test | val macro-F1 (nagrania) | test macro-F1 (nagrania) | test top-1 | test top-3 | Δ test vs ref | seed | urządzenie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261120-231932-cnn-rnn-cnn-s42 | cnn | 3.0 | basic | none |  | 762 | 910 | 825 | 0.067 | 0.049 | 0.083 | 0.292 | +0.000 | 42 | cuda |
| 20261120-231935-crnn-rnn-crnn-gru-s42 | crnn | 3.0 | basic | none | mean | 762 | 910 | 825 | 0.064 | 0.045 | 0.083 | 0.167 | -0.004 | 42 | cuda |
| 20261120-231938-crnn-rnn-crnn-lstm-s42 | crnn | 3.0 | basic | none | mean | 762 | 910 | 825 | 0.057 | 0.062 | 0.083 | 0.167 | +0.014 | 42 | cuda |

Legenda: `Δ test vs ref` to różnica test macro-F1 (na poziomie nagrania) względem runu referencyjnego `20261120-231932-cnn-rnn-cnn-s42`. `n/d` = pole niedostępne (np. top-3 przy mniej niż 3 klasach). Wiersze z `smoke` w uwagach są przebiegami technicznymi na 2 nagraniach na gatunek — nie są wynikami badania; dotyczy to także sekcji 3–4, bo korzystają z tych samych runów (`eval_test`/`eval_own`).

## 2. Runy i seedy

- `20261120-231932-cnn-rnn-cnn-s42` — seed 42, cuda, git fcc45579, uwagi: rnn:cnn; smoke
- `20261120-231935-crnn-rnn-crnn-gru-s42` — seed 42, cuda, git fcc45579, uwagi: rnn:crnn-gru; smoke
- `20261120-231938-crnn-rnn-crnn-lstm-s42` — seed 42, cuda, git fcc45579, uwagi: rnn:crnn-lstm; smoke

## 3. Pary mylące

| para | gatunek | 20261120-231932-cnn-rnn-cnn-s42 | 20261120-231935-crnn-rnn-crnn-gru-s42 | 20261120-231938-crnn-rnn-crnn-lstm-s42 |
|---|---|---|---|---|
| wrobel-mazurek | wróbel | 0.000 | 0.020 | 0.000 |
| wrobel-mazurek | mazurek | 0.200 | 0.000 | 0.118 |
| bogatka-modra | bogatka | 0.000 | 0.167 | 0.143 |
| bogatka-modra | modra | 0.000 | 0.000 | 0.130 |
| kos-kwiczol | kos | 0.000 | 0.061 | 0.105 |
| kos-kwiczol | kwiczoł | 0.000 | 0.043 | 0.000 |
| pierwiosnek-piecuszek | pierwiosnek | 0.175 | 0.110 | 0.167 |
| pierwiosnek-piecuszek | piecuszek | 0.107 | 0.133 | 0.135 |

## 4. Zbiór własny

_(brak nagrań własnych w `data/own/<gatunek>/` albo brak `eval_own` — to zadanie człowieka, SPEC §16; nic nie podstawiamy w zamian)_

## 5. Wnioski

_(uzupełnia człowiek — SPEC §11.3; agent nie wymyśla wniosków)_
