# Eksperymenty — bird_recognizer

> **Plik generowany przez `bird_recognizer report` — nie edytuj tabeli ręcznie.** Sekcje 1–4 pochodzą z `runs/index.csv`; sekcję „Wnioski” uzupełnia człowiek.

## 1. Tabela główna

| run | model | okno [s] | aug | pretrained | pooling | okna train | okna val | okna test | val macro-F1 (nagrania) | test macro-F1 (nagrania) | test top-1 | test top-3 | Δ test vs ref | seed | urządzenie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261121-172521-cnn-rnn-cnn-s42 | cnn | 3.0 | basic | none |  | 95898 | 24982 | 11020 | 0.686 | 0.704 | 0.736 | 0.883 | +0.000 | 42 | cuda |
| 20261121-180015-crnn-rnn-crnn-gru-s42 | crnn | 3.0 | basic | none | mean | 95898 | 24982 | 11020 | 0.722 | 0.707 | 0.739 | 0.912 | +0.004 | 42 | cuda |
| 20261121-184233-crnn-rnn-crnn-lstm-s42 | crnn | 3.0 | basic | none | mean | 95898 | 24982 | 11020 | 0.707 | 0.668 | 0.730 | 0.906 | -0.035 | 42 | cuda |

Legenda: `Δ test vs ref` to różnica test macro-F1 (na poziomie nagrania) względem runu referencyjnego `20261121-172521-cnn-rnn-cnn-s42`. `n/d` = pole niedostępne (np. top-3 przy mniej niż 3 klasach). Wiersze z `smoke` w uwagach są przebiegami technicznymi na 2 nagraniach na gatunek — nie są wynikami badania; dotyczy to także sekcji 3–4, bo korzystają z tych samych runów (`eval_test`/`eval_own`).

## 2. Runy i seedy

- `20261121-172521-cnn-rnn-cnn-s42` — seed 42, cuda, git 04d94912, uwagi: rnn:cnn
- `20261121-180015-crnn-rnn-crnn-gru-s42` — seed 42, cuda, git c6a73fdf, uwagi: rnn:crnn-gru
- `20261121-184233-crnn-rnn-crnn-lstm-s42` — seed 42, cuda, git 87884073, uwagi: rnn:crnn-lstm

## 3. Pary mylące

| para | gatunek | 20261121-172521-cnn-rnn-cnn-s42 | 20261121-180015-crnn-rnn-crnn-gru-s42 | 20261121-184233-crnn-rnn-crnn-lstm-s42 |
|---|---|---|---|---|
| wrobel-mazurek | wróbel | 0.500 | 0.500 | 0.200 |
| wrobel-mazurek | mazurek | 0.710 | 0.774 | 0.667 |
| bogatka-modra | bogatka | 0.674 | 0.676 | 0.727 |
| bogatka-modra | modra | 0.629 | 0.615 | 0.638 |
| kos-kwiczol | kos | 0.814 | 0.771 | 0.690 |
| kos-kwiczol | kwiczoł | 0.740 | 0.800 | 0.747 |
| pierwiosnek-piecuszek | pierwiosnek | 0.761 | 0.733 | 0.779 |
| pierwiosnek-piecuszek | piecuszek | 0.938 | 0.733 | 0.714 |

## 4. Zbiór własny

_(brak nagrań własnych w `data/own/<gatunek>/` albo brak `eval_own` — to zadanie człowieka, SPEC §16; nic nie podstawiamy w zamian)_

## 5. Wnioski

_(uzupełnia człowiek — SPEC §11.3; agent nie wymyśla wniosków)_
