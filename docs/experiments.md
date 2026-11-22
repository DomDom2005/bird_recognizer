# Eksperymenty — bird_recognizer

> **Plik generowany przez `bird_recognizer report` — nie edytuj tabeli ręcznie.** Sekcje 1–4 pochodzą z `runs/index.csv`; sekcję „Wnioski” uzupełnia człowiek.

## 1. Tabela główna

| run | model | okno [s] | aug | pretrained | pooling | okna train | okna val | okna test | val macro-F1 (nagrania) | test macro-F1 (nagrania) | test top-1 | test top-3 | Δ test vs ref | seed | urządzenie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261121-172521-cnn-rnn-cnn-s42 | cnn | 3.0 | basic | none |  | 95898 | 24982 | 11020 | 0.686 | 0.704 | 0.736 | 0.883 | +0.000 | 42 | cuda |
| 20261121-180015-crnn-rnn-crnn-gru-s42 | crnn | 3.0 | basic | none | mean | 95898 | 24982 | 11020 | 0.722 | 0.707 | 0.739 | 0.912 | +0.004 | 42 | cuda |
| 20261121-184233-crnn-rnn-crnn-lstm-s42 | crnn | 3.0 | basic | none | mean | 95898 | 24982 | 11020 | 0.707 | 0.668 | 0.730 | 0.906 | -0.035 | 42 | cuda |
| 20261121-192439-cnn-aug-basic-s42 | cnn | 3.0 | basic | none |  | 95898 | 24982 | 11020 | 0.686 | 0.704 | 0.736 | 0.883 | +0.000 | 42 | cuda |
| 20261121-200314-cnn-aug-none-s42 | cnn | 3.0 | none | none |  | 95898 | 24982 | 11020 | 0.667 | 0.685 | 0.704 | 0.879 | -0.019 | 42 | cuda |
| 20261121-203116-cnn-aug-full-s42 | cnn | 3.0 | full | none |  | 95898 | 24982 | 11020 | 0.675 | 0.661 | 0.681 | 0.883 | -0.043 | 42 | cuda |
| 20261121-210118-crnn-pool-mean-s42 | crnn | 3.0 | basic | none | mean | 95898 | 24982 | 11020 | 0.722 | 0.707 | 0.739 | 0.912 | +0.004 | 42 | cuda |
| 20261121-213122-crnn-pool-attention-s42 | crnn | 3.0 | basic | none | attention | 95898 | 24982 | 11020 | 0.715 | 0.716 | 0.762 | 0.925 | +0.013 | 42 | cuda |
| 20261121-220112-cnn-window-3s-s42 | cnn | 3.0 | basic | none |  | 95898 | 24982 | 11020 | 0.686 | 0.704 | 0.736 | 0.883 | +0.000 | 42 | cuda |
| 20261121-222548-cnn-window-5s-s42 | cnn | 5.0 | basic | none |  | 62034 | 16013 | 7019 | 0.708 | 0.712 | 0.762 | 0.899 | +0.008 | 42 | cuda |
| 20261121-225300-cnn-window-10s-s42 | cnn | 10.0 | basic | none |  | 32918 | 8340 | 3701 | 0.721 | 0.736 | 0.772 | 0.902 | +0.033 | 42 | cuda |
| 20261121-232349-transfer-pretrained-scratch-s42 | transfer | 3.0 | basic | none |  | 95898 | 24982 | 11020 | 0.685 | 0.644 | 0.720 | 0.925 | -0.059 | 42 | cuda |
| 20261121-235042-transfer-pretrained-imagenet-s42 | transfer | 3.0 | basic | imagenet |  | 95898 | 24982 | 11020 | 0.641 | 0.698 | 0.726 | 0.912 | -0.006 | 42 | cuda |

Legenda: `Δ test vs ref` to różnica test macro-F1 (na poziomie nagrania) względem runu referencyjnego `20261121-172521-cnn-rnn-cnn-s42`. `n/d` = pole niedostępne (np. top-3 przy mniej niż 3 klasach). Wiersze z `smoke` w uwagach są przebiegami technicznymi na 2 nagraniach na gatunek — nie są wynikami badania; dotyczy to także sekcji 3–4, bo korzystają z tych samych runów (`eval_test`/`eval_own`).

## 2. Runy i seedy

- `20261121-172521-cnn-rnn-cnn-s42` — seed 42, cuda, git 04d94912, uwagi: rnn:cnn
- `20261121-180015-crnn-rnn-crnn-gru-s42` — seed 42, cuda, git c6a73fdf, uwagi: rnn:crnn-gru
- `20261121-184233-crnn-rnn-crnn-lstm-s42` — seed 42, cuda, git 87884073, uwagi: rnn:crnn-lstm
- `20261121-192439-cnn-aug-basic-s42` — seed 42, cuda, git bf83acea, uwagi: aug:basic
- `20261121-200314-cnn-aug-none-s42` — seed 42, cuda, git bf83acea, uwagi: aug:none
- `20261121-203116-cnn-aug-full-s42` — seed 42, cuda, git bf83acea, uwagi: aug:full
- `20261121-210118-crnn-pool-mean-s42` — seed 42, cuda, git bf83acea, uwagi: pool:mean
- `20261121-213122-crnn-pool-attention-s42` — seed 42, cuda, git bf83acea, uwagi: pool:attention
- `20261121-220112-cnn-window-3s-s42` — seed 42, cuda, git bf83acea, uwagi: window:3s
- `20261121-222548-cnn-window-5s-s42` — seed 42, cuda, git bf83acea, uwagi: window:5s
- `20261121-225300-cnn-window-10s-s42` — seed 42, cuda, git bf83acea, uwagi: window:10s
- `20261121-232349-transfer-pretrained-scratch-s42` — seed 42, cuda, git 8c789559, uwagi: pretrained:scratch
- `20261121-235042-transfer-pretrained-imagenet-s42` — seed 42, cuda, git 8c789559, uwagi: pretrained:imagenet

## 3. Pary mylące

| para | gatunek | 20261121-172521-cnn-rnn-cnn-s42 | 20261121-180015-crnn-rnn-crnn-gru-s42 | 20261121-184233-crnn-rnn-crnn-lstm-s42 | 20261121-192439-cnn-aug-basic-s42 | 20261121-200314-cnn-aug-none-s42 | 20261121-203116-cnn-aug-full-s42 | 20261121-210118-crnn-pool-mean-s42 | 20261121-213122-crnn-pool-attention-s42 | 20261121-220112-cnn-window-3s-s42 | 20261121-222548-cnn-window-5s-s42 | 20261121-225300-cnn-window-10s-s42 | 20261121-232349-transfer-pretrained-scratch-s42 | 20261121-235042-transfer-pretrained-imagenet-s42 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wrobel-mazurek | wróbel | 0.500 | 0.500 | 0.200 | 0.500 | 0.600 | 0.333 | 0.500 | 0.333 | 0.500 | 0.182 | 0.400 | 0.167 | 0.250 |
| wrobel-mazurek | mazurek | 0.710 | 0.774 | 0.667 | 0.710 | 0.640 | 0.769 | 0.774 | 0.774 | 0.710 | 0.800 | 0.848 | 0.571 | 0.812 |
| bogatka-modra | bogatka | 0.674 | 0.676 | 0.727 | 0.674 | 0.730 | 0.675 | 0.676 | 0.685 | 0.674 | 0.785 | 0.761 | 0.667 | 0.605 |
| bogatka-modra | modra | 0.629 | 0.615 | 0.638 | 0.629 | 0.686 | 0.703 | 0.615 | 0.681 | 0.629 | 0.732 | 0.762 | 0.682 | 0.722 |
| kos-kwiczol | kos | 0.814 | 0.771 | 0.690 | 0.814 | 0.600 | 0.597 | 0.771 | 0.740 | 0.814 | 0.735 | 0.727 | 0.737 | 0.767 |
| kos-kwiczol | kwiczoł | 0.740 | 0.800 | 0.747 | 0.740 | 0.630 | 0.484 | 0.800 | 0.790 | 0.740 | 0.743 | 0.790 | 0.785 | 0.805 |
| pierwiosnek-piecuszek | pierwiosnek | 0.761 | 0.733 | 0.779 | 0.761 | 0.805 | 0.782 | 0.733 | 0.841 | 0.761 | 0.818 | 0.762 | 0.816 | 0.767 |
| pierwiosnek-piecuszek | piecuszek | 0.938 | 0.733 | 0.714 | 0.938 | 0.867 | 0.848 | 0.733 | 0.848 | 0.938 | 0.875 | 0.938 | 0.774 | 0.667 |

## 4. Zbiór własny

_(brak nagrań własnych w `data/own/<gatunek>/` albo brak `eval_own` — to zadanie człowieka, SPEC §16; nic nie podstawiamy w zamian)_

## 5. Wnioski

_(uzupełnia człowiek — SPEC §11.3; agent nie wymyśla wniosków)_
