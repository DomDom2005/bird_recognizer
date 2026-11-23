# Gatunki: ile danych i jakie wyniki

> Plik generowany przez `scripts/species_report.py` z katalogu, cache i plików
> `eval_test` runów ablacji. Stan: 2026-11-24. Najlepszy run: `20261121-225300-cnn-window-10s-s42`
> (CNN, okno 10 s). Liczby dotyczą **pełnego zbioru**; `n` to liczba nagrań.

## 1. Jak to policzono

- **Dane**: `data/catalog.jsonl` (katalog), `data/cache/*/meta.json` (cache), `data/splits.json` (podział).
- **Wyniki**: `runs/<run>/eval_test/per_class.csv` dla 13 runów macierzy ablacji;
  `best_*` to najlepszy run, `median/min/max` — rozrzut po wszystkich runach.
- **Support (`n`)**: liczba nagrań gatunku w zbiorze testowym (suma = 307); dla runów z innym oknem support jest ten sam, zmienia się liczba okien.
- Gatunki w kolejności z `configs/species.yaml` (kolejność = etykiety klas).

## 2. Ile danych ma każdy gatunek

| gatunek | katalog | cache | train | val | test | okna zachowane | okna razem | odrzucone | audio [min] | okna test |
|---|---|---|---|---|---|---|---|---|---|---|
| wróbel | 119 | 119 | 104 | 11 | 4 | 3 414 | 3 484 | 2,0% | 86 | 87 |
| mazurek | 112 | 109 | 77 | 15 | 17 | 5 234 | 5 416 | 3,4% | 135 | 499 |
| bogatka | 606 | 603 | 505 | 61 | 37 | 18 227 | 20 116 | 9,4% | 498 | 1 168 |
| modra | 182 | 182 | 144 | 18 | 20 | 5 254 | 6 078 | 13,6% | 151 | 521 |
| kos | 343 | 340 | 274 | 34 | 32 | 16 573 | 18 647 | 11,1% | 463 | 1 412 |
| kwiczoł | 276 | 276 | 176 | 57 | 43 | 28 297 | 30 383 | 6,9% | 758 | 2 624 |
| pierwiosnek | 319 | 319 | 226 | 45 | 48 | 13 494 | 15 456 | 12,7% | 384 | 1 148 |
| piecuszek | 164 | 162 | 111 | 36 | 15 | 6 828 | 9 366 | 27,1% | 233 | 360 |
| rudzik | 299 | 296 | 223 | 49 | 24 | 13 208 | 16 765 | 21,2% | 417 | 1 109 |
| zięba | 315 | 312 | 215 | 50 | 47 | 15 075 | 18 720 | 19,5% | 466 | 1 447 |
| szczygieł | 81 | 79 | 61 | 12 | 6 | 2 609 | 2 726 | 4,3% | 68 | 175 |
| dzwoniec | 101 | 100 | 71 | 15 | 14 | 3 687 | 3 958 | 6,8% | 98 | 470 |
| **razem** | **2 917** | **2 897** | — | — | 307 | **131 900** | **151 115** | — | **3 756** | **11 020** |

Różnica katalog ↔ cache (20 nagrań) to 19 nagrań odrzuconych w całości przez bramkę ciszy
i 1 błąd dekodowania; **żadne z nich nie jest w teście**.

## 3. Wyniki per gatunek (najlepszy run)

Run `20261121-225300-cnn-window-10s-s42` — CNN z oknem 10 s. `median/min/max F1` to rozrzut po 13 runach macierzy.

| gatunek | n (test) | F1 | precision | recall | mediana F1 | min–max F1 | najczęstsza pomyłka |
|---|---|---|---|---|---|---|---|
| piecuszek | 15 | **0,938** | 0,882 | 1,000 | 0,848 | 0,667–0,938 | — |
| dzwoniec | 14 | **0,867** | 0,812 | 0,929 | 0,788 | 0,647–0,929 | zięba (1) |
| zięba | 47 | **0,866** | 0,840 | 0,894 | 0,837 | 0,707–0,876 | bogatka (1) |
| mazurek | 17 | **0,848** | 0,875 | 0,824 | 0,769 | 0,571–0,848 | wróbel (2) |
| kwiczoł | 43 | **0,790** | 0,842 | 0,744 | 0,747 | 0,484–0,805 | kos (3) |
| modra | 20 | **0,762** | 0,727 | 0,800 | 0,681 | 0,615–0,762 | kos (2) |
| pierwiosnek | 48 | **0,762** | 0,889 | 0,667 | 0,767 | 0,733–0,841 | szczygieł (5) |
| bogatka | 37 | **0,761** | 0,794 | 0,730 | 0,676 | 0,605–0,785 | modra (5) |
| kos | 32 | **0,727** | 0,706 | 0,750 | 0,740 | 0,597–0,814 | kwiczoł (3) |
| rudzik | 24 | **0,638** | 0,652 | 0,625 | 0,625 | 0,500–0,667 | kos (2) |
| szczygieł | 6 | **0,476** | 0,333 | 0,833 | 0,556 | 0,308–0,800 | bogatka (1) |
| wróbel | 4 | **0,400** | 0,333 | 0,500 | 0,400 | 0,167–0,600 | mazurek (1) |

## 4. Co z tego wynika

- **Najmocniejsze gatunki:** piecuszek (0,938), dzwoniec (0,867), zięba (0,866), mazurek (0,848).
- **Najsłabsze gatunki:** wróbel (0,400, n=4), szczygieł (0,476, n=6), rudzik (0,638, n=24), kos (0,727, n=32).
- **Najbardziej zmienne w macierzy** (największy rozrzut min–max): szczygieł (0,308–0,800), wróbel (0,167–0,600), kwiczoł (0,484–0,805).
- Małe klasy (wróbel, szczygieł) mają duży wpływ na macro-F1: jedna pomyłka zmienia F1
  o kilkanaście punktów procentowych. Dlatego raportujemy `n` obok każdej liczby.

## 5. Pliki źródłowe

- `docs/gatunki-dane-i-wyniki.md` — ten dokument (generowany).
- `reports/per-species-data.csv` — tabela danych w CSV.
- `reports/per-species-results.csv` — tabela wyników w CSV.
- `runs/<run>/eval_test/per_class.csv` — surowe wyniki per klasa (każdy run osobno).
- `docs/experiments.md` — pełna tabela macierzy ablacji.
