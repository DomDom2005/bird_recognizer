# Przewodnik po projekcie — bird_recognizer

> Ten plik jest dla człowieka, nie dla agenta. Wyjaśnia **prostym językiem**: co to za projekt,
> jak działa, co znaczą terminy techniczne i jak elementy łączą się ze sobą.
> Nie zastępuje specyfikacji: pełne zasady są w [`project info/SPEC.md`](../project%20info/SPEC.md).

## 1. O co w tym projekcie chodzi (w trzech zdaniach)

Program ma rozpoznawać **12 gatunków polskich ptaków śpiewających** z nagrania dźwięku.
Dostaje 3-sekundowy wycinek nagrania, zamienia go na **obraz dźwięku** (spektrogram), a model
uczy się rozpoznawać wzór. Najważniejsze pytanie badawcze: **czy model, który patrzy na kolejność
dźwięków w czasie (RNN), wypada lepiej od takiego, który patrzy tylko na obraz bez czasu (CNN)?**

Projekt jest badawczy, nie produkcyjny: liczy się **odtwarzalność i uczciwe porównanie modeli**,
nie „najlepszy wynik za każdą cenę”.

## 2. Jak czytać ten dokument

- **Chcesz szybko zrozumieć, co działa:** sekcje 3, 4 i 9.
- **Chcesz wiedzieć, co znaczy jakiś termin:** sekcja 5 (słownik).
- **Chcesz wiedzieć, jak pliki się łączą:** sekcja 6.
- **Chcesz wiedzieć, co jest prawdziwym wynikiem, a co tylko testem:** sekcja 8.
- **Chcesz uruchomić coś sam:** sekcja 10 (ściąga komend).

## 3. Mapa projektu — co gdzie leży

| Miejsce | Co tam jest |
|---|---|
| `project info/SPEC.md` | **Jedno źródło prawdy**: wymagania, decyzje, karty etapów, bramki |
| `project info/PLAN.md` | Plan 8 etapów (po co, w jakiej kolejności) |
| `project info/AGENTS.md` | Zasady pracy agenta (komendy, pułapki, podział danych) |
| `docs/ai-journal.md` | Dziennik: co zrobiono, jak sprawdzono, gdzie się pomylono |
| `docs/api-notes.md` | Sprawdzone fakty o API xeno-canto |
| `docs/data-flow.md` | Diagram przepływu danych |
| `docs/experiments.md` | **Generowana** tabela wyników (nie edytuje się jej ręcznie) |
| `docs/manual-checklist.md` | Kroki, które musi wykonać człowiek (klucz, nagrania własne) |
| `docs/przewodnik.md` | Ten plik |
| `src/bird_recognizer/` | Kod programu (katalog główny repo) |
| `configs/default.yaml` | Parametry: audio, spektrogram, trening (np. okno 3 s) |
| `configs/species.yaml` | Lista 12 gatunków — **kolejność = numery klas** |
| `tests/` | Testy offline (bez sieci, na generowanych WAV-ach) |
| `data/` (poza repo) | Katalog nagrań, audio, podział, cache spektrogramów |
| `runs/` (poza repo) | Każdy trening: konfiguracja, model, metryki, `index.csv` |

**Ważne:** audio (`data/`) i wyniki (`runs/`) **nie są w repozytorium** — są za duże i mają
licencje; w repo jest kod, konfiguracja i dokumentacja.

## 4. Jak projekt działa — przepływ krok po kroku

```
xeno-canto API
   │  (1) spis nagrań: gatunek, nagrywający, jakość, link
   ▼
data/catalog.jsonl          (2) metadane wszystkich 2917 nagrań
   │
   ├──────────────► data/raw/<id>.mp3   (3) pobrane audio (8,2 GB, poza repo)
   │
   ▼
data/splits.json            (4) podział train/val/test po nagraniach i nagrywających
   │
   ▼
data/cache/<id>/mel.npy     (5) okna 3 s → obrazy dźwięku (log-mel), zapis float16
   │
   ├──► trening modeli (6)  → runs/<run_id>/  → runs/index.csv
   ├──► ablacje (7)         → runs/ablation_<grupa>.json
   ├──► ewaluacja (8)       → runs/<run>/eval_<split>/ + analysis/
   └──► demo (9)            → plik/nagranie → top-3 gatunki
                                   │
                                   ▼
                     docs/experiments.md (10) tabela wyników
```

**Po ludzku, etapami:**

1. **Spis nagrań** — program pyta serwis xeno-canto o nagrania 12 gatunków z Polski (jakość A/B)
   i zapisuje metadane do `data/catalog.jsonl`. To „książka adresowa” nagrań.
2. **Pobieranie audio** — pliki lądują w `data/raw/`. Pobieranie jest wznawialne i liczy
   „odcisk palca” pliku (sha256), żeby wykryć duplikaty.
3. **Podział danych** — nagrania dzielimy na **train** (nauka), **val** (podgląd podczas nauki)
   i **test** (egzamin). Dzielimy **grupami nagrywających**, nie pojedynczymi plikami.
4. **Okienkowanie i spektrogramy** — każde nagranie tniemy na okna 3 s (co 1,5 s), odrzucamy
   okna ciszy, a resztę zamieniamy na obraz dźwięku i zapisujemy w `data/cache`. To trwa długo,
   więc robi się to **raz** — potem treningi czytają gotowy cache.
5. **Trening modeli** — cztery rodziny: MFCC+SVM (klasyczny baseline), mały CNN, transfer
   (ResNet18/EfficientNet z ImageNet) i CRNN (CNN + RNN). Każdy trening to osobny **run**.
6. **Ablacje** — eksperymenty, w których zmieniamy **jedną rzecz** naraz i patrzymy, czy pomaga.
7. **Ewaluacja** — liczymy metryki na val/test, piszemy macierz pomyłek i analizę błędów.
8. **Demo** — wgrywasz plik, dostajesz top-3 gatunki + spektrogram.

## 5. Słownik pojęć (prosto)

### Dane i pliki

| Pojęcie | Najprościej | Gdzie w projekcie |
|---|---|---|
| **API** | „okienko”, przez które program prosi serwis o dane | xeno-canto API v3 |
| **JSON / JSONL** | tekstowy format danych; JSONL = jedna notatka na linię | `catalog.jsonl` |
| **YAML** | podobny format, używany do konfiguracji | `configs/*.yaml` |
| **`.env`** | plik z sekretami (klucz API), poza repo | `.env` |
| **klucz API** | osobisty „bilet wstępu” do xeno-canto | `.env` |
| **katalog** | spis wszystkich nagrań z metadanymi | `data/catalog.jsonl` |
| **sha256** | „odcisk palca” pliku; ten sam plik = ten sam odcisk | w katalogu i raportach |
| **cache** | gotowe przeliczone dane, żeby nie liczyć drugi raz | `data/cache` |
| **`config_hash`** | odcisk konfiguracji; zmiana parametrów = cache nieaktualny | `meta.json` |

### Dźwięk i spektrogram

| Pojęcie | Najprościej | Gdzie w projekcie |
|---|---|---|
| **mel-spektrogram** | obraz dźwięku: oś X = czas, oś Y = wysokość tonu, kolor = głośność | `data/cache/*/mel.npy` |
| **okno (window)** | wycinek nagrania, na który patrzy model (tu 3 s) | `window_s` |
| **hop** | o ile przesuwamy okno (tu 1,5 s) | `hop_s` |
| **bramkowanie ciszy** | odrzucanie okien bez śpiewu (cisza uczyłaby modelu ciszy) | `gating` |
| **resample** | przeskalowanie częstotliwości próbkowania (tu 32 kHz) | `audio.sample_rate` |
| **filtr górnoprzepustowy** | wycięcie najniższych, zakłócających tonów (tu <100 Hz) | `highpass_hz` |
| **normalizacja** | wyrównanie głośności każdego okna (średnia 0, odchylenie 1) | w cache |

### Modele

| Pojęcie | Najprościej | Gdzie w projekcie |
|---|---|---|
| **CNN** | sieć, która szuka wzorów na obrazie (kształtów na spektrogramie); nie widzi kolejności w czasie | `models/cnn.py` |
| **RNN (GRU/LSTM)** | sieć czytająca sekwencję krok po kroku i pamiętająca poprzednie kroki | `models/crnn.py` |
| **BiGRU / BiLSTM** | RNN czytający w obie strony (od początku i od końca) | CRNN |
| **CRNN** | CNN wyciąga cechy, RNN czyta je w czasie, pooling robi podsumowanie | `models/crnn.py` |
| **pooling mean / attention** | jak podsumować ramki: średnia vs „ucz się, na co patrzeć” | `--pool` |
| **MFCC** | stary, kompaktowy opis dźwięku (jak odcisk palca) | `models/mfcc_baseline.py` |
| **SVM / Random Forest** | klasyczne klasyfikatory (nie sieci neuronowe) | `--clf svm|rf` |
| **transfer / ImageNet** | model najpierw uczył się rozpoznawać obrazy, potem dostrajamy go do ptaków | `--pretrained` |
| **checkpoint** | zapisane „wagi” (wiedza) wytrenowanego modelu | `runs/*/best.pt` |
| **parametry** | liczba „pokręteł” w modelu; więcej = cięższy | `env.txt` |

### Trening

| Pojęcie | Najprościej | Gdzie w projekcie |
|---|---|---|
| **epoka (epoch)** | jeden pełny przegląd danych treningowych | `metrics.jsonl` |
| **batch** | paczka przykładów przetwarzana razem (tu 32) | `batch_size` |
| **loss** | kara za błąd; powinna maleć w miarę nauki | `metrics.jsonl` |
| **early stopping** | przerwij naukę, jeśli od 8 epok nie ma poprawy | `early_stopping_patience` |
| **augmentacja** | sztuczne utrudnianie danych, żeby model nie uczył się na pamięć | `train/augment.py` |
| **mixup** | sklejanie dwóch przykładów w jeden | preset `full` |
| **SpecAugment** | „zaklejanie” fragmentów spektrogramu | preset `full` |
| **seed** | liczba-startowa losowości; ten sam seed = powtarzalny eksperyment | `--seed`, `config.seed` |
| **determinizm** | ten sam wynik przy powtórzeniu (na CPU tak; GPU bywa kapryśne) | `set_seed` |
| **device / GPU** | czy liczymy na procesorze (CPU), czy na karcie graficznej (CUDA) | `--device` |
| **smoke** | szybki przebieg na 2 nagraniach/gatunek — **nie jest wynikiem badania** | `--smoke` |

### Wyniki i eksperymenty

| Pojęcie | Najprościej | Gdzie w projekcie |
|---|---|---|
| **train / val / test** | nauka / podgląd / egzamin | `splits.json` |
| **wyciek danych (leakage)** | ten sam materiał w nauce i egzaminie = oszukiwanie samego siebie | `check_leakage.py` |
| **accuracy** | procent trafień (mylące przy nierównych klasach) | `metrics.json` |
| **macro-F1** | jakość uśredniona po gatunkach, każdy liczy się równo — **metryka główna** | `metrics.json` |
| **balanced accuracy** | średnia „wykrywalność” po gatunkach | `eval` |
| **top-1 / top-3** | czy prawda jest typem nr 1 / w pierwszej trójce | `metrics.json` |
| **agregacja mean / vote** | jak połączyć typy z okien w typ całego nagrania | `aggregate` |
| **pokrycie (coverage)** | ile nagrań zostaje po odrzuceniu niepewnych | `coverage.csv` |
| **macierz pomyłek** | tabela prawda × predykcja (kto z kim mylony) | `confusion_matrix.csv` |
| **pary mylące** | pary gatunków najczęściej mylone (np. pierwiosnek/piecuszek) | `error-analysis` |
| **ablacja** | zmieniamy jeden element i patrzymy, czy pomaga | `ablate` |
| **run referencyjny** | punkt odniesienia w danej grupie ablacji | `ablation_<grupa>.json` |
| **run** | jeden trening z pełnym zapisem (config, seed, model, metryki) | `runs/<run_id>/` |
| **`runs/index.csv`** | jedna linia na każdy run — źródło tabeli wyników | `runs/` |
| **`git_sha`** | który commit kodu wytworzył dany run | `index.csv`, `env.txt` |

### Narzędzia

| Pojęcie | Najprościej |
|---|---|
| **Python / venv / pip** | język programu / osobne środowisko / instalator bibliotek |
| **CLI** | komendy wpisywane w terminalu (`bird_recognizer ...`) |
| **Gradio** | biblioteka do prostego okienka/aplikacji webowej (demo) |
| **pytest** | uruchamia testy automatyczne |
| **ruff** | sprawdza styl i błędy w kodzie |
| **git / commit / branch / remote** | system wersji / zapis wersji / gałąź / zdalne repo |
| **torch / CUDA** | biblioteka sieci neuronowych / tryb GPU |

## 6. Jak rzeczy się łączą — konkretne przykłady

**Konfiguracja ↔ cache.** Cache zapisuje `config_hash` z parametrów spektrogramu i okna.
Jeśli zmienisz `window_s`, `n_mels` albo `hop_length`, stary cache jest nieaktualny
(`prep cache --verify` to pokaże) i trzeba go przebudować. To dlatego eksperymenty z oknem
5 s i 10 s mają osobne katalogi `data/window5`, `data/window10`.

**Lista gatunków ↔ modele.** Kolejność w `configs/species.yaml` to numery klas (0–11).
Zmiana kolejności „unieważnia” zapisane modele — dlatego lista jest zamrożona decyzją D1.

**Split ↔ trening ↔ test.** Podział jest zapisany raz (`splits.json`) i wszystkie runy korzystają
z tego samego. Gdyby ktoś zmienił split, wyniki przestałyby być porównywalne.

**Run ↔ config ↔ kod.** Każdy run zapisuje pełny snapshot konfiguracji, seed, urządzenie
i `git_sha`. Bez tego „wynik” nie istnieje — nie da się go odtworzyć.

**Ablacja ↔ runy.** W jednej grupie wszystkie runy różnią się od referencji **dokładnie jednym**
elementem (np. tylko pooling). Zapis w `runs/ablation_<grupa>.json` jest dowodem tej różnicy.

**Ewaluacja ↔ run.** `eval` odtwarza model **z konfiguracji runu** (nie z bieżących `configs/`),
dlatego run z oknem 5 s liczy się na cache `data/window5`. Wyniki trafiają do
`runs/<run>/eval_<split>/` i nie nadpisują metryk treningu.

**`experiments.md` ↔ `index.csv` ↔ `eval`.** Tabela jest generowana z `index.csv`, a sekcje par
mylących i zbioru własnego — z plików `eval_test`/`eval_own`. Dlatego nie edytuje się jej ręcznie.

**Demo ↔ run.** `demo --run <id>` bierze model z tego konkretnego runu i pokazuje jego top-3.

## 7. Gdzie jesteśmy (stan 2026-11-21)

| Etap | Co daje | Stan |
|---|---|---|
| 1 | decyzje, szkielet, fakty o API | ✅ |
| 2 | katalog 2917 nagrań, pobrane audio, podział | ✅ |
| 3 | cache spektrogramów (2897 nagrań) | ✅ |
| 4 | baseline MFCC+SVM i mały CNN | ✅ |
| 5 | transfer (ResNet18) i CRNN | ✅ |
| 6 | macierz ablacji + generator tabeli | ✅ (kod), pełne runy w toku |
| 7 | ewaluacja, agregacja, analiza błędów | ✅ |
| 8 | demo, raport, domknięcie | 🔄 demo gotowe, raport w toku |
| 9 | rozszerzenia (opcjonalne, np. aplikacja) | ⏸ decyzja D12 |

**Stałe liczby:** 2917 nagrań w katalogu, 2897 z cache (20 odrzuconych: 19 cisza, 1 błąd
dekodowania; **0 w teście**), 131 900 okien, 6,06 GB cache, 12 gatunków, 4 pary mylące.
Sprzęt: RTX 5070 12 GB, `torch 2.15.0+cu132`; pełna epoka CNN na całym zbiorze ≈ 130 s.

**Co się teraz dzieje:** w tle liczy się pełna macierz ablacji (5 grup, po 15 epok z early stopping).
To potrwa kilka godzin. Dopiero po niej `docs/experiments.md` będzie zawierać **prawdziwe wyniki**.

## 8. Co jest wynikiem, a co tylko testem (bardzo ważne)

- **`smoke`** = szybki przebieg na 2 nagraniach na gatunek, 1 epoka, służy sprawdzeniu, że kod
  działa. Wiersze ze `smoke` w `docs/experiments.md` **nie są wynikami badania**.
- **Prawdziwy wynik** = run na pełnym zbiorze, bez `smoke`, z policzonym `test macro-F1`
  i pełnym budżetem epok. Tylko takie wiersze wolno cytować.
- **Wnioski** w `docs/experiments.md` uzupełnia **człowiek** — agent dostarcza tabelę i liczby,
  nie wymyśla interpretacji.
- **Liczby w README/SPEC** mają daty — katalog xeno-canto rośnie, więc przed nowym etapem
  odświeża się dane.

## 9. Najczęstsze pytania

**Dlaczego audio nie ma w repo?** Licencje Creative Commons (większość zakazuje użycia
komercyjnego) i rozmiar (8,2 GB). Repo zawiera kod, konfigurację i dokumentację.

**Dlaczego jest 20 nagrań bez cache?** 19 to nagrania bardzo ciche — bramka ciszy odrzuciła
wszystkie okna; 1 plik był uszkodzony. Test nie stracił żadnego nagrania.

**Dlaczego jeden duplikat audio?** To samo nagranie wgrano dwa razy (ten sam sha256);
oba są w train, więc nie jest to wyciek danych.

**Czemu liczby różnią się między CPU a GPU?** GPU bywa niedeterministyczny mimo seeda.
Dlatego każdy run raportuje urządzenie, a powtarzalność obiecujemy tylko na CPU.

**Czemu hashe commitów się zmieniły?** Poprawiono tożsamość autora w historii git
(konto `DomDom2005`); `git_sha` w runach przemapowano na nowe commity — drzewa kodu są te same.

**Co to D1 / D12?** Decyzje w SPEC: D1 = lista 12 gatunków, D12 = decyzja o ewentualnych
rozszerzeniach (np. aplikacja, embeddingi BirdNET).

**Co jeszcze zostało do „ukończenia planu”?** Domknięcie etapu 8 (pełne wyniki, raport, README),
zbiór własny nagrań z telefonu (etap 7, sekcja `own`) i wnioski pisane przez człowieka.

## 10. Ściąga — najważniejsze komendy

```bash
# dane
bird_recognizer data catalog          # metadane -> data/catalog.jsonl
bird_recognizer data download         # audio -> data/raw (wznawialne)
bird_recognizer data split            # podział -> data/splits.json
python scripts/check_leakage.py --data-dir data
bird_recognizer prep cache            # okna + spektrogramy -> data/cache
bird_recognizer prep cache --verify   # kontrola cache

# trening
bird_recognizer train mfcc --smoke
bird_recognizer train cnn --smoke
bird_recognizer train crnn --pool attention --smoke
bird_recognizer train transfer --arch resnet18 --scratch --smoke

# ablacje i raport
bird_recognizer ablate --group rnn --smoke
bird_recognizer report --runs RUN1 RUN2 RUN3 --out docs/experiments.md

# ewaluacja
bird_recognizer eval --run RUN --split test
bird_recognizer aggregate --run RUN --method vote
bird_recognizer error-analysis --run RUN

# demo
bird_recognizer demo --run RUN
bird_recognizer demo --run RUN --smoke

# jakość kodu
ruff check . && pytest
```

Jeśli komenda `bird_recognizer` nie jest w PATH: `python -m bird_recognizer.cli <komenda>`.

## 11. Gdzie szukać szczegółów

| Pytanie | Plik |
|---|---|
| „Jak to ma dokładnie działać?” | `project info/SPEC.md` |
| „Jakie są etapy i bramki?” | `project info/SPEC.md` §15 |
| „Co zrobiono i gdzie się pomylono?” | `docs/ai-journal.md` |
| „Jakie fakty sprawdzono o API?” | `docs/api-notes.md` |
| „Jakie wyniki wyszły?” | `docs/experiments.md` |
| „Co muszę zrobić ja?” | `docs/manual-checklist.md` |
| „Jak agent ma pracować?” | `project info/AGENTS.md` |
