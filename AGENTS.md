# AGENTS.md (root)

Pełne zasady pracy w tym repo: **`project info/AGENTS.md`** — przeczytaj go przed każdym zadaniem
(komendy, zasady, pułapki podziału danych, dziennik AI).

- Specyfikacja: `project info/SPEC.md`
- Plan etapów z bramkami: `project info/PLAN.md`
- Zweryfikowane fakty o xeno-canto API v3: `docs/api-notes.md` (powstaje w etapie 1)
- Dziennik AI: `docs/ai-journal.md` (powstaje w etapie 1)

Praca etapami: **status i bramki w `project info/SPEC.md` §15** (tabela 15.1, karty etapów 15.6–15.14).
Bez zielonej bramki etapu nie ma następnego etapu.

Skrót zasad, które obowiązują wszędzie w repo:

- Testy offline na mockach; żadnych prawdziwych żądań sieciowych ani pobierania audio.
- Klucz API tylko w `.env` poza repo; nie logować URL-i z `key=` i nie wklejać klucza do przykładów.
- Nagrania z xeno-canto nie trafiają do repo (licencje); commitujemy kod, metadane i wyniki.
- Podział danych tylko według nagrań i grup nagrywających; `check_leakage.py` przed treningiem.
- Dokumentacja https://xeno-canto.org/explore/api wygrywa ze SPEC.
- Kod i commity po angielsku; README, CLI, tabele wyników i raport po polsku.
