# api-notes.md — xeno-canto API v3

> Zasada: każdy fakt ma źródło, datę sprawdzenia i status. Dokumentacja wygrywa z `SPEC.md`
> (SPEC §0 pkt 1). Ten plik jest uzupełniany przez `scripts/probe_api.py`.

## 1. Źródła i data sprawdzenia

- Dokumentacja API v3: <https://xeno-canto.org/explore/api> — **2026-11-02**.
- Wyszukiwarka tagów: <https://xeno-canto.org/help/search> (odsyłacz z dokumentacji).
- Weryfikacja live: `python scripts/probe_api.py --live` — **2026-11-02**, 7 sprawdzeń.

## 2. Fakty potwierdzone

| # | Ustalenie | Status | Źródło |
|---|---|---|---|
| 1 | Endpoint `https://xeno-canto.org/api/3/recordings`, GET, JSON | `ZWERYFIKOWANE` | dokumentacja |
| 2 | Cztery parametry: `query` (wymagany), `key` (wymagany), `per_page` (50–500, domy. 100), `page` (1..numPages) | `ZWERYFIKOWANE` | dokumentacja |
| 3 | `query` przyjmuje wyłącznie tagi; styl v2 zwraca 400. Tagi wielowyrazowe/operatory w cudzysłowie | `ZWERYFIKOWANE` | dokumentacja + live |
| 4 | Skala `q` to A–E; `q:">C"` = A i B | `ZWERYFIKOWANE` | dokumentacja + live (119 nagrań wróbla) |
| 5 | `cnt:poland` działa jako tag kraju | `ZWERYFIKOWANE` | live: `cnt:poland grp:birds` → 25 129 |
| 6 | Odpowiedź: `numRecordings`, `numSpecies`, `page`, `numPages`, `recordings[]` | `ZWERYFIKOWANE` | live |
| 7 | Pola nagrania zgodne z dokumentacją (`id`, `gen`, `sp`, `rec`, `cnt`, `loc`, `type`, `q`, `lic`, `file`, `length`, `also`, `rmk`, …) | `ZWERYFIKOWANE` | live |
| 8 | Gatunki chronione: `_meta.redacted_fields`, brak `file` | `ZWERYFIKOWANE` | dokumentacja |
| 9 | `length` to `m:ss` (łańcuch) — 50/50 pobranych wartości, np. `"2:16"`; brak `h:mm:ss` w próbce | `ZWERYFIKOWANE` | live |
| 10 | Pobranie `file` **nie wymaga** klucza: HEAD → 200, `Content-Type: audio/wav` | `ZWERYFIKOWANE` | live |
| 11 | `page` poza `1..numPages` → HTTP 400 (`client_error`), nie pusta lista | `ZWERYFIKOWANE` | live |
| 12 | „Strict rate limit” zniesiony; brak liczby w dokumentacji. Nasze tempo: 1 req/s | `ZWERYFIKOWANE` | dokumentacja |
| 13 | Klucz osobisty, osobny klucz aplikacji zalecany; publikacja grozi unieważnieniem | `ZWERYFIKOWANE` | dokumentacja |

## 3. Rozbieżności dokumentacji z żywym API (ważne)

### 3.1 Tag `status` jest odrzucany (2026-11-02)

Dokumentacja podaje tag `status` (`identified`, `unidentified`, `questioned`, `unconfirmed`)
jako dodany 2026-07-17. **Live API zwraca na niego HTTP 400**:

```
{"error": "client_error",
 "message": "Xeno-canto API v3 only accepts queries using tags. Visit https://xeno-canto.org/explore/api ..."}
```

Sprawdzone warianty (wszystkie 400): `status:identified`, `status:"identified"`,
`status:"=identified"`, także w połączeniu z `sp:` i `cnt:`. `id:identified` jest akceptowane,
ale zwraca 1 149 828 nagrań — to nie jest filtr identyfikacji, więc nie używamy.

**Decyzja:** tag wyłączony domyślnie (`api.status: ""` w `configs/default.yaml`, pusty = pomijany).
Klucz w configu zostaje, żeby włączyć go bez zmiany kodu, gdy API to naprawi.
Konsekwencja: nie mamy filtra jakości identyfikacji — szum etykiet trafia do danych i musi
być widoczny w analizie błędów (`also`, `rmk`).

### 3.2 Kształt błędu jest płaski, nie zagnieżdżony

Dokumentacja pokazuje `{"error": {"code": "...", "message": "..."}}`. Live zwraca:

```json
{ "error": "client_error", "message": "Xeno-canto API v3 only accepts queries using tags. ..." }
```

Kod błędów w `errors.ApiError.code` musi obsługiwać oba kształty (proste `error` jako łańcuch
albo obiekt z `code`).

### 3.3 Rozszerzenie pliku bywa inne niż `mp3`

`file` wskazywał plik `audio/wav`. Przy pobieraniu rozszerzenie bierz z `file-name`, a gdy
go brak — z `Content-Type` odpowiedzi; nie zakładaj `mp3`.

## 4. Liczebność gatunków z D1 (live, 2026-11-02)

Zapytanie: `sp:"<łacina>" cnt:poland q:">C"` (bez taga `status`, patrz 3.1).

| Gatunek | Nagrania |
|---|---|
| wróbel | 119 |
| mazurek | 112 |
| bogatka | 606 |
| modra | 182 |
| kos | 342 |
| kwiczoł | 275 |
| pierwiosnek | 319 |
| piecuszek | 164 |
| rudzik | 299 |
| zięba | 315 |
| szczygieł | 81 |
| dzwoniec | 101 |

Wszystkie powyżej progu 50 nagrań (SPEC §4.3 pkt 5) — lista D1 jest wykonalna bez luzowania
filtra. Po pobraniu i odrzuceniu ciszy liczby okien będą niższe; to zmierzy etap 3.

## 5. Konsekwencje dla kodu

- Zapytania buduje wyłącznie `xc/client.py` (etap 2) z tagów, nigdy jako „goły” tekst.
- `_meta.redacted_fields` → nagranie pomijane jako `redacted`, nie jako błąd pobierania.
- Klucz jedzie w query stringu → każdy log URL-a musi przechodzić przez redakcję (SPEC §13).
- Parsowanie błędów: oba kształty payloadu (3.2).
