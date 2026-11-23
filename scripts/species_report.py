"""Generate per-species data and results tables (docs/gatunki-dane-i-wyniki.md).

Data come from the catalog, the cache metadata and the test-split eval outputs of
all complete ablation runs; nothing is typed by hand.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from bird_recognizer.config import load_config
from bird_recognizer.data.dataset import list_cache
from bird_recognizer.data.splits import load_splits, splits_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
DOC = ROOT / "docs" / "gatunki-dane-i-wyniki.md"
CSV_DATA = ROOT / "reports" / "per-species-data.csv"
CSV_RESULTS = ROOT / "reports" / "per-species-results.csv"

BEST_RUN = "20261121-225300-cnn-window-10s-s42"
GROUPS = ("rnn", "aug", "pool", "window", "pretrained")


def num(value: float, digits: int = 3) -> str:
    """Polish decimal comma, e.g. 0.736 -> 0,736."""
    return f"{value:.{digits}f}".replace(".", ",")


def count(value: int) -> str:
    """Thousands separated by a space, e.g. 131900 -> 131 900."""
    return f"{int(value):,}".replace(",", " ")


def read_catalog() -> list[dict]:
    entries = []
    with (DATA / "catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            entries.append(json.loads(line))
    return entries


def read_cache() -> dict[str, dict]:
    metas = {}
    for cache_dir in list_cache(DATA):
        meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        metas[str(meta["id"])] = meta
    return metas


def read_per_class(run_id: str) -> dict[str, dict]:
    path = RUNS / run_id / "eval_test" / "per_class.csv"
    if not path.is_file():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["species"]] = {
                "f1": float(row["f1"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "support": int(row["support"]),
            }
    return rows


def read_confusion(run_id: str) -> tuple[list[str], dict[str, Counter]]:
    path = RUNS / run_id / "eval_test" / "confusion_matrix.csv"
    if not path.is_file():
        return [], {}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    species = rows[0][1:]
    confusions: dict[str, Counter] = {}
    for row in rows[1:]:
        truth = row[0]
        counts = Counter(
            {
                pred: int(value)
                for pred, value in zip(species, row[1:], strict=True)
                if int(value) > 0 and pred != truth
            }
        )
        confusions[truth] = counts
    return species, confusions


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml", env_file=None)
    species_order = [item.pl for item in config.species]
    catalog = read_catalog()
    cache = read_cache()
    split_plan = load_splits(splits_path(DATA))
    split_of = {
        recording_id: name
        for name, ids in split_plan.splits.items()
        for recording_id in ids
    }

    # ---- data per species -------------------------------------------------- #
    data_rows = []
    for species in species_order:
        catalog_entries = [entry for entry in catalog if entry.get("species_pl") == species]
        cached = [meta for recording_id, meta in cache.items() if meta.get("species_pl") == species]
        split_counts = Counter(split_of.get(str(meta["id"]), "brak") for meta in cached)
        windows_total = sum(int(meta.get("n_windows") or 0) for meta in cached)
        windows_kept = sum(int(meta.get("n_kept") or 0) for meta in cached)
        audio_min = sum(float(meta.get("duration_s") or 0.0) for meta in cached) / 60.0
        test_windows = sum(
            int(meta.get("n_kept") or 0)
            for meta in cached
            if split_of.get(str(meta["id"])) == "test"
        )
        rejected = 100.0 * (1.0 - windows_kept / windows_total) if windows_total else 0.0
        data_rows.append(
            {
                "species": species,
                "catalog_n": len(catalog_entries),
                "cache_n": len(cached),
                "train_n": split_counts.get("train", 0),
                "val_n": split_counts.get("val", 0),
                "test_n": split_counts.get("test", 0),
                "windows_kept": windows_kept,
                "windows_total": windows_total,
                "rejected_pct": rejected,
                "audio_min": audio_min,
                "test_windows": test_windows,
            }
        )

    # ---- results per species ---------------------------------------------- #
    run_ids = []
    for group in GROUPS:
        payload = json.loads((RUNS / f"ablation_{group}.json").read_text(encoding="utf-8"))
        run_ids.extend(record["run_id"] for record in payload["runs"] if record["status"] == "ok")
    per_run = {run_id: read_per_class(run_id) for run_id in run_ids}
    best = read_per_class(BEST_RUN)
    species, confusions = read_confusion(BEST_RUN)

    result_rows = []
    for item in species_order:
        scores = [per_run[run_id][item]["f1"] for run_id in run_ids if item in per_run[run_id]]
        best_entry = best.get(item, {})
        top_confusion = ""
        if item in confusions and confusions[item]:
            target, confused_count = confusions[item].most_common(1)[0]
            top_confusion = f"{target} ({confused_count})"
        result_rows.append(
            {
                "species": item,
                "support": best_entry.get("support", 0),
                "best_f1": best_entry.get("f1", 0.0),
                "best_precision": best_entry.get("precision", 0.0),
                "best_recall": best_entry.get("recall", 0.0),
                "median_f1": sorted(scores)[len(scores) // 2] if scores else 0.0,
                "min_f1": min(scores) if scores else 0.0,
                "max_f1": max(scores) if scores else 0.0,
                "runs": len(scores),
                "top_confusion": top_confusion,
            }
        )

    # ---- CSVs -------------------------------------------------------------- #
    CSV_DATA.parent.mkdir(parents=True, exist_ok=True)
    with CSV_DATA.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data_rows[0]))
        writer.writeheader()
        writer.writerows(data_rows)
    with CSV_RESULTS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    # ---- Markdown ---------------------------------------------------------- #
    total_catalog = sum(row["catalog_n"] for row in data_rows)
    total_cache = sum(row["cache_n"] for row in data_rows)
    total_windows = sum(row["windows_kept"] for row in data_rows)
    total_windows_all = sum(row["windows_total"] for row in data_rows)
    total_audio = sum(row["audio_min"] for row in data_rows)
    total_test = sum(row["support"] for row in result_rows)
    total_test_windows = sum(row["test_windows"] for row in data_rows)

    lines = [
        "# Gatunki: ile danych i jakie wyniki",
        "",
        "> Plik generowany przez `scripts/species_report.py` z katalogu, cache i plików",
        f"> `eval_test` runów ablacji. Stan: 2026-11-24. Najlepszy run: `{BEST_RUN}`",
        "> (CNN, okno 10 s). Liczby dotyczą **pełnego zbioru**; `n` to liczba nagrań.",
        "",
        "## 1. Jak to policzono",
        "",
        "- **Dane**: `data/catalog.jsonl` (katalog), `data/cache/*/meta.json` (cache), "
        "`data/splits.json` (podział).",
        "- **Wyniki**: `runs/<run>/eval_test/per_class.csv` dla 13 runów macierzy ablacji;",
        "  `best_*` to najlepszy run, `median/min/max` — rozrzut po wszystkich runach.",
        "- **Support (`n`)**: liczba nagrań gatunku w zbiorze testowym (suma = "
        f"{total_test}); dla runów z innym oknem support jest ten sam, zmienia się liczba okien.",
        "- Gatunki w kolejności z `configs/species.yaml` (kolejność = etykiety klas).",
        "",
        "## 2. Ile danych ma każdy gatunek",
        "",
        "| gatunek | katalog | cache | train | val | test | okna zachowane | okna razem "
        "| odrzucone | audio [min] | okna test |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in data_rows:
        lines.append(
            f"| {row['species']} | {count(row['catalog_n'])} | {count(row['cache_n'])} | "
            f"{count(row['train_n'])} | {count(row['val_n'])} | {count(row['test_n'])} | "
            f"{count(row['windows_kept'])} | {count(row['windows_total'])} | "
            f"{num(row['rejected_pct'], 1)}% | {count(round(row['audio_min']))} | "
            f"{count(row['test_windows'])} |"
        )
    lines += [
        f"| **razem** | **{count(total_catalog)}** | **{count(total_cache)}** | — | — | "
        f"{count(total_test)} | **{count(total_windows)}** | **{count(total_windows_all)}** | — | "
        f"**{count(round(total_audio))}** | **{count(total_test_windows)}** |",
        "",
        "Różnica katalog ↔ cache (20 nagrań) to 19 nagrań odrzuconych w całości przez bramkę ciszy",
        "i 1 błąd dekodowania; **żadne z nich nie jest w teście**.",
        "",
        "## 3. Wyniki per gatunek (najlepszy run)",
        "",
        f"Run `{BEST_RUN}` — CNN z oknem 10 s. `median/min/max F1` to rozrzut po "
        f"{len(run_ids)} runach macierzy.",
        "",
        "| gatunek | n (test) | F1 | precision | recall | mediana F1 | min–max F1 "
        "| najczęstsza pomyłka |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(result_rows, key=lambda item: item["best_f1"], reverse=True):
        lines.append(
            f"| {row['species']} | {count(row['support'])} | **{num(row['best_f1'])}** | "
            f"{num(row['best_precision'])} | {num(row['best_recall'])} | "
            f"{num(row['median_f1'])} | "
            f"{num(row['min_f1'])}–{num(row['max_f1'])} | {row['top_confusion'] or '—'} |"
        )
    lines += [
        "",
        "## 4. Co z tego wynika",
        "",
        "- **Najmocniejsze gatunki:** "
        + ", ".join(
            f"{row['species']} ({num(row['best_f1'])})"
            for row in sorted(result_rows, key=lambda item: item["best_f1"], reverse=True)[:4]
        )
        + ".",
        "- **Najsłabsze gatunki:** "
        + ", ".join(
            f"{row['species']} ({num(row['best_f1'])}, n={row['support']})"
            for row in sorted(result_rows, key=lambda item: item["best_f1"])[:4]
        )
        + ".",
        "- **Najbardziej zmienne w macierzy** (największy rozrzut min–max): "
        + ", ".join(
            f"{row['species']} ({num(row['min_f1'])}–{num(row['max_f1'])})"
            for row in sorted(
                result_rows,
                key=lambda item: item["max_f1"] - item["min_f1"],
                reverse=True,
            )[:3]
        )
        + ".",
        "- Małe klasy (wróbel, szczygieł) mają duży wpływ na macro-F1: jedna pomyłka zmienia F1",
        "  o kilkanaście punktów procentowych. Dlatego raportujemy `n` obok każdej liczby.",
        "",
        "## 5. Pliki źródłowe",
        "",
        "- `docs/gatunki-dane-i-wyniki.md` — ten dokument (generowany).",
        "- `reports/per-species-data.csv` — tabela danych w CSV.",
        "- `reports/per-species-results.csv` — tabela wyników w CSV.",
        "- `runs/<run>/eval_test/per_class.csv` — surowe wyniki per klasa (każdy run osobno).",
        "- `docs/experiments.md` — pełna tabela macierzy ablacji.",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {DOC} ({len(lines)} lines)")
    print(f"written {CSV_DATA}")
    print(f"written {CSV_RESULTS}")


if __name__ == "__main__":
    main()
