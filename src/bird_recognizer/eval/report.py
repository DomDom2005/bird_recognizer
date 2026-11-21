"""Markdown result table generated from ``runs/index.csv`` (SPEC FR-14, §11.2-11.3).

The table is generated, never typed by hand: every number comes from a run directory
and its ``index.csv`` row. Section "Wnioski" is left for a human — the agent supplies
tables and numbers, not research conclusions.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..errors import DataError

GENERATED_NOTE = (
    "> **Plik generowany przez `bird_recognizer report` — nie edytuj tabeli ręcznie.** "
    "Sekcje 1–4 pochodzą z `runs/index.csv`; sekcję „Wnioski” uzupełnia człowiek."
)

MISSING_EVAL_HINT = (
    "_(brak eval_test/per_class.csv — uruchom "
    "`bird_recognizer eval --run <id> --split test`)_"
)

COLUMNS: tuple[tuple[str, str], ...] = (
    ("run_id", "run"),
    ("model", "model"),
    ("window_s", "okno [s]"),
    ("aug", "aug"),
    ("pretrained", "pretrained"),
    ("pool", "pooling"),
    ("n_train", "okna train"),
    ("n_val", "okna val"),
    ("n_test", "okna test"),
    ("val_macro_f1", "val macro-F1 (nagrania)"),
    ("test_macro_f1", "test macro-F1 (nagrania)"),
    ("test_top1", "test top-1"),
    ("test_top3", "test top-3"),
    ("delta", "Δ test vs ref"),
    ("seed", "seed"),
    ("device", "urządzenie"),
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return "n/d"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _delta(value: Any, reference: Any) -> str:
    if value in ("", None) or reference in ("", None):
        return "n/d"
    try:
        change = float(value) - float(reference)
    except (TypeError, ValueError):
        return "n/d"
    return f"{change:+.3f}"


def select_rows(
    rows: Sequence[dict[str, str]],
    run_ids: Sequence[str],
) -> list[dict[str, str]]:
    """Pick index rows in the requested order; every id must exist."""
    by_id = {row.get("run_id", ""): row for row in rows}
    missing = [run_id for run_id in run_ids if run_id not in by_id]
    if missing:
        raise DataError(
            f"brak runów w runs/index.csv: {missing}. Podaj run_id z `runs/index.csv`."
        )
    return [by_id[run_id] for run_id in run_ids]


def render_table(rows: Sequence[dict[str, str]], reference_id: str) -> str:
    """Markdown table of the selected runs, with a delta against the reference."""
    reference = next(
        (row for row in rows if row.get("run_id") == reference_id), rows[0] if rows else None
    )
    reference_f1 = reference.get("test_macro_f1", "") if reference else ""
    lines = [
        "| " + " | ".join(label for _key, label in COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for row in rows:
        cells: list[str] = []
        for key, _label in COLUMNS:
            if key == "delta":
                cells.append(_delta(row.get("test_macro_f1", ""), reference_f1))
            elif key in {"val_macro_f1", "test_macro_f1", "test_top1", "test_top3"}:
                cells.append(_fmt(row.get(key), 3))
            else:
                cells.append(str(row.get(key, "") or ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(
    rows: Sequence[dict[str, str]],
    *,
    reference_id: str | None = None,
    runs_dir: Path | None = None,
    species_pairs: dict[str, tuple[str, str]] | None = None,
    title: str = "Eksperymenty — bird_recognizer",
) -> str:
    """Full Markdown document: generated table, run list, placeholders for later stages."""
    if not rows:
        raise DataError("brak runów do raportu")
    reference_id = reference_id or rows[0]["run_id"]
    reference_row = next(
        (row for row in rows if row.get("run_id") == reference_id), rows[0]
    )
    run_list = "\n".join(
        f"- `{row['run_id']}` — seed {row.get('seed', '?')}, {row.get('device', '?')}, "
        f"git {row.get('git_sha', '?')[:8]}, uwagi: {row.get('notes') or '—'}"
        for row in rows
    )
    return "\n\n".join(
        [
            f"# {title}",
            GENERATED_NOTE,
            "## 1. Tabela główna\n\n" + render_table(rows, reference_id),
            "Legenda: `Δ test vs ref` to różnica test macro-F1 (na poziomie nagrania) względem "
            f"runu referencyjnego `{reference_row['run_id']}`. `n/d` = pole niedostępne "
            "(np. top-3 przy mniej niż 3 klasach). Wiersze z `smoke` w uwagach są przebiegami "
            "technicznymi na 2 nagraniach na gatunek — nie są wynikami badania; dotyczy to także "
            "sekcji 3–4, bo korzystają z tych samych runów (`eval_test`/`eval_own`).",
            "## 2. Runy i seedy\n\n" + run_list,
            "## 3. Pary mylące\n\n"
            + _pairs_section(rows, runs_dir, species_pairs or {}),
            "## 4. Zbiór własny\n\n" + _own_section(rows, runs_dir),
            "## 5. Wnioski\n\n_(uzupełnia człowiek — SPEC §11.3; agent nie wymyśla wniosków)_",
        ]
    ) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _pairs_section(
    rows: Sequence[dict[str, str]],
    runs_dir: Path | None,
    species_pairs: dict[str, tuple[str, str]],
) -> str:
    """F1 of both species of every D1 pair, from ``eval_test/per_class.csv``."""
    if runs_dir is None or not species_pairs:
        return MISSING_EVAL_HINT
    run_f1: dict[str, dict[str, float]] = {}
    for row in rows:
        per_class = _read_csv(Path(runs_dir) / row["run_id"] / "eval_test" / "per_class.csv")
        if not per_class:
            continue
        run_f1[row["run_id"]] = {
            item["species"]: float(item["f1"]) for item in per_class if item.get("f1")
        }
    if not run_f1:
        return MISSING_EVAL_HINT
    used_runs = [row["run_id"] for row in rows if row["run_id"] in run_f1]
    header = "| para | gatunek | " + " | ".join(used_runs) + " |"
    divider = "|---|---|" + "|".join("---" for _ in used_runs) + "|"
    lines = [header, divider]
    for pair, pair_species in species_pairs.items():
        for species in pair_species:
            cells = [
                _fmt(run_f1[run].get(species), 3) if species in run_f1[run] else "n/d"
                for run in used_runs
            ]
            lines.append(f"| {pair} | {species} | " + " | ".join(cells) + " |")
    missing = [row["run_id"] for row in rows if row["run_id"] not in run_f1]
    note = (
        f"\n\nBrak `eval_test` dla: {', '.join(f'`{item}`' for item in missing)}."
        if missing
        else ""
    )
    return "\n".join(lines) + note


def _own_section(rows: Sequence[dict[str, str]], runs_dir: Path | None) -> str:
    """Test vs own set, with ``own_delta`` (FR-10); own data is a human task."""
    if runs_dir is None:
        return "_(brak nagrań własnych — `data/own/<gatunek>/`; SPEC §16)_"
    lines = ["| run | test macro-F1 | own macro-F1 | own_delta |", "|---|---|---|---|"]
    found = False
    for row in rows:
        test_metrics = _read_json(Path(runs_dir) / row["run_id"] / "eval_test" / "metrics.json")
        own_metrics = _read_json(Path(runs_dir) / row["run_id"] / "eval_own" / "metrics.json")
        if not test_metrics or not own_metrics:
            continue
        found = True
        test_f1 = test_metrics["recording"]["macro_f1"]
        own_f1 = own_metrics["recording"]["macro_f1"]
        lines.append(
            f"| {row['run_id']} | {_fmt(test_f1, 3)} | {_fmt(own_f1, 3)} | "
            f"{_delta(own_f1, test_f1)} |"
        )
    if not found:
        return (
            "_(brak nagrań własnych w `data/own/<gatunek>/` albo brak `eval_own` — "
            "to zadanie człowieka, SPEC §16; nic nie podstawiamy w zamian)_"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    rows: Sequence[dict[str, str]],
    *,
    reference_id: str | None = None,
    runs_dir: Path | None = None,
    species_pairs: dict[str, tuple[str, str]] | None = None,
) -> Path:
    """Render the report and write it atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp"
    tmp.write_text(
        render_report(
            rows,
            reference_id=reference_id,
            runs_dir=runs_dir,
            species_pairs=species_pairs,
        ),
        encoding="utf-8",
    )
    tmp.replace(target)
    return target
