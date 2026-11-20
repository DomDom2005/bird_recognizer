"""Markdown result table generated from ``runs/index.csv`` (SPEC FR-14, §11.2-11.3).

The table is generated, never typed by hand: every number comes from a run directory
and its ``index.csv`` row. Section "Wnioski" is left for a human — the agent supplies
tables and numbers, not research conclusions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..errors import DataError

GENERATED_NOTE = (
    "> **Plik generowany przez `bird_recognizer report` — nie edytuj tabeli ręcznie.** "
    "Sekcje 1–4 pochodzą z `runs/index.csv`; sekcję „Wnioski” uzupełnia człowiek."
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
            "technicznymi na 2 nagraniach na gatunek — nie są wynikami badania.",
            "## 2. Runy i seedy\n\n" + run_list,
            "## 3. Pary mylące\n\n_(etap 7: tabela par z `per_class.csv`; tutaj tylko miejsce)_",
            "## 4. Zbiór własny\n\n_(etap 7: test vs own + `own_delta`)_",
            "## 5. Wnioski\n\n_(uzupełnia człowiek — SPEC §11.3; agent nie wymyśla wniosków)_",
        ]
    ) + "\n"


def write_report(
    path: Path,
    rows: Sequence[dict[str, str]],
    *,
    reference_id: str | None = None,
) -> Path:
    """Render the report and write it atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp"
    tmp.write_text(render_report(rows, reference_id=reference_id), encoding="utf-8")
    tmp.replace(target)
    return target
