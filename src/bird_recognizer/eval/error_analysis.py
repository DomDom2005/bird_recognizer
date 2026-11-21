"""Error analysis: confusing pairs, per-species recall and worst recordings (FR-12).

Works on already aggregated recording-level predictions, so it never re-runs the
model. Metadata from ``data/catalog.jsonl`` (quality, type, ``also``, remarks) and from
the cache (kept/all windows) is attached to every worst recording — a misclassified
file with a non-empty ``also`` is a label-noise suspect, not necessarily a model error.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .metrics import confusion_pairs, per_class_metrics

RMK_LIMIT = 140


@dataclass(frozen=True)
class RecordingPrediction:
    """One recording after aggregation."""

    recording_id: str
    truth: int
    predicted: int
    confidence: float
    p_true: float


def cache_stats(cache_root: Path, recording_id: str) -> dict[str, object]:
    """How many windows a recording had and how many survived silence gating."""
    meta_path = Path(cache_root) / recording_id / "meta.json"
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    windows = int(meta.get("n_windows") or 0)
    kept = int(meta.get("n_kept") or 0)
    rejected = 1.0 - kept / windows if windows else 0.0
    return {
        "windows": kept,
        "windows_total": windows,
        "rejected_ratio": round(rejected, 4),
    }


def _also_text(entry: Mapping[str, object]) -> str:
    also = entry.get("also") or []
    if isinstance(also, str):
        return also
    if isinstance(also, Sequence):
        return ", ".join(str(item) for item in also)
    return ""


def _worst_recordings(
    predictions: Sequence[RecordingPrediction],
    *,
    names: Mapping[int, str],
    catalog: Mapping[str, Mapping[str, object]],
    cache_root: Path,
    worst: int,
) -> list[dict[str, object]]:
    ordered = sorted(predictions, key=lambda item: (item.p_true, item.confidence))
    rows: list[dict[str, object]] = []
    for item in ordered[: max(1, worst)]:
        entry = catalog.get(item.recording_id, {})
        stats = cache_stats(cache_root, item.recording_id)
        rows.append(
            {
                "recording_id": item.recording_id,
                "truth": names.get(item.truth, str(item.truth)),
                "predicted": names.get(item.predicted, str(item.predicted)),
                "confidence": round(item.confidence, 4),
                "p_true": round(item.p_true, 4),
                "loss": round(1.0 - item.p_true, 4),
                "q": entry.get("q", ""),
                "type": entry.get("type", ""),
                "also": _also_text(entry),
                "rmk": str(entry.get("rmk") or "")[:RMK_LIMIT],
                "windows": stats.get("windows", ""),
                "rejected_ratio": stats.get("rejected_ratio", ""),
                "rec": entry.get("rec", ""),
                "url": entry.get("url") or f"https://xeno-canto.org/{item.recording_id}",
            }
        )
    return rows


def analyse(
    predictions: Sequence[RecordingPrediction],
    *,
    names: Mapping[int, str],
    catalog: Mapping[str, Mapping[str, object]],
    cache_root: Path,
    labels: Sequence[int],
    top: int = 10,
    worst: int = 10,
) -> dict[str, object]:
    """Build the analysis payload used by both the Markdown and CSV exports."""
    truth = [item.truth for item in predictions]
    predicted = [item.predicted for item in predictions]
    pairs = confusion_pairs(truth, predicted, names=names, top=top)
    per_species = per_class_metrics(
        truth, predicted, labels=list(labels), names=names
    )
    worst_rows = _worst_recordings(
        predictions, names=names, catalog=catalog, cache_root=cache_root, worst=worst
    )
    noise_suspects = [row for row in worst_rows if row["also"]]
    return {
        "n": len(predictions),
        "pairs": pairs,
        "per_species": per_species,
        "worst": worst_rows,
        "label_noise_suspects": len(noise_suspects),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def render_markdown(analysis: Mapping[str, object]) -> str:
    pairs = analysis["pairs"]
    per_species = analysis["per_species"]
    worst = analysis["worst"]
    lines = [
        "# Analiza błędów",
        "",
        f"Nagrania: **{analysis['n']}**; podejrzani o szum etykiet (`also` ≠ puste): "
        f"**{analysis['label_noise_suspects']}**.",
        "",
        "## Najczęstsze pary (prawda → predykcja)",
        "",
    ]
    if pairs:
        lines.append("| prawda | predykcja | liczba |")
        lines.append("|---|---|---|")
        for row in pairs:
            lines.append(
                f"| {row['truth_name']} | {row['predicted_name']} | {row['count']} |"
            )
    else:
        lines.append("_brak pomyłek_")
    lines += [
        "",
        "## Recall i F1 per gatunek",
        "",
        "| gatunek | recall | F1 | n |",
        "|---|---|---|---|",
    ]
    for row in per_species:
        lines.append(
            f"| {row['species']} | {row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |"
        )
    lines += [
        "",
        "## Nagrania z największą stratą",
        "",
        "| id | prawda | predykcja | pewność | strata | q | type | also | okna | odrzucone |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in worst:
        lines.append(
            f"| {row['recording_id']} | {row['truth']} | {row['predicted']} | "
            f"{row['confidence']:.3f} | {row['loss']:.3f} | {row['q']} | {row['type']} | "
            f"{row['also'] or '—'} | {row['windows']} | {row['rejected_ratio']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_analysis(out_dir: Path, analysis: Mapping[str, object]) -> dict[str, Path]:
    """Write ``errors.md`` plus three CSV files; returns their paths."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": target / "errors.md",
        "pairs": _write_csv(
            target / "pairs.csv",
            analysis["pairs"],
            ("truth_name", "predicted_name", "count"),
        ),
        "per_species": _write_csv(
            target / "per_species.csv",
            analysis["per_species"],
            ("species", "recall", "precision", "f1", "support"),
        ),
        "worst": _write_csv(
            target / "worst_recordings.csv",
            analysis["worst"],
            (
                "recording_id",
                "truth",
                "predicted",
                "confidence",
                "p_true",
                "loss",
                "q",
                "type",
                "also",
                "rmk",
                "windows",
                "rejected_ratio",
                "rec",
                "url",
            ),
        ),
    }
    paths["markdown"].write_text(render_markdown(analysis), encoding="utf-8")
    return paths
