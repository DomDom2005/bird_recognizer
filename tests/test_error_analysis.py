"""Error analysis: pairs, per-species recall, worst recordings (SPEC FR-12)."""

from __future__ import annotations

import json
from pathlib import Path

from bird_recognizer.eval.error_analysis import (
    RecordingPrediction,
    analyse,
    cache_stats,
    write_analysis,
)

NAMES = {0: "kos", 1: "kwiczoł", 2: "bogatka"}
LABELS = [0, 1, 2]


def _predictions() -> list[RecordingPrediction]:
    return [
        RecordingPrediction("r1", 0, 1, confidence=0.60, p_true=0.40),
        RecordingPrediction("r2", 1, 1, confidence=0.90, p_true=0.90),
        RecordingPrediction("r3", 2, 0, confidence=0.70, p_true=0.20),
    ]


def _catalog() -> dict[str, dict[str, object]]:
    return {
        "r1": {
            "q": "A",
            "type": "song",
            "also": ["Turdus merula"],
            "rmk": "x" * 300,
            "rec": "Anna",
            "url": "https://xeno-canto.org/r1",
        },
        "r2": {"q": "B", "type": "call", "also": [], "rmk": "", "rec": "Jan", "url": ""},
    }


def test_cache_stats_reads_kept_and_total_windows(tmp_dir: Path) -> None:
    record_dir = tmp_dir / "cache" / "r1"
    record_dir.mkdir(parents=True)
    (record_dir / "meta.json").write_text(
        json.dumps({"n_windows": 10, "n_kept": 8}), encoding="utf-8"
    )
    stats = cache_stats(tmp_dir / "cache", "r1")
    assert stats["windows"] == 8
    assert stats["windows_total"] == 10
    assert stats["rejected_ratio"] == 0.2


def test_cache_stats_missing_meta_is_empty(tmp_dir: Path) -> None:
    assert cache_stats(tmp_dir / "cache", "brak") == {}


def test_analyse_finds_pairs_recall_and_worst(tmp_dir: Path) -> None:
    (tmp_dir / "cache" / "r1").mkdir(parents=True)
    (tmp_dir / "cache" / "r1" / "meta.json").write_text(
        json.dumps({"n_windows": 10, "n_kept": 8}), encoding="utf-8"
    )
    analysis = analyse(
        _predictions(),
        names=NAMES,
        catalog=_catalog(),
        cache_root=tmp_dir / "cache",
        labels=LABELS,
    )
    assert analysis["n"] == 3
    pairs = {(row["truth_name"], row["predicted_name"]): row["count"] for row in analysis["pairs"]}
    assert pairs == {("kos", "kwiczoł"): 1, ("bogatka", "kos"): 1}
    recall = {row["species"]: row["recall"] for row in analysis["per_species"]}
    assert recall == {"kos": 0.0, "kwiczoł": 1.0, "bogatka": 0.0}
    worst_ids = [row["recording_id"] for row in analysis["worst"]]
    assert worst_ids == ["r3", "r1", "r2"], "najgorsze sortowane po 1 - p(prawda)"
    assert analysis["label_noise_suspects"] == 1  # tylko r1 ma 'also'
    first = analysis["worst"][1]
    assert first["also"] == "Turdus merula"
    assert first["windows"] == 8
    assert first["rejected_ratio"] == 0.2
    assert len(first["rmk"]) <= 140


def test_analyse_without_mistakes_has_empty_pairs() -> None:
    predictions = [RecordingPrediction("r1", 0, 0, 0.9, 0.9)]
    analysis = analyse(
        predictions,
        names=NAMES,
        catalog={},
        cache_root=Path("cache"),
        labels=LABELS,
    )
    assert analysis["pairs"] == []
    assert analysis["label_noise_suspects"] == 0


def test_write_analysis_creates_markdown_and_csv(tmp_dir: Path) -> None:
    analysis = analyse(
        _predictions(),
        names=NAMES,
        catalog=_catalog(),
        cache_root=tmp_dir / "cache",
        labels=LABELS,
    )
    paths = write_analysis(tmp_dir / "analysis", analysis)
    for key in ("markdown", "pairs", "per_species", "worst"):
        assert paths[key].is_file(), key
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "Najczęstsze pary" in text
    assert "kos" in text and "bogatka" in text
    assert "r3" in text
    header = paths["worst"].read_text(encoding="utf-8").splitlines()[0]
    assert "recording_id" in header and "rejected_ratio" in header
