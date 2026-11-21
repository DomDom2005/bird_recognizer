"""Evaluation CLI: eval, aggregate and error-analysis on a real tiny run (FR-10/12)."""

from __future__ import annotations

import json
from pathlib import Path

from bird_recognizer.cli import main
from support import make_catalog_with_audio, write_wav


def _argv(tmp_dir: Path, *rest: str) -> list[str]:
    return [
        "--env-file",
        str(tmp_dir / "nonexistent.env"),
        "--data-dir",
        str(tmp_dir / "data"),
        "--runs-dir",
        str(tmp_dir / "runs"),
        "--device",
        "cpu",
        *rest,
    ]


def _seed_run(tmp_dir: Path) -> str:
    data_dir = tmp_dir / "data"
    make_catalog_with_audio(data_dir, per_species=4)
    assert main(_argv(tmp_dir, "prep", "cache")) == 0
    assert main(_argv(tmp_dir, "data", "split")) == 0
    assert main(_argv(tmp_dir, "--smoke", "train", "cnn")) == 0
    return next(path.name for path in (tmp_dir / "runs").iterdir() if path.is_dir())


def test_eval_writes_metrics_confusion_and_predictions(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    assert main(_argv(tmp_dir, "eval", "--run", run_id, "--split", "test")) == 0
    out_dir = tmp_dir / "runs" / run_id / "eval_test"
    for name in (
        "metrics.json",
        "per_class.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "predictions.csv",
    ):
        assert (out_dir / name).is_file(), name
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    assert payload["n_recordings"] > 0
    assert payload["n_windows"] >= payload["n_recordings"]
    assert 0.0 <= payload["recording"]["macro_f1"] <= 1.0
    assert 0.0 <= payload["vote"]["macro_f1"] <= 1.0
    assert payload["recording"]["aggregation"] == "mean"
    assert payload["coverage"][0]["coverage"] == 1.0


def test_aggregate_writes_comparison_and_coverage(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    assert main(_argv(tmp_dir, "aggregate", "--run", run_id, "--method", "vote")) == 0
    out_dir = tmp_dir / "runs" / run_id / "eval_test"
    aggregation = (out_dir / "aggregation.csv").read_text(encoding="utf-8")
    coverage = (out_dir / "coverage.csv").read_text(encoding="utf-8")
    assert "mean,vote" not in aggregation  # one row per method, not a joined string
    assert "mean" in aggregation and "vote" in aggregation
    assert "threshold" in coverage


def test_error_analysis_writes_reports(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    assert main(_argv(tmp_dir, "error-analysis", "--run", run_id)) == 0
    out_dir = tmp_dir / "runs" / run_id / "analysis"
    for name in ("errors.md", "pairs.csv", "per_species.csv", "worst_recordings.csv"):
        assert (out_dir / name).is_file(), name
    markdown = (out_dir / "errors.md").read_text(encoding="utf-8")
    assert "Najczęstsze pary" in markdown
    assert "Recall i F1 per gatunek" in markdown
    assert "Nagrania z największą stratą" in markdown


def test_eval_own_split_uses_own_recordings(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    own_dir = tmp_dir / "data" / "own" / "kos"
    own_dir.mkdir(parents=True)
    write_wav(own_dir / "ptak.wav", seconds=6.0, freq=800.0)
    assert main(_argv(tmp_dir, "eval", "--run", run_id, "--split", "own")) == 0
    metrics = json.loads(
        (tmp_dir / "runs" / run_id / "eval_own" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["split"] == "own"
    assert metrics["n_recordings"] == 1
    assert metrics["window"]["n"] > 0


def test_eval_own_without_data_exits_three(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    assert main(_argv(tmp_dir, "eval", "--run", run_id, "--split", "own")) == 3


def test_report_includes_pairs_and_own_sections(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    assert main(_argv(tmp_dir, "eval", "--run", run_id, "--split", "test")) == 0
    own_dir = tmp_dir / "data" / "own" / "kos"
    own_dir.mkdir(parents=True)
    write_wav(own_dir / "ptak.wav", seconds=6.0, freq=800.0)
    assert main(_argv(tmp_dir, "eval", "--run", run_id, "--split", "own")) == 0
    out = tmp_dir / "docs" / "experiments.md"
    assert main(_argv(tmp_dir, "report", "--runs", run_id, "--out", str(out))) == 0
    text = out.read_text(encoding="utf-8")
    assert "kos-kwiczol" in text, "sekcja par mylących ma F1 per gatunek z eval_test"
    assert "own_delta" in text


def test_eval_unknown_run_exits_three(tmp_dir: Path) -> None:
    _seed_run(tmp_dir)
    assert main(_argv(tmp_dir, "eval", "--run", "nie-ma-takiego-runu")) == 3


def test_eval_rejects_mfcc_run(tmp_dir: Path) -> None:
    _seed_run(tmp_dir)
    assert main(_argv(tmp_dir, "--smoke", "train", "mfcc")) == 0
    mfcc_run = next(
        path.name
        for path in (tmp_dir / "runs").iterdir()
        if path.is_dir() and path.name.endswith("-mfcc-svm-s42")
    )
    assert main(_argv(tmp_dir, "eval", "--run", mfcc_run)) == 4


def test_eval_without_checkpoint_exits_four(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    (tmp_dir / "runs" / run_id / "best.pt").unlink()
    assert main(_argv(tmp_dir, "eval", "--run", run_id)) == 4


def test_own_split_skips_unknown_species_and_broken_audio(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    own_dir = tmp_dir / "data" / "own"
    (own_dir / "kos").mkdir(parents=True)
    write_wav(own_dir / "kos" / "dobry.wav", seconds=6.0, freq=800.0)
    (own_dir / "kos" / "zly.wav").write_bytes(b"to nie jest audio")
    (own_dir / "nieznany").mkdir()
    write_wav(own_dir / "nieznany" / "x.wav", seconds=6.0)
    assert main(_argv(tmp_dir, "eval", "--run", run_id, "--split", "own")) == 0
    metrics = json.loads(
        (tmp_dir / "runs" / run_id / "eval_own" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["n_recordings"] == 1


def test_report_marks_missing_eval_sections(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    out = tmp_dir / "docs" / "experiments.md"
    assert main(_argv(tmp_dir, "report", "--runs", run_id, "--out", str(out))) == 0
    text = out.read_text(encoding="utf-8")
    assert "per_class.csv" in text, "brak eval_test musi być jawnie zaznaczony"
    assert "brak nagrań własnych" in text
    assert "eval_own" in text
