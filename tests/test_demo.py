"""Demo wiring tests: prediction path and CLI smoke (SPEC FR-13)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from bird_recognizer import demo
from bird_recognizer.cli import main
from bird_recognizer.config import load_config
from bird_recognizer.eval import evaluate as evaluate_module
from support import make_catalog_with_audio, write_wav

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _bundle(tmp_dir: Path, run_id: str):
    base = load_config(env_file=None)
    config = replace(
        base,
        paths=replace(
            base.paths, data_dir=tmp_dir / "data", runs_dir=tmp_dir / "runs"
        ),
    )
    return evaluate_module.load_run(
        config, run_id, device=torch.device("cpu"), runs_dir=tmp_dir / "runs"
    )


def test_predict_file_returns_sorted_top3(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    bundle = _bundle(tmp_dir, run_id)
    audio = tmp_dir / "ptak.wav"
    write_wav(audio, seconds=6.0, freq=900.0)
    top, starts, signal = demo.predict_file(bundle, audio)
    assert len(top) == 3
    # 6 s przy hop 1,5 s daje starty 0/1,5/3/4,5; ostatnie okno jest krótsze, ale ważne
    assert len(starts) == 4
    probabilities = [probability for _name, probability in top]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert signal.ndim == 1 and signal.size > 0


def test_predict_for_ui_reports_broken_audio(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    bundle = _bundle(tmp_dir, run_id)
    broken = tmp_dir / "zly.wav"
    broken.write_bytes(b"to nie jest audio")
    label, figure, message = demo.predict_for_ui(bundle, str(broken))
    assert label == {} and figure is None
    assert "BŁĄD" in message


def test_predict_for_ui_asks_for_a_file(tmp_dir: Path) -> None:
    run_id = _seed_run(tmp_dir)
    bundle = _bundle(tmp_dir, run_id)
    label, figure, message = demo.predict_for_ui(bundle, None)
    assert label == {} and figure is None
    assert "Wgraj plik" in message


def test_cli_demo_smoke_prints_top3(tmp_dir: Path, capsys) -> None:
    run_id = _seed_run(tmp_dir)
    assert main(_argv(tmp_dir, "demo", "--run", run_id, "--smoke")) == 0
    output = capsys.readouterr().out
    assert "Demo smoke" in output
    assert "plik syntetyczny" in output
    assert "UWAGA" in output


def test_bundled_demo_models_are_complete() -> None:
    roots = [
        REPO_ROOT / "artifacts" / "demo-model",
        REPO_ROOT / "artifacts" / "demo-model-crnn",
    ]
    scores = []
    for root in roots:
        for name in ("best.pt", "config.yaml", "metrics.json", "README.md"):
            assert (root / name).is_file(), f"{root.name}/{name}"
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["model"] in {"cnn", "crnn"}
        scores.append(metrics["test"]["recording_macro_f1"])
    assert all(0.68 <= score <= 0.77 for score in scores)


def test_cli_demo_uses_bundled_model_by_default(tmp_dir: Path, capsys) -> None:
    assert main(_argv(tmp_dir, "demo", "--smoke")) == 0
    output = capsys.readouterr().out
    assert "Demo smoke" in output
    assert "device=cpu" in output


def test_cli_demo_runs_bundled_crnn(tmp_dir: Path, capsys) -> None:
    crnn = REPO_ROOT / "artifacts" / "demo-model-crnn"
    assert main(_argv(tmp_dir, "demo", "--run", str(crnn), "--smoke")) == 0
    assert "Demo smoke" in capsys.readouterr().out
