"""Run bookkeeping tests (SPEC §7.3, §11.2)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from bird_recognizer.train.runner import (
    INDEX_COLUMNS,
    Run,
    base_index_row,
    config_to_dict,
    git_sha,
    load_index,
)


def _config_with_runs(config, tmp_dir: Path):
    return replace(config, paths=replace(config.paths, runs_dir=tmp_dir / "runs"))


def test_create_run_writes_snapshot_and_environment(config, tmp_dir: Path) -> None:
    config = _config_with_runs(config, tmp_dir)
    run = Run.create(config, model="cnn", device="cpu")

    assert run.directory.is_dir()
    assert (run.directory / "config.yaml").is_file()
    assert (run.directory / "env.txt").is_file()
    assert "-cnn-s42" in run.run_id

    snapshot = yaml.safe_load((run.directory / "config.yaml").read_text(encoding="utf-8"))
    assert snapshot["audio"]["sample_rate"] == 32000
    assert snapshot["paths"]["runs_dir"] == str(tmp_dir / "runs")

    environment = (run.directory / "env.txt").read_text(encoding="utf-8")
    assert "git_sha:" in environment
    assert "torch:" in environment
    assert "device: cpu" in environment


def test_run_ids_are_unique(config, tmp_dir: Path) -> None:
    config = _config_with_runs(config, tmp_dir)
    first = Run.create(config, model="cnn")
    second = Run.create(config, model="cnn")
    assert first.run_id != second.run_id


def test_log_epoch_appends_jsonl(config, tmp_dir: Path) -> None:
    config = _config_with_runs(config, tmp_dir)
    run = Run.create(config, model="cnn")
    run.log_epoch({"epoch": 1, "val_macro_f1": 0.5})
    run.log_epoch({"epoch": 2, "val_macro_f1": 0.6})
    lines = (run.directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["val_macro_f1"] == 0.6


def test_save_metrics_writes_json(config, tmp_dir: Path) -> None:
    config = _config_with_runs(config, tmp_dir)
    run = Run.create(config, model="cnn")
    run.save_metrics({"val": {"recording_macro_f1": 0.42}, "paths": config.paths})
    payload = json.loads((run.directory / "metrics.json").read_text(encoding="utf-8"))
    assert payload["val"]["recording_macro_f1"] == 0.42
    assert isinstance(payload["paths"]["data_dir"], str)


def test_append_index_writes_header_and_rows(config, tmp_dir: Path) -> None:
    config = _config_with_runs(config, tmp_dir)
    first = Run.create(config, model="cnn")
    first.append_index(base_index_row(first, config, val_macro_f1=0.5))
    second = Run.create(config, model="mfcc")
    second.append_index(base_index_row(second, config, val_macro_f1=0.7))

    rows = load_index(tmp_dir / "runs")
    assert len(rows) == 2
    assert list(rows[0].keys()) == list(INDEX_COLUMNS)
    assert rows[0]["val_macro_f1"] == "0.5"
    assert rows[1]["run_id"] == second.run_id


def test_config_to_dict_is_yaml_friendly(config) -> None:
    snapshot = config_to_dict(config)
    assert isinstance(snapshot["paths"]["data_dir"], str)
    assert snapshot["species"][0]["latin"] == "Passer domesticus"
    yaml.safe_dump(snapshot, allow_unicode=True)


def test_git_sha_is_a_string() -> None:
    sha = git_sha()
    assert isinstance(sha, str) and sha
