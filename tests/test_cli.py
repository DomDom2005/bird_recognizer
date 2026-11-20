"""CLI tests: help, exit codes, split on fixtures (SPEC §8)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from bird_recognizer.cli import main
from support import make_catalog_with_audio

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic_data"


def _argv(tmp_dir: Path, *rest: str) -> list[str]:
    return [
        "--env-file",
        str(tmp_dir / "nonexistent.env"),
        "--data-dir",
        str(tmp_dir / "data"),
        "--runs-dir",
        str(tmp_dir / "runs"),
        *rest,
    ]


def _seed_catalog(tmp_dir: Path) -> Path:
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE_DIR / "catalog.jsonl", data_dir / "catalog.jsonl")
    return data_dir


# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #
def test_top_level_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_data_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["data", "--help"])
    assert excinfo.value.code == 0


# --------------------------------------------------------------------------- #
# Exit codes (SPEC §8: 0 ok, 1 config/user, 2 api, 3 data)
# --------------------------------------------------------------------------- #
def test_unknown_species_exits_one(tmp_dir: Path) -> None:
    assert main(_argv(tmp_dir, "data", "catalog", "--species", "nie-ma")) == 1


def test_split_without_catalog_exits_three(tmp_dir: Path) -> None:
    assert main(_argv(tmp_dir, "data", "split")) == 3


def test_download_without_catalog_exits_three(tmp_dir: Path) -> None:
    assert main(_argv(tmp_dir, "data", "download")) == 3


# --------------------------------------------------------------------------- #
# Happy path on the synthetic fixture (no network)
# --------------------------------------------------------------------------- #
def test_split_writes_splits_file(tmp_dir: Path) -> None:
    data_dir = _seed_catalog(tmp_dir)
    assert main(_argv(tmp_dir, "data", "split")) == 0
    assert (data_dir / "splits.json").is_file()


def test_split_without_force_keeps_existing_file(tmp_dir: Path) -> None:
    data_dir = _seed_catalog(tmp_dir)
    assert main(_argv(tmp_dir, "data", "split")) == 0
    path = data_dir / "splits.json"
    before = path.read_text(encoding="utf-8")
    assert main(_argv(tmp_dir, "data", "split")) == 0
    assert path.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------- #
# prep cache (etap 3)
# --------------------------------------------------------------------------- #
def test_prep_cache_builds_and_verifies(tmp_dir: Path) -> None:
    data_dir = tmp_dir / "data"
    make_catalog_with_audio(data_dir)
    assert main(_argv(tmp_dir, "prep", "cache")) == 0
    assert (data_dir / "cache").is_dir()
    assert main(_argv(tmp_dir, "prep", "cache", "--verify")) == 0


def test_prep_cache_verify_without_cache_exits_three(tmp_dir: Path) -> None:
    assert main(_argv(tmp_dir, "prep", "cache", "--verify")) == 3


# --------------------------------------------------------------------------- #
# train (etap 4)
# --------------------------------------------------------------------------- #
def _seed_trainable(tmp_dir: Path) -> Path:
    data_dir = tmp_dir / "data"
    make_catalog_with_audio(data_dir, per_species=4)
    assert main(_argv(tmp_dir, "prep", "cache")) == 0
    assert main(_argv(tmp_dir, "data", "split")) == 0
    return data_dir


def test_train_mfcc_smoke_writes_run(tmp_dir: Path) -> None:
    data_dir = _seed_trainable(tmp_dir)
    assert main(_argv(tmp_dir, "--smoke", "train", "mfcc")) == 0
    assert (tmp_dir / "runs" / "index.csv").is_file()
    run_dirs = [path for path in (tmp_dir / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "metrics.json").is_file()
    assert (run_dirs[0] / "model.joblib").is_file()
    assert data_dir.is_dir()


def test_train_cnn_smoke_writes_checkpoint(tmp_dir: Path) -> None:
    _seed_trainable(tmp_dir)
    assert main(_argv(tmp_dir, "--smoke", "train", "cnn")) == 0
    run_dirs = [path for path in (tmp_dir / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "best.pt").is_file()
    assert (run_dirs[0] / "metrics.json").is_file()
    assert (run_dirs[0] / "metrics.jsonl").is_file()
    index = (tmp_dir / "runs" / "index.csv").read_text(encoding="utf-8")
    assert "cnn" in index


def test_train_without_splits_exits_three(tmp_dir: Path) -> None:
    data_dir = tmp_dir / "data"
    make_catalog_with_audio(data_dir, per_species=4)
    assert main(_argv(tmp_dir, "prep", "cache")) == 0
    assert main(_argv(tmp_dir, "--smoke", "train", "cnn")) == 3


def test_train_cnn_smoke_honours_aug_flag(tmp_dir: Path) -> None:
    from bird_recognizer.train.runner import load_index

    _seed_trainable(tmp_dir)
    assert main(_argv(tmp_dir, "--smoke", "train", "cnn", "--aug", "none")) == 0
    run_dir = _run_dirs(tmp_dir)[0]
    snapshot = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert snapshot["augment"]["preset"] == "none"
    rows = load_index(tmp_dir / "runs")
    assert rows[0]["aug"] == "none"


# --------------------------------------------------------------------------- #
# train (etap 5): CRNN i transfer
# --------------------------------------------------------------------------- #
def _run_dirs(tmp_dir: Path) -> list[Path]:
    return [path for path in (tmp_dir / "runs").iterdir() if path.is_dir()]


def test_train_crnn_smoke_writes_run_with_pool(tmp_dir: Path) -> None:
    _seed_trainable(tmp_dir)
    assert main(_argv(tmp_dir, "--smoke", "train", "crnn", "--pool", "attention")) == 0
    run_dirs = _run_dirs(tmp_dir)
    assert len(run_dirs) == 1
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model"] == "crnn"
    assert metrics["options"]["pool"] == "attention"
    assert (run_dirs[0] / "best.pt").is_file()
    assert "crnn" in (tmp_dir / "runs" / "index.csv").read_text(encoding="utf-8")


def test_train_transfer_scratch_smoke_writes_run(tmp_dir: Path) -> None:
    _seed_trainable(tmp_dir)
    argv = _argv(tmp_dir, "--smoke", "train", "transfer", "--arch", "resnet18", "--scratch")
    assert main(argv) == 0
    run_dirs = _run_dirs(tmp_dir)
    assert len(run_dirs) == 1
    assert "resnet18-scratch" in run_dirs[0].name
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["options"] == {"arch": "resnet18", "pretrained": False}
    assert "none" in (tmp_dir / "runs" / "index.csv").read_text(encoding="utf-8")


def test_transfer_scratch_and_pretrained_are_separate_runs(
    tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchvision.models import get_model

    from bird_recognizer.models import transfer as transfer_module

    monkeypatch.setattr(
        transfer_module,
        "_build_backbone",
        lambda arch, pretrained: get_model(arch, weights=None),
    )
    _seed_trainable(tmp_dir)
    assert main(_argv(tmp_dir, "--smoke", "train", "transfer", "--scratch")) == 0
    assert main(_argv(tmp_dir, "--smoke", "train", "transfer", "--pretrained")) == 0
    names = [path.name for path in _run_dirs(tmp_dir)]
    assert any("resnet18-scratch" in name for name in names)
    assert any("resnet18-pretrained" in name for name in names)
    learning_rates = {
        yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))["training"]["lr"]
        for path in _run_dirs(tmp_dir)
    }
    assert learning_rates == {0.001, 0.0003}
