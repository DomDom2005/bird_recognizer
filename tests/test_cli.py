"""CLI tests: help, exit codes, split on fixtures (SPEC §8)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from bird_recognizer.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic_data"


def _argv(tmp_dir: Path, *rest: str) -> list[str]:
    return [
        "--env-file",
        str(tmp_dir / "nonexistent.env"),
        "--data-dir",
        str(tmp_dir / "data"),
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
