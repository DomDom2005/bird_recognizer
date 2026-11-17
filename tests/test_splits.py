"""Split tests: grouping, determinism, coverage and leakage (SPEC FR-3)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from bird_recognizer.data.splits import (
    SPLIT_NAMES,
    SplitPlan,
    check_leakage,
    load_splits,
    make_and_save,
    make_split,
    save_splits,
    splits_path,
)
from bird_recognizer.errors import DataError
from bird_recognizer.xc.catalog import load_catalog
from support import entry, write_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic_data"


def _entries(species=("kos", "bogatka"), recordists: int = 6, per_group: int = 2):
    items: list[dict] = []
    counter = 0
    for species_name in species:
        for number in range(recordists):
            for _ in range(per_group):
                counter += 1
                items.append(
                    entry(
                        f"XC{counter:04d}",
                        species_pl=species_name,
                        recordist=f"{species_name}-rec-{number}",
                    )
                )
    return items


def _script_module():
    path = REPO_ROOT / "scripts" / "check_leakage.py"
    spec = importlib.util.spec_from_file_location("check_leakage_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_split_partitions_every_recording_once() -> None:
    items = _entries()
    plan = make_split(items)
    assigned = [recording_id for name in SPLIT_NAMES for recording_id in plan.splits[name]]
    assert len(assigned) == len(set(assigned)) == len(items)
    assert set(assigned) == {item["id"] for item in items}


def test_same_seed_gives_identical_split() -> None:
    items = _entries()
    first = make_split(items, seed=42)
    second = make_split(items, seed=42)
    assert first.splits == second.splits
    assert first.counts == second.counts


def test_train_keeps_the_majority_share() -> None:
    """Regression: an abs-distance greedy starved train on real, uneven data."""
    items = _entries(recordists=8, per_group=2)
    plan = make_split(items, seed=42)
    train_share = len(plan.splits["train"]) / len(items)
    assert 0.55 <= train_share <= 0.85, plan.counts


def test_uneven_recordist_sizes_do_not_starve_train() -> None:
    items: list[dict] = []
    counter = 0
    # One prolific recordist (40 recordings) plus many small ones.
    for _ in range(40):
        counter += 1
        items.append(entry(f"XC{counter:04d}", species_pl="kos", recordist="prolific"))
    for number in range(10):
        for _ in range(2):
            counter += 1
            items.append(
                entry(f"XC{counter:04d}", species_pl="bogatka", recordist=f"small-{number}")
            )
    plan = make_split(items, seed=42)
    assert len(plan.splits["train"]) >= len(items) * 0.5, plan.counts


def test_every_species_reaches_minimum_test_size() -> None:
    items = _entries(recordists=8, per_group=2)
    plan = make_split(items, min_test_per_species=3)
    test_ids = set(plan.splits["test"])
    test_species = {item["species_pl"] for item in items if item["id"] in test_ids}
    assert test_species == {"kos", "bogatka"}
    for species_name in test_species:
        count = sum(
            1
            for item in items
            if item["id"] in test_ids and item["species_pl"] == species_name
        )
        assert count >= 3


def test_single_group_species_produces_warning() -> None:
    items = _entries(recordists=4, per_group=2)
    items.append(entry("XC9999", species_pl="rzadki", recordist="samotny"))
    plan = make_split(items, min_test_per_species=3)
    assert any("rzadki" in warning for warning in plan.warnings)


def test_redacted_entries_are_excluded() -> None:
    items = _entries(recordists=4)
    items.append(entry("XC9998", species_pl="kos", recordist="red", redacted=True))
    plan = make_split(items)
    assigned = {recording_id for name in SPLIT_NAMES for recording_id in plan.splits[name]}
    assert "XC9998" not in assigned


def test_empty_catalog_is_an_error() -> None:
    with pytest.raises(DataError):
        make_split([])


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_make_and_save_keeps_existing_without_force(tmp_dir: Path) -> None:
    data_dir = tmp_dir / "data"
    write_catalog(data_dir / "catalog.jsonl", _entries(recordists=4))
    first = make_and_save(data_dir, seed=42)
    assert first.seed == 42
    empty = SplitPlan(splits={"train": [], "val": [], "test": []}, seed=7)
    save_splits(splits_path(data_dir), empty)

    kept = make_and_save(data_dir, seed=42)
    assert kept.seed == 7
    forced = make_and_save(data_dir, seed=42, force=True)
    assert forced.seed == 42


def test_split_file_roundtrip(tmp_dir: Path) -> None:
    plan = SplitPlan(splits={"train": ["a"], "val": ["b"], "test": ["c"]}, seed=3)
    path = tmp_dir / "splits.json"
    save_splits(path, plan)
    loaded = load_splits(path)
    assert loaded.splits == plan.splits
    assert loaded.seed == 3


def test_load_splits_missing_or_broken(tmp_dir: Path) -> None:
    with pytest.raises(DataError, match="Brak pliku"):
        load_splits(tmp_dir / "nope.json")
    broken = tmp_dir / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(DataError, match="JSON"):
        load_splits(broken)


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_clean_plan_has_no_violations() -> None:
    items = _entries(recordists=4)
    plan = make_split(items, seed=42)
    assert check_leakage(items, plan) == []


def test_duplicate_recording_is_detected() -> None:
    items = [entry("XC1"), entry("XC2")]
    plan = SplitPlan(splits={"train": ["XC1"], "val": [], "test": ["XC1"]})
    violations = check_leakage(items, plan)
    assert any("XC1" in violation for violation in violations)


def test_shared_recordist_is_detected() -> None:
    items = [entry("XC1", recordist="Anna"), entry("XC2", recordist="Anna")]
    plan = SplitPlan(splits={"train": ["XC1"], "val": [], "test": ["XC2"]})
    assert any("Anna" in violation for violation in check_leakage(items, plan))


def test_shared_hash_is_detected() -> None:
    items = [
        entry("XC1", recordist="A", sha256_audio="same"),
        entry("XC2", recordist="B", sha256_audio="same"),
    ]
    plan = SplitPlan(splits={"train": ["XC1"], "val": [], "test": ["XC2"]})
    assert any("sha256" in violation for violation in check_leakage(items, plan))


def test_unknown_recording_is_detected() -> None:
    plan = SplitPlan(splits={"train": ["XC404"], "val": [], "test": []})
    assert any("nie istnieje" in violation for violation in check_leakage([entry("XC1")], plan))


# --------------------------------------------------------------------------- #
# Fixture and script
# --------------------------------------------------------------------------- #
def test_synthetic_fixture_is_clean() -> None:
    entries = load_catalog(FIXTURE_DIR / "catalog.jsonl")
    plan = load_splits(FIXTURE_DIR / "splits.json")
    assert check_leakage(entries, plan) == []


def test_script_returns_zero_for_clean_fixture() -> None:
    assert _script_module().main(["--data-dir", str(FIXTURE_DIR)]) == 0


def test_script_returns_two_for_leaking_split(tmp_dir: Path) -> None:
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_DIR / "catalog.jsonl", data_dir / "catalog.jsonl")
    payload = json.loads((FIXTURE_DIR / "splits.json").read_text(encoding="utf-8"))
    payload["splits"]["train"].append("XC2005")  # XC2005 jest już w teście
    (data_dir / "splits.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _script_module().main(["--data-dir", str(data_dir)]) == 2


def test_script_returns_one_for_missing_input(tmp_dir: Path) -> None:
    assert _script_module().main(["--data-dir", str(tmp_dir / "brak")]) == 1
