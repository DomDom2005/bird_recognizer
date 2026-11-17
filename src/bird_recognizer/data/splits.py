"""Leak-proof train/val/test split (SPEC FR-3, §7.2, §15.7).

The split groups *whole recordings* and keeps every recordist (`rec`) in exactly
one split. It also checks file hashes, because xeno-canto sometimes hosts the same
file twice under different catalogue numbers.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import DataError
from ..xc.catalog import catalog_path, load_catalog

LOGGER = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
SPLITS_FILENAME = "splits.json"
MIN_TEST_PER_SPECIES = 3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def splits_path(data_dir: Path) -> Path:
    return data_dir / SPLITS_FILENAME


def group_key(entry: dict[str, Any]) -> str:
    """Recordist is the grouping key; fall back to the recording id."""
    recordist = str(entry.get("rec") or "").strip()
    return recordist if recordist else f"id:{entry['id']}"


# --------------------------------------------------------------------------- #
# Plan model
# --------------------------------------------------------------------------- #
@dataclass
class SplitPlan:
    """Result of split construction, serialisable to ``splits.json``."""

    splits: dict[str, list[str]]
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seed: int = 42
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "seed": self.seed,
            "created_at": self.created_at or _now(),
            "strategy": {"by": "recording", "group": "recordist", "stratify": "species"},
            "splits": {name: list(self.splits.get(name, [])) for name in SPLIT_NAMES},
            "counts": self.counts,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitPlan:
        splits = payload.get("splits")
        if not isinstance(splits, dict):
            raise DataError("splits.json: brak sekcji 'splits'")
        return cls(
            splits={name: [str(x) for x in splits.get(name, [])] for name in SPLIT_NAMES},
            counts=payload.get("counts") or {},
            warnings=list(payload.get("warnings") or []),
            seed=int(payload.get("seed", 42)),
            created_at=str(payload.get("created_at", "")),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_splits(path: Path, plan: SplitPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(
        json.dumps(plan.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_splits(path: Path) -> SplitPlan:
    if not path.exists():
        raise DataError(f"Brak pliku {path}. Uruchom `bird_recognizer data split`.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: niepoprawny JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataError(f"{path}: oczekiwano obiektu JSON")
    return SplitPlan.from_dict(payload)


# --------------------------------------------------------------------------- #
# Split construction
# --------------------------------------------------------------------------- #
def _group_sizes(entries: list[dict[str, Any]]) -> dict[str, int]:
    sizes: dict[str, int] = defaultdict(int)
    for entry in entries:
        sizes[group_key(entry)] += 1
    return dict(sizes)


def _group_species(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    species: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in entries:
        species[group_key(entry)][str(entry.get("species_pl", ""))] += 1
    return {key: dict(value) for key, value in species.items()}


def _split_species_counts(
    split_of: dict[str, str],
    group_species: dict[str, dict[str, int]],
    split: str,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for group, assigned in split_of.items():
        if assigned == split:
            for species, number in group_species[group].items():
                counts[species] += number
    return dict(counts)


def _can_move(
    group: str,
    split_of: dict[str, str],
    group_species: dict[str, dict[str, int]],
) -> bool:
    """Refuse a move that would empty the source split of any species."""
    source = split_of[group]
    source_counts = _split_species_counts(split_of, group_species, source)
    for species, number in group_species[group].items():
        if source_counts.get(species, 0) - number < 1:
            return False
    return True


def make_split(
    entries: list[dict[str, Any]],
    *,
    seed: int = 42,
    min_test_per_species: int = MIN_TEST_PER_SPECIES,
) -> SplitPlan:
    """Assign whole recordist groups to train/val/test, then repair coverage."""
    usable = [entry for entry in entries if not entry.get("redacted")]
    if not usable:
        raise DataError("Katalog nie zawiera nagrań nadających się do podziału")

    group_sizes = _group_sizes(usable)
    group_species = _group_species(usable)
    all_species = sorted({str(entry.get("species_pl", "")) for entry in usable})
    total = len(usable)

    rng = random.Random(seed)
    targets = {
        "train": total * TRAIN_RATIO,
        "val": total * VAL_RATIO,
        "test": total * (1.0 - TRAIN_RATIO - VAL_RATIO),
    }

    def _shuffled(groups: list[str]) -> list[str]:
        return sorted(groups, key=lambda group: (group_sizes[group], rng.random()))

    # Fill val/test from the *small* groups first: it maximises how many different
    # recordists land in the evaluation splits (xeno-canto has recordists with
    # hundreds of files, and a one-recordist test set would measure one session).
    # Both evaluation splits are filled together so neither is left with a handful
    # of huge groups; leftovers go to train, which needs the most minutes anyway.
    split_of: dict[str, str] = {}
    budgets = {name: targets[name] for name in ("val", "test")}
    remaining: list[str] = []
    for group in _shuffled(list(group_sizes)):
        size = group_sizes[group]
        candidates = [name for name in ("val", "test") if budgets[name] >= size]
        if not candidates:
            remaining.append(group)
            continue
        chosen = max(candidates, key=lambda name: budgets[name])
        split_of[group] = chosen
        budgets[chosen] -= size
    for group in remaining:
        split_of[group] = "train"

    # Repair: every species needs >=1 recording in val and >=min in test, while
    # train must keep >=1 recording of every species it had.
    required = {"val": 1, "test": max(1, min_test_per_species)}
    for _ in range(len(group_sizes) + 1):
        moved = False
        for split in ("test", "val"):
            counts = _split_species_counts(split_of, group_species, split)
            for species in all_species:
                if counts.get(species, 0) >= required[split]:
                    continue
                donors = [
                    group
                    for group, assigned_to in split_of.items()
                    if assigned_to == "train" and species in group_species[group]
                ]
                donors.sort(key=lambda group: (group_sizes[group], group))
                for donor in donors:
                    if not _can_move(donor, split_of, group_species):
                        continue
                    split_of[donor] = split
                    moved = True
                    counts = _split_species_counts(split_of, group_species, split)
                    break
        if not moved:
            break

    splits = {
        name: sorted(
            entry["id"] for entry in usable if split_of[group_key(entry)] == name
        )
        for name in SPLIT_NAMES
    }

    warnings: list[str] = []
    test_counts = _split_species_counts(split_of, group_species, "test")
    val_counts = _split_species_counts(split_of, group_species, "val")
    train_counts = _split_species_counts(split_of, group_species, "train")
    for species in all_species:
        if train_counts.get(species, 0) == 0:
            warnings.append(f"{species}: brak nagrań w train")
        if val_counts.get(species, 0) < 1:
            warnings.append(f"{species}: brak nagrań w val")
        if test_counts.get(species, 0) < min_test_per_species:
            warnings.append(
                f"{species}: test ma {test_counts.get(species, 0)} "
                f"< {min_test_per_species} nagrań"
            )
    for warning in warnings:
        LOGGER.warning(warning)

    counts: dict[str, dict[str, int]] = {}
    for name in SPLIT_NAMES:
        ids = set(splits[name])
        species_in_split = {
            str(entry.get("species_pl", "")) for entry in usable if entry["id"] in ids
        }
        groups_in_split = {group_key(entry) for entry in usable if entry["id"] in ids}
        counts[name] = {
            "recordings": len(ids),
            "species": len(species_in_split),
            "groups": len(groups_in_split),
        }

    plan = SplitPlan(
        splits=splits,
        counts=counts,
        warnings=warnings,
        seed=seed,
        created_at=_now(),
    )
    LOGGER.info(
        "Podział: train=%s, val=%s, test=%s (seed=%s)",
        counts["train"]["recordings"],
        counts["val"]["recordings"],
        counts["test"]["recordings"],
        seed,
    )
    return plan


def make_and_save(
    data_dir: Path,
    *,
    seed: int = 42,
    force: bool = False,
    min_test_per_species: int = MIN_TEST_PER_SPECIES,
) -> SplitPlan:
    """Build the split from ``catalog.jsonl``; keep an existing one without ``force``."""
    path = splits_path(data_dir)
    if path.exists() and not force:
        plan = load_splits(path)
        LOGGER.warning(
            "%s już istnieje (seed=%s) — zachowuję. Użyj --force, by przeliczyć.",
            path,
            plan.seed,
        )
        return plan
    entries = load_catalog(catalog_path(data_dir))
    if not entries:
        raise DataError(
            f"Brak katalogu w {data_dir}. Najpierw uruchom `bird_recognizer data catalog`."
        )
    plan = make_split(entries, seed=seed, min_test_per_species=min_test_per_species)
    save_splits(path, plan)
    LOGGER.info("Zapisano %s", path)
    return plan


# --------------------------------------------------------------------------- #
# Leakage check (used by scripts/check_leakage.py and tests)
# --------------------------------------------------------------------------- #
def check_leakage(
    entries: list[dict[str, Any]],
    plan: SplitPlan,
) -> list[str]:
    """Return human-readable violations of the three disjointness rules."""
    violations: list[str] = []
    catalog = {str(entry["id"]): entry for entry in entries}
    seen: dict[str, str] = {}

    for split in SPLIT_NAMES:
        for recording_id in plan.splits.get(split, []):
            if recording_id in seen:
                violations.append(
                    f"nagranie {recording_id} występuje w {seen[recording_id]} i {split}"
                )
            else:
                seen[recording_id] = split

    for recording_id, split in seen.items():
        if recording_id not in catalog:
            violations.append(f"nagranie {recording_id} ({split}) nie istnieje w katalogu")

    rec_split: dict[str, str] = {}
    sha_split: dict[str, str] = {}
    for recording_id, split in seen.items():
        entry = catalog.get(recording_id)
        if entry is None:
            continue
        recordist = group_key(entry)
        previous = rec_split.get(recordist)
        if previous is not None and previous != split:
            violations.append(
                f"nagrywający '{recordist}' występuje w {previous} i {split}"
            )
        else:
            rec_split[recordist] = split
        digest = str(entry.get("sha256_audio") or "")
        if digest:
            previous_sha = sha_split.get(digest)
            if previous_sha is not None and previous_sha != split:
                violations.append(
                    f"ten sam plik (sha256 {digest[:12]}…) jest w {previous_sha} i {split}"
                )
            else:
                sha_split[digest] = split

    extra = set(plan.splits) - set(SPLIT_NAMES)
    if extra:
        violations.append(f"nieznane sekcje podziału: {sorted(extra)}")
    return violations
