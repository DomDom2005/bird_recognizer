"""Fail loudly when the train/val/test split leaks (SPEC FR-3, §15.7).

Checks three disjointness rules: recording id, recordist (`rec`) and audio hash.
Exit codes: 0 — clean, 1 — missing/invalid input, 2 — leakage found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bird_recognizer.data.splits import (  # noqa: E402
    SPLIT_NAMES,
    check_leakage,
    load_splits,
    splits_path,
)
from bird_recognizer.errors import AppError  # noqa: E402
from bird_recognizer.logging_setup import configure_console  # noqa: E402
from bird_recognizer.xc.catalog import catalog_path, load_catalog  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_leakage",
        description="Sprawdza, czy podział train/val/test nie przecieka.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = build_parser().parse_args(argv)
    catalog_file = args.catalog or catalog_path(args.data_dir)
    splits_file = args.splits or splits_path(args.data_dir)

    try:
        entries = load_catalog(catalog_file)
        plan = load_splits(splits_file)
    except AppError as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 1

    print(f"Katalog: {len(entries)} wpisów ({catalog_file})")
    for name in SPLIT_NAMES:
        counts = plan.counts.get(name, {})
        print(f"  {name}: {len(plan.splits.get(name, []))} nagrań, counts={counts}")

    violations = check_leakage(entries, plan)
    if violations:
        print(f"\nWYCIEK DANYCH ({len(violations)}):", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 2
    print("\nOK: brak wycieku (id, nagrywający, sha256)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
