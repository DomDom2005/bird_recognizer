"""Command line interface (SPEC §8)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, Config, get_api_key, load_config
from .data.splits import make_and_save
from .errors import AppError
from .logging_setup import configure_console, setup_logging
from .xc.catalog import build_catalog, catalog_path
from .xc.client import XCClient
from .xc.download import download_audio

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree: global options + ``data`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="bird_recognizer",
        description="Klasyfikacja gatunku ptaka z nagrania: dane, trening, ewaluacja.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="katalog nagrań, pobieranie audio, podział")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    catalog = data_sub.add_parser("catalog", help="pobierz metadane z xeno-canto")
    catalog.add_argument("--species", nargs="*", default=None, help="nazwy PL lub łacińskie")
    catalog.add_argument("--refresh", action="store_true", help="odbuduj katalog od zera")

    download = data_sub.add_parser("download", help="pobierz pliki audio")
    download.add_argument("--species", nargs="*", default=None)
    download.add_argument("--limit", type=int, default=None)
    download.add_argument("--force", action="store_true")
    download.add_argument("--retry-failed", action="store_true")

    split = data_sub.add_parser("split", help="podział train/val/test")
    split.add_argument("--force", action="store_true", help="nadpisz istniejący podział")

    return parser


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    updates: dict[str, object] = {}
    if args.data_dir is not None:
        updates["paths"] = replace(config.paths, data_dir=args.data_dir)
    if args.seed is not None:
        updates["seed"] = args.seed
    return replace(config, **updates) if updates else config


def _cmd_catalog(config: Config, args: argparse.Namespace) -> int:
    client = XCClient(config)
    result = build_catalog(
        config,
        client,
        data_dir=config.paths.data_dir,
        species_filter=args.species,
        refresh=args.refresh,
    )
    print(f"Katalog: {len(result.entries)} wpisów -> {catalog_path(config.paths.data_dir)}")
    for species, info in result.queries.items():
        print(
            f"  {species}: {info['num_recordings']} nagrań w API, "
            f"zapisano {info['saved']} (stron: {info['pages']})"
        )
    for warning in result.warnings:
        print(f"UWAGA: {warning}")
    return 0


def _cmd_download(config: Config, args: argparse.Namespace) -> int:
    client = XCClient(config)
    report = download_audio(
        config,
        client,
        data_dir=config.paths.data_dir,
        species_filter=args.species,
        limit=args.limit,
        force=args.force,
        retry_failed=args.retry_failed,
    )
    print(
        "Pobieranie: "
        f"ok={len(report.succeeded)}, pominięte={len(report.skipped)}, "
        f"zredagowane={len(report.redacted)}, błędy={len(report.errors)}, "
        f"rozmiar={report.total_bytes / 1024 / 1024:.1f} MB"
    )
    if report.duplicates:
        print(f"UWAGA: {len(report.duplicates)} duplikatów plików (ten sam sha256)")
    if report.stopped:
        print(f"UWAGA: przerwano ({report.stopped}) — uruchom ponownie później")
        return 2
    return 0


def _cmd_split(config: Config, args: argparse.Namespace) -> int:
    plan = make_and_save(config.paths.data_dir, seed=config.seed, force=args.force)
    for name in ("train", "val", "test"):
        counts = plan.counts.get(name, {})
        print(
            f"{name}: {counts.get('recordings', 0)} nagrań, "
            f"{counts.get('species', 0)} gatunków, {counts.get('groups', 0)} nagrywających"
        )
    for warning in plan.warnings:
        print(f"UWAGA: {warning}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; maps :class:`AppError` to the exit codes from SPEC §8."""
    configure_console()
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, env_file=args.env_file)
        config = _apply_overrides(config, args)
        key = get_api_key()
        setup_logging("DEBUG" if args.verbose else "INFO", secrets=[key] if key else [])
        if args.command == "data":
            if args.data_command == "catalog":
                return _cmd_catalog(config, args)
            if args.data_command == "download":
                return _cmd_download(config, args)
            if args.data_command == "split":
                return _cmd_split(config, args)
        LOGGER.error("Nieznana komenda: %s", args.command)
        return 1
    except AppError as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("Przerwano.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
