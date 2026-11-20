"""Run bookkeeping: run_id, config snapshot, environment and ``runs/index.csv``.

Every training command creates one directory under ``runs/`` with everything needed
to reproduce the number later (SPEC §7.3, §11.2).
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from ..config import Config
from ..errors import ModelError

LOGGER = logging.getLogger(__name__)

INDEX_COLUMNS = (
    "run_id",
    "created_at",
    "model",
    "pretrained",
    "pool",
    "window_s",
    "aug",
    "seed",
    "split_version",
    "n_train",
    "n_val",
    "n_test",
    "val_macro_f1",
    "test_macro_f1",
    "test_top1",
    "test_top3",
    "own_macro_f1",
    "own_delta",
    "duration_s",
    "device",
    "git_sha",
    "notes",
)


def git_sha() -> str:
    """Current commit, or ``unknown`` outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def environment_info(device: str) -> dict[str, Any]:
    import numpy
    import torch

    return {
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "sklearn": _version("scikit-learn"),
        "platform": platform.platform(),
        "device": device,
        "torch_threads": torch.get_num_threads(),
    }


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _stringify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(item) for item in value]
    if is_dataclass(value):
        return _stringify(asdict(value))
    return value


def config_to_dict(config: Config) -> dict[str, Any]:
    """JSON/YAML-safe snapshot of the whole configuration."""
    return _stringify(asdict(config))


class Run:
    """One training/evaluation run on disk."""

    def __init__(
        self,
        run_id: str,
        directory: Path,
        model: str,
        seed: int,
        device: str,
    ) -> None:
        self.run_id = run_id
        self.directory = directory
        self.model = model
        self.seed = seed
        self.device = device

    @classmethod
    def create(
        cls,
        config: Config,
        *,
        model: str,
        tag: str = "",
        seed: int | None = None,
        device: str = "cpu",
    ) -> Run:
        effective_seed = config.seed if seed is None else seed
        runs_dir = Path(config.paths.runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        suffix = f"-{tag}" if tag else ""
        base = f"{stamp}-{model}{suffix}-s{effective_seed}"
        run_id = base
        counter = 1
        while (runs_dir / run_id).exists():
            counter += 1
            run_id = f"{base}-{counter}"
        directory = runs_dir / run_id
        directory.mkdir(parents=True)
        run = cls(run_id, directory, model, effective_seed, device)
        run._write_config(config)
        run._write_environment()
        return run

    def path(self, name: str) -> Path:
        return self.directory / name

    def _write_config(self, config: Config) -> None:
        snapshot = config_to_dict(config)
        self.path("config.yaml").write_text(
            yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )

    def _write_environment(self) -> None:
        info = environment_info(self.device)
        lines = [f"{key}: {value}" for key, value in sorted(info.items())]
        self.path("env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def log_epoch(self, record: dict[str, Any]) -> None:
        with self.path("metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def save_metrics(self, payload: dict[str, Any]) -> None:
        self.path("metrics.json").write_text(
            json.dumps(_stringify(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def append_index(self, row: dict[str, Any]) -> Path:
        index_path = self.directory.parent / "index.csv"
        new_file = not index_path.exists()
        values = [row.get(column, "") for column in INDEX_COLUMNS]
        with index_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(INDEX_COLUMNS)
            writer.writerow(values)
        return index_path


def base_index_row(run: Run, config: Config, **overrides: Any) -> dict[str, Any]:
    """Sensible defaults for the index row; overrides win."""
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": run.model,
        "window_s": config.audio.window_s,
        "aug": config.augment.preset,
        "seed": run.seed,
        "device": run.device,
        "git_sha": git_sha(),
    }
    row.update(overrides)
    return row


def load_index(runs_dir: Path) -> list[dict[str, str]]:
    path = Path(runs_dir) / "index.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_metrics(payload: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ModelError(f"brak pól w metrykach: {missing}")
