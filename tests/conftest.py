"""Shared pytest fixtures (SPEC §12, NFR-1).

Network access is blocked for the whole test session: a test that tries to open a
real socket must fail. Mock HTTP responses instead.
"""

from __future__ import annotations

import shutil
import socket
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
SCRATCH_ROOT = REPO_ROOT / ".tmp"


@pytest.fixture
def tmp_dir() -> Iterator[Path]:
    """Writable scratch directory inside the repo.

    Why not ``tmp_path``: this environment's sandbox treats directories created by
    ``tempfile.mkdtemp`` (mode 0o700) as non-writable — both in the OS temp dir and
    inside the workspace — so ``tmp_path`` fixtures fail with ``PermissionError``.
    ``Path.mkdir`` + a random suffix works, and ``.tmp/`` is gitignored.
    """
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCRATCH_ROOT / f"pytest-{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)



@pytest.fixture(scope="session", autouse=True)
def _block_network() -> Iterator[None]:
    """Make every socket connection attempt raise (SPEC NFR-1)."""

    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Network access is blocked in tests (SPEC NFR-1). Use mocks.")

    saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = saved


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_path() -> Path:
    return CONFIG_PATH


@pytest.fixture(scope="session")
def config(config_path: Path) -> Any:
    """Real configuration from ``configs/`` (no ``.env`` side effects)."""
    from bird_recognizer.config import load_config

    return load_config(config_path, env_file=None)
