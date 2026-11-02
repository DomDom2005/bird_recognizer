"""Proves that the session-wide network block actually works (SPEC NFR-1)."""

from __future__ import annotations

import socket

import pytest


def test_create_connection_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="blocked"):
        socket.create_connection(("xeno-canto.org", 443), timeout=1)


def test_socket_connect_is_blocked() -> None:
    sock = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="blocked"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()
