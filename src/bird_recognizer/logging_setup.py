"""Logging setup with secret redaction (SPEC §13).

The xeno-canto API key travels in the query string, so *any* logged URL can leak
it. The filter below masks the key, tokens and authorization headers, and also
masks literal secret values registered at startup.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from typing import TextIO

REDACTED = "***"

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # key=... / api_key=... / token=... in a URL or a log line
    re.compile(r"(?i)\b(key|api_key|apikey|token|access_token|refresh_token|client_secret)=([^&\s\"'<>]+)"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)(\S+)"),
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)(\S+)"),
)


class RedactionFilter(logging.Filter):
    """Mask known secrets and secret-looking parameters in log records."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets: tuple[str, ...] = tuple(s for s in secrets if s)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        original = message
        for secret in self._secrets:
            message = message.replace(secret, REDACTED)
        for pattern in _SECRET_PATTERNS:
            message = pattern.sub(lambda m: f"{m.group(1)}={REDACTED}", message)
        if message != original:
            # Already formatted; clear args so formatMessage() does not re-apply them.
            record.msg = message
            record.args = None
        return True


def setup_logging(
    level: str | int = "INFO",
    *,
    stream: TextIO | None = None,
    secrets: Iterable[str] = (),
) -> logging.Logger:
    """Configure the root logger once; return it."""
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handler.addFilter(RedactionFilter(secrets))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper() if isinstance(level, str) else level)
    return root


def configure_console() -> None:
    """Force UTF-8 on stdout/stderr so Polish text survives Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue

