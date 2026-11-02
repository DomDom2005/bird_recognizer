"""Exception hierarchy (SPEC §9).

Every error carries an ``exit_code`` so the CLI can map exceptions to process
exit codes without a second lookup table (SPEC §8: 0-4).
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors."""

    exit_code: int = 1


class ConfigError(AppError):
    """Invalid or missing configuration."""

    exit_code: int = 1


class MissingApiKeyError(ConfigError):
    """No API key in the environment.

    Subclass of :class:`ConfigError` (SPEC §9 calls it a config error) but exits
    with code 2, because the fix is credentials, not configuration (SPEC §8).
    """

    exit_code: int = 2


class ApiError(AppError):
    """The API returned an error or an unexpected response."""

    exit_code: int = 2

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.status is not None:
            parts.append(f"(HTTP {self.status})")
        if self.code:
            parts.append(f"[{self.code}]")
        return " ".join(parts)


class AuthError(ApiError):
    """Missing, invalid or revoked API key."""


class RateLimitError(ApiError):
    """Rate limited or service unavailable after all retries."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message, status=status, code=code)
        self.retry_after = retry_after


class DataError(AppError):
    """Corrupt, undecodable or missing data (audio, catalog, split)."""

    exit_code: int = 3


class ModelError(AppError):
    """Training, checkpoint or model failure (including OOM)."""

    exit_code: int = 4
