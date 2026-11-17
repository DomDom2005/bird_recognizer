"""xeno-canto API v3 client: queries, pagination, retries (SPEC §4, FR-1, §9).

This is the only module in the package that may touch the network. Everything is
injectable, so tests run offline with a fake transport (NFR-1).
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any

from ..config import ApiConfig, Config, Species, get_api_key
from ..errors import ApiError, AuthError, MissingApiKeyError, RateLimitError

LOGGER = logging.getLogger(__name__)

USER_AGENT = "bird_recognizer/0.1.0 (research project; xeno-canto API v3)"
_KEY_RE = re.compile(r"(?i)(key=)[^&\s]+")


def mask_url(url: str) -> str:
    """Hide ``key=...`` before a URL reaches any log destination (SPEC §13)."""
    return _KEY_RE.sub(r"\1***", url)


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HttpResponse:
    """Status, headers and raw body of one HTTP exchange."""

    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""


class HttpTransport:
    """Minimal urllib transport; injectable for tests."""

    def request(self, url: str, *, timeout: float, method: str = "GET") -> HttpResponse:
        request = urllib.request.Request(
            url,
            method=method,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json, audio/*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return HttpResponse(
                    status=response.status,
                    headers=dict(response.headers),
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                headers=dict(exc.headers or {}),
                body=exc.read(),
            )
        except urllib.error.URLError as exc:
            raise ApiError(f"błąd sieci: {exc.reason}") from exc


# --------------------------------------------------------------------------- #
# Query building
# --------------------------------------------------------------------------- #
def build_query(species: Species, api: ApiConfig) -> str:
    """Assemble the v3 query from search tags only (SPEC §4.1)."""
    tags = [f'sp:"{species.latin}"', f"cnt:{api.country}", f"q:{api.quality}"]
    if api.status:
        tags.append(f"status:{api.status}")
    return " ".join(tags)


def build_url(
    api: ApiConfig,
    query: str,
    *,
    key: str | None,
    page: int | None = None,
    per_page: int | None = None,
) -> str:
    """Encode a v3 request URL; the key travels in the query string (SPEC §4.1)."""
    pairs: list[tuple[str, str]] = [
        ("query", query),
        ("per_page", str(clamp_per_page(per_page))),
    ]
    if page is not None:
        pairs.append(("page", str(page)))
    if key:
        pairs.append(("key", key))
    params = urllib.parse.urlencode(pairs, quote_via=urllib.parse.quote, safe=':"<>')
    return f"{api.base_url}?{params}"


def clamp_per_page(value: int | None, default: int = 100) -> int:
    """xeno-canto accepts ``per_page`` in 50..500 (SPEC §4.1)."""
    raw = default if value is None else value
    return max(50, min(500, raw))


# --------------------------------------------------------------------------- #
# Response parsing helpers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SearchPage:
    """One page of search results."""

    page: int
    num_pages: int
    num_recordings: int
    recordings: tuple[dict[str, Any], ...]


def parse_error_payload(body: bytes) -> tuple[str | None, str | None]:
    """Return ``(code, message)`` from either documented error shape (SPEC §4.1)."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    error = payload.get("error")
    message = payload.get("message")
    if isinstance(error, str):
        return error, message if isinstance(message, str) else None
    if isinstance(error, dict):
        code = error.get("code")
        detail = error.get("message")
        return (
            code if isinstance(code, str) else None,
            detail if isinstance(detail, str) else None,
        )
    return None, message if isinstance(message, str) else None


def parse_retry_after(value: str | None) -> float | None:
    """Parse ``Retry-After`` as seconds or an HTTP date (SPEC §9)."""
    if not value:
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    return max(0.0, when.timestamp() - time.time())


def _retry_after_from(headers: Mapping[str, str]) -> float | None:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            return parse_retry_after(value)
    return None


def _backoff_seconds(base: float, attempt: int) -> float:
    """Exponential backoff with a small jitter (SPEC §9)."""
    return base * (2 ** (attempt - 1)) * (1.0 + random.uniform(0.0, 0.1))


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class XCClient:
    """Throttled xeno-canto client with explicit error mapping."""

    def __init__(
        self,
        config: Config,
        *,
        api_key: str | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._api_key = api_key if api_key is not None else get_api_key()
        self._transport = transport if transport is not None else HttpTransport()
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.request_count = 0

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def _require_key(self) -> str:
        if not self._api_key:
            raise MissingApiKeyError(
                "Brak klucza API xeno-canto. Ustaw BIRD_XC_API_KEY w `.env` "
                "(wzór: .env.example). Klucz jest osobisty, nie commituj go."
            )
        return self._api_key

    def _throttle(self) -> None:
        interval = self._config.api.request_interval_s
        if self._last_request is not None and interval > 0:
            wait = interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _fetch(self, url: str) -> HttpResponse:
        self._throttle()
        self.request_count += 1
        LOGGER.debug("GET %s", mask_url(url))
        return self._transport.request(url, timeout=self._config.api.read_timeout_s)

    def _decode_page(self, response: HttpResponse, page: int) -> SearchPage:
        try:
            payload = json.loads(response.body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(
                "odpowiedź xeno-canto nie jest poprawnym JSON-em",
                status=response.status,
            ) from exc
        if not isinstance(payload, dict):
            raise ApiError("nieoczekiwany kształt odpowiedzi", status=response.status)
        recordings = payload.get("recordings")
        if not isinstance(recordings, list):
            raise ApiError("brak listy 'recordings' w odpowiedzi", status=response.status)
        try:
            num_pages = int(payload.get("numPages") or 1)
            num_recordings = int(payload.get("numRecordings") or len(recordings))
        except (TypeError, ValueError) as exc:
            raise ApiError("niepoprawne liczniki w odpowiedzi", status=response.status) from exc
        return SearchPage(
            page=page,
            num_pages=max(1, num_pages),
            num_recordings=max(0, num_recordings),
            recordings=tuple(recordings),
        )

    def _request_json(self, url: str, *, page: int) -> SearchPage:
        api = self._config.api
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self._fetch(url)
            except ApiError as exc:
                if attempts >= 3:
                    raise ApiError(f"błąd sieci po {attempts} próbach: {exc}") from exc
                delay = _backoff_seconds(1.0, attempts)
                LOGGER.warning("Błąd sieci (%s), ponawiam za %.1f s", exc, delay)
                self._sleep(delay)
                continue

            if response.status == 200:
                return self._decode_page(response, page)

            if response.status in (401, 403):
                raise AuthError(
                    "xeno-canto odrzuciło klucz API (HTTP "
                    f"{response.status}); sprawdź BIRD_XC_API_KEY",
                    status=response.status,
                )

            if response.status in (429, 503):
                if attempts >= max(1, api.max_retries):
                    raise RateLimitError(
                        f"limit zapytań xeno-canto przekroczony ({attempts} prób)",
                        retry_after=_retry_after_from(response.headers),
                        status=response.status,
                    )
                retry_after = _retry_after_from(response.headers)
                delay = (
                    retry_after
                    if retry_after is not None
                    else _backoff_seconds(5.0, attempts)
                )
                LOGGER.warning(
                    "HTTP %s od xeno-canto (próba %s/%s), czekam %.1f s",
                    response.status,
                    attempts,
                    max(1, api.max_retries),
                    delay,
                )
                self._sleep(delay)
                continue

            if 500 <= response.status < 600:
                if attempts >= 4:
                    raise ApiError(
                        f"xeno-canto: błąd serwera po {attempts} próbach",
                        status=response.status,
                    )
                delay = _backoff_seconds(2.0, attempts)
                LOGGER.warning(
                    "HTTP %s od xeno-canto, ponawiam za %.1f s", response.status, delay
                )
                self._sleep(delay)
                continue

            code, message = parse_error_payload(response.body)
            raise ApiError(
                message or "błąd odpowiedzi xeno-canto",
                status=response.status,
                code=code,
            )

    # -- public API ------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        page: int = 1,
        per_page: int | None = None,
    ) -> SearchPage:
        """Fetch one page of results."""
        key = self._require_key()
        if page < 1:
            raise ApiError("numer strony musi być >= 1")
        url = build_url(
            self._config.api,
            query,
            key=key,
            page=page,
            per_page=per_page,
        )
        return self._request_json(url, page=page)

    def iter_recordings(
        self,
        query: str,
        *,
        per_page: int | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield every recording of a query, walking ``1..numPages``."""
        page = 1
        while True:
            result = self.search(query, page=page, per_page=per_page)
            yield from result.recordings
            if page >= result.num_pages:
                return
            if max_pages is not None and page >= max_pages:
                return
            page += 1

    def fetch_bytes(self, url: str) -> tuple[bytes, Mapping[str, str]]:
        """Download a media file (no API key, no JSON parsing — SPEC §4.3 pkt 2)."""
        self._throttle()
        self.request_count += 1
        LOGGER.debug("GET %s", url)
        response = self._transport.request(
            url, timeout=self._config.api.download_timeout_s
        )
        if response.status in (401, 403):
            raise AuthError(
                f"pobranie pliku odrzucone (HTTP {response.status})",
                status=response.status,
            )
        if response.status in (429, 503):
            raise RateLimitError(
                f"limit pobierania pliku (HTTP {response.status})",
                retry_after=_retry_after_from(response.headers),
                status=response.status,
            )
        if response.status >= 400:
            raise ApiError(
                f"pobranie pliku nie powiodło się (HTTP {response.status})",
                status=response.status,
            )
        if not response.body:
            raise ApiError(f"pusty plik (HTTP {response.status})", status=response.status)
        return response.body, response.headers
