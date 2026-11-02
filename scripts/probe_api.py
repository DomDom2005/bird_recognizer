"""Verify the xeno-canto API facts listed in SPEC §4.3.

Two modes:

* ``--dry-run`` (default) — no key, no network. Validates the config and prints
  the exact queries that would be sent. Safe for tests and stage-1 gate.
* ``--live`` — one request per check, spaced by ``api.request_interval_s``, then a
  short findings report to copy into ``docs/api-notes.md``.

Usage::

    python scripts/probe_api.py --dry-run
    python scripts/probe_api.py --live --json .tmp/api-probe.json

The API key is never printed: every URL goes through :func:`mask_key`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bird_recognizer.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    ApiConfig,
    Config,
    Species,
    get_api_key,
    load_config,
)
from bird_recognizer.errors import ApiError, AppError, MissingApiKeyError  # noqa: E402

USER_AGENT = "bird_recognizer/0.1.0 (research project; xeno-canto API v3)"
_KEY_RE = re.compile(r"(?i)(key=)[^&\s]+")
_MSS_RE = re.compile(r"^\d{1,2}:\d{2}$")
_HMS_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


# --------------------------------------------------------------------------- #
# Queries and URLs
# --------------------------------------------------------------------------- #
def build_query(species: Species, api: ApiConfig) -> str:
    """Assemble the v3 query from search tags only (SPEC §4.1).

    The ``status`` tag is emitted only when configured: the live API rejected
    ``status:identified`` with HTTP 400 on 2026-11-02 (docs/api-notes.md §3).
    """
    tags = [
        f'sp:"{species.latin}"',
        f"cnt:{api.country}",
        f"q:{api.quality}",
    ]
    if api.status:
        tags.append(f"status:{api.status}")
    return " ".join(tags)


def build_url(
    config: Config,
    query: str,
    *,
    page: int | None = None,
    per_page: int | None = None,
    key: str | None = None,
) -> str:
    pairs: list[tuple[str, str]] = [
        ("query", query),
        ("per_page", str(per_page if per_page is not None else config.api.per_page)),
    ]
    if page is not None:
        pairs.append(("page", str(page)))
    if key:
        pairs.append(("key", key))
    params = urllib.parse.urlencode(pairs, quote_via=urllib.parse.quote, safe=':"<>')
    return f"{config.api.base_url}?{params}"


def mask_key(text: str) -> str:
    """Hide ``key=...`` so nothing secret reaches a terminal or a file."""
    return _KEY_RE.sub(r"\1***", text)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Probe:
    """Minimal throttled client; only this script is allowed to use the network."""

    def __init__(self, config: Config, key: str | None) -> None:
        self.config = config
        self.key = key
        self._last_request: float | None = None
        self.results: list[dict[str, Any]] = []

    def _throttle(self) -> None:
        interval = self.config.api.request_interval_s
        if self._last_request is not None:
            wait = interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()

    def _request(self, url: str, *, method: str = "GET") -> tuple[int, bytes, str]:
        request = urllib.request.Request(
            url, method=method, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        try:
            timeout = self.config.api.read_timeout_s
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read(), response.geturl()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            code = _error_code(body)
            raise ApiError("xeno-canto: błąd odpowiedzi", status=exc.code, code=code) from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"błąd sieci: {exc.reason}") from exc

    def recordings(
        self,
        query: str,
        *,
        page: int | None = None,
        per_page: int | None = None,
    ) -> dict:
        self._throttle()
        url = build_url(self.config, query, page=page, per_page=per_page, key=self.key)
        status, body, _ = self._request(url)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiError(f"odpowiedź nie jest JSON (HTTP {status})", status=status) from exc
        if not isinstance(payload, dict):
            raise ApiError(f"nieoczekiwany kształt odpowiedzi (HTTP {status})", status=status)
        return payload

    def head_file(self, url: str) -> tuple[int, str]:
        try:
            request = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=self.config.api.read_timeout_s) as resp:
                return resp.status, resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type", "") if exc.headers else ""
        except urllib.error.URLError as exc:
            raise ApiError(f"błąd sieci przy pobieraniu pliku: {exc.reason}") from exc

    def record(self, check: str, status: str, detail: str) -> None:
        self.results.append({"check": check, "status": status, "detail": detail})
        print(f"[{status}] {check}: {detail}")


def _error_code(body: bytes) -> str | None:
    """Error payload comes as ``{"error": "client_error", "message": ...}``.

    The documentation shows a nested ``{"error": {"code": ...}}``; accept both.
    """
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        code = error.get("code")
        return code if isinstance(code, str) else None
    return None


# --------------------------------------------------------------------------- #
# Checks (SPEC §4.3)
# --------------------------------------------------------------------------- #
def check_country_tag(probe: Probe) -> None:
    query = f"cnt:{probe.config.api.country} grp:birds"
    payload = probe.recordings(query, per_page=50)
    total = int(payload.get("numRecordings", 0))
    status = "OK" if total > 0 else "TODO"
    probe.record("4.3.1 tag kraju", status, f"{query!r} → numRecordings={total}")


def check_quality_operator(probe: Probe, species: Species, condition: str) -> None:
    query = f'sp:"{species.latin}" cnt:{probe.config.api.country} q:{probe.config.api.quality} '
    query += f"{condition}".strip()
    payload = probe.recordings(query, per_page=50)
    probe.record(
        "4.3.4 operator jakości",
        "OK" if int(payload.get("numRecordings", 0)) > 0 else "TODO",
        f"q:{probe.config.api.quality} → numRecordings={payload.get('numRecordings')}",
    )


def check_status_tag(probe: Probe, species: Species) -> None:
    """The documented ``status`` tag is rejected live on 2026-11-02."""
    base = f'sp:"{species.latin}" cnt:{probe.config.api.country} q:{probe.config.api.quality}'
    try:
        payload = probe.recordings(f"{base} status:identified", per_page=50)
    except ApiError as exc:
        probe.record(
            "4.3.3 tag status",
            "ROZBIEŻNOŚĆ",
            f"status:identified → {exc}; tag jest w dokumentacji (2026-07-17), "
            "ale API go odrzuca — wyłączony w configs/default.yaml",
        )
        return
    probe.record(
        "4.3.3 tag status",
        "OK",
        f"status:identified → numRecordings={payload.get('numRecordings')}",
    )


def check_length_format(probe: Probe, species: Species) -> None:
    query = build_query(species, probe.config.api)
    payload = probe.recordings(query, per_page=50)
    lengths = [str(rec.get("length", "")) for rec in payload.get("recordings", [])]
    shapes = {"m:ss": 0, "h:mm:ss": 0, "inne": 0}
    examples: dict[str, str] = {}
    for value in lengths:
        key = "m:ss" if _MSS_RE.match(value) else "h:mm:ss" if _HMS_RE.match(value) else "inne"
        shapes[key] += 1
        examples.setdefault(key, value)
    probe.record(
        "4.3.2 format length",
        "OK" if shapes["m:ss"] or shapes["h:mm:ss"] else "TODO",
        f"{shapes}, przykłady: {examples}",
    )


def check_file_download(probe: Probe, species: Species) -> None:
    query = build_query(species, probe.config.api)
    payload = probe.recordings(query, per_page=50)
    recordings = payload.get("recordings", [])
    with_file = [rec for rec in recordings if rec.get("file")]
    if not with_file:
        probe.record("4.3.5 pobieranie pliku", "TODO", "brak nagrań z polem 'file'")
        return
    url = str(with_file[0]["file"])
    status, content_type = probe.head_file(url)
    needs_key = status in (401, 403)
    probe.record(
        "4.3.5 pobieranie pliku",
        "WARN" if needs_key else "OK",
        f"HEAD bez klucza → HTTP {status}, Content-Type={content_type!r}, "
        f"{'wymaga klucza' if needs_key else 'nie wymaga klucza'}",
    )


def check_page_out_of_range(probe: Probe, species: Species) -> None:
    query = build_query(species, probe.config.api)
    first = probe.recordings(query, per_page=50)
    pages = int(first.get("numPages", 0))
    try:
        payload = probe.recordings(query, page=pages + 1, per_page=50)
    except ApiError as exc:
        probe.record(
            "4.3.6 page poza zakresem",
            "OK",
            f"page={pages + 1} → ApiError: {exc}",
        )
        return
    probe.record(
        "4.3.6 page poza zakresem",
        "OK",
        f"page={pages + 1} → HTTP 200, recordings={len(payload.get('recordings', []))}, "
        f"numPages={payload.get('numPages')}",
    )


def check_species_availability(probe: Probe) -> None:
    rows: list[str] = []
    for species in probe.config.species:
        query = build_query(species, probe.config.api)
        try:
            payload = probe.recordings(query, per_page=50)
            rows.append(f"{species.pl}: {payload.get('numRecordings')}")
        except ApiError as exc:
            rows.append(f"{species.pl}: BŁĄD ({exc})")
    probe.record("4.3.7 dostępność gatunków", "OK", "; ".join(rows))


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _dry_run(config: Config) -> int:
    key = get_api_key()
    print("Tryb dry-run: żadnych requestów sieciowych.\n")
    print(f"Config: {config.config_path} ({len(config.species)} gatunków, "
          f"sr={config.audio.sample_rate} Hz, okno={config.audio.window_s} s)\n")
    for species in config.species:
        print(f"  {species.pl:<13} → {build_query(species, config.api)}")
    print("\nPrzykładowy URL (klucz zamaskowany):")
    print("  " + mask_key(build_url(config, build_query(config.species[0], config.api), key=key)))
    if key:
        print("\nKlucz API: obecny w środowisku (wartość nie jest wypisywana).")
    else:
        print("\nKlucz API: brak — tryb --live jest niedostępny.")
    return 0


def _live(config: Config, json_path: Path | None) -> int:
    try:
        key = get_api_key(required=True)
    except MissingApiKeyError as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 2

    probe = Probe(config, key)
    species = config.species[0]
    checks = (
        ("4.3.1 tag kraju", lambda: check_country_tag(probe)),
        ("4.3.2 format length", lambda: check_length_format(probe, species)),
        ("4.3.3 tag status", lambda: check_status_tag(probe, species)),
        ("4.3.4 operator jakości", lambda: check_quality_operator(probe, species, "")),
        ("4.3.5 pobieranie pliku", lambda: check_file_download(probe, species)),
        ("4.3.6 page poza zakresem", lambda: check_page_out_of_range(probe, species)),
        ("4.3.7 dostępność gatunków", lambda: check_species_availability(probe)),
    )
    print(f"Tryb live: {len(checks)} sprawdzeń, ~{config.api.request_interval_s:.1f} s "
          f"przerwy między requestami. Klucz API: obecny (wartość nie jest wypisywana).\n")
    for name, check in checks:
        try:
            check()
        except AppError as exc:
            probe.record(name, "BŁĄD", str(exc))
        except Exception as exc:  # noqa: BLE001 - raport ma przetrwać dowolny błąd
            probe.record(name, "BŁĄD", f"{type(exc).__name__}: {mask_key(str(exc))}")

    print("\nUstalenia do przepisania w docs/api-notes.md (data: dzisiejsza).")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {"results": probe.results, "base_url": config.api.base_url},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Zapisano: {json_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weryfikacja faktów o xeno-canto API v3 (SPEC §4.3)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--species-config", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="walidacja configu i wypisanie zapytań bez sieci (domyślne)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="wysyła requesty (wymaga klucza API w środowisku)",
    )
    parser.add_argument("--json", type=Path, default=None, help="zapisz wyniki do pliku JSON")
    args = parser.parse_args(argv)
    if args.live and args.dry_run:
        parser.error("wybierz --dry-run albo --live")
    return args


def _configure_stdout() -> None:
    """Polish text on Windows consoles: force UTF-8 instead of the legacy codepage."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)
    try:
        config = load_config(
            args.config, species_path=args.species_config, env_file=args.env_file
        )
    except AppError as exc:
        print(f"BŁĄD konfiguracji: {exc}", file=sys.stderr)
        return 1
    if args.live:
        return _live(config, args.json)
    return _dry_run(config)


if __name__ == "__main__":
    raise SystemExit(main())
