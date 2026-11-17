"""Download audio files for catalogued recordings (SPEC FR-2, §9).

Resumable by design: an existing non-empty file is skipped, so an interrupted run
can simply be started again. Every outcome (ok, skipped, redacted, error) lands in
``data/download_report.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import ApiError, DataError, RateLimitError
from .catalog import catalog_path, load_catalog, save_catalog
from .client import XCClient

LOGGER = logging.getLogger(__name__)

RAW_DIRNAME = "raw"
REPORT_FILENAME = "download_report.json"

_KNOWN_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
_CONTENT_TYPE_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
}


def raw_dir(data_dir: Path) -> Path:
    return data_dir / RAW_DIRNAME


def report_path(data_dir: Path) -> Path:
    return data_dir / REPORT_FILENAME


def extension_for(file_name: str, content_type: str | None) -> str:
    """Prefer the original file name, fall back to ``Content-Type`` (SPEC §4.3)."""
    suffix = Path(file_name).suffix.lower() if file_name else ""
    if suffix in _KNOWN_EXTENSIONS:
        return suffix
    if content_type:
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type in _CONTENT_TYPE_EXTENSIONS:
            return _CONTENT_TYPE_EXTENSIONS[media_type]
    return ".bin"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _existing_file(directory: Path, recording_id: str) -> Path | None:
    for candidate in sorted(directory.glob(f"{recording_id}.*")):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Report model
# --------------------------------------------------------------------------- #
@dataclass
class DownloadReport:
    """Outcome of one download run."""

    requested: int = 0
    succeeded: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    redacted: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    duplicates: dict[str, str] = field(default_factory=dict)
    stopped: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(int(item.get("size", 0)) for item in self.succeeded)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "counts": {
                "requested": self.requested,
                "succeeded": len(self.succeeded),
                "skipped": len(self.skipped),
                "redacted": len(self.redacted),
                "errors": len(self.errors),
                "bytes": self.total_bytes,
            },
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "redacted": self.redacted,
            "errors": self.errors,
            "duplicates": self.duplicates,
            "stopped": self.stopped,
        }


def save_report(data_dir: Path, report: DownloadReport) -> Path:
    path = report_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def load_report(data_dir: Path) -> dict[str, Any] | None:
    path = report_path(data_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: niepoprawny JSON raportu: {exc}") from exc
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _select_entries(
    entries: list[dict[str, Any]],
    *,
    species_filter: list[str] | None,
    error_ids: set[str] | None,
) -> list[dict[str, Any]]:
    wanted = {name.strip().lower() for name in species_filter or []}
    selected: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("redacted"):
            continue
        if wanted and str(entry.get("species_pl", "")).lower() not in wanted:
            continue
        if error_ids is not None and entry["id"] not in error_ids:
            continue
        selected.append(entry)
    return selected


def download_audio(
    config: Config,
    client: XCClient,
    *,
    data_dir: Path,
    species_filter: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> DownloadReport:
    """Download missing audio for catalogued, non-redacted recordings."""
    entries = load_catalog(catalog_path(data_dir))
    if not entries:
        raise DataError(
            f"Brak katalogu w {data_dir}. Najpierw uruchom `bird_recognizer data catalog`."
        )

    previous = load_report(data_dir) if retry_failed else None
    error_ids: set[str] | None = None
    if retry_failed:
        if previous is None:
            raise DataError("Brak wcześniejszego raportu — nie ma czego ponawiać.")
        error_ids = {
            str(item.get("id"))
            for item in previous.get("errors", [])
            if item.get("id")
        }
        if not error_ids:
            LOGGER.info("Poprzedni raport nie zawiera błędów — nic do zrobienia.")
            report = DownloadReport()
            save_report(data_dir, report)
            return report

    report = DownloadReport(redacted=sorted(e["id"] for e in entries if e.get("redacted")))
    selected = _select_entries(entries, species_filter=species_filter, error_ids=error_ids)
    if limit is not None:
        selected = selected[: max(0, limit)]
    report.requested = len(selected)
    LOGGER.info(
        "Do pobrania: %s nagrań (%s w katalogu, %s zredagowanych)",
        len(selected),
        len(entries),
        len(report.redacted),
    )

    directory = raw_dir(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {entry["id"]: entry for entry in entries}
    hashes: dict[str, str] = {}
    updated = False

    for index, entry in enumerate(selected, start=1):
        recording_id = entry["id"]
        existing = _existing_file(directory, recording_id)
        if existing is not None and not force:
            report.skipped.append(
                {"id": recording_id, "reason": "exists", "path": str(existing)}
            )
            LOGGER.debug("[%s/%s] %s już jest", index, len(selected), recording_id)
            continue

        file_url = entry.get("file")
        if not file_url:
            report.errors.append(
                {"id": recording_id, "status": None, "message": "brak URL pliku"}
            )
            continue

        try:
            body, headers = client.fetch_bytes(str(file_url))
        except RateLimitError as exc:
            report.errors.append(
                {"id": recording_id, "status": exc.status, "message": str(exc)}
            )
            report.stopped = "rate_limit"
            LOGGER.error("Limit zapytań — przerywam pobieranie: %s", exc)
            break
        except ApiError as exc:
            report.errors.append(
                {"id": recording_id, "status": exc.status, "message": str(exc)}
            )
            LOGGER.warning("[%s/%s] %s: %s", index, len(selected), recording_id, exc)
            continue

        content_type = next(
            (value for name, value in headers.items() if name.lower() == "content-type"),
            "",
        )
        extension = extension_for(str(entry.get("file_name", "")), content_type)
        target = directory / f"{recording_id}{extension}"
        _atomic_write_bytes(target, body)
        digest = _sha256(body)

        if digest in hashes and hashes[digest] != recording_id:
            report.duplicates[digest] = hashes[digest]
            LOGGER.warning(
                "Duplikat pliku: %s ma ten sam hash co %s", recording_id, hashes[digest]
            )
        else:
            hashes[digest] = recording_id

        report.succeeded.append(
            {
                "id": recording_id,
                "path": str(target),
                "sha256": digest,
                "size": len(body),
                "content_type": content_type,
            }
        )
        if by_id[recording_id].get("sha256_audio") != digest:
            by_id[recording_id]["sha256_audio"] = digest
            updated = True
        LOGGER.info(
            "[%s/%s] %s -> %s (%.1f kB)",
            index,
            len(selected),
            recording_id,
            target.name,
            len(body) / 1024,
        )

    if updated:
        save_catalog(catalog_path(data_dir), entries)
    save_report(data_dir, report)
    return report
