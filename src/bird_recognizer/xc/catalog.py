"""Catalog of xeno-canto recordings: build, merge, persist (SPEC FR-1, §7.1).

The catalog is a JSON Lines file — one recording object per line — plus a small
metadata file with the queries that produced it. No secrets are ever written.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Config, Species
from ..errors import ConfigError, DataError
from .client import XCClient, build_query

LOGGER = logging.getLogger(__name__)

CATALOG_FILENAME = "catalog.jsonl"
META_FILENAME = "catalog.meta.json"


# --------------------------------------------------------------------------- #
# Paths and IO
# --------------------------------------------------------------------------- #
def catalog_path(data_dir: Path) -> Path:
    return data_dir / CATALOG_FILENAME


def meta_path(data_dir: Path) -> Path:
    return data_dir / META_FILENAME


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via ``.tmp`` + ``os.replace`` so a crash cannot truncate the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_catalog(path: Path) -> list[dict[str, Any]]:
    """Read the catalog; raise :class:`DataError` on the first broken line."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{number}: niepoprawny JSON: {exc}") from exc
            if not isinstance(entry, dict) or "id" not in entry:
                raise DataError(f"{path}:{number}: brak pola 'id'")
            entries.append(entry)
    return entries


def save_catalog(path: Path, entries: list[dict[str, Any]]) -> None:
    """Persist entries deterministically (sorted by species, then id)."""
    ordered = sorted(entries, key=lambda item: (item.get("species_pl", ""), item["id"]))
    lines = [json.dumps(entry, ensure_ascii=False, sort_keys=True) for entry in ordered]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


# --------------------------------------------------------------------------- #
# Mapping API recording -> catalog entry (SPEC §7.1)
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _redacted(recording: dict[str, Any]) -> bool:
    meta = recording.get("_meta")
    return isinstance(meta, dict) and bool(meta.get("redacted_fields"))


def record_to_entry(
    recording: dict[str, Any],
    *,
    species_pl: str,
    fetched_at: str,
) -> dict[str, Any]:
    """Map one API recording to the catalog schema."""
    recording_id = str(recording.get("id", "")).strip()
    if not recording_id:
        raise DataError("nagranie bez pola 'id' w odpowiedzi API")
    latin = " ".join(
        part for part in (recording.get("gen"), recording.get("sp")) if part
    ).strip()
    file_url = recording.get("file") or None
    return {
        "id": recording_id,
        "species_latin": latin,
        "species_pl": species_pl,
        "species_en": recording.get("en") or "",
        "group": recording.get("grp") or "",
        "rec": recording.get("rec") or "",
        "cnt": recording.get("cnt") or "",
        "loc": recording.get("loc") or "",
        "lat": _to_float(recording.get("lat")),
        "lon": _to_float(recording.get("lon")),
        "date": recording.get("date") or "",
        "time": recording.get("time") or "",
        "uploaded": recording.get("uploaded") or "",
        "length": recording.get("length") or "",
        "type": recording.get("type") or "",
        "sex": recording.get("sex") or "",
        "stage": recording.get("stage") or "",
        "q": recording.get("q") or "",
        "status": recording.get("status") or "",
        "lic": recording.get("lic") or "",
        "file": None if _redacted(recording) else file_url,
        "file_name": recording.get("file-name") or "",
        "smp": recording.get("smp") or "",
        "also": list(recording.get("also") or []),
        "rmk": recording.get("rmk") or "",
        "redacted": _redacted(recording),
        "sha256_audio": "",
        "fetched_at": fetched_at,
    }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CatalogBuild:
    """Summary of one catalog run."""

    entries: list[dict[str, Any]]
    queries: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def counts_by_species(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            species = entry.get("species_pl", "")
            counts[species] = counts.get(species, 0) + 1
        return counts


def _select_species(config: Config, species_filter: list[str] | None) -> list[Species]:
    if not species_filter:
        return list(config.species)
    wanted = {name.strip().lower() for name in species_filter}
    selected = [
        species
        for species in config.species
        if species.pl.lower() in wanted or species.latin.lower() in wanted
    ]
    missing = wanted - {s.pl.lower() for s in selected} - {s.latin.lower() for s in selected}
    if missing:
        known = ", ".join(species.pl for species in config.species)
        raise ConfigError(f"Nieznane gatunki: {sorted(missing)}. Znane: {known}")
    return selected


def build_catalog(
    config: Config,
    client: XCClient,
    *,
    data_dir: Path,
    species_filter: list[str] | None = None,
    refresh: bool = False,
    per_page: int | None = None,
) -> CatalogBuild:
    """Fetch metadata for every selected species and merge it into the catalog."""
    selected = _select_species(config, species_filter)
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    path = catalog_path(data_dir)

    existing: dict[str, dict[str, Any]] = {}
    if not refresh:
        existing = {entry["id"]: entry for entry in load_catalog(path)}

    queries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for species in selected:
        query = build_query(species, config.api)
        saved = 0
        page = 1
        num_recordings = 0
        while True:
            result = client.search(query, page=page, per_page=per_page)
            num_recordings = result.num_recordings
            for recording in result.recordings:
                entry = record_to_entry(
                    recording, species_pl=species.pl, fetched_at=fetched_at
                )
                previous = existing.get(entry["id"])
                if previous and not entry["sha256_audio"]:
                    entry["sha256_audio"] = previous.get("sha256_audio", "")
                existing[entry["id"]] = entry
                saved += 1
            if page >= result.num_pages:
                break
            page += 1
        queries[species.pl] = {
            "query": query,
            "num_recordings": num_recordings,
            "saved": saved,
            "pages": page,
        }
        LOGGER.info("%s: %s nagrań (stron: %s)", species.pl, num_recordings, page)
        if num_recordings == 0:
            warnings.append(f"{species.pl}: zapytanie zwróciło 0 nagrań")

    entries = list(existing.values())
    save_catalog(path, entries)
    meta = {
        "generated_at": fetched_at,
        "base_url": config.api.base_url,
        "per_page": per_page or config.api.per_page,
        "species": queries,
        "total_entries": len(entries),
        "redacted": sum(1 for entry in entries if entry.get("redacted")),
        "warnings": warnings,
    }
    _atomic_write_text(
        meta_path(data_dir), json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True)
    )
    LOGGER.info("Katalog: %s wpisów -> %s", len(entries), path)
    return CatalogBuild(entries=entries, queries=queries, warnings=warnings)
