"""Catalog tests: merge, redaction, metadata and atomic IO (SPEC FR-1, §7.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird_recognizer.errors import ConfigError, DataError
from bird_recognizer.xc.catalog import (
    build_catalog,
    catalog_path,
    load_catalog,
    meta_path,
    record_to_entry,
    save_catalog,
)
from bird_recognizer.xc.client import XCClient
from support import FakeTransport, json_response, load_fixture

PAGES = ("parus_page1.json", "parus_page2.json", "parus_page3.json")


def _client(config, pages=PAGES, key: str = "test-key"):
    transport = FakeTransport([json_response(load_fixture(name)) for name in pages])
    client = XCClient(config, api_key=key, transport=transport, sleep=lambda _s: None)
    return client, transport


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #
def test_record_to_entry_maps_spec_fields() -> None:
    recording = load_fixture("parus_page1.json")["recordings"][0]
    entry = record_to_entry(recording, species_pl="bogatka", fetched_at="2026-11-02T00:00:00+00:00")
    assert entry["id"] == "XC1001"
    assert entry["species_latin"] == "Parus major"
    assert entry["species_pl"] == "bogatka"
    assert entry["rec"] == "Jan Kowalski"
    assert entry["lat"] == 52.7
    assert entry["length"] == "2:16"
    assert entry["also"] == ["Turdus merula"]
    assert entry["redacted"] is False
    assert entry["sha256_audio"] == ""


def test_record_to_entry_requires_id() -> None:
    with pytest.raises(DataError, match="id"):
        record_to_entry({"gen": "Parus", "sp": "major"}, species_pl="bogatka", fetched_at="x")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def test_build_marks_redacted_and_keeps_order(config, tmp_dir: Path) -> None:
    client, _ = _client(config)
    result = build_catalog(
        config, client, data_dir=tmp_dir, species_filter=["bogatka"], refresh=True
    )
    assert len(result.entries) == 4
    redacted = [item for item in result.entries if item["redacted"]]
    assert len(redacted) == 1
    assert redacted[0]["id"] == "XC1004"
    assert redacted[0]["file"] is None
    assert redacted[0]["loc"] == ""
    assert all(item["species_pl"] == "bogatka" for item in result.entries)


def test_build_writes_meta_without_secret(config, tmp_dir: Path) -> None:
    client, _ = _client(config)
    build_catalog(config, client, data_dir=tmp_dir, species_filter=["bogatka"], refresh=True)
    meta_text = meta_path(tmp_dir).read_text(encoding="utf-8")
    meta = json.loads(meta_text)
    assert "test-key" not in meta_text
    assert "key=" not in meta_text
    assert meta["total_entries"] == 4
    assert meta["species"]["bogatka"]["num_recordings"] == 3
    assert 'sp:"Parus major"' in meta["species"]["bogatka"]["query"]


def test_catalog_file_is_sorted_jsonl(config, tmp_dir: Path) -> None:
    client, _ = _client(config)
    build_catalog(config, client, data_dir=tmp_dir, species_filter=["bogatka"], refresh=True)
    path = catalog_path(tmp_dir)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = [json.loads(line) for line in lines]
    assert [item["id"] for item in entries] == sorted(item["id"] for item in entries)
    assert load_catalog(path) == entries


def test_rebuild_preserves_sha_and_is_idempotent(config, tmp_dir: Path) -> None:
    client, _ = _client(config)
    build_catalog(config, client, data_dir=tmp_dir, species_filter=["bogatka"], refresh=True)
    path = catalog_path(tmp_dir)
    entries = load_catalog(path)
    entries[0]["sha256_audio"] = "deadbeef"
    save_catalog(path, entries)

    client2, _ = _client(config)
    build_catalog(config, client2, data_dir=tmp_dir, species_filter=["bogatka"])
    reloaded = load_catalog(path)
    assert len(reloaded) == 4
    by_id = {item["id"]: item for item in reloaded}
    assert by_id[entries[0]["id"]]["sha256_audio"] == "deadbeef"


def test_refresh_drops_stale_entries(config, tmp_dir: Path) -> None:
    client, _ = _client(config)
    build_catalog(config, client, data_dir=tmp_dir, species_filter=["bogatka"], refresh=True)
    path = catalog_path(tmp_dir)
    entries = load_catalog(path)
    entries.append(
        {
            "id": "XC9999",
            "species_pl": "bogatka",
            "species_latin": "Parus major",
            "rec": "stary",
            "redacted": False,
            "file": "https://xeno-canto.org/9999/download",
            "file_name": "XC9999.mp3",
            "sha256_audio": "",
            "fetched_at": "2026-01-01T00:00:00+00:00",
        }
    )
    save_catalog(path, entries)

    client2, _ = _client(config)
    build_catalog(config, client2, data_dir=tmp_dir, species_filter=["bogatka"], refresh=True)
    ids = {item["id"] for item in load_catalog(path)}
    assert "XC9999" not in ids


def test_unknown_species_filter_fails_before_any_request(config, tmp_dir: Path) -> None:
    client, transport = _client(config)
    with pytest.raises(ConfigError, match="Nieznane gatunki"):
        build_catalog(config, client, data_dir=tmp_dir, species_filter=["nie-ma"])
    assert transport.calls == []


def test_atomic_write_leaves_no_temp_file(config, tmp_dir: Path) -> None:
    client, _ = _client(config)
    build_catalog(config, client, data_dir=tmp_dir, species_filter=["bogatka"], refresh=True)
    leftovers = list(tmp_dir.glob(".*.tmp"))
    assert leftovers == []
