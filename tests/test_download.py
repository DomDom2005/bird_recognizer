"""Download tests: resume, hashes, redaction, errors (SPEC FR-2)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bird_recognizer.errors import DataError
from bird_recognizer.xc.catalog import catalog_path, load_catalog
from bird_recognizer.xc.client import XCClient
from bird_recognizer.xc.download import (
    download_audio,
    extension_for,
    load_report,
    raw_dir,
    report_path,
)
from support import FakeTransport, entry, file_response, write_catalog


def _setup(tmp_dir: Path, entries: list[dict]) -> Path:
    data_dir = tmp_dir / "data"
    write_catalog(catalog_path(data_dir), entries)
    return data_dir


def _client(config, responses, key: str = "test-key"):
    transport = FakeTransport(responses)
    client = XCClient(config, api_key=key, transport=transport, sleep=lambda _s: None)
    return client, transport


# --------------------------------------------------------------------------- #
# Extensions
# --------------------------------------------------------------------------- #
def test_extension_prefers_file_name_then_content_type() -> None:
    assert extension_for("XC1.wav", "audio/mpeg") == ".wav"
    assert extension_for("nieznany.dat", "audio/flac") == ".flac"
    assert extension_for("", "audio/mpeg") == ".mp3"
    assert extension_for("", None) == ".bin"


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_download_writes_file_updates_catalog_and_report(config, tmp_dir: Path) -> None:
    entries = [entry("XC1")]
    data_dir = _setup(tmp_dir, entries)
    payload = b"ID3-fake-audio"
    client, _ = _client(config, [file_response(payload)])

    report = download_audio(config, client, data_dir=data_dir)

    target = raw_dir(data_dir) / "XC1.mp3"
    assert target.is_file()
    assert target.read_bytes() == payload
    assert report.succeeded[0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert load_catalog(catalog_path(data_dir))[0]["sha256_audio"] == hashlib.sha256(
        payload
    ).hexdigest()
    saved = load_report(data_dir)
    assert saved is not None
    assert saved["counts"]["succeeded"] == 1
    assert report_path(data_dir).is_file()


def test_download_uses_content_type_when_name_is_missing(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC2", file_name="")])
    client, _ = _client(config, [file_response(b"RIFF", content_type="audio/wav")])
    report = download_audio(config, client, data_dir=data_dir)
    assert report.succeeded[0]["path"].endswith("XC2.wav")


# --------------------------------------------------------------------------- #
# Resume and filtering
# --------------------------------------------------------------------------- #
def test_download_skips_existing_file(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC1")])
    directory = raw_dir(data_dir)
    directory.mkdir(parents=True)
    (directory / "XC1.mp3").write_bytes(b"already-here")

    client, transport = _client(config, [])
    report = download_audio(config, client, data_dir=data_dir)

    assert report.skipped[0] == {
        "id": "XC1",
        "reason": "exists",
        "path": str(directory / "XC1.mp3"),
    }
    assert transport.calls == []


def test_redacted_recordings_are_not_requested(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC1"), entry("XC9", redacted=True)])
    client, transport = _client(config, [file_response(b"audio")])
    report = download_audio(config, client, data_dir=data_dir)
    assert report.redacted == ["XC9"]
    assert len(transport.calls) == 1


def test_retry_failed_only_repeats_errors(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC1"), entry("XC2")])
    bad_client, _ = _client(
        config, [file_response(b"", status=500), file_response(b"audio-2")]
    )
    first = download_audio(config, bad_client, data_dir=data_dir)
    assert [item["id"] for item in first.errors] == ["XC1"]
    assert [item["id"] for item in first.succeeded] == ["XC2"]

    good_client, transport = _client(config, [file_response(b"audio-1")])
    second = download_audio(config, good_client, data_dir=data_dir, retry_failed=True)
    assert [item["id"] for item in second.succeeded] == ["XC1"]
    assert len(transport.calls) == 1


def test_limit_caps_the_batch(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC1"), entry("XC2")])
    client, transport = _client(config, [file_response(b"a")])
    report = download_audio(config, client, data_dir=data_dir, limit=1)
    assert len(transport.calls) == 1
    assert len(report.succeeded) == 1


def test_force_redownloads_existing_file(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC1")])
    directory = raw_dir(data_dir)
    directory.mkdir(parents=True)
    (directory / "XC1.mp3").write_bytes(b"old")
    client, _ = _client(config, [file_response(b"new")])
    report = download_audio(config, client, data_dir=data_dir, force=True)
    assert report.succeeded
    assert (directory / "XC1.mp3").read_bytes() == b"new"


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #
def test_error_is_recorded_and_loop_continues(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC1"), entry("XC2")])
    client, _ = _client(
        config, [file_response(b"", status=500), file_response(b"audio-2")]
    )
    report = download_audio(config, client, data_dir=data_dir)
    assert len(report.errors) == 1
    assert report.errors[0]["id"] == "XC1"
    assert report.errors[0]["status"] == 500
    assert len(report.succeeded) == 1


def test_rate_limit_stops_the_batch(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC1"), entry("XC2")])
    client, transport = _client(config, [file_response(b"", status=429)])
    report = download_audio(config, client, data_dir=data_dir)
    assert report.stopped == "rate_limit"
    assert len(report.errors) == 1
    assert len(transport.calls) == 1


def test_missing_catalog_is_an_error(config, tmp_dir: Path) -> None:
    client, _ = _client(config, [])
    with pytest.raises(DataError, match="Brak katalogu"):
        download_audio(config, client, data_dir=tmp_dir / "data")


def test_retry_failed_without_report_is_an_error(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC1")])
    client, _ = _client(config, [])
    with pytest.raises(DataError, match="raport"):
        download_audio(config, client, data_dir=data_dir, retry_failed=True)


def test_entry_without_file_url_is_an_error(config, tmp_dir: Path) -> None:
    data_dir = _setup(tmp_dir, [entry("XC1", file=None)])
    client, transport = _client(config, [])
    report = download_audio(config, client, data_dir=data_dir)
    assert report.errors[0]["message"] == "brak URL pliku"
    assert transport.calls == []
