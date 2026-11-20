"""Shared test helpers: fake transport, object factories and tiny WAV files."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import numpy as np

from bird_recognizer.xc.client import HttpResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FakeTransport:
    """Queue of canned responses; records every URL it was asked for."""

    def __init__(self, responses: list[HttpResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def request(self, url: str, *, timeout: float, method: str = "GET") -> HttpResponse:
        self.calls.append(url)
        if not self._responses:
            raise AssertionError(f"nieoczekiwany request: {url}")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def json_response(
    payload: Any,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> HttpResponse:
    merged = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        merged.update(headers)
    return HttpResponse(
        status=status,
        headers=merged,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def file_response(
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "audio/mpeg",
) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers={"Content-Type": content_type},
        body=body,
    )


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / "api" / name).read_text(encoding="utf-8"))


def recording(
    recording_id: str = "XC1",
    *,
    latin: tuple[str, str] = ("Turdus", "merula"),
    english: str = "Common Blackbird",
    recordist: str = "Anna Nowak",
    redacted: bool = False,
    file_url: str | None = None,
    file_name: str | None = None,
    quality: str = "A",
    **extra: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": recording_id,
        "gen": latin[0],
        "sp": latin[1],
        "en": english,
        "grp": "birds",
        "rec": recordist,
        "cnt": "Poland",
        "loc": "Białowieża",
        "lat": "52.7",
        "lon": "23.8",
        "alt": "150",
        "type": "song",
        "sex": "male",
        "stage": "adult",
        "status": "identified",
        "date": "2024-05-01",
        "time": "06:30",
        "uploaded": "2024-05-02",
        "length": "2:16",
        "smp": "44100",
        "q": quality,
        "lic": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "url": f"https://xeno-canto.org/{recording_id}",
        "file": file_url or f"https://xeno-canto.org/{recording_id}/download",
        "file-name": file_name or f"XC{recording_id}.mp3",
        "also": [],
        "rmk": "",
    }
    item.update(extra)
    if redacted:
        item["_meta"] = {
            "redacted_fields": {
                "loc": "restricted_species",
                "lat": "restricted_species",
                "lon": "restricted_species",
                "file": "restricted_species",
            }
        }
        item["file"] = ""
        item["loc"] = ""
    return item


def page(
    recordings: list[dict[str, Any]],
    *,
    number: int = 1,
    num_pages: int = 1,
    num_recordings: int | None = None,
) -> dict[str, Any]:
    return {
        "numRecordings": str(num_recordings if num_recordings is not None else len(recordings)),
        "numSpecies": "1",
        "page": number,
        "numPages": num_pages,
        "recordings": recordings,
    }


def entry(
    recording_id: str = "XC1",
    *,
    species_pl: str = "kos",
    recordist: str = "Anna Nowak",
    sha256_audio: str = "",
    redacted: bool = False,
    file_name: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": recording_id,
        "species_latin": "Turdus merula",
        "species_pl": species_pl,
        "species_en": "Common Blackbird",
        "group": "birds",
        "rec": recordist,
        "cnt": "Poland",
        "loc": "Białowieża",
        "lat": 52.7,
        "lon": 23.8,
        "date": "2024-05-01",
        "time": "06:30",
        "uploaded": "2024-05-02",
        "length": "2:16",
        "type": "song",
        "sex": "male",
        "stage": "adult",
        "q": "A",
        "status": "identified",
        "lic": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "file": None if redacted else f"https://xeno-canto.org/{recording_id}/download",
        "file_name": file_name if file_name is not None else f"XC{recording_id}.mp3",
        "smp": "44100",
        "also": [],
        "rmk": "",
        "redacted": redacted,
        "sha256_audio": sha256_audio,
        "fetched_at": "2026-11-02T00:00:00+00:00",
    }
    item.update(extra)
    return item


def write_catalog(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in entries]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Audio for offline tests (SPEC FR-15: no real recordings, no network)
# --------------------------------------------------------------------------- #
def write_wav(
    path: Path,
    *,
    seconds: float = 6.0,
    sample_rate: int = 32000,
    freq: float = 1000.0,
    kind: str = "tone",
    amplitude: float = 0.3,
    seed: int = 0,
) -> Path:
    """Write a deterministic 16-bit mono WAV (tone / noise / silence / silence-then-tone)."""
    if kind not in {"tone", "noise", "silence", "silence-then-tone"}:
        raise ValueError(f"nieznany rodzaj sygnału: {kind}")
    samples = max(1, int(round(seconds * sample_rate)))
    timeline = np.arange(samples, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    if kind == "tone":
        signal = amplitude * np.sin(2.0 * np.pi * freq * timeline)
    elif kind == "noise":
        signal = amplitude * 0.3 * rng.standard_normal(samples)
    elif kind == "silence":
        signal = np.zeros(samples)
    else:
        signal = np.zeros(samples)
        half = samples // 2
        signal[half:] = amplitude * np.sin(2.0 * np.pi * freq * timeline[half:])
    pcm = np.clip(signal, -1.0, 1.0)
    frames = (pcm * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames.tobytes())
    return path


def make_catalog_with_audio(
    data_dir: Path,
    *,
    species: tuple[tuple[str, str], ...] = (("kos", "Turdus merula"), ("bogatka", "Parus major")),
    per_species: int = 2,
    seconds: float = 6.0,
    kind: str = "tone",
) -> list[dict[str, Any]]:
    """Catalog + ``data/raw`` WAV files for offline cache tests."""
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for species_index, (species_pl, latin) in enumerate(species):
        for number in range(per_species):
            recording_id = f"XC{100 + species_index * 10 + number}"
            write_wav(
                raw / f"{recording_id}.wav",
                seconds=seconds,
                freq=800.0 + species_index * 400.0,
                kind=kind,
                seed=number,
            )
            item = entry(
                recording_id,
                species_pl=species_pl,
                recordist=f"rec-{species_index}",
                file_name=f"{recording_id}.wav",
            )
            item["species_latin"] = latin
            entries.append(item)
    write_catalog(data_dir / "catalog.jsonl", entries)
    return entries
