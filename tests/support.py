"""Shared helpers for stage-2 tests: fake transport and object factories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
