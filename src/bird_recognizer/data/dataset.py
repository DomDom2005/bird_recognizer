"""Spectrogram cache and dataset access (SPEC FR-4, §7.4).

Cache layout: ``data/cache/<id>/mel.npy`` (float16, ``(n_kept, n_mels, frames)``)
and ``meta.json`` (geometry, RMS of *all* windows, which were kept, ``config_hash``).
The cache is fully reproducible from ``data/raw`` — deleting it never loses data.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..errors import DataError
from ..xc.catalog import catalog_path, load_catalog
from .audio import load_audio
from .spectrogram import CACHE_VERSION, config_hash, mel_spectrogram
from .windows import iter_window_signals, plan_windows, valid_frames

LOGGER = logging.getLogger(__name__)

CACHE_DIRNAME = "cache"
MEL_FILENAME = "mel.npy"
META_FILENAME = "meta.json"
RAW_DIRNAME = "raw"


def cache_root(data_dir: Path) -> Path:
    return Path(data_dir) / CACHE_DIRNAME


def recording_cache_dir(data_dir: Path, recording_id: str) -> Path:
    return cache_root(data_dir) / recording_id


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class CacheReport:
    """Outcome of one ``prep cache`` run."""

    processed: int = 0
    skipped: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    windows_total: int = 0
    windows_kept: int = 0
    audio_seconds: float = 0.0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    per_species: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def rejected_ratio(self) -> float:
        if not self.windows_total:
            return 0.0
        return 1.0 - self.windows_kept / self.windows_total

    def count_skip(self, reason: str) -> None:
        self.skipped += 1
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def add_windows(self, species: str, total: int, kept: int) -> None:
        self.windows_total += total
        self.windows_kept += kept
        stats = self.per_species.setdefault(species, {"windows": 0, "kept": 0})
        stats["windows"] += total
        stats["kept"] += kept

    def as_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "skipped_reasons": dict(sorted(self.skipped_reasons.items())),
            "errors": list(self.errors),
            "windows_total": self.windows_total,
            "windows_kept": self.windows_kept,
            "rejected_ratio": round(self.rejected_ratio, 4),
            "audio_seconds": round(self.audio_seconds, 1),
            "per_species": dict(sorted(self.per_species.items())),
        }


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _find_audio(raw_dir: Path, recording_id: str) -> Path | None:
    for candidate in sorted(raw_dir.glob(f"{recording_id}.*")):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _write_meta(path: Path, meta: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def _read_meta(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: niepoprawny JSON: {exc}") from exc
    return payload if isinstance(payload, dict) else None


def _cache_is_current(meta: Mapping[str, Any], fingerprint: str, sha256_audio: str) -> bool:
    if meta.get("cache_version") != CACHE_VERSION:
        return False
    if meta.get("config_hash") != fingerprint:
        return False
    cached_sha = str(meta.get("sha256_audio") or "")
    if sha256_audio and cached_sha and sha256_audio != cached_sha:
        return False
    return int(meta.get("n_kept") or 0) > 0


def _select_entries(
    entries: list[dict[str, Any]],
    species: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    wanted = {name.strip().lower() for name in species or []}
    selected = [
        entry
        for entry in entries
        if not entry.get("redacted")
        and (not wanted or str(entry.get("species_pl", "")).lower() in wanted)
    ]
    return selected[: max(0, limit)] if limit is not None else selected


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_cache(
    config: Config,
    data_dir: Path,
    *,
    species: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
) -> CacheReport:
    """Cut every catalogued recording into windows and cache its log-mel frames."""
    catalog_file = catalog_path(data_dir)
    if not catalog_file.is_file():
        raise DataError(
            f"Brak katalogu {catalog_file}. Najpierw uruchom `bird_recognizer data catalog`."
        )
    entries = _select_entries(load_catalog(catalog_file), species, limit)
    if not entries:
        raise DataError("Wybór jest pusty — sprawdź --species / --limit i katalog.")

    fingerprint = config_hash(config)
    raw_dir = Path(data_dir) / RAW_DIRNAME
    report = CacheReport()

    for index, entry in enumerate(entries, start=1):
        recording_id = str(entry["id"])
        cache_dir = recording_cache_dir(data_dir, recording_id)
        meta = _read_meta(cache_dir / META_FILENAME)
        if not force and meta is not None and _cache_is_current(
            meta, fingerprint, str(entry.get("sha256_audio") or "")
        ):
            report.count_skip("up_to_date")
            continue

        audio_file = _find_audio(raw_dir, recording_id)
        if audio_file is None:
            report.count_skip("missing_audio")
            continue

        try:
            signal = load_audio(audio_file, config.audio)
            plan = plan_windows(signal, config.audio, config.gating)
        except DataError as exc:
            report.errors.append({"id": recording_id, "reason": str(exc)})
            LOGGER.warning("%s: %s", recording_id, exc)
            continue

        if plan.n_kept == 0:
            report.errors.append({"id": recording_id, "reason": "empty_after_gating"})
            LOGGER.warning("%s: wszystkie okna odrzucone przez bramkę ciszy", recording_id)
            continue

        frames: list[np.ndarray] = []
        valid: list[int] = []
        for window_index, window in iter_window_signals(signal, plan, only_kept=True):
            frames.append(
                mel_spectrogram(window, config.audio.sample_rate, config.spectrogram)
            )
            valid.append(
                valid_frames(plan.lengths[window_index], config.spectrogram.hop_length)
            )
        mel = np.stack(frames).astype(np.float16)
        _write_npy(cache_dir / MEL_FILENAME, mel)
        meta_payload: dict[str, Any] = {
            "id": recording_id,
            "species_pl": entry.get("species_pl", ""),
            "species_latin": entry.get("species_latin", ""),
            "cache_version": CACHE_VERSION,
            "config_hash": fingerprint,
            "sha256_audio": entry.get("sha256_audio", ""),
            "sample_rate": config.audio.sample_rate,
            "window_s": config.audio.window_s,
            "hop_s": config.audio.hop_s,
            "window_samples": plan.window_samples,
            "hop_samples": plan.hop_samples,
            "n_fft": config.spectrogram.n_fft,
            "hop_length": config.spectrogram.hop_length,
            "n_mels": config.spectrogram.n_mels,
            "fmin": config.spectrogram.fmin,
            "fmax": config.spectrogram.fmax,
            "starts_s": list(plan.starts_s()),
            "lengths": list(plan.lengths),
            "rms": [round(value, 6) for value in plan.rms],
            "kept": list(plan.kept),
            "valid_frames": valid,
            "shape": list(mel.shape),
            "n_windows": plan.n_windows,
            "n_kept": plan.n_kept,
            "duration_s": round(signal.size / config.audio.sample_rate, 3),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _write_meta(cache_dir / META_FILENAME, meta_payload)

        report.processed += 1
        report.audio_seconds += signal.size / config.audio.sample_rate
        report.add_windows(str(entry.get("species_pl", "")), plan.n_windows, plan.n_kept)
        LOGGER.debug(
            "[%s/%s] %s: %s okien, zachowane %s",
            index,
            len(entries),
            recording_id,
            plan.n_windows,
            plan.n_kept,
        )
        if report.processed % 50 == 0:
            LOGGER.info(
                "cache: %s/%s przetworzonych, %s pominiętych, %s błędów",
                report.processed,
                len(entries),
                report.skipped,
                len(report.errors),
            )

    LOGGER.info(
        "cache: gotowe — przetworzone %s, pominięte %s, błędy %s, okna %s/%s",
        report.processed,
        report.skipped,
        len(report.errors),
        report.windows_kept,
        report.windows_total,
    )
    return report


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_cache(config: Config, data_dir: Path) -> list[str]:
    """Return a list of human-readable problems; empty list means the cache is valid."""
    root = cache_root(data_dir)
    if not root.is_dir():
        return [f"brak katalogu cache: {root}"]
    fingerprint = config_hash(config)
    problems: list[str] = []
    for record_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        recording_id = record_dir.name
        meta = _read_meta(record_dir / META_FILENAME)
        mel_file = record_dir / MEL_FILENAME
        if meta is None:
            problems.append(f"{recording_id}: brak {META_FILENAME}")
            continue
        if not mel_file.is_file():
            problems.append(f"{recording_id}: brak {MEL_FILENAME}")
            continue
        if meta.get("cache_version") != CACHE_VERSION:
            problems.append(f"{recording_id}: cache_version != {CACHE_VERSION}")
        if meta.get("config_hash") != fingerprint:
            problems.append(f"{recording_id}: config_hash nieaktualny")
        array = np.load(mel_file, mmap_mode="r")
        if tuple(array.shape) != tuple(meta.get("shape") or ()):
            problems.append(
                f"{recording_id}: kształt {array.shape} != meta {tuple(meta.get('shape') or ())}"
            )
        if array.dtype != np.float16:
            problems.append(f"{recording_id}: dtype {array.dtype} != float16")
        if int(meta.get("n_kept") or 0) < 1:
            problems.append(f"{recording_id}: n_kept < 1")
    return problems


# --------------------------------------------------------------------------- #
# Dataset access
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CachedRecording:
    """One cached recording ready for training/evaluation."""

    id: str
    species_pl: str
    label: int
    mel: np.ndarray  # (n_kept, n_mels, frames) float32
    mask: np.ndarray  # (n_kept, frames) bool — zero-padded tail frames are False
    meta: Mapping[str, Any]


def list_cache(data_dir: Path) -> list[Path]:
    root = cache_root(data_dir)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if (path / MEL_FILENAME).is_file())


def load_recording(cache_dir: Path, labels: Mapping[str, int]) -> CachedRecording:
    """Load one cached recording as float32 with its label and length mask."""
    meta = _read_meta(cache_dir / META_FILENAME)
    if meta is None:
        raise DataError(f"{cache_dir}: brak {META_FILENAME}")
    mel_file = cache_dir / MEL_FILENAME
    if not mel_file.is_file():
        raise DataError(f"{cache_dir}: brak {MEL_FILENAME}")
    mel = np.load(mel_file).astype(np.float32)
    if mel.ndim != 3:
        raise DataError(f"{cache_dir}: oczekiwano (n, n_mels, frames), jest {mel.shape}")
    species_pl = str(meta.get("species_pl", ""))
    if species_pl not in labels:
        raise DataError(
            f"{cache_dir}: gatunek '{species_pl}' nie występuje w configs/species.yaml"
        )
    frames = mel.shape[-1]
    valid = [int(value) for value in meta.get("valid_frames", [])]
    mask = np.zeros(mel.shape[:2], dtype=bool)
    if valid:
        for row in range(min(mel.shape[0], len(valid))):
            mask[row, : min(valid[row], frames)] = True
    else:
        mask[:, :] = True
    return CachedRecording(
        id=str(meta.get("id", cache_dir.name)),
        species_pl=species_pl,
        label=int(labels[species_pl]),
        mel=mel,
        mask=mask,
        meta=meta,
    )


def iter_cached_windows(
    data_dir: Path,
    labels: Mapping[str, int],
) -> Iterator[tuple[np.ndarray, int, np.ndarray]]:
    """Yield ``(mel_window, label, mask_window)`` for every cached, kept window."""
    for record_dir in list_cache(data_dir):
        recording = load_recording(record_dir, labels)
        for index in range(recording.mel.shape[0]):
            yield recording.mel[index], recording.label, recording.mask[index]
