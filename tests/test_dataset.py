"""Cache build, verification and dataset access (SPEC FR-4, §7.4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bird_recognizer.data.dataset import (
    MEL_FILENAME,
    META_FILENAME,
    WindowDataset,
    build_cache,
    build_window_index,
    iter_cached_windows,
    list_cache,
    load_recording,
    recording_cache_dir,
    verify_cache,
)
from bird_recognizer.data.spectrogram import config_hash
from bird_recognizer.errors import DataError
from support import make_catalog_with_audio


def _data_dir(tmp_dir: Path) -> Path:
    data_dir = tmp_dir / "data"
    make_catalog_with_audio(data_dir)
    return data_dir


def test_build_creates_float16_cache(config, tmp_dir: Path) -> None:
    data_dir = _data_dir(tmp_dir)
    report = build_cache(config, data_dir)

    assert report.processed == 4
    assert report.windows_total > 0
    assert report.windows_kept > 0
    assert not report.errors
    assert len(list_cache(data_dir)) == 4

    cache_dir = list_cache(data_dir)[0]
    mel = np.load(cache_dir / MEL_FILENAME)
    assert mel.dtype == np.float16
    assert mel.ndim == 3
    meta = json.loads((cache_dir / META_FILENAME).read_text(encoding="utf-8"))
    assert meta["config_hash"] == config_hash(config)
    assert meta["n_kept"] == mel.shape[0]
    assert len(meta["rms"]) == meta["n_windows"]
    assert meta["shape"] == list(mel.shape)
    assert len(meta["valid_frames"]) == mel.shape[0]


def test_second_build_skips_current_cache(config, tmp_dir: Path) -> None:
    data_dir = _data_dir(tmp_dir)
    build_cache(config, data_dir)
    report = build_cache(config, data_dir)
    assert report.processed == 0
    assert report.skipped == 4
    assert report.skipped_reasons == {"up_to_date": 4}


def test_force_rebuild_is_deterministic(config, tmp_dir: Path) -> None:
    data_dir = _data_dir(tmp_dir)
    build_cache(config, data_dir)
    first = {
        path.parent.name: np.load(path / MEL_FILENAME) for path in list_cache(data_dir)
    }
    build_cache(config, data_dir, force=True)
    second = {
        path.parent.name: np.load(path / MEL_FILENAME) for path in list_cache(data_dir)
    }
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_missing_audio_is_skipped(config, tmp_dir: Path) -> None:
    data_dir = _data_dir(tmp_dir)
    (data_dir / "raw" / "XC100.wav").unlink()
    report = build_cache(config, data_dir)
    assert report.skipped_reasons.get("missing_audio") == 1
    assert report.processed == 3


def test_corrupt_audio_records_error(config, tmp_dir: Path) -> None:
    data_dir = _data_dir(tmp_dir)
    (data_dir / "raw" / "XC100.wav").write_bytes(b"to nie audio")
    report = build_cache(config, data_dir)
    assert report.processed == 3
    assert report.errors and report.errors[0]["id"] == "XC100"


def test_verify_clean_then_tampered(config, tmp_dir: Path) -> None:
    data_dir = _data_dir(tmp_dir)
    build_cache(config, data_dir)
    assert verify_cache(config, data_dir) == []

    meta_file = list_cache(data_dir)[0] / META_FILENAME
    payload = json.loads(meta_file.read_text(encoding="utf-8"))
    payload["config_hash"] = "stary-hash"
    meta_file.write_text(json.dumps(payload), encoding="utf-8")
    problems = verify_cache(config, data_dir)
    assert any("config_hash" in problem for problem in problems)


def test_verify_reports_missing_cache(config, tmp_dir: Path) -> None:
    problems = verify_cache(config, tmp_dir / "pusto")
    assert problems and "brak katalogu cache" in problems[0]


def test_load_recording_gives_label_and_mask(config, tmp_dir: Path) -> None:
    data_dir = _data_dir(tmp_dir)
    build_cache(config, data_dir)
    labels = config.species_labels()
    recording = load_recording(recording_cache_dir(data_dir, "XC100"), labels)
    assert recording.label == labels["kos"]
    assert recording.mel.dtype == np.float32
    assert recording.mask.shape == recording.mel.shape[:2]
    assert recording.mask.dtype == bool
    assert recording.mask.any()


def test_unknown_species_is_an_error(config, tmp_dir: Path) -> None:
    data_dir = _data_dir(tmp_dir)
    build_cache(config, data_dir, limit=1)
    with pytest.raises(DataError, match="species.yaml"):
        load_recording(recording_cache_dir(data_dir, "XC100"), {"bogatka": 0})


def test_iter_cached_windows_counts_kept_windows(config, tmp_dir: Path) -> None:
    data_dir = _data_dir(tmp_dir)
    report = build_cache(config, data_dir)
    labels = config.species_labels()
    items = list(iter_cached_windows(data_dir, labels))
    assert len(items) == report.windows_kept
    window, label, mask = items[0]
    assert window.shape[0] == config.spectrogram.n_mels
    assert isinstance(label, int)
    assert mask.dtype == bool


def test_window_dataset_with_lengths_reports_padding(config, tmp_dir: Path) -> None:
    data_dir = tmp_dir / "data-lengths"
    make_catalog_with_audio(data_dir, seconds=4.0)
    build_cache(config, data_dir)
    refs = build_window_index(config, data_dir)
    dataset = WindowDataset(refs, with_lengths=True)
    samples = [dataset[index] for index in range(len(dataset))]
    assert all(len(sample) == 3 for sample in samples), "oczekiwano (mel, label, długość)"
    frames = samples[0][0].shape[-1]
    lengths = [sample[2] for sample in samples]
    assert max(lengths) == frames
    assert min(lengths) < frames, "krótkie ostatnie okno powinno mieć mniej ważnych ramek"
    assert len(WindowDataset(refs)[0]) == 2


def test_build_without_catalog_raises(config, tmp_dir: Path) -> None:
    with pytest.raises(DataError, match="Brak katalogu"):
        build_cache(config, tmp_dir / "data")


def test_species_filter_limits_work(config, tmp_dir: Path) -> None:
    data_dir = tmp_dir / "data-kos"
    make_catalog_with_audio(data_dir)
    report = build_cache(config, data_dir, species=["kos"])
    assert report.processed == 2
    assert set(report.per_species) == {"kos"}


def test_limit_limits_work(config, tmp_dir: Path) -> None:
    data_dir = tmp_dir / "data-limit"
    make_catalog_with_audio(data_dir)
    report = build_cache(config, data_dir, limit=1)
    assert report.processed == 1


def test_gating_rejects_silent_windows(config, tmp_dir: Path) -> None:
    data_dir = tmp_dir / "data-gated"
    make_catalog_with_audio(
        data_dir,
        species=(("kos", "Turdus merula"),),
        per_species=1,
        seconds=8.0,
        kind="silence-then-tone",
    )
    report = build_cache(config, data_dir)
    assert report.processed == 1
    assert report.windows_total > report.windows_kept
    assert report.rejected_ratio > 0.0
    assert report.per_species["kos"]["windows"] == report.windows_total
