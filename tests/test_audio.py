"""Audio tests: decoding, resampling and high-pass filtering (SPEC FR-4).

All signals are generated locally by ``tests/support.write_wav`` — no real
recordings, no network (FR-15).
"""

from __future__ import annotations

import numpy as np
import pytest

from bird_recognizer.data.audio import highpass, load_audio, read_audio, resample
from bird_recognizer.errors import DataError
from support import write_wav


def test_read_audio_returns_mono_float32(tmp_dir) -> None:
    path = write_wav(tmp_dir / "a.wav", seconds=1.0)
    signal, sample_rate = read_audio(path)
    assert sample_rate == 32000
    assert signal.ndim == 1
    assert signal.dtype == np.float32
    assert abs(signal.size - 32000) <= 2


def test_load_audio_resamples_to_configured_rate(config, tmp_dir) -> None:
    path = write_wav(tmp_dir / "b.wav", seconds=2.0, sample_rate=22050)
    signal = load_audio(path, config.audio)
    assert abs(signal.size / config.audio.sample_rate - 2.0) < 0.05


def test_highpass_attenuates_below_cutoff(config) -> None:
    sample_rate = config.audio.sample_rate
    timeline = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    low = (0.5 * np.sin(2.0 * np.pi * 40.0 * timeline)).astype(np.float32)
    filtered = highpass(low, sample_rate, 100.0, 4)
    before = np.sqrt(np.mean(low ** 2))
    after = np.sqrt(np.mean(filtered ** 2))
    assert after < 0.2 * before


def test_highpass_passes_above_cutoff(config) -> None:
    sample_rate = config.audio.sample_rate
    timeline = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    high = (0.5 * np.sin(2.0 * np.pi * 2000.0 * timeline)).astype(np.float32)
    filtered = highpass(high, sample_rate, 100.0, 4)
    before = np.sqrt(np.mean(high ** 2))
    after = np.sqrt(np.mean(filtered ** 2))
    assert after > 0.8 * before


def test_highpass_returns_short_signal_unchanged() -> None:
    short = np.ones(10, dtype=np.float32)
    out = highpass(short, 32000, 100.0, 4)
    np.testing.assert_allclose(out, short)


def test_resample_is_noop_for_same_rate() -> None:
    signal = np.arange(16, dtype=np.float32)
    np.testing.assert_array_equal(resample(signal, 32000, 32000), signal)


def test_corrupt_file_raises_data_error(config, tmp_dir) -> None:
    path = tmp_dir / "bad.mp3"
    path.write_bytes(b"to nie jest audio")
    with pytest.raises(DataError, match="audio"):
        load_audio(path, config.audio)
