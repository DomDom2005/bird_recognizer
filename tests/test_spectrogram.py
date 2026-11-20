"""log-mel spectrogram tests (SPEC §10.1)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bird_recognizer.data.spectrogram import config_hash, mel_spectrogram
from bird_recognizer.errors import DataError


def _window(config, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = int(config.audio.window_s * config.audio.sample_rate)
    return rng.standard_normal(samples).astype(np.float32)


def test_shape_dtype_and_finiteness(config) -> None:
    mel = mel_spectrogram(_window(config), config.audio.sample_rate, config.spectrogram)
    assert mel.shape[0] == config.spectrogram.n_mels
    assert 180 <= mel.shape[1] <= 200
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all()


def test_per_window_normalisation(config) -> None:
    mel = mel_spectrogram(_window(config), config.audio.sample_rate, config.spectrogram)
    assert abs(float(mel.mean())) < 1e-3
    assert abs(float(mel.std()) - 1.0) < 1e-2


def test_deterministic_for_same_input(config) -> None:
    window = _window(config)
    first = mel_spectrogram(window, config.audio.sample_rate, config.spectrogram)
    second = mel_spectrogram(window, config.audio.sample_rate, config.spectrogram)
    np.testing.assert_array_equal(first, second)


def test_silent_window_becomes_zeros(config) -> None:
    samples = int(config.audio.window_s * config.audio.sample_rate)
    mel = mel_spectrogram(
        np.zeros(samples, dtype=np.float32),
        config.audio.sample_rate,
        config.spectrogram,
    )
    assert np.allclose(mel, 0.0)


def test_config_hash_is_stable_and_sensitive(config) -> None:
    assert config_hash(config) == config_hash(config)
    other_window = replace(config, audio=replace(config.audio, window_s=5.0))
    assert config_hash(other_window) != config_hash(config)
    other_gating = replace(config, gating=replace(config.gating, rel_floor=0.2))
    assert config_hash(other_gating) != config_hash(config)


def test_fmax_above_nyquist_is_rejected_at_runtime(config) -> None:
    broken = replace(config.spectrogram, fmax=20000.0)
    with pytest.raises(DataError, match="Nyquista"):
        mel_spectrogram(_window(config), config.audio.sample_rate, broken)
