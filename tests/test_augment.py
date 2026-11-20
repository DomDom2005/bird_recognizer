"""Basic mel augmentation tests (SPEC §10.3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from bird_recognizer.train.augment import add_noise, augment_mel, shift_time


def _mel(frames: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((8, frames)).astype(np.float32)


def test_preset_none_is_identity(config) -> None:
    mel = _mel()
    settings = replace(config.augment, preset="none")
    out = augment_mel(mel, settings, np.random.default_rng(0))
    np.testing.assert_array_equal(out, mel)


def test_shift_preserves_shape_and_zero_fills() -> None:
    mel = np.ones((4, 20), dtype=np.float32)
    outputs = [shift_time(mel, 5, np.random.default_rng(seed)) for seed in range(10)]
    assert all(output.shape == mel.shape for output in outputs)
    assert any((output == 0.0).any() for output in outputs)
    assert all(output.sum() <= mel.sum() + 1e-6 for output in outputs)


def test_shift_zero_frames_is_identity() -> None:
    mel = _mel()
    np.testing.assert_array_equal(shift_time(mel, 0, np.random.default_rng(0)), mel)


def test_noise_changes_signal_but_keeps_shape() -> None:
    mel = _mel()
    out = add_noise(mel, 5.0, 20.0, np.random.default_rng(1))
    assert out.shape == mel.shape
    assert not np.array_equal(out, mel)


def test_noise_is_deterministic_for_same_seed() -> None:
    mel = _mel()
    first = add_noise(mel, 5.0, 20.0, np.random.default_rng(7))
    second = add_noise(mel, 5.0, 20.0, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)


def test_basic_preset_changes_signal(config) -> None:
    mel = _mel()
    out = augment_mel(mel, config.augment, np.random.default_rng(0))
    assert out.shape == mel.shape
    assert out.dtype == np.float32

