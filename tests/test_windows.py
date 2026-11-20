"""Window planning and silence gating (SPEC FR-4, §10.2)."""

from __future__ import annotations

import numpy as np
import pytest

from bird_recognizer.data.windows import iter_window_signals, plan_windows, valid_frames
from bird_recognizer.errors import DataError


def _constant(seconds: float, sample_rate: int, value: float = 0.1) -> np.ndarray:
    return np.full(int(seconds * sample_rate), value, dtype=np.float32)


def test_ten_seconds_gives_seven_windows_with_short_tail(config) -> None:
    sample_rate = config.audio.sample_rate
    signal = _constant(10.0, sample_rate)
    plan = plan_windows(signal, config.audio, config.gating)
    assert plan.n_windows == 7
    assert plan.starts_s()[0] == 0.0
    assert plan.starts_s()[1] == 1.5
    assert plan.lengths[-1] == int(1.0 * sample_rate)
    assert all(plan.kept)


def test_three_seconds_keeps_short_tail(config) -> None:
    sample_rate = config.audio.sample_rate
    plan = plan_windows(_constant(3.0, sample_rate), config.audio, config.gating)
    assert plan.n_windows == 2
    assert plan.lengths[-1] == int(1.5 * sample_rate)


def test_gating_drops_silence(config) -> None:
    sample_rate = config.audio.sample_rate
    total = np.zeros(int(8.0 * sample_rate), dtype=np.float32)
    timeline = np.arange(total.size, dtype=np.float32) / sample_rate
    tone = 0.5 * np.sin(2.0 * np.pi * 1000.0 * timeline)
    total[int(4.0 * sample_rate) :] = tone[int(4.0 * sample_rate) :]
    plan = plan_windows(total, config.audio, config.gating)
    assert not plan.kept[0]
    assert plan.kept[-1]
    assert plan.n_kept < plan.n_windows


def test_all_silence_is_rejected(config) -> None:
    signal = np.zeros(int(6.0 * config.audio.sample_rate), dtype=np.float32)
    plan = plan_windows(signal, config.audio, config.gating)
    assert plan.n_kept == 0
    assert plan.kept_ratio == 0.0


def test_windows_are_padded_to_full_length(config) -> None:
    sample_rate = config.audio.sample_rate
    signal = _constant(10.0, sample_rate)
    plan = plan_windows(signal, config.audio, config.gating)
    windows = list(iter_window_signals(signal, plan))
    assert len(windows) == plan.n_kept
    assert all(window.size == plan.window_samples for _, window in windows)


def test_only_kept_false_yields_all_windows(config) -> None:
    sample_rate = config.audio.sample_rate
    signal = _constant(10.0, sample_rate)
    plan = plan_windows(signal, config.audio, config.gating)
    all_windows = list(iter_window_signals(signal, plan, only_kept=False))
    assert len(all_windows) == plan.n_windows


def test_empty_signal_raises(config) -> None:
    with pytest.raises(DataError):
        plan_windows(np.array([], dtype=np.float32), config.audio, config.gating)


def test_valid_frames_math() -> None:
    assert valid_frames(96000, 512) == 188
    assert valid_frames(48000, 512) == 94
    assert valid_frames(1, 512) == 1
