"""Windowing and silence gating (SPEC FR-4, §10.2).

Windows are cut with a fixed hop; the short last window is kept (and later padded),
because xeno-canto recordings rarely end on a clean boundary. A window is kept only
if its RMS clears ``max(abs_floor, rel_floor * percentile(rms))`` — otherwise the
model would learn silence.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..config import AudioConfig, GatingConfig
from ..errors import DataError

LOGGER = logging.getLogger(__name__)

#: Windows shorter than this fraction of the target are dropped (tail noise).
MIN_WINDOW_FRACTION = 0.1


@dataclass(frozen=True)
class WindowPlan:
    """Where the windows are and which ones contain sound."""

    starts: tuple[int, ...]
    lengths: tuple[int, ...]
    rms: tuple[float, ...]
    kept: tuple[bool, ...]
    window_samples: int
    hop_samples: int
    sample_rate: int

    @property
    def n_windows(self) -> int:
        return len(self.starts)

    @property
    def n_kept(self) -> int:
        return sum(self.kept)

    @property
    def kept_ratio(self) -> float:
        return self.n_kept / self.n_windows if self.n_windows else 0.0

    def starts_s(self) -> tuple[float, ...]:
        return tuple(round(start / self.sample_rate, 3) for start in self.starts)


def _rms(window: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))


def plan_windows(
    signal: np.ndarray,
    audio: AudioConfig,
    gating: GatingConfig,
) -> WindowPlan:
    """Cut the signal into windows and mark the ones that contain sound."""
    if signal.size == 0:
        raise DataError("pusty sygnał audio")
    window_samples = int(round(audio.window_s * audio.sample_rate))
    hop_samples = int(round(audio.hop_s * audio.sample_rate))
    if window_samples <= 0 or hop_samples <= 0:
        raise DataError("window_s i hop_s muszą dawać dodatnią liczbę próbek")

    starts = list(range(0, signal.size, hop_samples))
    min_length = max(1, int(MIN_WINDOW_FRACTION * window_samples))
    lengths: list[int] = []
    used_starts: list[int] = []
    for start in starts:
        length = min(window_samples, signal.size - start)
        if length >= min_length:
            used_starts.append(start)
            lengths.append(length)
    if not used_starts:
        raise DataError("sygnał krótszy niż minimalne okno")

    rms = tuple(
        _rms(signal[start : start + length])
        for start, length in zip(used_starts, lengths, strict=True)
    )
    reference = float(np.percentile(np.asarray(rms, dtype=np.float64), gating.percentile))
    threshold = max(gating.abs_floor, gating.rel_floor * reference)
    kept = tuple(value >= threshold and value > 0.0 for value in rms)
    return WindowPlan(
        starts=tuple(used_starts),
        lengths=tuple(lengths),
        rms=rms,
        kept=kept,
        window_samples=window_samples,
        hop_samples=hop_samples,
        sample_rate=audio.sample_rate,
    )


def iter_window_signals(
    signal: np.ndarray,
    plan: WindowPlan,
    *,
    only_kept: bool = True,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, window)``; short windows are zero-padded to ``window_samples``."""
    for index, (start, length, keep) in enumerate(
        zip(plan.starts, plan.lengths, plan.kept, strict=True)
    ):
        if only_kept and not keep:
            continue
        window = signal[start : start + length]
        if length < plan.window_samples:
            window = np.pad(window, (0, plan.window_samples - length))
        yield index, window.astype(np.float32, copy=False)


def valid_frames(length_samples: int, hop_length: int) -> int:
    """Frames of a window that come from real audio (zero padding excluded)."""
    return max(1, 1 + length_samples // hop_length)
