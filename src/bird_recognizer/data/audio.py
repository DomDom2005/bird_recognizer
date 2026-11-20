"""Audio loading and conditioning (SPEC FR-4, §10.1).

Decoding uses ``soundfile`` (libsndfile reads wav/flac/ogg **and mp3**); resampling
uses ``torchaudio.functional.resample``. The high-pass filter removes wind rumble
below 100 Hz with a Butterworth filter applied forward-backward (zero phase).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as torchaudio_functional
from scipy.signal import butter, filtfilt

from ..config import AudioConfig
from ..errors import DataError

LOGGER = logging.getLogger(__name__)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode a file to ``(mono float32, sample_rate)``; raise :class:`DataError`."""
    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise DataError(f"nie mogę odczytać audio {path.name}: {exc}") from exc
    if data.size == 0:
        raise DataError(f"pusty plik audio: {path.name}")
    mono = data.mean(axis=1).astype(np.float32, copy=False)
    return mono, int(sample_rate)


def resample(signal: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Resample to ``target_rate`` with torchaudio's polyphase filter."""
    if sample_rate == target_rate:
        return np.ascontiguousarray(signal, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32))
    resampled = torchaudio_functional.resample(tensor, sample_rate, target_rate)
    return resampled.numpy().astype(np.float32, copy=False)


def highpass(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass; returns the signal unchanged if too short."""
    if cutoff_hz <= 0:
        return np.ascontiguousarray(signal, dtype=np.float32)
    nyquist = sample_rate / 2.0
    if not 0 < cutoff_hz < nyquist:
        raise DataError(
            f"highpass_hz={cutoff_hz} musi być w zakresie (0, {nyquist}) Hz"
        )
    numerator, denominator = butter(order, cutoff_hz / nyquist, btype="high")
    padlen = 3 * (max(len(numerator), len(denominator)) - 1)
    if signal.size <= padlen:
        LOGGER.debug("Sygnał za krótki na filtr (%s <= %s) — pomijam HPF", signal.size, padlen)
        return np.ascontiguousarray(signal, dtype=np.float32)
    filtered = filtfilt(numerator, denominator, signal)
    return np.ascontiguousarray(filtered, dtype=np.float32)


def load_audio(path: Path, audio: AudioConfig) -> np.ndarray:
    """Read, mix to mono, resample to the configured rate and high-pass filter."""
    signal, sample_rate = read_audio(path)
    signal = resample(signal, sample_rate, audio.sample_rate)
    signal = highpass(
        signal,
        sample_rate=audio.sample_rate,
        cutoff_hz=audio.highpass_hz,
        order=audio.highpass_order,
    )
    LOGGER.debug(
        "%s: %.1f s @ %s Hz -> %.1f s @ %s Hz",
        path.name,
        signal.size / audio.sample_rate,
        sample_rate,
        signal.size / audio.sample_rate,
        audio.sample_rate,
    )
    return signal
