"""log-mel spectrogram with per-window normalisation (SPEC §10.1).

Parameters come from ``configs/default.yaml``; the transforms are cached per process
so the cache build does not rebuild them for every window.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torchaudio.transforms as transforms

from ..config import Config, SpectrogramConfig
from ..errors import DataError

LOGGER = logging.getLogger(__name__)

#: Bump when the spectrogram definition changes; stored in every cache's meta.
CACHE_VERSION = 1


@lru_cache(maxsize=8)
def _transforms(
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    power: float,
    top_db: float,
) -> tuple[Any, Any]:
    mel = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=power,
        center=True,
        pad_mode="reflect",
    )
    to_db = transforms.AmplitudeToDB(stype="power", top_db=top_db)
    return mel, to_db


def mel_spectrogram(
    window: np.ndarray,
    sample_rate: int,
    config: SpectrogramConfig,
) -> np.ndarray:
    """Return a normalised log-mel spectrogram of shape ``(n_mels, frames)``."""
    if config.fmax >= sample_rate / 2:
        raise DataError(
            f"fmax={config.fmax} musi być poniżej Nyquista ({sample_rate / 2} Hz)"
        )
    mel, to_db = _transforms(
        sample_rate,
        config.n_fft,
        config.win_length,
        config.hop_length,
        config.n_mels,
        config.fmin,
        config.fmax,
        config.power,
        config.top_db,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
    spectrogram = to_db(mel(tensor))
    array = spectrogram.numpy().astype(np.float32, copy=False)
    if config.normalise:
        mean = float(array.mean())
        std = float(array.std())
        array = (
            (array - mean) / std
            if std > 1e-6
            else np.zeros_like(array, dtype=np.float32)
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def config_hash(config: Config) -> str:
    """Stable fingerprint of everything that changes the cached spectrograms."""
    audio = config.audio
    spectrogram = config.spectrogram
    gating = config.gating
    payload = {
        "cache_version": CACHE_VERSION,
        "sample_rate": audio.sample_rate,
        "highpass_hz": audio.highpass_hz,
        "highpass_order": audio.highpass_order,
        "window_s": audio.window_s,
        "hop_s": audio.hop_s,
        "n_fft": spectrogram.n_fft,
        "win_length": spectrogram.win_length,
        "hop_length": spectrogram.hop_length,
        "n_mels": spectrogram.n_mels,
        "fmin": spectrogram.fmin,
        "fmax": spectrogram.fmax,
        "power": spectrogram.power,
        "top_db": spectrogram.top_db,
        "normalise": spectrogram.normalise,
        "abs_floor": gating.abs_floor,
        "rel_floor": gating.rel_floor,
        "percentile": gating.percentile,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
