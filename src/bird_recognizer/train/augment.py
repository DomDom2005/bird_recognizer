"""Basic mel augmentations (SPEC §10.3, wariant ``basic``).

Applied to the cached, per-window normalised log-mel frames, so they cannot change
the global level: ``gain`` would cancel out under normalisation and is therefore
not implemented yet (see docs/ai-journal.md, odstępstwo 23). Mixup and SpecAugment
belong to stage 5 (``preset: full`` currently falls back to the basic set).
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import AugmentConfig

LOGGER = logging.getLogger(__name__)

_FULL_WARNED = False


def shift_time(
    mel: np.ndarray,
    max_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shift the time axis by up to ``max_frames`` (zero fill, no wrap-around)."""
    if max_frames <= 0:
        return mel
    shift = int(rng.integers(-max_frames, max_frames + 1))
    if shift == 0:
        return mel
    out = np.zeros_like(mel)
    if shift > 0:
        out[:, shift:] = mel[:, :-shift]
    else:
        out[:, :shift] = mel[:, -shift:]
    return out


def add_noise(
    mel: np.ndarray,
    snr_min_db: float,
    snr_max_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Additive white noise scaled to a random SNR (dB) relative to window power."""
    snr_db = float(rng.uniform(snr_min_db, snr_max_db))
    power = float(np.mean(mel**2))
    if power <= 0.0:
        return mel
    noise_std = float(np.sqrt(power / (10.0 ** (snr_db / 10.0))))
    noise = rng.normal(0.0, noise_std, size=mel.shape)
    return (mel + noise).astype(np.float32)


def augment_mel(
    mel: np.ndarray,
    config: AugmentConfig,
    rng: np.random.Generator,
    *,
    frames_per_second: float = 62.5,
) -> np.ndarray:
    """Apply the configured augmentation preset to one window."""
    global _FULL_WARNED
    if config.preset == "none":
        return mel
    if config.preset == "full" and not _FULL_WARNED:
        LOGGER.warning(
            "preset 'full' nie jest jeszcze zaimplementowany (mixup i SpecAugment "
            "dochodzą w etapie 5) — używam wariantu basic"
        )
        _FULL_WARNED = True
    out = shift_time(mel, int(round(config.shift_s * frames_per_second)), rng)
    out = add_noise(out, config.noise_snr_min, config.noise_snr_max, rng)
    return out.astype(np.float32)
