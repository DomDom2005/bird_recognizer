"""Mel augmentations (SPEC §10.3, ``--aug {none,basic,full}``).

Applied to the cached, per-window normalised log-mel frames, so a global ``gain``
would cancel out under normalisation and is deliberately not implemented (see
docs/ai-journal.md, odstępstwo 23). ``basic`` is a time shift plus additive noise;
``full`` adds SpecAugment masks per window and mixup at the batch level (stage 5).
Validation and test data always stay clean — the training loop is the only caller.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..config import AugmentConfig


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


def spec_augment(
    mel: np.ndarray,
    config: AugmentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Zero out frequency bands and time spans of one window (SpecAugment).

    The cache is normalised per window, so zero is the per-band mean — the usual
    SpecAugment fill value on normalised features. A mask never covers the whole
    axis: it would erase the window instead of hiding a part of it.
    """
    out = mel.copy()
    n_mels, frames = out.shape
    for _ in range(config.freq_masks):
        width = int(rng.integers(1, config.freq_mask_width + 1))
        if 0 < width < n_mels:
            start = int(rng.integers(0, n_mels - width + 1))
            out[start : start + width, :] = 0.0
    for _ in range(config.time_masks):
        width = int(rng.integers(1, config.time_mask_width + 1))
        if 0 < width < frames:
            start = int(rng.integers(0, frames - width + 1))
            out[:, start : start + width] = 0.0
    return out


def mixup_batch(
    inputs: Tensor,
    labels: Tensor,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Mix a training batch with a random permutation of itself (mixup).

    Returns ``(mixed_inputs, targets_a, targets_b, lam)`` so the caller can use the
    convex combination of two cross-entropies. ``alpha <= 0`` or a batch of one is a
    no-op (``lam == 1.0``), which keeps the ablation "mixup off" honest.
    """
    if alpha <= 0 or inputs.shape[0] < 2:
        return inputs, labels, labels, 1.0
    lam = float(rng.beta(alpha, alpha))
    order = torch.from_numpy(rng.permutation(inputs.shape[0]))
    mixed = (lam * inputs + (1.0 - lam) * inputs[order]).to(inputs.dtype)
    return mixed, labels, labels[order], lam


def augment_mel(
    mel: np.ndarray,
    config: AugmentConfig,
    rng: np.random.Generator,
    *,
    frames_per_second: float = 62.5,
) -> np.ndarray:
    """Apply the configured augmentation preset to one window."""
    if config.preset == "none":
        return mel
    out = shift_time(mel, int(round(config.shift_s * frames_per_second)), rng)
    out = add_noise(out, config.noise_snr_min, config.noise_snr_max, rng)
    if config.preset == "full":
        out = spec_augment(out, config, rng)
    return out.astype(np.float32)
