"""SpecAugment and mixup tests for the ``full`` augmentation preset (SPEC §10.3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from bird_recognizer.train.augment import augment_mel, mixup_batch, spec_augment


def _mel(n_mels: int = 16, frames: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_mels, frames)) + 0.5).astype(np.float32)


def _masked_rows(mel: np.ndarray) -> list[int]:
    return [row for row in range(mel.shape[0]) if np.all(mel[row] == 0.0)]


def _masked_columns(mel: np.ndarray) -> list[int]:
    return [col for col in range(mel.shape[1]) if np.all(mel[:, col] == 0.0)]


def test_spec_augment_limits_frequency_and_time_masks(config) -> None:
    out = spec_augment(_mel(), config.augment, np.random.default_rng(0))
    assert out.shape == (16, 64)
    rows = _masked_rows(out)
    columns = _masked_columns(out)
    assert rows, "oczekiwano co najmniej jednej zamaskowanej grupy pasm"
    # Masks may touch, so consecutive runs are checked instead of a strict count.
    assert len(rows) <= config.augment.freq_masks * config.augment.freq_mask_width
    assert len(columns) <= config.augment.time_masks * config.augment.time_mask_width


def test_spec_augment_is_deterministic_for_same_seed(config) -> None:
    mel = _mel()
    first = spec_augment(mel, config.augment, np.random.default_rng(11))
    second = spec_augment(mel, config.augment, np.random.default_rng(11))
    np.testing.assert_array_equal(first, second)


def test_spec_augment_keeps_original_untouched(config) -> None:
    mel = _mel()
    spec_augment(mel, config.augment, np.random.default_rng(0))
    assert np.count_nonzero(mel) > 0
    assert not np.array_equal(mel, np.zeros_like(mel))


def test_spec_augment_without_masks_is_identity(config) -> None:
    settings = replace(config.augment, freq_masks=0, time_masks=0)
    mel = _mel()
    np.testing.assert_array_equal(spec_augment(mel, settings, np.random.default_rng(0)), mel)


def test_full_preset_applies_specaugment(config) -> None:
    settings = replace(config.augment, preset="full", shift_s=0.0)
    out = augment_mel(_mel(), settings, np.random.default_rng(0))
    assert out.dtype == np.float32
    assert np.count_nonzero(out == 0.0) > 0
    assert np.count_nonzero(out) > 0


def test_basic_preset_does_not_mask(config) -> None:
    settings = replace(config.augment, preset="basic", shift_s=0.0)
    out = augment_mel(_mel(), settings, np.random.default_rng(0))
    assert np.count_nonzero(out == 0.0) == 0


# --------------------------------------------------------------------------- #
# Mixup
# --------------------------------------------------------------------------- #
def test_mixup_is_a_convex_combination_of_two_samples() -> None:
    inputs = torch.tensor([[0.0], [2.0], [4.0], [6.0]])
    labels = torch.tensor([0, 1, 2, 3])
    mixed, targets_a, targets_b, lam = mixup_batch(
        inputs, labels, 0.4, np.random.default_rng(3)
    )
    assert 0.0 < lam < 1.0
    torch.testing.assert_close(targets_a, labels)
    assert sorted(targets_b.tolist()) == [0, 1, 2, 3]
    for row in range(inputs.shape[0]):
        candidates = [
            lam * inputs[row] + (1.0 - lam) * inputs[other]
            for other in range(inputs.shape[0])
        ]
        assert any(torch.allclose(mixed[row], candidate) for candidate in candidates)


def test_mixup_with_zero_alpha_is_a_no_op() -> None:
    inputs = torch.arange(6.0).reshape(3, 2)
    labels = torch.tensor([0, 1, 2])
    mixed, targets_a, targets_b, lam = mixup_batch(
        inputs, labels, 0.0, np.random.default_rng(0)
    )
    assert lam == 1.0
    torch.testing.assert_close(mixed, inputs)
    torch.testing.assert_close(targets_a, labels)
    torch.testing.assert_close(targets_b, labels)


def test_mixup_of_single_sample_is_a_no_op() -> None:
    inputs = torch.tensor([[1.0, 2.0]])
    labels = torch.tensor([1])
    mixed, _targets_a, _targets_b, lam = mixup_batch(
        inputs, labels, 0.4, np.random.default_rng(0)
    )
    assert lam == 1.0
    torch.testing.assert_close(mixed, inputs)
