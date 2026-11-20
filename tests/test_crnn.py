"""CRNN tests: variable time length, pooling and the length mask (SPEC FR-7)."""

from __future__ import annotations

import pytest
import torch

from bird_recognizer.models.crnn import (
    CRNN,
    AttentionPooling,
    MeanPooling,
    frames_mask,
)
from bird_recognizer.models_registry import available_models, build_model


def test_output_shape_for_three_second_window() -> None:
    model = CRNN(n_classes=12)
    assert model(torch.zeros(2, 1, 128, 188)).shape == (2, 12)


def test_variable_time_length_is_supported() -> None:
    model = CRNN(n_classes=4)
    for frames in (188, 50, 17):
        assert model(torch.zeros(1, 1, 128, frames)).shape == (1, 4)


def test_frames_mask_maps_input_frames_to_backbone_steps() -> None:
    mask = frames_mask(torch.tensor([188, 50, 1]), steps=12)
    assert mask.shape == (3, 12)
    assert [int(row.sum()) for row in mask] == [11, 3, 1]


def test_mean_pooling_ignores_padded_steps() -> None:
    sequence = torch.zeros(1, 4, 2)
    sequence[0, 0] = torch.tensor([1.0, 1.0])
    sequence[0, 1] = torch.tensor([3.0, 3.0])
    sequence[0, 2:] = 100.0
    mask = torch.tensor([[True, True, False, False]])
    torch.testing.assert_close(
        MeanPooling()(sequence, mask), torch.tensor([[2.0, 2.0]])
    )


def test_attention_pooling_ignores_padded_steps() -> None:
    torch.manual_seed(0)
    pooling = AttentionPooling(4)
    sequence = torch.randn(1, 5, 4)
    contaminated = sequence.clone()
    contaminated[0, 2:] = 1e4
    mask = torch.tensor([[True, True, False, False, False]])
    torch.testing.assert_close(
        pooling(contaminated, mask), pooling(sequence, mask)
    )


def test_full_length_mask_matches_no_mask() -> None:
    torch.manual_seed(0)
    model = CRNN(n_classes=3, pool="attention")
    model.eval()
    inputs = torch.randn(2, 1, 128, 188)
    lengths = torch.tensor([188, 188])
    with torch.no_grad():
        torch.testing.assert_close(model(inputs, lengths), model(inputs))


def test_pooling_variants_give_different_outputs() -> None:
    inputs = torch.randn(3, 1, 128, 188)
    torch.manual_seed(0)
    mean_model = CRNN(n_classes=5, pool="mean")
    torch.manual_seed(0)
    attention_model = CRNN(n_classes=5, pool="attention")
    mean_model.eval()
    attention_model.eval()
    assert isinstance(mean_model.pool, MeanPooling)
    assert isinstance(attention_model.pool, AttentionPooling)
    with torch.no_grad():
        assert not torch.allclose(mean_model(inputs), attention_model(inputs))


def test_unknown_pooling_is_rejected() -> None:
    with pytest.raises(ValueError, match="pool"):
        CRNN(n_classes=3, pool="max")


def test_unknown_recurrent_layer_is_rejected() -> None:
    with pytest.raises(ValueError, match="rnn"):
        CRNN(n_classes=3, rnn="transformer")


def test_lstm_variant_has_expected_output_shape() -> None:
    model = CRNN(n_classes=6, rnn="lstm")
    assert model(torch.zeros(2, 1, 128, 50)).shape == (2, 6)


def test_registry_builds_crnn_with_options() -> None:
    assert "crnn" in available_models()
    model = build_model("crnn", 5, pool="attention", rnn="lstm")
    assert isinstance(model, CRNN)
    assert model(torch.zeros(1, 1, 128, 188)).shape == (1, 5)
