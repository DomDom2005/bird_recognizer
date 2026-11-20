"""CNN model and registry tests (SPEC FR-6, §15.9)."""

from __future__ import annotations

import pytest
import torch

from bird_recognizer.errors import ModelError
from bird_recognizer.models.cnn import SmallCNN
from bird_recognizer.models_registry import available_models, build_model, register_model


def test_output_shape_for_three_second_window() -> None:
    model = SmallCNN(n_classes=12)
    logits = model(torch.zeros(2, 1, 128, 188))
    assert logits.shape == (2, 12)


def test_variable_time_length_is_supported() -> None:
    model = SmallCNN(n_classes=4)
    for frames in (188, 50, 17):
        logits = model(torch.zeros(1, 1, 128, frames))
        assert logits.shape == (1, 4)


def test_parameter_count_is_small() -> None:
    model = SmallCNN(n_classes=12)
    assert 300_000 < model.n_parameters() < 600_000


def test_single_class_is_rejected() -> None:
    with pytest.raises(ValueError):
        SmallCNN(n_classes=1)


def test_registry_builds_cnn() -> None:
    assert "cnn" in available_models()
    model = build_model("cnn", 7)
    assert isinstance(model, SmallCNN)


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(ModelError, match="nieznany model"):
        build_model("transformer", 3)


def test_registry_allows_later_registration() -> None:
    register_model("tiny-test", lambda n: torch.nn.Linear(4, n))
    try:
        model = build_model("tiny-test", 3)
        assert model.out_features == 3
    finally:
        from bird_recognizer import models_registry

        models_registry._FACTORIES.pop("tiny-test", None)
