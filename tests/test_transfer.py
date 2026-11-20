"""Transfer model tests (SPEC FR-8); no ImageNet download in the test suite."""

from __future__ import annotations

import pytest
import torch

from bird_recognizer.errors import ModelError
from bird_recognizer.models import transfer
from bird_recognizer.models.transfer import _build_backbone as original_build_backbone
from bird_recognizer.models.transfer import build_transfer
from bird_recognizer.models_registry import available_models, build_model


@pytest.fixture(autouse=True)
def _offline_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the same architectures without ImageNet weights (socket-free tests)."""

    def fake_backbone(arch: str, pretrained: bool) -> torch.nn.Module:
        from torchvision.models import get_model

        return get_model(arch, weights=None)

    monkeypatch.setattr(transfer, "_build_backbone", fake_backbone)


def test_resnet18_scratch_uses_single_channel_input() -> None:
    model = build_transfer(12, arch="resnet18", pretrained=False)
    assert model.backbone.conv1.in_channels == 1
    assert model.pretrained is False
    assert model(torch.zeros(2, 1, 128, 188)).shape == (2, 12)


def test_efficientnet_b0_scratch_uses_single_channel_input() -> None:
    model = build_transfer(7, arch="efficientnet_b0", pretrained=False)
    assert model.backbone.features[0][0].in_channels == 1
    assert model(torch.zeros(1, 1, 128, 188)).shape == (1, 7)


def test_pretrained_first_conv_sums_rgb_channels() -> None:
    from torchvision.models import get_model

    torch.manual_seed(0)
    reference = get_model("resnet18", weights=None)
    expected = reference.conv1.weight.sum(dim=1, keepdim=True)
    torch.manual_seed(0)
    model = build_transfer(5, arch="resnet18", pretrained=True)
    assert model.pretrained is True
    torch.testing.assert_close(model.backbone.conv1.weight, expected)


def test_head_output_shape_follows_number_of_classes() -> None:
    model = build_transfer(3, arch="resnet18")
    assert model(torch.zeros(3, 1, 128, 50)).shape == (3, 3)


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ModelError, match="architektura"):
        build_transfer(3, arch="vgg16")


def test_download_failure_becomes_a_readable_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchvision.models as tv

    def failing(*_args: object, **_kwargs: object) -> None:
        raise OSError("brak sieci")

    monkeypatch.setattr(tv, "get_model", failing)
    with pytest.raises(ModelError, match="--scratch"):
        original_build_backbone("resnet18", pretrained=True)


def test_registry_builds_transfer_offline_by_default() -> None:
    assert "transfer" in available_models()
    model = build_model("transfer", 4)
    assert model.pretrained is False
    assert model(torch.zeros(1, 1, 128, 188)).shape == (1, 4)


def test_registry_accepts_transfer_options() -> None:
    model = build_model("transfer", 4, arch="efficientnet_b0", pretrained=False)
    assert model.arch == "efficientnet_b0"
