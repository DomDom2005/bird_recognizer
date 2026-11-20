"""ImageNet transfer backbones with a new head (SPEC FR-8).

``resnet18`` and ``efficientnet_b0`` from torchvision, adapted to a single-channel
log-mel input and a fresh ``n_classes`` head. ``--pretrained`` downloads the
ImageNet weights once into the torch cache and initialises the first convolution
by summing its RGB weights into one channel (a single-channel mel carries the same
energy the three channels would); ``--scratch`` builds the identical architecture
without any ImageNet weights, which is the "transfer vs from scratch" ablation.

Decision (journal, stage 5): the input is *not* normalised with the ImageNet
per-channel mean/std. The cache is already normalised per window, and a second,
RGB-specific normalisation would silently change the input statistics the
preprocessing promises in SPEC §10.1.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from ..errors import ModelError

LOGGER = logging.getLogger(__name__)

ARCHITECTURES = ("resnet18", "efficientnet_b0")


def _weights_enum(arch: str):
    from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

    if arch == "resnet18":
        return ResNet18_Weights.IMAGENET1K_V1
    if arch == "efficientnet_b0":
        return EfficientNet_B0_Weights.IMAGENET1K_V1
    raise ModelError(f"nieznana architektura '{arch}'; dostępne: {list(ARCHITECTURES)}")


def _build_backbone(arch: str, pretrained: bool) -> nn.Module:
    """torchvision model, ImageNet weights when requested (offline cache or download)."""
    from torchvision.models import get_model

    weights = _weights_enum(arch) if pretrained else None
    try:
        return get_model(arch, weights=weights)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelError(
            f"nie mogę zbudować '{arch}' z wagami ImageNet: {exc}. "
            "Wagi trafiają do ~/.cache/torch; bez sieci użyj --scratch."
        ) from exc


def _single_channel_conv(conv: nn.Conv2d, *, pretrained: bool) -> nn.Conv2d:
    """A 1-channel copy of ``conv``; RGB weights are summed when pre-trained."""
    new = nn.Conv2d(
        1,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
    return new


def _adapt_input(model: nn.Module, arch: str, *, pretrained: bool) -> None:
    if arch == "resnet18":
        model.conv1 = _single_channel_conv(model.conv1, pretrained=pretrained)
    else:  # efficientnet_b0: features[0][0] is the stem convolution
        stem = model.features[0][0]
        model.features[0][0] = _single_channel_conv(stem, pretrained=pretrained)


def _replace_head(model: nn.Module, arch: str, n_classes: int) -> None:
    if arch == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        # Classifier is Dropout + Linear; keep the dropout, replace the linear.
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)


class TransferClassifier(nn.Module):
    """Thin wrapper so every model family exposes ``forward`` and ``n_parameters``."""

    def __init__(self, backbone: nn.Module, *, arch: str, pretrained: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.arch = arch
        self.pretrained = bool(pretrained)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_transfer(
    n_classes: int,
    *,
    arch: str = "resnet18",
    pretrained: bool = False,
) -> TransferClassifier:
    """Build a transfer model; ``pretrained=False`` is the offline-safe default."""
    if arch not in ARCHITECTURES:
        raise ModelError(f"nieznana architektura '{arch}'; dostępne: {list(ARCHITECTURES)}")
    if n_classes < 2:
        raise ValueError("n_classes musi być >= 2")
    backbone = _build_backbone(arch, pretrained)
    _adapt_input(backbone, arch, pretrained=pretrained)
    _replace_head(backbone, arch, n_classes)
    LOGGER.debug(
        "transfer: %s, pretrained=%s, %s parametrów",
        arch,
        pretrained,
        sum(parameter.numel() for parameter in backbone.parameters()),
    )
    return TransferClassifier(backbone, arch=arch, pretrained=pretrained)
