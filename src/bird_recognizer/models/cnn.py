"""Small CNN baseline without recurrence (SPEC FR-6).

Four convolutional blocks (32/64/128/256) with batch norm and 2x2 pooling, global
average pooling and dropout. It accepts any time length thanks to adaptive pooling,
so 3 s and 5 s windows both work.
"""

from __future__ import annotations

from torch import nn


class SmallCNN(nn.Module):
    """Convolutional baseline; recording prediction is the mean over windows."""

    def __init__(
        self,
        n_classes: int,
        *,
        in_channels: int = 1,
        dropout: float = 0.3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes musi być >= 2")
        channels = [base_channels * (2**index) for index in range(4)]
        layers: list[nn.Module] = []
        previous = in_channels
        for channel in channels:
            layers.extend(
                [
                    nn.Conv2d(previous, channel, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            previous = channel
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], n_classes),
        )

    def forward(self, inputs):  # noqa: ANN001, ANN201 - torch signature
        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
