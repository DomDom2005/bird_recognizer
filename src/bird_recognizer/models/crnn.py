"""CRNN: the stage-4 CNN backbone, a BiGRU over time and temporal pooling (SPEC FR-7).

The backbone is the exact convolutional stack of :class:`~bird_recognizer.models.cnn.SmallCNN`
(so "same backbone as the CNN" is true by construction, not by copy-paste). Each
time step of its output is projected to ``projection`` features, fed through a
bidirectional GRU (1 layer, dropout applied after the RNN) and reduced by either
mean or attention pooling over time. Padded tail frames of short windows are
excluded through a per-window length mask, both in pooling and by never attending
to them.
"""

from __future__ import annotations

import torch
from torch import nn

from .cnn import SmallCNN

#: Every 2x2 max pool halves both axes; the backbone has four of them.
_POOL_FACTOR = 2**4


class MeanPooling(nn.Module):
    """Mean over time steps; padded steps are excluded via ``mask``."""

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return sequence.mean(dim=1)
        weights = mask.unsqueeze(-1).to(sequence.dtype)
        total = weights.sum(dim=1).clamp(min=1.0)
        return (sequence * weights).sum(dim=1) / total


class AttentionPooling(nn.Module):
    """Learned attention over time steps; padded steps get ``-inf`` score."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(sequence).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            # A fully padded row would be all -inf; lengths are clamped to >= 1, but
            # nan_to_num keeps a corrupt mask from poisoning the whole batch.
            weights = torch.nan_to_num(weights, nan=0.0)
        return (sequence * weights.unsqueeze(-1)).sum(dim=1)


POOLINGS: dict[str, type[nn.Module]] = {
    "mean": MeanPooling,
    "attention": AttentionPooling,
}

RNN_KINDS: dict[str, type[nn.RNNBase]] = {"gru": nn.GRU, "lstm": nn.LSTM}


def frames_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    """``(batch, steps)`` bool mask for valid (non-padding) time steps.

    ``lengths`` are valid input frames (``n_mels`` axis untouched); four 2x2 pools
    turn them into ``lengths // 16`` backbone steps. A length of one frame still
    yields one valid step, never an all-masked row.
    """
    valid = torch.as_tensor(lengths, dtype=torch.long).reshape(-1)
    valid = torch.div(valid.clamp(min=1), _POOL_FACTOR, rounding_mode="floor")
    valid = valid.clamp(min=1, max=max(1, steps))
    positions = torch.arange(steps, device=valid.device)
    return positions.unsqueeze(0) < valid.unsqueeze(1)


class CRNN(nn.Module):
    """CNN backbone + BiGRU + temporal pooling; accepts variable time lengths."""

    accepts_lengths = True

    def __init__(
        self,
        n_classes: int,
        *,
        n_mels: int = 128,
        in_channels: int = 1,
        base_channels: int = 32,
        hidden_size: int = 128,
        projection: int = 256,
        dropout: float = 0.3,
        pool: str = "mean",
        rnn: str = "gru",
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes musi być >= 2")
        if pool not in POOLINGS:
            raise ValueError(f"pool musi być jednym z {sorted(POOLINGS)}, jest '{pool}'")
        if rnn not in RNN_KINDS:
            raise ValueError(f"rnn musi być jednym z {sorted(RNN_KINDS)}, jest '{rnn}'")
        backbone = SmallCNN(
            n_classes=n_classes,
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
        )
        # Reuse the stage-4 convolutional stack verbatim; the temporary head is unused.
        self.features = backbone.features
        channels = base_channels * 8
        freq_bins = max(1, n_mels // _POOL_FACTOR)
        self.project = nn.Linear(channels * freq_bins, projection)
        self.sequence = RNN_KINDS[rnn](
            projection, hidden_size, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = (
            AttentionPooling(hidden_size * 2)
            if pool == "attention"
            else MeanPooling()
        )
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.features(inputs)  # (B, C, F, T')
        batch, channels, freq, steps = features.shape
        sequence = features.permute(0, 3, 1, 2).reshape(batch, steps, channels * freq)
        sequence, _ = self.sequence(self.project(sequence))
        sequence = self.dropout(sequence)
        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=sequence.device)
        mask = frames_mask(lengths, steps) if lengths is not None else None
        return self.classifier(self.pool(sequence, mask))

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
