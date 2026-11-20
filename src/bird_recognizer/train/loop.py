"""Training loop for mel-spectrogram classifiers (SPEC FR-9, §10.5).

Works for every registered family (CNN, CRNN, transfer): class-weighted
cross-entropy, gradient clipping, cosine schedule, early stopping on validation
macro-F1 and the best checkpoint are driven by ``configs/default.yaml``. Windows may
carry a valid-length mask (zero-padded tails); models that declare
``accepts_lengths = True`` receive it and ignore the padding.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import WindowDataset
from ..errors import ModelError
from ..eval.metrics import aggregate_by_group, macro_f1
from .augment import augment_mel, mixup_batch

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch; trade parallel speed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def class_weights(labels: Sequence[int], n_classes: int) -> torch.Tensor:
    """Balanced class weights; classes absent from the batch get weight 1.0."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(float)
    weights = np.ones(n_classes, dtype=float)
    present = counts > 0
    if present.any():
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(
    dataset: WindowDataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _unpack_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Support datasets with and without the per-window valid-length mask."""
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        inputs, labels, lengths = batch
        return inputs, labels, lengths
    inputs, labels = batch
    return inputs, labels, None


def forward_logits(
    model: nn.Module,
    inputs: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the model with lengths only when it declares support for them."""
    if lengths is not None and getattr(model, "accepts_lengths", False):
        return model(inputs, lengths)
    return model(inputs)


def _augment_batch(batch: torch.Tensor, config: Config, rng: np.random.Generator) -> torch.Tensor:
    frames_per_second = config.audio.sample_rate / config.spectrogram.hop_length
    arrays = batch.numpy()
    augmented = np.stack(
        [
            augment_mel(
                arrays[row],
                config.augment,
                rng,
                frames_per_second=frames_per_second,
            )
            for row in range(arrays.shape[0])
        ]
    )
    return torch.from_numpy(augmented.astype(np.float32))


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    lr: float
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "val_macro_f1": round(self.val_macro_f1, 6),
            "lr": self.lr,
            "seconds": round(self.seconds, 3),
        }


@dataclass
class TrainResult:
    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_val_macro_f1: float = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    stopped_early: bool = False
    parameters: int = 0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    truths: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels, lengths = _unpack_batch(batch)
            logits = forward_logits(model, inputs.unsqueeze(1).to(device), lengths)
            losses.append(float(criterion(logits, labels.to(device)).item()))
            truths.extend(int(value) for value in labels.tolist())
            predictions.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())
    if not losses:
        raise ModelError("pusty loader walidacyjny")
    return float(np.mean(losses)), macro_f1(truths, predictions)


def _copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_model(
    model: nn.Module,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    config: Config,
    device: torch.device,
    *,
    epochs: int | None = None,
    augmentation: bool = True,
    run: Any | None = None,
    seed: int | None = None,
) -> TrainResult:
    """Train ``model`` and return the history plus the best validation state.

    The caller must create ``model`` *after* seeding (``set_seed(config.seed)``),
    otherwise the initial weights differ between runs with the same config.
    ``preset: full`` adds mixup at the batch level (SPEC §10.3); validation and test
    data are never augmented.
    """
    training = config.training
    effective_seed = config.seed if seed is None else seed
    set_seed(effective_seed)
    rng = np.random.default_rng(effective_seed)
    total_epochs = max(1, epochs if epochs is not None else training.epochs)
    use_mixup = augmentation and config.augment.preset == "full"
    mixup_alpha = config.augment.mixup_alpha

    train_loader = make_loader(
        train_dataset, training.batch_size, shuffle=True, seed=effective_seed
    )
    val_loader = make_loader(val_dataset, training.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.lr, weight_decay=training.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        if training.scheduler == "cosine"
        else None
    )
    weights = class_weights(train_dataset.labels, len(config.species)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=training.label_smoothing
    )

    result = TrainResult(parameters=sum(p.numel() for p in model.parameters()))
    best_f1 = -1.0
    patience = 0
    patience_limit = max(1, training.early_stopping_patience)

    for epoch in range(1, total_epochs + 1):
        started = time.perf_counter()
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch_inputs, labels, lengths = _unpack_batch(batch)
            inputs = batch_inputs.unsqueeze(1)
            if augmentation:
                inputs = _augment_batch(batch_inputs, config, rng).unsqueeze(1)
            targets_a, targets_b, lam = labels, labels, 1.0
            if use_mixup:
                mixed, targets_a, targets_b, lam = mixup_batch(
                    inputs.squeeze(1), labels, mixup_alpha, rng
                )
                inputs = mixed.unsqueeze(1)
            logits = forward_logits(model, inputs.to(device), lengths)
            loss = (
                lam * criterion(logits, targets_a.to(device))
                + (1.0 - lam) * criterion(logits, targets_b.to(device))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        record = EpochRecord(
            epoch=epoch,
            train_loss=float(np.mean(losses)) if losses else 0.0,
            val_loss=val_loss,
            val_macro_f1=val_f1,
            lr=float(optimizer.param_groups[0]["lr"]),
            seconds=time.perf_counter() - started,
        )
        result.history.append(record)
        if run is not None:
            run.log_epoch(record.as_dict())
        LOGGER.info(
            "epoka %s/%s: train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f (%.1f s)",
            epoch,
            total_epochs,
            record.train_loss,
            record.val_loss,
            record.val_macro_f1,
            record.seconds,
        )

        if val_f1 > best_f1 + training.early_stopping_min_delta:
            best_f1 = val_f1
            result.best_epoch = epoch
            result.best_val_macro_f1 = val_f1
            result.best_state = _copy_state(model)
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                result.stopped_early = True
                LOGGER.info("early stopping po %s epokach bez poprawy", patience)
                break
        if scheduler is not None:
            scheduler.step()

    if result.best_state is None:
        result.best_state = _copy_state(model)
        result.best_epoch = len(result.history)
        result.best_val_macro_f1 = result.history[-1].val_macro_f1
    return result


def predict_probs(
    model: nn.Module,
    dataset: WindowDataset,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    loader = make_loader(dataset, batch_size, shuffle=False)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs, _labels, lengths = _unpack_batch(batch)
            logits = forward_logits(model, inputs.unsqueeze(1).to(device), lengths)
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks)


def predict_recordings(
    model: nn.Module,
    dataset: WindowDataset,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    """Mean window probabilities per recording (SPEC FR-6/FR-10)."""
    probs = predict_probs(model, dataset, device)
    return aggregate_by_group(probs, dataset.groups())


#: Stage-4 name; still used by older scripts and tests.
train_cnn = train_model
