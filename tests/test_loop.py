"""Training loop, determinism and metrics tests (SPEC FR-9, §10.6)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from bird_recognizer.eval.metrics import (
    accuracy,
    aggregate_by_group,
    macro_f1,
    top_k_accuracy,
)
from bird_recognizer.models.cnn import SmallCNN
from bird_recognizer.train.loop import class_weights, set_seed, train_cnn


class ToyDataset:
    """Duck-typed ``WindowDataset``: random mel windows, few classes, one group each."""

    def __init__(self, n: int = 24, classes: int = 3, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.data = rng.standard_normal((n, 16, 64)).astype(np.float32)
        self.labels = [index % classes for index in range(n)]
        self.recording_ids = [f"rec-{index}" for index in range(n)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return self.data[index], self.labels[index]

    def groups(self) -> list[str]:
        return list(self.recording_ids)


def _cpu() -> torch.device:
    return torch.device("cpu")


def test_set_seed_is_reproducible() -> None:
    set_seed(7)
    first = torch.randn(4)
    set_seed(7)
    second = torch.randn(4)
    torch.testing.assert_close(first, second)


def test_class_weights_are_inverse_frequency() -> None:
    weights = class_weights([0, 0, 0, 1], 3)
    assert float(weights[2]) == 1.0  # class absent from training
    assert float(weights[0]) < float(weights[1])
    assert abs(float(weights[0]) * 3 - float(weights[1])) < 1e-6


def test_train_cnn_records_history_and_best_state(config) -> None:
    model = SmallCNN(n_classes=len(config.species))
    result = train_cnn(
        model, ToyDataset(), ToyDataset(seed=1), config, _cpu(), epochs=1
    )
    assert len(result.history) == 1
    assert result.best_state is not None
    assert 0.0 <= result.best_val_macro_f1 <= 1.0
    assert result.parameters > 0
    assert result.history[0].train_loss > 0.0


def test_training_is_deterministic_for_same_seed(config) -> None:
    set_seed(config.seed)
    first_model = SmallCNN(n_classes=len(config.species))
    first = train_cnn(first_model, ToyDataset(), ToyDataset(seed=1), config, _cpu(), epochs=1)
    set_seed(config.seed)
    second_model = SmallCNN(n_classes=len(config.species))
    second = train_cnn(
        second_model, ToyDataset(), ToyDataset(seed=1), config, _cpu(), epochs=1
    )
    assert first.best_val_macro_f1 == second.best_val_macro_f1
    assert first.history[0].train_loss == second.history[0].train_loss


def test_early_stopping_stops_before_max_epochs(config) -> None:
    training = replace(
        config.training,
        early_stopping_patience=1,
        early_stopping_min_delta=0.9,
    )
    impatient = replace(config, training=training)
    result = train_cnn(
        SmallCNN(n_classes=len(config.species)),
        ToyDataset(),
        ToyDataset(seed=1),
        impatient,
        _cpu(),
        epochs=5,
    )
    assert result.stopped_early
    assert len(result.history) <= 2


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_macro_f1_known_values() -> None:
    # class 0: TP=2, FN=0, FP=1 -> F1=0.8; class 1: TP=1, FN=1, FP=0 -> F1=2/3
    assert abs(macro_f1([0, 0, 1, 1], [0, 0, 1, 0]) - 0.733333) < 1e-5


def test_accuracy_counts_correct_predictions() -> None:
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_top_k_known_scores() -> None:
    scores = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    labels = [0, 1, 2]
    assert top_k_accuracy([0, 1], scores, k=1, labels=labels) == 1.0
    assert top_k_accuracy([0, 2], scores, k=2, labels=labels) == 0.5


def test_top_k_is_none_when_k_meets_class_count() -> None:
    scores = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    assert top_k_accuracy([0, 1], scores, k=3) is None


def test_aggregate_by_group_averages_scores() -> None:
    scores = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    order, matrix = aggregate_by_group(scores, ["a", "a", "b"])
    assert order == ["a", "b"]
    np.testing.assert_allclose(matrix, [[2.0, 0.0], [0.0, 2.0]])
