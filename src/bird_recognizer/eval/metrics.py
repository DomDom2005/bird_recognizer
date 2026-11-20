"""Classification metrics shared by training and evaluation (SPEC §10.6).

Pure functions over numpy arrays. Window-level metrics drive early stopping;
recording-level metrics are the headline numbers (SPEC §10.6).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


def macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
) -> float:
    """Macro-averaged F1 (equal weight per class, imbalanced-class metric)."""
    label_list = list(labels) if labels is not None else None
    return float(
        f1_score(y_true, y_pred, labels=label_list, average="macro", zero_division=0)
    )


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def top_k_accuracy(
    y_true: Sequence[int],
    scores: np.ndarray,
    *,
    k: int = 3,
    labels: Sequence[int] | None = None,
) -> float | None:
    """Top-k accuracy; ``None`` when k >= the number of classes (trivially perfect)."""
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 2:
        raise ValueError(f"oczekiwano macierzy (n, klasy), jest {score_array.shape}")
    if score_array.shape[1] <= k:
        return None
    label_list = list(labels) if labels is not None else None
    return float(
        top_k_accuracy_score(
            y_true,
            score_array,
            k=k,
            labels=label_list,
            normalize=True,
        )
    )


def confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    return confusion_matrix(
        y_true, y_pred, labels=list(labels) if labels is not None else None
    )


def per_class_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
) -> dict[int, float]:
    scores = f1_score(
        y_true,
        y_pred,
        labels=list(labels),
        average=None,
        zero_division=0,
    )
    return {int(label): float(score) for label, score in zip(labels, scores, strict=True)}


def aggregate_by_group(
    scores: np.ndarray,
    groups: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Average per-window scores into one score vector per recording.

    ``groups`` holds one recording id per row; the returned order is first-seen.
    """
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape[0] != len(groups):
        raise ValueError(
            f"liczba wierszy ({score_array.shape[0]}) != liczba grup ({len(groups)})"
        )
    order: list[str] = []
    buckets: dict[str, np.ndarray] = {}
    for row, group in enumerate(groups):
        if group not in buckets:
            buckets[group] = score_array[row].copy()
            order.append(group)
        else:
            buckets[group] += score_array[row]
    counts = {group: 0 for group in order}
    for group in groups:
        counts[group] += 1
    matrix = np.stack([buckets[group] / counts[group] for group in order])
    return order, matrix
