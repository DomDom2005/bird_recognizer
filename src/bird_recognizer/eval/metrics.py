"""Classification metrics shared by training and evaluation (SPEC §10.6).

Pure functions over numpy arrays. Window-level metrics drive early stopping;
recording-level metrics are the headline numbers (SPEC §10.6).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
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


def balanced_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
) -> float:
    """Macro recall: every class counts equally, robust to class imbalance."""
    return float(
        recall_score(
            y_true,
            y_pred,
            labels=list(labels) if labels else None,
            average="macro",
            zero_division=0,
        )
    )


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


def per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
    names: Mapping[int, str] | None = None,
) -> list[dict[str, object]]:
    """Precision, recall, F1 and support for every class (SPEC FR-10)."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        average=None,
        zero_division=0,
    )
    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "label": int(label),
                "species": (names or {}).get(int(label), str(label)),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    return rows


def confusion_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    names: Mapping[int, str] | None = None,
    top: int = 10,
) -> list[dict[str, object]]:
    """Most frequent (truth -> prediction) mistakes; only real confusions count."""
    counts = Counter(
        (int(truth), int(prediction))
        for truth, prediction in zip(y_true, y_pred, strict=True)
        if int(truth) != int(prediction)
    )
    rows: list[dict[str, object]] = []
    for (truth, predicted), count in counts.most_common(max(1, top)):
        rows.append(
            {
                "truth": truth,
                "predicted": predicted,
                "truth_name": (names or {}).get(truth, str(truth)),
                "predicted_name": (names or {}).get(predicted, str(predicted)),
                "count": int(count),
            }
        )
    return rows


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
