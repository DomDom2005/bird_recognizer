"""Recording-level aggregation and confidence coverage (SPEC FR-10, §10.6).

``mean`` averages the window probabilities of one recording (the headline method);
``vote`` counts window argmax labels and breaks ties by summed probability. The
confidence threshold never changes a prediction — it only selects which recordings
are reported, so "coverage vs accuracy" stays honest.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)


def vote_by_group(
    scores: np.ndarray,
    groups: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Majority vote of window labels per recording; ties by summed probability."""
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape[0] != len(groups):
        raise ValueError(
            f"liczba wierszy ({score_array.shape[0]}) != liczba grup ({len(groups)})"
        )
    order: list[str] = []
    counts: dict[str, dict[int, int]] = defaultdict(dict)
    totals: dict[str, np.ndarray] = {}
    for row, group in enumerate(groups):
        if group not in totals:
            order.append(group)
            totals[group] = np.zeros(score_array.shape[1], dtype=np.float64)
        label = int(np.argmax(score_array[row]))
        counts[group][label] = counts[group].get(label, 0) + 1
        totals[group] += score_array[row]
    labels = np.zeros(len(order), dtype=np.int64)
    for index, group in enumerate(order):
        per_label = counts[group]
        best = max(per_label.values())
        candidates = [label for label, count in per_label.items() if count == best]
        if len(candidates) == 1:
            labels[index] = candidates[0]
        else:
            probabilities = [float(totals[group][label]) for label in candidates]
            labels[index] = candidates[int(np.argmax(probabilities))]
    return order, labels


def coverage_curve(
    truth: Sequence[int],
    scores: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, object]]:
    """For each confidence threshold: how many recordings stay and how accurate they are."""
    score_array = np.asarray(scores, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.int64)
    if score_array.shape[0] != truth_array.shape[0]:
        raise ValueError(
            f"liczba wierszy ({score_array.shape[0]}) != liczba etykiet ({truth_array.shape[0]})"
        )
    predictions = score_array.argmax(axis=1)
    confidence = score_array.max(axis=1)
    total = int(truth_array.shape[0])
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        selected = confidence >= float(threshold)
        count = int(selected.sum())
        accuracy = (
            float((predictions[selected] == truth_array[selected]).mean())
            if count
            else None
        )
        rows.append(
            {
                "threshold": float(threshold),
                "n": count,
                "coverage": round(count / total, 6) if total else 0.0,
                "accuracy": None if accuracy is None else round(accuracy, 6),
            }
        )
    return rows
