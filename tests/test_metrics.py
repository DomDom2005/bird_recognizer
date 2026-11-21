"""Metric definitions on known matrices (SPEC §10.6, gate of stage 7)."""

from __future__ import annotations

import numpy as np
import pytest

from bird_recognizer.eval.metrics import (
    balanced_accuracy,
    confusion,
    confusion_pairs,
    macro_f1,
    per_class_metrics,
)


def _known_matrix() -> tuple[list[int], list[int]]:
    """3 classes; F1 per class: 2/3, 3/4, 2/3 -> macro = 25/36."""
    truth: list[int] = []
    predicted: list[int] = []
    rows = {(0, 0): 2, (0, 1): 1, (1, 1): 3, (1, 2): 1, (2, 2): 2, (2, 0): 1}
    for (true_label, pred_label), count in rows.items():
        truth.extend([true_label] * count)
        predicted.extend([pred_label] * count)
    return truth, predicted


def test_macro_f1_matches_hand_computed_value() -> None:
    truth, predicted = _known_matrix()
    assert abs(macro_f1(truth, predicted) - 25 / 36) < 1e-9
    matrix = confusion(truth, predicted, labels=[0, 1, 2])
    np.testing.assert_array_equal(
        matrix, np.array([[2, 1, 0], [0, 3, 1], [1, 0, 2]])
    )


def test_balanced_accuracy_is_mean_recall() -> None:
    truth, predicted = _known_matrix()
    assert abs(balanced_accuracy(truth, predicted) - 25 / 36) < 1e-9


def test_per_class_metrics_has_support_and_names() -> None:
    truth, predicted = _known_matrix()
    names = {0: "kos", 1: "kwiczoł", 2: "bogatka"}
    rows = per_class_metrics(truth, predicted, labels=[0, 1, 2], names=names)
    assert [row["species"] for row in rows] == ["kos", "kwiczoł", "bogatka"]
    assert [row["support"] for row in rows] == [3, 4, 3]
    assert rows[0]["f1"] == pytest.approx(2 / 3)
    assert rows[1]["f1"] == pytest.approx(3 / 4)
    assert rows[2]["f1"] == pytest.approx(2 / 3)


def test_confusion_pairs_returns_most_frequent_mistakes() -> None:
    truth = [0, 0, 0, 1, 1, 2, 2]
    predicted = [0, 1, 1, 1, 1, 0, 2]
    names = {0: "kos", 1: "kwiczoł", 2: "bogatka"}
    pairs = confusion_pairs(truth, predicted, names=names, top=10)
    assert pairs[0]["truth_name"] == "kos"
    assert pairs[0]["predicted_name"] == "kwiczoł"
    assert pairs[0]["count"] == 2
    assert all(row["truth"] != row["predicted"] for row in pairs)


def test_per_class_metrics_does_not_hide_missing_class() -> None:
    rows = per_class_metrics([0, 0], [0, 0], labels=[0, 1], names={0: "a", 1: "b"})
    assert rows[1]["support"] == 0
    assert rows[1]["recall"] == 0.0
