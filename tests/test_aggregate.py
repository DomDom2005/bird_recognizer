"""Recording aggregation: mean, vote and coverage (SPEC FR-10, §10.6)."""

from __future__ import annotations

import numpy as np

from bird_recognizer.eval.aggregate import coverage_curve, vote_by_group
from bird_recognizer.eval.metrics import aggregate_by_group


def test_mean_aggregation_averages_window_probabilities() -> None:
    scores = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    order, matrix = aggregate_by_group(scores, ["a", "a", "b"])
    assert order == ["a", "b"]
    np.testing.assert_allclose(matrix, [[0.45, 0.4, 0.15], [0.1, 0.3, 0.6]])


def test_vote_counts_window_labels_and_breaks_ties_by_probability() -> None:
    scores = np.array(
        [
            [0.70, 0.20, 0.10],  # a: vote 0
            [0.60, 0.30, 0.10],  # b: vote 0
            [0.60, 0.30, 0.10],  # a: vote 0
            [0.20, 0.70, 0.10],  # b: vote 1 -> tie 1:1, sums favour 1
        ]
    )
    order, labels = vote_by_group(scores, ["a", "b", "a", "b"])
    assert order == ["a", "b"]
    assert labels.tolist() == [0, 1]


def test_vote_requires_one_group_per_row() -> None:
    with np.testing.assert_raises(ValueError):
        vote_by_group(np.zeros((2, 3)), ["a"])


def test_coverage_curve_keeps_prediction_and_reports_selection() -> None:
    scores = np.array([[0.9, 0.1], [0.4, 0.35], [0.2, 0.8]])
    truth = [0, 1, 1]
    rows = coverage_curve(truth, scores, thresholds=(0.0, 0.5, 0.85))
    assert rows[0]["coverage"] == 1.0
    assert rows[0]["accuracy"] == round(2 / 3, 6)
    # 0.9 and 0.8 clear the 0.5 threshold, the 0.4 one does not
    assert rows[1]["n"] == 2
    assert rows[1]["coverage"] == round(2 / 3, 6)
    assert rows[1]["accuracy"] == 1.0
    assert rows[2]["n"] == 1


def test_coverage_curve_empty_selection_is_none() -> None:
    rows = coverage_curve([0], np.array([[0.4, 0.3]]), thresholds=(0.9,))
    assert rows[0]["n"] == 0
    assert rows[0]["accuracy"] is None
