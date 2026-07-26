from __future__ import annotations

import numpy as np
import pytest

from src.analysis.prefix_reliability_metrics import routing_stress_metrics
from src.models.prefix_reliability import (
    action_probabilities,
    fit_action_calibration,
    prefix_class_vote,
    prefix_probability_ensemble,
)


def test_action_probabilities_are_per_action_and_temperature_scaled() -> None:
    scores = np.array(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    probabilities = action_probabilities(scores, temperatures=[1.0, 2.0])
    assert probabilities.shape == scores.shape
    assert np.allclose(probabilities.sum(axis=2), 1.0)
    assert probabilities[0, 0, 0] > probabilities[0, 1, 1]
    assert not probabilities.flags.writeable


def test_prefix_probability_ensemble_resets_at_video_boundary() -> None:
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.1, 0.9], [0.2, 0.8]],
            [[0.1, 0.9], [0.2, 0.8]],
        ]
    )
    prefix = prefix_probability_ensemble(
        probabilities, video_ids=["a", "a", "b", "b"], retention=1.0
    )
    current = prefix_probability_ensemble(
        probabilities, video_ids=["a", "a", "b", "b"], retention=0.0
    )
    assert prefix.predictions.tolist() == [0, 0, 1, 1]
    assert current.predictions.tolist() == [0, 0, 1, 1]
    assert np.allclose(prefix.class_state[2], [0.15, 0.85])
    assert np.allclose(prefix.state_l1, [1.0, 2.0, 1.0, 2.0])


def test_prefix_class_vote_uses_declared_action_weights() -> None:
    predictions = np.array([[0, 1], [0, 1], [1, 1]], dtype=np.int64)
    trace = prefix_class_vote(
        predictions,
        n_classes=2,
        video_ids=["v", "v", "v"],
        action_weights=[0.8, 0.2],
    )
    assert trace.predictions.tolist() == [0, 0, 0]
    assert np.allclose(trace.state_l1, [1.0, 2.0, 3.0])


def test_validation_calibration_downweights_systematically_wrong_action() -> None:
    labels = np.tile(np.array([0, 1], dtype=np.int64), 80)
    scores = np.empty((labels.size, 2, 2), dtype=np.float64)
    scores[:, 0, :] = -2.0
    scores[np.arange(labels.size), 0, labels] = 2.0
    scores[:, 1, :] = -2.0
    scores[np.arange(labels.size), 1, 1 - labels] = 2.0
    calibration = fit_action_calibration(
        [scores], [labels], temperature_bounds=(0.1, 5.0), stacking_l2=1e-4
    )
    assert calibration.stacking_weights[0] > 0.99
    assert calibration.stacking_weights[1] < 0.01
    assert calibration.action_nll[0] < calibration.action_nll[1]
    assert calibration.n_frames == labels.size
    assert not calibration.stacking_weights.flags.writeable


def test_routing_stress_metrics_reports_wrong_lock_and_switch_latency() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    predictions = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    stable_wrong = np.zeros(labels.size, dtype=np.int64)
    metrics = routing_stress_metrics(
        labels,
        predictions,
        stable_wrong_predictions=stable_wrong,
        switch_index=3,
        stability_frames=2,
    )
    assert metrics.accuracy == pytest.approx(5 / 7)
    assert metrics.wrong_lock_fraction == pytest.approx(0.5)
    assert metrics.post_switch_accuracy == pytest.approx(0.5)
    assert metrics.switch_latency == 2.0


@pytest.mark.parametrize("retention", [-0.01, 1.01])
def test_prefix_ensemble_rejects_invalid_retention(retention: float) -> None:
    probabilities = np.full((2, 2, 2), 0.5)
    with pytest.raises(ValueError, match="retention"):
        prefix_probability_ensemble(
            probabilities, video_ids=["v", "v"], retention=retention
        )
