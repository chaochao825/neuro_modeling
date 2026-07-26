from __future__ import annotations

import numpy as np
import pytest

from src.analysis.change_point_metrics import change_point_metrics
from src.models.change_aware_prefix import (
    JSDChangeConfig,
    circularly_shift_resets,
    fixed_forgetting_accumulator,
    jensen_shannon_divergence,
    jsd_change_accumulator,
    scheduled_reset_accumulator,
    sliding_window_accumulator,
)


def _switched_evidence() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    evidence = np.vstack(
        [
            np.tile([0.95, 0.05], (20, 1)),
            np.tile([0.05, 0.95], (20, 1)),
        ]
    )
    labels = np.concatenate(
        [np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64)]
    )
    streams = np.repeat(np.asarray(["stream"], dtype=str), 40)
    switches = np.zeros(40, dtype=np.bool_)
    switches[20] = True
    return evidence, labels, streams, switches


def test_jsd_is_symmetric_bounded_and_zero_on_identity() -> None:
    first = np.asarray([0.9, 0.1])
    second = np.asarray([0.1, 0.9])
    assert jensen_shannon_divergence(first, first) == pytest.approx(0.0)
    assert jensen_shannon_divergence(first, second) == pytest.approx(
        jensen_shannon_divergence(second, first)
    )
    assert 0.0 < jensen_shannon_divergence(first, second) < np.log(2.0)
    with pytest.raises(ValueError, match="non-negative"):
        jensen_shannon_divergence([1.0, -1.0], [0.5, 0.5])


def test_change_reset_recovers_after_hidden_switch_without_labels() -> None:
    evidence, labels, streams, switches = _switched_evidence()
    cumulative = fixed_forgetting_accumulator(
        evidence, stream_ids=streams, retention=1.0
    )
    config = JSDChangeConfig(
        fast_retention=0.0,
        jsd_threshold=0.1,
        patience=1,
        min_run_frames=4,
    )
    reset = jsd_change_accumulator(evidence, stream_ids=streams, config=config)
    score_only = jsd_change_accumulator(
        evidence, stream_ids=streams, config=config, enable_reset=False
    )

    assert np.flatnonzero(reset.alarm_flags).tolist() == [20]
    assert np.flatnonzero(reset.reset_flags).tolist() == [20]
    assert reset.run_lengths[20] == 1
    assert not np.any(score_only.reset_flags)
    assert np.any(score_only.alarm_flags)
    assert np.mean(reset.predictions == labels) > np.mean(
        cumulative.predictions == labels
    )
    assert np.mean(reset.predictions == labels) > np.mean(
        score_only.predictions == labels
    )

    metrics = change_point_metrics(
        reset.predictions,
        labels,
        alarm_flags=reset.alarm_flags,
        switch_flags=switches,
        post_switch_window=8,
        detection_tolerance=4,
    )
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.post_switch_accuracy == pytest.approx(1.0)
    assert metrics.detection_precision == pytest.approx(1.0)
    assert metrics.detection_recall == pytest.approx(1.0)
    assert metrics.median_detection_delay == pytest.approx(0.0)
    assert metrics.false_alarms_per_1000 == pytest.approx(0.0)


def test_fixed_forgetting_sliding_and_observable_boundaries_are_exact() -> None:
    evidence = np.asarray(
        [[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]],
        dtype=np.float64,
    )
    streams = np.asarray(["a", "a", "b", "b"])
    forgetting = fixed_forgetting_accumulator(
        evidence, stream_ids=streams, retention=0.5
    )
    assert np.allclose(forgetting.class_state[1], 0.5 * evidence[0] + evidence[1])
    assert np.allclose(forgetting.class_state[2], evidence[2])
    assert not np.any(forgetting.reset_flags)

    window = sliding_window_accumulator(
        np.vstack([evidence[:2], evidence[:2]]),
        stream_ids=np.repeat("one", 4),
        window_frames=2,
    )
    assert np.allclose(window.class_state[2], evidence[1] + evidence[0])
    assert np.allclose(window.class_state[3], evidence[0] + evidence[1])


def test_matched_shift_preserves_reset_count_and_changes_timing() -> None:
    evidence, _, streams, _ = _switched_evidence()
    schedule = np.zeros(40, dtype=np.bool_)
    schedule[[10, 20, 30]] = True
    shifted = circularly_shift_resets(schedule, stream_ids=streams, offset=3)
    assert np.sum(shifted) == np.sum(schedule)
    assert not np.array_equal(shifted, schedule)
    trace = scheduled_reset_accumulator(
        evidence, stream_ids=streams, reset_schedule=shifted
    )
    assert np.array_equal(trace.reset_flags, shifted)
    with pytest.raises(ValueError, match="nonzero"):
        circularly_shift_resets(schedule, stream_ids=streams, offset=0)


def test_metric_matching_does_not_reuse_one_alarm() -> None:
    predictions = np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int64)
    labels = np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int64)
    switches = np.asarray([False, False, True, True, False, False])
    alarms = np.asarray([True, False, False, True, False, False])
    metrics = change_point_metrics(
        predictions,
        labels,
        alarm_flags=alarms,
        switch_flags=switches,
        post_switch_window=2,
        detection_tolerance=2,
    )
    assert metrics.n_switches == 2
    assert metrics.n_alarms == 2
    assert metrics.n_matched_switches == 1
    assert metrics.detection_recall == pytest.approx(0.5)
    assert metrics.detection_precision == pytest.approx(0.5)
    assert metrics.false_alarms_per_1000 == pytest.approx(1000.0 / 6.0)


def test_accumulators_fail_closed_on_invalid_probability_inputs() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        fixed_forgetting_accumulator(
            [[0.2, 0.2], [0.5, 0.5]], stream_ids=["x", "x"], retention=1.0
        )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        JSDChangeConfig(
            fast_retention=1.0,
            jsd_threshold=0.1,
            patience=1,
            min_run_frames=1,
        )
