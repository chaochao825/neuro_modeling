from __future__ import annotations

import numpy as np
import pytest

from src.analysis.memory_demand_metrics import (
    qualify_memory_demand,
    source_video_accuracy,
    switch_reachability,
)


def test_source_video_accuracy_keeps_video_as_unit() -> None:
    frame = source_video_accuracy(
        np.array([0, 1, 1, 1, 0, 0]),
        np.array([0, 0, 1, 1, 0, 1]),
        source_video_ids=np.array(["a", "a", "b", "b", "c", "c"]),
        switch_flags=np.array([False, False, True, False, True, False]),
        post_switch_window=2,
    ).set_index("source_video_id")
    assert frame.loc["a", "accuracy"] == pytest.approx(0.5)
    assert frame.loc["b", "post_switch_accuracy"] == pytest.approx(1.0)
    assert frame.loc["c", "post_switch_accuracy"] == pytest.approx(0.5)


def test_switch_reachability_matches_rising_alarms_causally() -> None:
    score = np.array([0.0, 0.1, 0.2, 0.9, 0.8, 0.1, 0.2, 0.95, 0.1])
    switches = np.zeros(9, dtype=bool)
    switches[[3, 7]] = True
    result = switch_reachability(
        score,
        switch_flags=switches,
        stream_ids=np.repeat("s", 9),
        threshold=0.5,
        detection_tolerance=1,
        refractory_frames=2,
        min_run_frames=2,
    )
    assert result.recall == pytest.approx(1.0)
    assert result.n_alarms == 2
    assert result.false_alarms_per_1000 == 0.0
    assert result.median_delay == 0.0
    assert result.auc == pytest.approx(1.0)


def test_silent_controller_fails_reachability() -> None:
    score = np.zeros(12)
    switches = np.zeros(12, dtype=bool)
    switches[6] = True
    result = switch_reachability(
        score,
        switch_flags=switches,
        stream_ids=np.repeat("s", 12),
        threshold=0.5,
        detection_tolerance=2,
        refractory_frames=2,
        min_run_frames=2,
    )
    assert result.recall == 0.0
    assert np.isnan(result.median_delay)


def test_qualification_requires_every_gate() -> None:
    reachability = switch_reachability(
        np.array([0.0, 0.1, 0.9, 0.1, 0.9]),
        switch_flags=np.array([False, False, True, False, True]),
        stream_ids=np.repeat("s", 5),
        threshold=0.5,
        detection_tolerance=1,
        refractory_frames=1,
        min_run_frames=2,
    )
    result = qualify_memory_demand(
        current_frame_natural_accuracy=0.60,
        best_accumulator_natural_accuracy=0.66,
        best_fixed_hidden_accuracy=0.70,
        oracle_hidden_accuracy=0.76,
        best_fixed_post_switch_accuracy=0.60,
        cumulative_post_switch_accuracy=0.50,
        reachability=reachability,
        stable_gain_mcid=0.02,
        oracle_headroom_mcid=0.02,
        cumulative_harm_mcid=0.03,
        min_reachability_auc=0.60,
        min_reachability_recall=0.50,
        max_false_alarms_per_1000=100.0,
        max_median_delay=2.0,
    )
    assert result.passed
    failed = qualify_memory_demand(
        current_frame_natural_accuracy=0.60,
        best_accumulator_natural_accuracy=0.61,
        best_fixed_hidden_accuracy=0.70,
        oracle_hidden_accuracy=0.76,
        best_fixed_post_switch_accuracy=0.60,
        cumulative_post_switch_accuracy=0.50,
        reachability=reachability,
        stable_gain_mcid=0.02,
        oracle_headroom_mcid=0.02,
        cumulative_harm_mcid=0.03,
        min_reachability_auc=0.60,
        min_reachability_recall=0.50,
        max_false_alarms_per_1000=100.0,
        max_median_delay=2.0,
    )
    assert not failed.passed
    assert not failed.stable_accumulation_gate

