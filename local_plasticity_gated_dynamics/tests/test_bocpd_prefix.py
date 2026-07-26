from __future__ import annotations

import numpy as np
import pytest

from src.models.bocpd_prefix import BOCPDConfig, bocpd_prefix_accumulator
from src.models.change_aware_prefix import fixed_forgetting_accumulator


def _config(**updates: object) -> BOCPDConfig:
    values: dict[str, object] = {
        "hazard": 0.05,
        "prior_concentration": 1.0,
        "alarm_threshold": 0.25,
        "min_run_frames": 2,
        "max_run_length": 32,
    }
    values.update(updates)
    return BOCPDConfig(**values)  # type: ignore[arg-type]


def test_bocpd_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="hazard"):
        _config(hazard=0.0)
    with pytest.raises(ValueError, match="max_run_length"):
        _config(min_run_frames=8, max_run_length=4)


def test_hard_reset_and_score_only_share_alarms_but_not_resets() -> None:
    first = np.tile([0.99, 0.01], (8, 1))
    second = np.tile([0.01, 0.99], (8, 1))
    evidence = np.vstack([first, second])
    streams = np.repeat("one-hidden-stream", len(evidence))
    hard = bocpd_prefix_accumulator(
        evidence, stream_ids=streams, config=_config(), mode="hard_reset"
    )
    score = bocpd_prefix_accumulator(
        evidence, stream_ids=streams, config=_config(), mode="score_only"
    )
    assert np.array_equal(hard.alarm_flags, score.alarm_flags)
    assert hard.alarm_flags[8:].any()
    assert hard.reset_flags.sum() == hard.alarm_flags.sum()
    assert not score.reset_flags.any()
    assert hard.predictions[9] == 1
    assert score.predictions[9] == 0


def test_posterior_accumulator_is_causal() -> None:
    prefix = np.tile([0.8, 0.2], (6, 1))
    first = np.vstack([prefix, np.tile([0.1, 0.9], (4, 1))])
    second = np.vstack([prefix, np.tile([0.9, 0.1], (4, 1))])
    streams = np.repeat("stream", 10)
    one = bocpd_prefix_accumulator(
        first, stream_ids=streams, config=_config(), mode="posterior"
    )
    two = bocpd_prefix_accumulator(
        second, stream_ids=streams, config=_config(), mode="posterior"
    )
    assert np.allclose(one.class_state[:6], two.class_state[:6])
    assert not one.alarm_flags.any()
    assert np.all(one.class_state.sum(axis=1) >= 1.0 - 1e-9)


def test_change_reset_beats_unbounded_prefix_on_clean_switch() -> None:
    evidence = np.vstack(
        [np.tile([0.99, 0.01], (16, 1)), np.tile([0.01, 0.99], (16, 1))]
    )
    streams = np.repeat("stream", 32)
    hard = bocpd_prefix_accumulator(
        evidence, stream_ids=streams, config=_config(), mode="hard_reset"
    )
    cumulative = fixed_forgetting_accumulator(
        evidence, stream_ids=streams, retention=1.0
    )
    labels = np.r_[np.zeros(16, dtype=int), np.ones(16, dtype=int)]
    assert np.mean(hard.predictions == labels) > np.mean(cumulative.predictions == labels)
