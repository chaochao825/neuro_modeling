from __future__ import annotations

import numpy as np
import pytest

from src.analysis.sparse_feedback_metrics import (
    binary_belief_scores,
    post_switch_cost,
    stream_accuracy,
    switch_diagnostics,
)


def test_sparse_feedback_metrics_use_the_whole_paired_stream() -> None:
    states = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=int)
    belief = np.array([0.1, 0.1, 0.1, 0.2, 0.8, 0.9, 0.8, 0.1])
    rewards = np.array([1, 1, 1, 0, 1, 1, 0, 1], dtype=float)
    assert stream_accuracy(rewards) == 0.75
    scores = binary_belief_scores(states, belief)
    assert 0.0 <= scores["context_brier"] <= 1.0
    assert scores["belief_state_accuracy"] == 0.75
    switches = switch_diagnostics(states, belief, stability_window=1)
    assert switches["hidden_switch_count"] == 2
    assert switches["median_switch_latency"] == 1.0
    assert switches["false_switch_rate"] > 0.0
    assert np.isfinite(post_switch_cost(states, rewards, window=1))
    with pytest.raises(ValueError, match="paired"):
        binary_belief_scores(states, belief[:-1])
