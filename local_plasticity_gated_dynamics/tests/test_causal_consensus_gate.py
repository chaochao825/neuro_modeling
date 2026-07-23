from __future__ import annotations

import numpy as np

from src.models.causal_consensus_gate import (
    CausalConsensusConfig,
    CausalConsensusGate,
)


def test_consensus_gate_uses_no_labels_future_frames_or_bptt() -> None:
    predictions = np.asarray(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
        ]
    )
    gate = CausalConsensusGate(
        4,
        2,
        config=CausalConsensusConfig(tie_break_order=(1, 3, 0, 2)),
    )
    trace = gate.trace(
        predictions,
        video_ids=["v0"] * 4,
        action_event_l1=np.ones((4, 4)),
    )
    assert np.array_equal(trace.actions, [1, 1, 1, 1])
    assert np.array_equal(trace.predictions, [1, 1, 1, 1])
    assert np.array_equal(trace.full_bank_event_l1, [4.0] * 4)
    assert trace.used_query_labels is False
    assert trace.used_future_frames is False
    assert trace.used_autograd is False
    assert trace.used_bptt is False


def test_video_boundary_and_memoryless_intervention_reset_state() -> None:
    predictions = np.asarray([[0, 1], [1, 1], [0, 1], [1, 1]])
    videos = np.asarray(["a", "a", "b", "b"])
    gate = CausalConsensusGate(
        2,
        2,
        config=CausalConsensusConfig(tie_break_order=(0, 1)),
    )
    trace = gate.trace(predictions, video_ids=videos)
    assert trace.count_state_l1.tolist() == [2.0, 4.0, 2.0, 4.0]

    memoryless = CausalConsensusGate(
        2,
        2,
        config=CausalConsensusConfig(reset_each_frame=True, tie_break_order=(1, 0)),
    ).trace(predictions, video_ids=videos)
    assert np.array_equal(memoryless.actions, [1, 1, 1, 1])
    assert np.array_equal(memoryless.count_state_l1, [2.0] * 4)


def test_delayed_gate_cannot_use_an_observation_before_it_arrives() -> None:
    predictions = np.asarray([[0, 1], [1, 1], [0, 1]])
    gate = CausalConsensusGate(
        2,
        2,
        config=CausalConsensusConfig(delay_frames=2, tie_break_order=(0, 1)),
    )
    trace = gate.trace(predictions, video_ids=["v"] * 3)
    assert np.array_equal(trace.count_state_l1, [0.0, 0.0, 2.0])
    assert np.array_equal(trace.actions[:2], [0, 0])
