from __future__ import annotations

import numpy as np
import pytest

from src.plasticity.reward_only_selector import (
    RewardOnlyDeltaSelector,
    balanced_probe_schedule,
)


def test_balanced_probe_schedule_is_seeded_and_exact() -> None:
    first = balanced_probe_schedule(20, seed=7)
    second = balanced_probe_schedule(20, seed=7)
    assert np.array_equal(first, second)
    assert np.bincount(first, minlength=2).tolist() == [10, 10]
    assert first.flags.writeable is False
    with pytest.raises(ValueError, match="even"):
        balanced_probe_schedule(3, seed=7)


def test_selector_accepts_only_executed_scalar_reward_and_uses_local_delta() -> None:
    selector = RewardOnlyDeltaSelector(q_prior=0.5, tie_break_action=0)
    selector.observe(0, 0.0)
    selector.observe(1, 1.0)
    selector.observe(0, 0.0)
    selector.observe(1, 1.0)
    receipt = selector.receipt()
    assert receipt.selected_action == 1
    assert np.array_equal(receipt.action_counts, [2, 2])
    assert np.allclose(receipt.q_values, [0.0, 1.0])
    assert receipt.used_true_context is False
    assert receipt.used_counterfactual_reward is False
    assert receipt.used_autograd is False
    assert receipt.used_bptt is False
    assert len(receipt.observation_fingerprint) == 64
    with pytest.raises(TypeError, match="scalar"):
        selector.observe(0, [0.0, 1.0])  # type: ignore[arg-type]


def test_selector_requires_every_action_before_deployment() -> None:
    selector = RewardOnlyDeltaSelector()
    selector.observe(0, 1.0)
    with pytest.raises(RuntimeError, match="every action"):
        selector.select()
