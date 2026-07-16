"""Focused receipts for the Exp23 frozen-trajectory compatibility cell."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments import exp23_closed_loop_local_controller as exp23


class _MutableRecurrentNetwork:
    def __init__(self) -> None:
        self.recurrent_weights = np.array(
            [[0.0, 1.0], [-2.0, 0.0]],
            dtype=np.float64,
        )


def test_recurrent_checkpoint_is_an_independent_copy_with_bitwise_hash() -> None:
    network = _MutableRecurrentNetwork()
    checkpoint = exp23._capture_recurrent_weight_checkpoint(network)
    original_fingerprint = checkpoint.fingerprint

    network.recurrent_weights[0, 1] += 0.125

    assert checkpoint.fingerprint == original_fingerprint
    assert checkpoint.weights[0, 1] == 1.0
    assert not checkpoint.weights.flags.writeable
    with pytest.raises(RuntimeError, match="recurrent weights changed"):
        exp23._assert_recurrent_weight_checkpoint_unchanged(
            network,
            checkpoint,
            phase="focused test",
        )


def test_legacy_off_policy_key_has_an_explicit_block_local_method_label() -> None:
    assert (
        exp23._condition_training_algorithm("current_off_policy")
        == exp23.FROZEN_TRAJECTORY_BLOCK_LOCAL_METHOD
    )
    assert exp23._condition_training_algorithm("local_eprop") == "local_eprop"


def test_frozen_trajectory_proposal_is_not_mislabeled_as_exp22(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = SimpleNamespace(
        task=SimpleNamespace(episode_ids=np.array([7], dtype=int))
    )
    splits = SimpleNamespace(
        dev=SimpleNamespace(subset=lambda indices: episode)
    )
    network = SimpleNamespace(n_units=2)
    calls: list[tuple[np.ndarray, str]] = []

    def fake_forward(
        network: object,
        dataset: object,
        posterior: np.ndarray,
        axis: np.ndarray,
        readout: object,
        config: dict[str, object],
        *,
        mode: str,
        task_variant: str,
    ) -> SimpleNamespace:
        del network, dataset, posterior, readout, config, task_variant
        calls.append((np.asarray(axis).copy(), mode))
        return SimpleNamespace(gradient=np.array([3.0, 4.0]))

    monkeypatch.setattr(
        exp23,
        "_episode_indices",
        lambda dataset: [np.array([0], dtype=int)],
    )
    monkeypatch.setattr(exp23, "_forward_with_gradient", fake_forward)
    updates, audit = exp23._frozen_trajectory_block_local_updates(
        network,
        splits,
        np.array([0.75]),
        SimpleNamespace(),
        {
            "off_policy": {
                "learning_rate": 2.0,
                "total_budget": 5.0,
                "budget_norm": "l2",
                "budget_tolerance": 1e-12,
                "tau_eligibility_steps": 3.0,
                "error_clip": 2.0,
            }
        },
        "current",
    )

    np.testing.assert_array_equal(calls[0][0], np.zeros(2))
    assert calls[0][1] == "local"
    np.testing.assert_allclose(updates[7], np.array([-3.0, -4.0]))
    assert audit["off_policy_method"] == (
        exp23.FROZEN_TRAJECTORY_BLOCK_LOCAL_METHOD
    )
    assert audit["off_policy_condition_key_is_legacy_alias"]
    assert not audit["off_policy_exp22_proposal_reused"]
    assert audit["off_policy_proposal_budget_norm"] == "l2"
    assert audit["off_policy_proposal_budget_attained"]
    assert audit["off_policy_proposal_budget_cap_respected"]
    assert set(audit["off_policy_legacy_no_op_config_fields"]) == {
        "error_clip",
        "tau_eligibility_steps",
    }

    _, underfilled = exp23._frozen_trajectory_block_local_updates(
        network,
        splits,
        np.array([0.75]),
        SimpleNamespace(),
        {
            "off_policy": {
                "learning_rate": 2.0,
                "total_budget": 20.0,
                "budget_norm": "l2",
                "budget_tolerance": 1e-12,
            }
        },
        "current",
    )
    assert not underfilled["off_policy_proposal_budget_attained"]
    assert underfilled["off_policy_proposal_budget_cap_respected"]
