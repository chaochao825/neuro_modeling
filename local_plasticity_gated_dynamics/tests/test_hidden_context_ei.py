"""Contracts for the rank-one hidden-belief to frozen E/I bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.common import load_json_config
from experiments.exp10_hidden_context_ei_bridge import run_seed
from src.analysis.end_to_end_energy import end_to_end_compute_proxy
from src.models.belief_gain import (
    balanced_gain_axis,
    belief_gain_trajectory,
    gain_control_rank,
)


def test_balanced_gain_is_rank_one_and_cue_neutral() -> None:
    excitatory = np.array([True, True, True, True, False, False])
    axis = balanced_gain_axis(excitatory, seed=7)
    np.testing.assert_allclose(np.mean(axis[excitatory]), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.mean(axis[~excitatory]), 0.0, atol=1e-15)
    posterior = np.array([0.1, 0.5, 0.9])
    epochs = np.array(["cue", "sensory", "delay"])
    trajectory = belief_gain_trajectory(posterior, epochs, axis, strength=0.6)
    np.testing.assert_allclose(trajectory.gains[:, 0], 1.0)
    assert trajectory.control_rank == 1
    assert np.min(trajectory.gains) > 0.0

    neutral = belief_gain_trajectory(np.full(3, 0.5), epochs, axis, strength=0.6)
    assert gain_control_rank(neutral.gains) == 0
    np.testing.assert_allclose(neutral.gains, 1.0)


def test_end_to_end_compute_counts_declared_gate_not_posterior_amplitude() -> None:
    features = np.array([[1.0, -2.0], [0.5, 1.0]])
    weights = np.array([2.0, -1.0])
    summary = end_to_end_compute_proxy(
        n_trials=2,
        input_event_sum=10.0,
        recurrent_event_sum=20.0,
        firing_sum=30.0,
        readout_features=features,
        readout_weights=weights,
        gate_operations_per_trial=8.0,
        gate_state_updates_per_trial=2.0,
    )
    assert summary.input_weighted_events == 5.0
    assert summary.recurrent_weighted_events == 10.0
    assert summary.receiver_firing_magnitude == 15.0
    assert summary.readout_weighted_events == 3.0
    assert summary.total_compute_proxy == 28.0
    assert "not_atp" in summary.interpretation


def test_exp10_smoke_preserves_pairing_and_intervention_checkpoints(
    tmp_path: Path,
) -> None:
    config = load_json_config("configs/smoke/exp10_hidden_context_ei_bridge.json")
    path = run_seed(config, 0, str(tmp_path))
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 7
    assert {record["status"] for record in records} == {"complete"}
    assert len({record["network_init_id"] for record in records}) == 1
    assert len({record["gain_axis_id"] for record in records}) == 1
    assert all(not record["gate_test_accessed_true_context"] for record in records)
    assert all(record["preprocessing_fit_train_only"] for record in records)
    assert all(not record["base_conditions_share_readout"] for record in records)
    assert all(not record["efficiency_claim_eligible"] for record in records)
    assert all(record["plasticity_l1_cost"] == 0.0 for record in records)
    no_gate = next(record for record in records if record["gate_model"] == "no_gate")
    assert no_gate["effective_control_rank"] == 0
    md = next(
        record
        for record in records
        if record["gate_model"] == "md_recurrent_belief"
        and record["intervention"] == "none"
    )
    interventions = [record for record in records if record["intervention"] != "none"]
    assert len(interventions) == 3
    assert all(
        record["readout_checkpoint_id"] == md["readout_checkpoint_id"]
        for record in interventions
    )
    assert all(
        record["gate_checkpoint_id"] == md["gate_checkpoint_id"]
        for record in interventions
    )
    assert all(record["intervention_postfit"] for record in interventions)
    assert (path / "status.json").exists()
