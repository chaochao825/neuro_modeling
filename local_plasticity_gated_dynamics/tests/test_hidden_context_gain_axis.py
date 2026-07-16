"""Contracts for dev-only local gain-axis eligibility and proposal tapes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import exp19_belief_ei_effective_dynamics as exp19
from src.models.belief_gain import balanced_gain_axis
from src.models.context_belief import MDRecurrentBeliefGate
from src.tasks.hidden_context import (
    generate_hidden_context,
    make_hidden_context_random_tape,
)
from src.training.hidden_context_ei import fit_receiver_readout, simulate_receiver
from src.training.hidden_context_gain_axis import (
    build_gain_axis_local_tape,
    make_gain_axis_proposal_tape,
)
from src.training.hidden_context_gate import split_hidden_context_dataset
from src.utils.reproducibility import derive_seed


def _setup(seed: int = 0):
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "smoke"
        / "exp19_belief_ei_effective_dynamics.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    task = exp19._task_config(config)
    tape = make_hidden_context_random_tape(task, seed=seed)
    dataset = generate_hidden_context(task, seed=seed, random_tape=tape)
    splits = split_hidden_context_dataset(
        dataset,
        outer_test_fraction=float(config["outer_test_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=seed,
    )
    network = exp19._network(config, task, seed)
    zero_axis = np.zeros(network.n_units)
    gate = MDRecurrentBeliefGate(
        seed=derive_seed(seed, "gain-axis-test", "gate"),
        **dict(config["md_gate"]),
    ).fit(splits.train.gate)
    train_prediction = gate.predict(splits.train.gate)
    dev_prediction = gate.predict(splits.dev.gate)
    common = dict(
        gain_axis=zero_axis,
        gain_strength=0.0,
        integration_substeps=int(config["integration_substeps"]),
        trial_batch_size=32,
        pathway_gating=True,
        population_gain=True,
        record_substeps=True,
    )
    train_sim = simulate_receiver(
        network,
        splits.train,
        train_prediction.context_probability,
        **common,
    )
    readout = fit_receiver_readout(
        train_sim,
        splits.train.task,
        alpha=float(config["readout_alpha"]),
    )
    dev_sim = simulate_receiver(
        network,
        splits.dev,
        dev_prediction.context_probability,
        **common,
    )
    return config, splits, network, readout, dev_sim, dev_prediction


def test_local_tape_is_dev_only_neutral_and_deterministic() -> None:
    config, splits, network, readout, dev_sim, _ = _setup()
    first = build_gain_axis_local_tape(
        network,
        dev_sim,
        splits.dev.task,
        readout,
        integration_substeps=int(config["integration_substeps"]),
        tau_eligibility_steps=3.0,
    )
    second = build_gain_axis_local_tape(
        network,
        dev_sim,
        splits.dev.task,
        readout,
        integration_substeps=int(config["integration_substeps"]),
        tau_eligibility_steps=3.0,
    )
    np.testing.assert_array_equal(first.eligibility, second.eligibility)
    np.testing.assert_array_equal(first.task_error, second.task_error)
    assert first.fingerprint == second.fingerprint
    assert first.eligibility.shape == (
        splits.dev.task.trial_ids.size,
        network.n_units,
    )
    assert set(first.trial_ids).isdisjoint(set(splits.train.task.trial_ids))
    assert set(first.trial_ids).isdisjoint(set(splits.test.task.trial_ids))
    assert not first.eligibility.flags.writeable
    assert not first.feedback_coefficients.flags.writeable
    assert not first.feedback_schedule.flags.writeable
    assert first.feedback_policy == "readout_aligned"
    assert "no_weight_transport_free_claim" in first.local_feedback_scope


def test_feedback_coefficients_are_applied_before_same_neuron_eligibility() -> None:
    config, splits, network, readout, dev_sim, _ = _setup(seed=3)
    aligned = build_gain_axis_local_tape(
        network,
        dev_sim,
        splits.dev.task,
        readout,
        integration_substeps=int(config["integration_substeps"]),
        tau_eligibility_steps=3.0,
    )
    sign_reversed = build_gain_axis_local_tape(
        network,
        dev_sim,
        splits.dev.task,
        readout,
        integration_substeps=int(config["integration_substeps"]),
        tau_eligibility_steps=3.0,
        feedback_coefficients=-aligned.feedback_coefficients,
        feedback_policy="test_sign_reversed_readout_feedback",
    )

    np.testing.assert_allclose(sign_reversed.eligibility, -aligned.eligibility)
    np.testing.assert_allclose(
        sign_reversed.feedback_schedule,
        -aligned.feedback_schedule,
    )
    np.testing.assert_array_equal(sign_reversed.task_error, aligned.task_error)
    np.testing.assert_array_equal(sign_reversed.neutral_scores, aligned.neutral_scores)
    assert sign_reversed.fingerprint != aligned.fingerprint
    assert sign_reversed.feedback_policy == "test_sign_reversed_readout_feedback"


def test_proposal_uses_only_scalar_third_factor_and_matches_oracle_when_equal() -> None:
    config, splits, network, readout, dev_sim, dev_prediction = _setup(seed=1)
    local = build_gain_axis_local_tape(
        network,
        dev_sim,
        splits.dev.task,
        readout,
        integration_substeps=int(config["integration_substeps"]),
        tau_eligibility_steps=2.0,
    )
    posterior_third = 2.0 * dev_prediction.context_probability - 1.0
    learned = make_gain_axis_proposal_tape(
        local,
        posterior_third,
        learning_rate=0.05,
        error_clip=1.0,
    )
    duplicate = make_gain_axis_proposal_tape(
        local,
        posterior_third.copy(),
        learning_rate=0.05,
        error_clip=1.0,
    )
    np.testing.assert_array_equal(learned.proposals, duplicate.proposals)
    assert learned.fingerprint == duplicate.fingerprint
    expected = (
        0.05
        * local.eligibility
        * np.clip(local.task_error, -1.0, 1.0)[:, None]
        * posterior_third[:, None]
    )
    np.testing.assert_allclose(learned.proposals, expected)
    assert not learned.proposals.flags.writeable
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        make_gain_axis_proposal_tape(
            local,
            np.full(local.task_error.size, 1.1),
            learning_rate=0.05,
            error_clip=1.0,
        )


def test_local_tape_rejects_non_neutral_gain() -> None:
    config, splits, network, readout, _, dev_prediction = _setup(seed=2)
    axis = balanced_gain_axis(network.excitatory_mask, seed=9)
    nonneutral = simulate_receiver(
        network,
        splits.dev,
        dev_prediction.context_probability,
        axis,
        gain_strength=0.5,
        integration_substeps=int(config["integration_substeps"]),
        trial_batch_size=32,
        pathway_gating=True,
        population_gain=True,
        record_substeps=True,
    )
    with pytest.raises(ValueError, match="neutral population gain"):
        build_gain_axis_local_tape(
            network,
            nonneutral,
            splits.dev.task,
            readout,
            integration_substeps=int(config["integration_substeps"]),
            tau_eligibility_steps=3.0,
        )
