from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.context_integration import (
    ContextIntegrationConfig,
    generate_context_integration,
)
from src.training.context_local import (
    Phase2Condition,
    balanced_block_split,
    build_phase2_conditions,
    run_context_condition,
)


PROJECT = Path(__file__).resolve().parents[1]


def _smoke_data(profile: str = "exp02_context_ei_oracle_gate.json"):
    config = json.loads((PROJECT / "configs" / "smoke" / profile).read_text())
    batch = generate_context_integration(ContextIntegrationConfig(**config["task"]), seed=0)
    train, test = balanced_block_split(batch, test_fraction=0.25, seed=11)
    return config, train, test


def test_required_condition_matrix_and_block_safe_split() -> None:
    names = [condition.name for condition in build_phase2_conditions("oracle")]
    assert names == [
        "local",
        "bptt",
        "readout-only",
        "no-gate",
        "no-homeostasis",
        "full-feedback",
        "shuffled-feedback",
        "separate-network",
    ]
    _, train, test = _smoke_data()
    assert not np.intersect1d(train.block_ids, test.block_ids).size
    assert set(train.contexts) == {0, 1}
    assert set(test.contexts) == {0, 1}
    contiguous_switches = np.flatnonzero(
        (np.diff(test.trial_ids) == 1) & (np.diff(test.contexts) != 0)
    )
    assert contiguous_switches.size >= 1


def test_formal_episode_split_tracks_requested_trial_fraction() -> None:
    config = json.loads(
        (PROJECT / "configs" / "formal" / "exp02_context_ei_oracle_gate.json").read_text()
    )
    batch = generate_context_integration(ContextIntegrationConfig(**config["task"]), seed=7)
    train, test = balanced_block_split(
        batch,
        test_fraction=float(config["test_fraction"]),
        seed=19,
        switch_window=int(config["training"]["switch_window"]),
    )
    assert not np.intersect1d(train.block_ids, test.block_ids).size
    assert abs(test.inputs.shape[0] / batch.inputs.shape[0] - config["test_fraction"]) <= 0.01


def test_oracle_local_training_reports_all_required_metric_families() -> None:
    config, train, test = _smoke_data()
    condition = build_phase2_conditions("oracle")[0]
    result = run_context_condition(
        train,
        test,
        condition,
        config["architectures"][1],
        config["training"],
        seed=3,
    )

    required = {
        "accuracy",
        "switch_cost",
        "forgetting",
        "raw_update_effective_rank",
        "masked_update_effective_rank",
        "applied_update_effective_rank",
        "total_update_effective_rank",
        "activity_participation_ratio",
        "jacobian_max_real_part",
        "context_subspace_overlap",
        "reduced_heldout_r2",
        "firing_rate_energy",
        "synaptic_event_energy",
        "plasticity_update_energy",
    }
    assert required <= result.metrics.keys()
    assert result.metrics["used_autograd"] is False
    assert result.metrics["training_algorithm"] == "causal_online_three_factor"
    assert result.metrics["gate_context_accuracy"] == 1.0
    assert result.metrics["switch_cost_estimable"] is True
    assert result.activity.shape == (*test.inputs.shape[:2], 16)
    assert np.isfinite(result.metrics["raw_update_effective_rank"])
    expected_forgetting = np.mean(
        np.asarray(result.metrics["retention_before_by_context"])
        - np.asarray(result.metrics["retention_after_by_context"])
    )
    assert result.metrics["forgetting"] == pytest.approx(expected_forgetting)
    assert "before any joint replay" in result.metrics["forgetting_definition"]
    assert result.metrics["plasticity_weight_event_count"] > 0
    assert (
        result.metrics["homeostasis_plasticity_weight_event_count"]
        >= result.metrics["homeostasis_local_plasticity_weight_event_count"]
    )
    assert result.metrics["plasticity_update_energy_per_weight_event"] == pytest.approx(
        result.metrics["plasticity_update_energy"]
        / result.metrics["plasticity_weight_event_count"]
    )
    dense_event_count = (
        result.metrics["task_update_count"]
        + result.metrics["homeostasis_update_count"]
    ) * 16 * 16
    assert result.metrics["plasticity_weight_event_count"] < dense_event_count


def test_local_training_is_deterministic_for_fixed_seed() -> None:
    config, train, test = _smoke_data()
    condition = build_phase2_conditions("oracle")[0]
    first = run_context_condition(
        train,
        test,
        condition,
        config["architectures"][0],
        config["training"],
        seed=23,
    )
    second = run_context_condition(
        train,
        test,
        condition,
        config["architectures"][0],
        config["training"],
        seed=23,
    )

    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_array_equal(first.activity, second.activity)


def test_paired_controls_share_initialization_and_non_dale_homeostasis_is_rejected() -> None:
    config, train, test = _smoke_data()
    conditions = {item.name: item for item in build_phase2_conditions("oracle")}
    training = dict(config["training"])
    training.update(recurrent_learning_rate=0.0, weight_decay=0.0)
    readout_only = run_context_condition(
        train,
        test,
        conditions["readout-only"],
        config["architectures"][1],
        training,
        seed=29,
    )
    no_homeostasis = run_context_condition(
        train,
        test,
        conditions["no-homeostasis"],
        config["architectures"][1],
        training,
        seed=29,
    )
    np.testing.assert_array_equal(readout_only.predictions, no_homeostasis.predictions)
    assert readout_only.metrics["initialization_id"] == no_homeostasis.metrics[
        "initialization_id"
    ]

    with pytest.raises(ValueError, match="only an interpretable control"):
        run_context_condition(
            train,
            test,
            conditions["no-homeostasis"],
            config["architectures"][0],
            training,
            seed=29,
        )

    signed_ei = {**config["architectures"][1], "activation": "tanh"}
    with pytest.raises(ValueError, match="requires rectified_tanh"):
        run_context_condition(
            train,
            test,
            conditions["local"],
            signed_ei,
            training,
            seed=29,
        )


def test_learned_gate_runs_oracle_then_hebbian_md_stage() -> None:
    config, train, test = _smoke_data("exp03_context_ei_learned_gate.json")
    condition = build_phase2_conditions("learned")[0]
    result = run_context_condition(
        train,
        test,
        condition,
        config["architectures"][0],
        config["training"],
        seed=9,
    )

    assert result.metrics["gate_stage"] == "hebbian_pfc_to_md_then_frozen_inference"
    assert 0.0 <= result.metrics["gate_context_accuracy"] <= 1.0
    assert result.metrics["used_autograd"] is False


def test_shuffled_feedback_does_not_shuffle_readout_teacher() -> None:
    config, train, test = _smoke_data()
    aligned = Phase2Condition(
        "readout-aligned", "local", False, "oracle", False, "low_dimensional"
    )
    shuffled = Phase2Condition(
        "readout-shuffled", "local", False, "oracle", False, "shuffled"
    )
    aligned_result = run_context_condition(
        train,
        test,
        aligned,
        config["architectures"][0],
        config["training"],
        seed=13,
    )
    shuffled_result = run_context_condition(
        train,
        test,
        shuffled,
        config["architectures"][0],
        config["training"],
        seed=13,
    )
    np.testing.assert_allclose(aligned_result.predictions, shuffled_result.predictions)


def test_local_training_module_has_no_direct_torch_or_autograd_import() -> None:
    source = (PROJECT / "src" / "training" / "context_local.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "torch" or name.startswith("torch.") for name in imports)
    assert ".backward(" not in source
    assert "torch.autograd" not in source
