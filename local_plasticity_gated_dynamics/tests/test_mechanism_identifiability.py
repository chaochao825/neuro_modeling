from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tasks.context_integration import (
    ContextIntegrationConfig,
    generate_context_integration,
)
from src.training.context_local import balanced_block_split
from src.training.mechanism_identifiability import (
    build_mechanism_conditions,
    prepare_paired_resources,
    run_mechanism_condition,
)


PROJECT = Path(__file__).resolve().parents[1]


def _resources():
    config = json.loads(
        (PROJECT / "configs" / "smoke" / "exp02_context_ei_oracle_gate.json").read_text(
            encoding="utf-8"
        )
    )
    batch = generate_context_integration(
        ContextIntegrationConfig(**config["task"]), seed=5
    )
    train, test = balanced_block_split(batch, test_fraction=0.25, seed=17)
    resources = prepare_paired_resources(
        train,
        test,
        config["architectures"][1],
        {**config["training"], "train_epochs": 2},
        seed=23,
    )
    return config, train, test, resources


def test_p0_condition_grid_separates_geometry_and_components() -> None:
    conditions = build_mechanism_conditions()
    assert len(conditions) == 32
    for norm in ("l1", "l2"):
        panel = [item for item in conditions if item.budget_norm == norm]
        geometry = [
            item.feedback_mode
            for item in panel
            if item.mechanism == "task+homeostasis+normalization"
        ]
        assert geometry == ["aligned", "random", "orthogonal", "shuffled", "full"]
        task_only_geometry = [
            item.feedback_mode for item in panel if item.mechanism == "task-only"
        ]
        assert task_only_geometry == [
            "aligned",
            "random",
            "orthogonal",
            "shuffled",
            "full",
        ]
        assert {item.mechanism for item in panel} == {
            "task+homeostasis+normalization",
            "task+homeostasis",
            "task-only",
            "task+normalization",
            "homeostasis-only",
            "homeostasis+normalization",
            "normalization-only",
            "frozen-recurrent",
        }


def test_real_branch_arrays_and_replay_tape_have_stable_fingerprints() -> None:
    config, train, test, first = _resources()
    second = prepare_paired_resources(
        train,
        test,
        config["architectures"][1],
        {**config["training"], "train_epochs": 2},
        seed=23,
    )
    for field in (
        "initialization_id",
        "readout_training_id",
        "gate_id",
        "homeostasis_signal_id",
        "noise_contract_id",
        "replay_contract_id",
    ):
        assert getattr(first, field) == getattr(second, field)
    assert first.tape.trial_order_id == second.tape.trial_order_id
    assert first.tape.noise_id == second.tape.noise_id
    assert first.tape.split_id == second.tape.split_id
    assert first.tape.replay_contract_id == second.tape.replay_contract_id
    assert first.tape.train_data_id == second.tape.train_data_id
    assert first.tape.test_data_id == second.tape.test_data_id
    np.testing.assert_array_equal(first.readout, second.readout)
    np.testing.assert_array_equal(first.oracle_gains, second.oracle_gains)
    for array in (
        first.reference_rate_tape,
        first.feedback_signal_tape,
        first.shuffled_feedback_signal_tape,
        first.homeostasis_reference_weights,
    ):
        assert array.flags.writeable is False

    source = first.tape.shuffled_source_indices
    assert np.all(source != np.arange(source.size))
    for trial, shuffled in enumerate(source):
        assert train.block_ids[trial] == train.block_ids[shuffled]
        assert train.contexts[trial] == train.contexts[shuffled]
        np.testing.assert_array_equal(
            first.shuffled_feedback_signal_tape[:, trial],
            first.feedback_signal_tape[:, shuffled],
        )
    changed_inputs = np.array(train.inputs, copy=True)
    changed_inputs[0, 0, 0] += 0.5
    changed_train = replace(train, inputs=changed_inputs)
    frozen = next(
        item
        for item in build_mechanism_conditions(("l1",))
        if item.mechanism == "frozen-recurrent"
    )
    with pytest.raises(ValueError, match="train data"):
        run_mechanism_condition(
            changed_train,
            test,
            first,
            frozen,
            {**config["training"], "train_epochs": 2},
        )


def test_feedback_controls_share_source_and_aligned_shuffled_projector() -> None:
    _, _, _, resources = _resources()
    design = resources.feedback
    features = np.concatenate(
        [np.asarray([1.0, 0.2, -0.3, 1.0, 0.0]), np.linspace(0.0, 1.0, 16)]
    )
    source = design.shared_encoder @ features
    np.testing.assert_allclose(
        design.project(source, "aligned"), design.project(source, "shuffled")
    )
    np.testing.assert_allclose(design.project(source, "full"), source)
    assert design.encoder_id
    assert design.design_id
    assert np.linalg.norm(design.task_basis.T @ design.orthogonal_basis) < 1e-10


def test_p0_local_module_has_no_torch_or_autograd_path() -> None:
    path = PROJECT / "src" / "training" / "mechanism_identifiability.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "torch" or name.startswith("torch.") for name in imports)
    assert ".backward(" not in source


def test_branch_order_does_not_change_results_and_frozen_is_exact() -> None:
    config, train, test, resources = _resources()
    training = {
        **config["training"],
        "train_epochs": 2,
        "total_update_budget": 1e-5,
        "network_noise_std": resources.training_noise_std,
        "evaluation_noise_std": resources.evaluation_noise_std,
    }
    lookup = {item.name: item for item in build_mechanism_conditions(("l1",))}
    aligned = run_mechanism_condition(
        train,
        test,
        resources,
        lookup["task-homeostasis-normalization__aligned__l1"],
        training,
    )
    shuffled = run_mechanism_condition(
        train,
        test,
        resources,
        lookup["task-homeostasis-normalization__shuffled__l1"],
        training,
    )
    repeated = run_mechanism_condition(
        train,
        test,
        resources,
        lookup["task-homeostasis-normalization__aligned__l1"],
        training,
    )
    frozen = run_mechanism_condition(
        train,
        test,
        resources,
        lookup["frozen-recurrent__aligned__l1"],
        training,
    )
    np.testing.assert_array_equal(aligned.predictions, repeated.predictions)
    np.testing.assert_array_equal(aligned.activity, repeated.activity)
    for key in (
        "initialization_id",
        "readout_training_id",
        "gate_id",
        "trial_order_id",
        "noise_id",
        "replay_contract_id",
        "homeostasis_signal_id",
        "feedback_encoder_id",
    ):
        assert aligned.metrics[key] == shuffled.metrics[key]
    assert (
        aligned.metrics["homeostasis_component_id"]
        == shuffled.metrics["homeostasis_component_id"]
    )
    assert (
        aligned.metrics["feedback_signal_tape_id"]
        != shuffled.metrics["feedback_signal_tape_id"]
    )
    assert aligned.metrics["feedback_projector_rank"] == 4
    assert aligned.metrics["feedback_projected_signal_span"] <= 4
    assert (
        aligned.metrics["feedback_credit_source_span"]
        >= aligned.metrics["feedback_projected_signal_span"]
    )
    assert aligned.metrics["budget_match_valid"] is True
    assert shuffled.metrics["budget_match_valid"] is True
    assert aligned.metrics["shuffled_fixed_point_fraction"] == 0.0

    mismatched_noise = {
        **training,
        "evaluation_noise_std": resources.evaluation_noise_std + 0.01,
    }
    with pytest.raises(ValueError, match="evaluation_noise_std"):
        run_mechanism_condition(
            train,
            test,
            resources,
            lookup["frozen-recurrent__aligned__l1"],
            mismatched_noise,
        )
    assert frozen.metrics["frozen_exact"] is True
    assert frozen.metrics["recurrent_changed"] is False
    assert frozen.metrics["task_component_l1"] == 0.0
    assert frozen.metrics["homeostasis_component_l1"] == 0.0
    assert frozen.metrics["normalization_component_l1"] == 0.0
    assert aligned.metrics["dale_valid"] is True
