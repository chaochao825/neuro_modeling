from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.common import load_json_config
from experiments.exp08_rank_stage_validation import (
    STAGE_NAMES,
    build_rank_stage_conditions,
    construct_feedback_basis,
    evaluate_rank_stage_condition,
    prepare_shared_resources,
    run_seed,
)


PROJECT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT / "configs" / "smoke" / "exp08_rank_stage_validation.json"
FORMAL_CONFIG = PROJECT / "configs" / "formal" / "exp08_rank_stage_validation.json"


def _records(run_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_path / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_formal_rank_stage_grid_has_required_scale_and_retains_invalid_cells() -> None:
    config = load_json_config(FORMAL_CONFIG)
    assert config["seeds"] == list(range(30))
    assert config["feedback_dims"] == [1, 2, 4, 8, 16, 32, "full"]
    assert config["feedback_angles_degrees"] == [0, 15, 30, 45, 60, 75, 90]
    assert config["parameterizations"] == [
        "direct",
        "multiplicative",
        "full-per-synapse",
    ]
    assert config["trajectory"]["train_trials"] == config["trajectory"]["test_trials"]
    assert config["hankel"]["n_permutations"] >= 100

    conditions = build_rank_stage_conditions(config)
    assert len(conditions) == 7 * 7 * 3
    invalid = [condition for condition in conditions if not condition.geometry_valid]
    assert len(invalid) == 6 * 3
    assert all(condition.requested_feedback_dim == "full" for condition in invalid)
    assert all(condition.feedback_angle_degrees != 0.0 for condition in invalid)


def test_feedback_geometry_realizes_requested_principal_angle() -> None:
    frame, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(8, 8)))
    task = frame[:, :2]
    complement = frame[:, 2:]

    basis, geometry = construct_feedback_basis(task, complement, 2, 45.0)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(
        geometry["feedback_principal_angles_degrees"], [45.0, 45.0], atol=1e-10
    )
    assert geometry["actual_feedback_dim"] == 2
    assert geometry["feedback_alignment_fraction"] == pytest.approx(0.5)

    full, full_geometry = construct_feedback_basis(task, complement, "full", 0.0)
    np.testing.assert_array_equal(full, np.column_stack([task, complement]))
    assert full_geometry["actual_feedback_dim"] == 8
    with pytest.raises(ValueError, match="nonzero subspace angle"):
        construct_feedback_basis(task, complement, "full", 15.0)


def test_exp08_fixed_seed_and_condition_replay_are_exact() -> None:
    config = load_json_config(SMOKE_CONFIG)
    first = prepare_shared_resources(config, seed=11)
    second = prepare_shared_resources(config, seed=11)
    assert first.shared_resource_id == second.shared_resource_id
    np.testing.assert_array_equal(first.initial_weights, second.initial_weights)
    np.testing.assert_array_equal(first.train_noise, second.train_noise)

    condition = next(
        item
        for item in build_rank_stage_conditions(config)
        if item.geometry_valid and item.parameterization == "direct"
    )
    first_metrics = evaluate_rank_stage_condition(config, first, condition, seed=11)
    second_metrics = evaluate_rank_stage_condition(config, second, condition, seed=11)
    assert first_metrics == second_metrics


def test_exp08_smoke_retains_complete_grid_and_audit_contract(tmp_path: Path) -> None:
    config = load_json_config(SMOKE_CONFIG)
    run_path = run_seed(config, 0, str(tmp_path))
    planned = json.loads(
        (run_path / "planned_conditions.json").read_text(encoding="utf-8")
    )
    records = _records(run_path)

    assert len(planned) == 3 * 2 * 3
    assert len(records) == len(planned)
    assert sum(record["status"] == "invalid" for record in records) == 3
    assert sum(record["status"] == "complete" for record in records) == 15
    assert {record["status"] for record in records} == {"complete", "invalid"}
    invalid = [record for record in records if record["status"] == "invalid"]
    assert all(record["requested_feedback_dim"] == "full" for record in invalid)
    assert all("principal angle" in record["reason"] for record in invalid)

    complete = [record for record in records if record["status"] == "complete"]
    assert {record["parameterization"] for record in complete} == {
        "direct",
        "multiplicative",
        "full-per-synapse",
    }
    shared_identifiers = (
        "initialization_id",
        "mask_id",
        "state_id",
        "noise_id",
        "trajectory_tape_id",
        "feedback_frame_id",
        "edge_modulation_id",
        "hankel_feature_id",
        "shared_resource_id",
    )
    for name in shared_identifiers:
        assert len({record[name] for record in complete}) == 1
    for geometry_cell in {
        (record["requested_feedback_dim"], record["feedback_angle_degrees"])
        for record in complete
    }:
        paired = [
            record
            for record in complete
            if (
                record["requested_feedback_dim"],
                record["feedback_angle_degrees"],
            )
            == geometry_cell
        ]
        assert len({record["feedback_basis_id"] for record in paired}) == 1
        assert len({record["feedback_channel_prefix_id"] for record in paired}) == 1
        assert len({record["feedback_modulator_id"] for record in paired}) == 1

    required_fields = {
        "masked_identity_equal",
        "masked_identity_max_abs_residual",
        "masked_identity_rank_details",
        "lowdim_credit_tangent_dimension",
        "lowdim_credit_tangent_effective_dimension",
        "jacobian_outlier_count",
        "jacobian_bulk_right_edge",
        "activity_participation_ratio",
        "hankel_raw_numerical_rank",
        "hankel_raw_effective_rank",
        "hankel_noise_adjusted_dimension",
        "hankel_dimension_thresholds",
        "parameterization_costs",
        "parameterization_implementation",
        "parameterization_control_tangent_dimension",
        "parameterization_control_tangent_definition",
        "lowdim_tangent_is_full_parameterization_tangent",
        "dale_projection_changed_synapse_count",
        "dale_boundary_synapse_count",
    }
    required_fields.update(
        f"{stage}_{kind}"
        for stage in STAGE_NAMES
        for kind in ("numerical_rank", "effective_rank")
    )
    required_fields.update(
        f"parameterization_{stage}_{kind}_rank"
        for stage in (
            "raw_control",
            "masked_control",
            "applied_control",
            "raw_weight",
            "masked_weight",
            "applied_weight_pre_dale",
            "dale_applied_weight",
        )
        for kind in ("numerical", "effective")
    )
    for record in complete:
        assert required_fields <= record.keys()
        assert record["masked_identity_equal"] is True
        assert record["masked_identity_exact_rank_preservation_expected"] is True
        assert record["masked_identity_numerically_preserves_mask_rank"] is True
        assert record["dale_valid_after_update"] is True
        assert record["hankel_preprocessor_train_only"] is True
        assert record["hankel_test_activity_used_for_fit"] is False
        assert (
            record["activity_centering_scope"] == "train_fitted_mean_applied_to_heldout"
        )
        assert record["hankel_threshold_source"] == (
            "train_fitted_within_trial_permutation"
        )
        assert record["statistics_unit"] == "seed"
        assert record["used_autograd"] is False
        assert record["cross_parameterization_budget_matched"] is False
        assert record["cross_parameterization_dynamic_metrics_scope"] == (
            "descriptive_only_until_applied_update_budget_is_matched"
        )
        assert (
            record["hankel_train_trial_count"] == config["trajectory"]["train_trials"]
        )
        assert record["hankel_test_trial_count"] == config["trajectory"]["test_trials"]
        assert record["train_activity_shape"] == [
            config["trajectory"]["train_trials"],
            config["trajectory"]["time_steps"],
            config["architecture"]["n_units"],
        ]
        assert record["test_activity_shape"] == [
            config["trajectory"]["test_trials"],
            config["trajectory"]["time_steps"],
            config["architecture"]["n_units"],
        ]
        assert len(record["hankel_dimension_thresholds"]) == (
            config["hankel"]["past_lags"] * config["hankel"]["feature_count"]
        )
        assert record["hankel_train_window_count"] == record["hankel_test_window_count"]

    full_per_synapse = [
        record
        for record in complete
        if record["parameterization"] == "full-per-synapse"
    ]
    assert all(
        record["parameterization_credit_dimension"] > record["actual_feedback_dim"]
        for record in full_per_synapse
    )
    assert all(
        record["parameterization_control_tangent_dimension"]
        == record["parameterization_credit_dimension"]
        and record["lowdim_tangent_is_full_parameterization_tangent"] is False
        and record["full_per_synapse_control_space_exhaustively_sampled"] is False
        for record in full_per_synapse
    )
    assert all(
        record["lowdim_tangent_is_full_parameterization_tangent"] is True
        for record in complete
        if record["parameterization"] != "full-per-synapse"
    )
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["condition_failures"] == 0
    assert status["condition_invalid"] == 3


def test_multiplicative_control_bound_is_not_mislabeled_as_dale_projection() -> None:
    config = copy.deepcopy(load_json_config(SMOKE_CONFIG))
    config["update"]["learning_rate"] = 1.0
    config["update"]["max_abs_log_step"] = 1e-5
    resources = prepare_shared_resources(config, seed=3)
    condition = next(
        item
        for item in build_rank_stage_conditions(config)
        if item.geometry_valid and item.parameterization == "multiplicative"
    )
    metrics = evaluate_rank_stage_condition(config, resources, condition, seed=3)

    assert metrics["parameterization_control_bound_active"] is True
    assert metrics["control_bound_changed_synapse_count"] > 0
    assert metrics["control_bound_correction_numerical_rank"] > 0
    assert metrics["dale_projection_changed_synapse_count"] == 0
    assert metrics["dale_projection_correction_numerical_rank"] == 0
    assert metrics["parameterization_applied_weight_pre_dale_numerical_rank"] > 0


def test_exp08_shared_resource_failure_is_retained_for_every_cell(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(load_json_config(SMOKE_CONFIG))
    config["architecture"]["n_units"] = 1
    run_path = run_seed(config, 0, str(tmp_path))
    planned = json.loads(
        (run_path / "planned_conditions.json").read_text(encoding="utf-8")
    )
    records = _records(run_path)

    assert len(planned) == 18
    assert len(records) == len(planned)
    assert sum(record["status"] == "failed" for record in records) == 15
    assert sum(record["status"] == "invalid" for record in records) == 3
    assert all(
        record["error_type"] == "ValueError"
        for record in records
        if record["status"] == "failed"
    )
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
    assert status["condition_failures"] == 15
    assert status["condition_invalid"] == 3
