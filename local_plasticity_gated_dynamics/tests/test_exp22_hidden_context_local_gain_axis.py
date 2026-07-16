"""Smoke contracts for the local hidden-context gain-axis audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import exp22_hidden_context_local_gain_axis as exp22


def _config() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "smoke"
        / "exp22_hidden_context_local_gain_axis.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_exp22_smoke_is_paired_budget_matched_and_leakage_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_count = {"readout": 0}
    original = exp22.fit_receiver_readout

    def counted(*args: object, **kwargs: object):
        fit_count["readout"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(exp22, "fit_receiver_readout", counted)
    path = exp22.run_seed(_config(), 0, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    planned = json.loads((path / "planned_conditions.json").read_text(encoding="utf-8"))
    records = _records(path)

    assert status["status"] == "complete"
    assert fit_count == {"readout": 1}
    assert len(planned) == len(records) == 11
    grid_fields = (
        "condition",
        "feedback_condition",
        "budget_norm",
        "third_factor_source",
    )
    assert {tuple(row[field] for field in grid_fields) for row in planned} == {
        tuple(row[field] for field in grid_fields) for row in records
    }
    assert all(row["status"] == "complete" for row in records)
    for field in (
        "network_init_id",
        "gate_checkpoint_id",
        "readout_checkpoint_id",
        "split_id",
        "shared_noise_id",
        "dev_neutral_trajectory_id",
        "aligned_dev_local_tape_id",
        "dev_local_tape_id",
        "dev_trial_order_id",
        "learned_third_factor_id",
        "planned_condition_grid_id",
        "experiment_protocol_id",
    ):
        assert len({str(row[field]) for row in records}) == 1
    assert all(int(row["planned_condition_count"]) == 11 for row in records)
    assert all(row["statistics_unit"] == "seed" for row in records)
    assert all(row["split_unit"] == "episode" for row in records)
    assert all(row["gate_fit_train_only"] for row in records)
    assert all(row["readout_fit_train_only"] for row in records)
    assert all(row["gain_axis_fit_dev_only"] for row in records)
    assert all(not row["test_used_for_axis_fit"] for row in records)
    assert all(not row["gain_axis_learning_closed_loop"] for row in records)
    assert all(
        not row["dev_trajectory_recomputed_after_each_update"] for row in records
    )
    assert all(row["off_policy_frozen_trajectory_proposal_audit"] for row in records)
    assert all(row["train_dev_test_episode_disjoint"] for row in records)
    assert all(not row["gate_fit_accessed_true_context"] for row in records)
    assert all(not row["gate_test_accessed_true_context"] for row in records)
    assert all(not row["axis_test_truth_accessed"] for row in records)
    assert all(
        row["test_gain_control_source"] == "learned_belief_posterior" for row in records
    )
    assert all(not row["used_bptt"] for row in records)
    assert all(not row["used_autograd"] for row in records)
    assert all(not row["recurrent_learning"] for row in records)
    assert all(not row["homeostasis_learning"] for row in records)
    assert all(not row["normalization_learning"] for row in records)
    assert all(
        not row["generic_recurrent_three_factor_claim_eligible"] for row in records
    )
    assert all(
        not row["closed_loop_local_plasticity_claim_eligible"] for row in records
    )
    assert all(not row["gain_axis_local_plasticity_claim_eligible"] for row in records)
    assert all(not row["weight_transport_free_claim"] for row in records)
    assert all(row["proposal_scale_predeclared_in_config"] for row in records)
    assert all(not row["proposal_scale_fit_on_test"] for row in records)
    assert all(not row["budget_controller_can_amplify_proposals"] for row in records)
    assert all(not row["budget_controller_used"] for row in records)
    assert all(not row["budget_matcher_can_amplify_proposals"] for row in records)
    assert all(
        not row["proposal_coordinate_permutation_after_eligibility"] for row in records
    )

    by_condition = {str(row["condition"]): row for row in records}
    frozen = by_condition["frozen_zero"]
    assert not frozen["gain_axis_learning"]
    assert frozen["budget_attained"]
    assert frozen["budget_total"] == 0.0
    assert frozen["gain_axis_coefficient_l1"] == 0.0
    assert frozen["gain_axis_coefficient_l2"] == 0.0
    assert frozen["frozen_zero_update_budget_baseline"]
    assert not frozen["gain_axis_off_policy_proposal_claim_eligible"]
    assert frozen["condition_dev_local_tape_id"] is None
    assert frozen["condition_third_factor_id"] is None
    assert frozen["feedback_coefficients_id"] is None
    assert frozen["budget_scaling_policy"] == "none_zero_update_baseline"
    assert not frozen["gain_axis_three_factor_rule_used_for_eligibility"]
    assert not frozen["fixed_readout_feedback_coefficients_used"]
    assert not frozen["feedback_transform_applied_before_local_eligibility"]
    assert not frozen["budget_preserves_event_relative_magnitude"]

    learned = [row for row in records if row["condition"] != "frozen_zero"]
    assert all(row["gain_axis_learning"] for row in learned)
    assert all(
        row["gain_axis_three_factor_rule_used_for_eligibility"] for row in learned
    )
    assert all(row["fixed_readout_feedback_coefficients_used"] for row in learned)
    assert all(
        row["feedback_transform_applied_before_local_eligibility"] for row in learned
    )
    assert all(row["budget_preserves_event_relative_magnitude"] for row in learned)
    assert all(not row["frozen_zero_update_budget_baseline"] for row in learned)
    assert all(row["recurrent_weights_bitwise_frozen"] for row in learned)
    assert all(row["budget_attained"] for row in learned)
    assert all(
        float(row["budget_selected_applied"])
        == pytest.approx(float(row["budget_total"]), abs=1e-9)
        for row in learned
    )
    assert all(row["budget_secondary_norm_is_diagnostic_only"] for row in learned)
    assert all(not row["budget_simultaneous_dual_norm_match"] for row in learned)
    assert all(
        int(row["budget_processed_events"]) == int(row["budget_planned_events"])
        for row in learned
    )
    assert all(int(row["budget_raw_nonzero_events"]) > 0 for row in learned)
    assert all(int(row["budget_applied_nonzero_events"]) > 0 for row in learned)
    assert all(int(row["budget_scaled_down_event_count"]) > 0 for row in learned)
    assert all(
        float(row["budget_remaining"]) <= float(row["budget_tolerance"])
        for row in learned
    )
    assert all(
        row["budget_scaling_policy"]
        == "single_global_downscale_preserves_event_relative_magnitude"
        for row in learned
    )
    assert all(0.0 < float(row["budget_global_scale_factor"]) <= 1.0 for row in learned)
    assert all(str(row["budget_path_application_id"]) for row in learned)
    assert {str(row["budget_selected_norm"]) for row in learned} == {"l1", "l2"}

    for norm in ("l1", "l2"):
        assert by_condition[f"aligned_local_{norm}"][
            "gain_axis_off_policy_proposal_claim_eligible"
        ]
        for feedback in (
            "random_signed_feedback",
            "shuffled_feedback",
            "orthogonal_feedback",
        ):
            assert not by_condition[f"{feedback}_{norm}"]["dev_truth_accessed_for_axis"]
        oracle = by_condition[f"oracle_third_factor_{norm}"]
        assert oracle["dev_truth_accessed_for_axis"]
        assert not oracle["gain_axis_local_plasticity_claim_eligible"]
        assert oracle["condition_third_factor_id"] != oracle["learned_third_factor_id"]

        aligned = by_condition[f"aligned_local_{norm}"]
        oracle = by_condition[f"oracle_third_factor_{norm}"]
        assert (
            aligned["condition_dev_local_tape_id"]
            == oracle["condition_dev_local_tape_id"]
        )
        assert aligned["feedback_coefficients_id"] == oracle["feedback_coefficients_id"]
        assert aligned["feedback_angle_to_aligned_degrees"] == pytest.approx(0.0)
        assert oracle["feedback_angle_to_aligned_degrees"] == pytest.approx(0.0)
        orthogonal = by_condition[f"orthogonal_feedback_{norm}"]
        for feedback in (
            "aligned_local",
            "random_signed_feedback",
            "shuffled_feedback",
            "orthogonal_feedback",
        ):
            row = by_condition[f"{feedback}_{norm}"]
            assert row["condition_third_factor_id"] == row["learned_third_factor_id"]
        assert orthogonal["feedback_angle_to_aligned_degrees"] == pytest.approx(
            90.0,
            abs=1e-8,
        )
        assert orthogonal[
            "feedback_coefficient_angle_to_aligned_degrees"
        ] == pytest.approx(90.0, abs=1e-8)
        for epoch in ("sensory", "delay", "response"):
            assert orthogonal[
                f"feedback_{epoch}_angle_to_aligned_degrees"
            ] == pytest.approx(90.0, abs=1e-8)
        assert orthogonal["feedback_coefficient_l2"] == pytest.approx(
            aligned["feedback_coefficient_l2"]
        )

    for feedback in exp22.FEEDBACK_CONDITIONS:
        left = by_condition[f"{feedback}_l1"]
        right = by_condition[f"{feedback}_l2"]
        assert (
            left["condition_dev_local_tape_id"] == right["condition_dev_local_tape_id"]
        )
        assert left["feedback_coefficients_id"] == right["feedback_coefficients_id"]
        assert left["feedback_schedule_id"] == right["feedback_schedule_id"]
        assert left["gain_axis_proposal_id"] == right["gain_axis_proposal_id"]
        assert "same_neuron_local_state" in str(left["condition_local_feedback_scope"])
    assert (
        len(
            {
                by_condition[f"{feedback}_l1"]["condition_dev_local_tape_id"]
                for feedback in (
                    "aligned_local",
                    "random_signed_feedback",
                    "shuffled_feedback",
                    "orthogonal_feedback",
                )
            }
        )
        == 4
    )


def test_exp22_setup_failure_preserves_the_registered_factorial_grid(
    tmp_path: Path,
) -> None:
    config = _config()
    config["gate_timing"] = "predictive_prior_before_current_cue"
    path = exp22.run_seed(config, 7, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    records = _records(path)

    assert status["status"] == "complete_with_failures"
    assert len(records) == 11
    assert {str(row["condition"]) for row in records} == {
        str(row["condition"]) for row in exp22._planned_conditions(config)
    }
    assert all(row["status"] == "failed" for row in records)
    assert all(row["error_type"] == "ValueError" for row in records)
    assert all("gate_timing must equal" in str(row["error"]) for row in records)


def test_exp22_malformed_budget_preserves_all_registered_cells(
    tmp_path: Path,
) -> None:
    config = _config()
    config["gain_axis_learning"]["budgets"] = {"l1": 0.5}
    path = exp22.run_seed(config, 8, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    records = _records(path)

    assert status["status"] == "complete_with_failures"
    assert len(records) == 11
    assert all(row["status"] == "failed" for row in records)
    assert all(row["error_type"] == "ValueError" for row in records)
    assert all(
        "budgets must define exactly l1 and l2" in str(row["error"]) for row in records
    )


def test_exp22_one_feedback_failure_does_not_select_other_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = exp22._condition_proposal

    def fail_random(feedback: str, *args: object, **kwargs: object):
        if feedback == "random_signed_feedback":
            raise RuntimeError("synthetic random-feedback failure")
        return original(feedback, *args, **kwargs)

    monkeypatch.setattr(exp22, "_condition_proposal", fail_random)
    path = exp22.run_seed(_config(), 0, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    records = _records(path)
    failed = {str(row["condition"]) for row in records if row["status"] == "failed"}

    assert status["status"] == "complete_with_failures"
    assert len(records) == 11
    assert failed == {
        "random_signed_feedback_l1",
        "random_signed_feedback_l2",
    }
    assert all(
        row["status"] == "complete" for row in records if row["condition"] not in failed
    )


def test_exp22_feedback_tape_failure_is_isolated_and_retains_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = exp22.build_gain_axis_local_tape

    def fail_random_tape(*args: object, **kwargs: object):
        if kwargs.get("feedback_policy") == "random_signed_readout_feedback":
            raise RuntimeError("synthetic random feedback tape failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(exp22, "build_gain_axis_local_tape", fail_random_tape)
    path = exp22.run_seed(_config(), 0, tmp_path)
    records = _records(path)
    failed = [row for row in records if row["status"] == "failed"]

    assert {str(row["condition"]) for row in failed} == {
        "random_signed_feedback_l1",
        "random_signed_feedback_l2",
    }
    assert all(row["error_type"] == "RuntimeError" for row in failed)
    assert all(
        row["error"] == "synthetic random feedback tape failure" for row in failed
    )
    assert all(
        row["status"] == "complete"
        for row in records
        if row["condition"]
        not in {
            "random_signed_feedback_l1",
            "random_signed_feedback_l2",
        }
    )


def test_exp22_oracle_proposal_failure_does_not_select_non_oracle_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = exp22.make_gain_axis_proposal_tape
    calls = {"count": 0}

    def fail_oracle(*args: object, **kwargs: object):
        calls["count"] += 1
        if calls["count"] == 5:
            raise RuntimeError("synthetic oracle-only failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(exp22, "make_gain_axis_proposal_tape", fail_oracle)
    path = exp22.run_seed(_config(), 0, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    records = _records(path)
    failed = {str(row["condition"]) for row in records if row["status"] == "failed"}

    assert status["status"] == "complete_with_failures"
    assert failed == {
        "oracle_third_factor_l1",
        "oracle_third_factor_l2",
    }
    oracle_failures = [row for row in records if row["condition"] in failed]
    assert all(row["error_type"] == "RuntimeError" for row in oracle_failures)
    assert all(
        row["error"] == "synthetic oracle-only failure" for row in oracle_failures
    )
    assert {
        str(row["condition"]) for row in records if row["status"] == "complete"
    } == {
        "frozen_zero",
        "aligned_local_l1",
        "random_signed_feedback_l1",
        "shuffled_feedback_l1",
        "orthogonal_feedback_l1",
        "aligned_local_l2",
        "random_signed_feedback_l2",
        "shuffled_feedback_l2",
        "orthogonal_feedback_l2",
    }
