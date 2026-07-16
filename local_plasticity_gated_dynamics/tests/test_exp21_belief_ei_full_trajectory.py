"""Smoke contracts for the every-substep E/I trajectory audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import exp21_belief_ei_full_trajectory as exp21


def _config() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "smoke"
        / "exp21_belief_ei_full_trajectory.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _record(path: Path) -> dict[str, object]:
    lines = (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_exp21_v2_configs_use_physical_x_perturbation_geometry() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    obsolete = {
        "derivative_tolerance",
        "tangent_pullback_residual_tolerance",
        "minimum_initial_normal_purity",
    }
    for profile in ("smoke", "formal"):
        config = json.loads(
            (root / profile / "exp21_belief_ei_full_trajectory.json").read_text(
                encoding="utf-8"
            )
        )
        perturbation = config["perturbation"]
        assert perturbation["geometry"] == ("joint_state_pca_physical_x_projection_v2")
        assert not obsolete & perturbation.keys()


def test_exp21_smoke_is_train_safe_paired_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    assert config["perturbation"]["geometry"] == (
        "joint_state_pca_physical_x_projection_v2"
    )
    assert (
        not {
            "derivative_tolerance",
            "tangent_pullback_residual_tolerance",
            "minimum_initial_normal_purity",
        }
        & config["perturbation"].keys()
    )
    counts = {"readout": 0}
    original = exp21.fit_receiver_readout

    def counted(*args: object, **kwargs: object):
        counts["readout"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(exp21, "fit_receiver_readout", counted)
    path = exp21.run_seed(config, 0, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    row = _record(path)

    assert status["status"] == "complete"
    assert row["status"] == "complete"
    assert counts == {"readout": 1}
    assert row["condition"] == "md_combined_intact_full_trajectory"
    assert row["experiment_protocol_version"] == "exp21_v2"
    assert row["training_algorithm"] == (
        "md_filtered_belief_full_substep_controlled_affine_koopman_audit_v2"
    )
    assert isinstance(row["experiment_protocol_id"], str)
    assert len(row["experiment_protocol_id"]) == 64
    assert row["statistics_unit"] == "seed"
    assert row["split_unit"] == "episode"
    assert row["primary_receiver_state_reset_scope"] == "every_trial_zero_state"
    assert row["sensitivity_receiver_state_reset_scope"] == "episode_start_only"
    assert row["primary_exp19_physical_semantics_preserved"]
    assert row["episode_continuous_changes_receiver_state_policy"]
    assert not row["used_bptt"]
    assert not row["used_autograd"]
    assert not row["recurrent_learning"]
    assert row["full_trajectory_model"]
    assert not row["full_trajectory_lds"]
    assert not row["autonomous_rollout"]
    assert row["known_future_exogenous_controls_in_rollout"]
    assert row["preprocessing_fit_train_only"]
    assert row["operator_fit_train_only"]
    assert not row["gate_test_accessed_true_context"]
    assert row["trial_reset_trajectory_sequence_scope"] == "trial_reset_state"
    assert (
        row["episode_continuous_trajectory_sequence_scope"]
        == "episode_continuous_state"
    )
    assert row["trial_reset_train_trajectory_shape"][1] == 15
    assert row["trial_reset_test_trajectory_shape"][1] == 15
    assert row["episode_continuous_train_trajectory_shape"][1] > 15
    assert row["episode_continuous_test_trajectory_shape"][1] > 15
    assert row["trial_reset_paired_models_share_state_pca"]
    assert row["episode_continuous_paired_models_share_state_pca"]
    assert row["trial_reset_state_affine_operator_design_full_rank"]
    assert row["episode_continuous_state_affine_operator_design_full_rank"]
    for prefix in ("trial_reset", "episode_continuous"):
        assert row[f"{prefix}_total_operator_design_full_rank"]
        assert row[f"{prefix}_total_operator_design_rank"] == 19
        assert row[f"{prefix}_total_operator_design_columns"] == 19
        assert row[f"{prefix}_total_operator_identifiable_columns"] == 19
        assert row[f"{prefix}_total_operator_unconstrained_columns"] == 20
        assert row[f"{prefix}_total_operator_constraint"] == (
            "shared_neutral_cue_coefficient"
        )
        assert row[f"{prefix}_total_operator_mode"] == ("full_shared_neutral_cue")
        receipt = row[f"{prefix}_total_operator_identifiability_fingerprint"]
        assert isinstance(receipt, str)
        assert len(receipt) == 64
    assert row["trial_reset_population_exogenous_control_delta_frobenius"] == 0.0
    assert row["episode_continuous_population_exogenous_control_delta_frobenius"] == 0.0
    assert row["trial_reset_total_full_n_rollout_windows"] > 0
    assert row["episode_continuous_total_full_n_rollout_windows"] > 0
    assert row["readout_reused_across_state_policies"]
    assert row["network_reused_across_state_policies"]
    assert row["gate_reused_across_state_policies"]
    assert row["gain_axis_reused_across_state_policies"]
    for prefix in ("trial_reset", "episode_continuous"):
        assert row[f"{prefix}_perturbation_geometry"] == (
            "joint_state_pca_physical_x_projection_v2"
        )
        assert row[f"{prefix}_perturbation_report_scope"] == (
            "physical_x_projection_finite_amplitude_recovery"
        )
        assert row[f"{prefix}_perturbation_reference_fraction_policy"] == (
            "sampled_over_planned_eligible_sampled_over_sampled_"
            "eligible_reference_over_planned"
        )
        assert row[f"{prefix}_perturbation_status"] in {
            "complete",
            "ineligible",
        }
        if row[f"{prefix}_perturbation_status"] == "complete":
            assert row[f"{prefix}_perturbation_tangent_basis_space"] == (
                "train_joint_state_pca_physical_x_projection"
            )
            assert row[f"{prefix}_perturbation_tangent_basis_rank"] == 4
            assert (
                0.0
                < row[f"{prefix}_perturbation_tangent_basis_x_block_energy_fraction"]
                <= 1.0
            )
            interpretation = row[f"{prefix}_perturbation_interpretation"]
            assert "finite_amplitude_recovery" in interpretation
            assert "physical_x_projection" in interpretation
            assert row[f"{prefix}_perturbation_sampled_reference_fraction"] == 1.0
            assert row[
                f"{prefix}_perturbation_eligible_sampled_reference_fraction"
            ] == pytest.approx(
                row[f"{prefix}_perturbation_eligible_reference_count"]
                / row[f"{prefix}_perturbation_sampled_reference_count"]
            )
            assert row[f"{prefix}_perturbation_eligible_reference_fraction"] == (
                pytest.approx(
                    row[f"{prefix}_perturbation_eligible_reference_count"]
                    / row[f"{prefix}_perturbation_planned_reference_count"]
                )
            )
            assert (
                row[f"{prefix}_perturbation_baseline_replay_max_abs_error"]
                <= row[f"{prefix}_perturbation_baseline_replay_tolerance"]
            )
    assert row["attractor_anchor_fit_scope"] == "training_trajectory_only"
    assert "not_global_attractor_proof" in row["attractor_interpretation"]


def test_exp21_setup_failure_still_emits_registered_cell(tmp_path: Path) -> None:
    config = _config()
    config["gate_timing"] = "predictive_prior_before_current_cue"
    path = exp21.run_seed(config, 3, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    row = _record(path)
    assert status["status"] == "complete_with_failures"
    assert row["status"] == "failed"
    assert row["condition"] == "md_combined_intact_full_trajectory"
    assert row["error_type"] == "ValueError"
