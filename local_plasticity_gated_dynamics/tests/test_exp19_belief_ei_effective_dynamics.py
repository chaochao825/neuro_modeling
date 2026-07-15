from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import exp19_belief_ei_effective_dynamics as exp19


def _config() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "smoke"
        / "exp19_belief_ei_effective_dynamics.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_exp19_smoke_reuses_one_intact_readout_and_dynamics_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_counts = {"readout": 0, "dynamics": 0}
    original_readout = exp19.fit_receiver_readout
    original_dynamics = exp19.fit_soft_epoch_dynamics

    def counted_readout(*args: object, **kwargs: object):
        fit_counts["readout"] += 1
        return original_readout(*args, **kwargs)

    def counted_dynamics(*args: object, **kwargs: object):
        fit_counts["dynamics"] += 1
        return original_dynamics(*args, **kwargs)

    monkeypatch.setattr(exp19, "fit_receiver_readout", counted_readout)
    monkeypatch.setattr(exp19, "fit_soft_epoch_dynamics", counted_dynamics)
    path = exp19.run_seed(_config(), 0, tmp_path)

    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert fit_counts == {"readout": 1, "dynamics": 1}
    records = _records(path)
    planned = json.loads(
        (path / "planned_conditions.json").read_text(encoding="utf-8")
    )
    expected = {spec.condition for spec in exp19.ALL_CONDITION_SPECS}
    grid_fields = (
        "condition",
        "model_family",
        "controller_mode",
        "belief_intervention",
        "gate_model",
    )
    planned_grid = {tuple(row[field] for field in grid_fields) for row in planned}
    observed_grid = {tuple(row[field] for field in grid_fields) for row in records}
    assert {str(row["condition"]) for row in planned} == expected
    assert {str(row["condition"]) for row in records} == expected
    assert observed_grid == planned_grid
    assert all(row["status"] == "complete" for row in records)
    assert len({str(row["gate_checkpoint_id"]) for row in records}) == 1
    assert all(int(row["gate_fit_trial_count"]) > 0 for row in records)
    assert all(int(row["gate_fit_episode_count"]) > 0 for row in records)
    assert all(0.0 < float(row["estimated_context_hazard"]) < 0.5 for row in records)
    assert all(
        0.5 < float(row["estimated_cue_reliability"]) < 1.0 for row in records
    )
    assert all(isinstance(row["moment_anchor_identifiable"], bool) for row in records)
    assert all(float(row["cue_signal_z_score"]) >= 0.0 for row in records)
    assert len({str(row["planned_condition_grid_id"]) for row in records}) == 1
    assert all(int(row["planned_condition_count"]) == len(expected) for row in records)
    assert {str(row["observed_grid_cell_id"]) for row in records} == expected

    ei_rows = [row for row in records if row["uses_ei_receiver"]]
    assert len(ei_rows) == len(exp19.EI_CONDITION_SPECS)
    assert len({str(row["gate_checkpoint_id"]) for row in ei_rows}) == 1
    assert len({str(row["network_init_id"]) for row in ei_rows}) == 1
    assert len({str(row["readout_checkpoint_id"]) for row in ei_rows}) == 1
    assert len({str(row["dynamics_checkpoint_id"]) for row in ei_rows}) == 1
    assert len({str(row["split_id"]) for row in ei_rows}) == 1
    assert len({str(row["shared_noise_id"]) for row in ei_rows}) == 1
    assert len({str(row["intact_belief_trajectory_id"]) for row in ei_rows}) == 1
    assert all(row["readout_reused_from_intact_train"] for row in ei_rows)
    assert all(row["dynamics_reused_from_intact_train"] for row in ei_rows)
    assert all(row["preprocessing_fit_train_only"] for row in ei_rows)
    assert all(row["dynamics_fit_train_only"] for row in ei_rows)
    assert all(not row["dynamics_heldout_used_for_fit"] for row in ei_rows)
    assert all(row["gate_timing"] == exp19.GATE_TIMING for row in records)
    assert all(row["current_cue_accessed_for_same_trial"] for row in records)
    assert all(row["cue_available_before_receiver_control"] for row in records)
    assert all(not row["receiver_received_cue_channels"] for row in records)
    assert all(
        float(row["test_episode_start_abs_belief_from_half"]) > 0.0
        for row in records
    )

    n_units = int(_config()["network"]["n_units"])
    assert all(int(row["physical_raw_rank"]) >= int(0.9 * n_units) for row in ei_rows)
    assert all(row["physical_dale_valid"] for row in ei_rows)
    assert all(int(row["dale_excitatory_violation_count"]) == 0 for row in ei_rows)
    assert all(int(row["dale_inhibitory_violation_count"]) == 0 for row in ei_rows)
    assert all(row["coarse_dynamics_applicable"] for row in ei_rows)
    assert all(
        "population_gain_sensory_delay_response"
        in str(row["receiver_control_epoch_scope"])
        for row in ei_rows
    )
    assert all(not row["full_trajectory_lds"] for row in ei_rows)
    assert all(
        "not_full_trajectory_lds" in str(row["approximation_scope"])
        for row in ei_rows
    )
    assert all(float(row["heldout_normalized_closure_mse"]) >= 0.0 for row in ei_rows)
    assert all(int(row["local_state_tangent_dimension"]) == 4 for row in ei_rows)
    assert all(
        float(row["rate_to_state_pullback_residual"]) >= 0.0 for row in ei_rows
    )
    assert all(
        int(row["rate_to_state_derivative_active_count"]) >= 4 for row in ei_rows
    )
    assert all(
        "pseudoinverse_preactivation_pullback" in str(row["normal_basis_scope"])
        for row in ei_rows
    )
    eligibility = {bool(row["normal_stability_eligible"]) for row in ei_rows}
    assert len(eligibility) == 1
    if eligibility == {True}:
        assert all(int(row["normal_dimension"]) == n_units - 4 for row in ei_rows)
    else:
        assert all(row["normal_dimension"] is None for row in ei_rows)
        assert all(
            row["normal_stability_ineligibility_reason"]
            == "rate_basis_pullback_rank_or_residual_threshold_failed"
            for row in ei_rows
        )

    by_condition = {str(row["condition"]): row for row in records}
    assert by_condition["md_combined_intact"][
        "empirical_combined_control_trajectory_rank"
    ] == 1
    assert by_condition["md_combined_clamp"][
        "empirical_combined_control_trajectory_rank"
    ] == 0
    assert by_condition["md_disconnected"][
        "combined_effective_control_dimension"
    ] == 0
    assert by_condition["md_disconnected"][
        "empirical_combined_control_trajectory_rank"
    ] == 0
    assert by_condition["md_population_only"]["pathway_control_rank"] == 0
    assert by_condition["md_pathway_only"]["population_gain_control_rank"] == 0

    direct = by_condition["direct_evidence_mix"]
    assert direct["direct_baseline_separate_from_ei"]
    assert not direct["uses_ei_receiver"]
    assert not direct["coarse_dynamics_applicable"]
    assert direct["dynamics_checkpoint_id"] is None
    assert direct["physical_raw_rank"] is None


def test_exp19_setup_failure_preserves_the_complete_registered_grid(
    tmp_path: Path,
) -> None:
    config = _config()
    config["gate_timing"] = "predictive_prior_before_current_cue"
    path = exp19.run_seed(config, 4, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
    records = _records(path)
    assert len(records) == len(exp19.ALL_CONDITION_SPECS)
    assert {str(row["condition"]) for row in records} == {
        spec.condition for spec in exp19.ALL_CONDITION_SPECS
    }
    assert all(row["status"] == "failed" for row in records)
    assert all(row["error_type"] == "ValueError" for row in records)
    assert all("gate_timing must equal" in str(row["error"]) for row in records)


def test_exp19_ineligible_normal_audit_retains_behavioral_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_normal(*args: object, **kwargs: object) -> object:
        raise np.linalg.LinAlgError("synthetic normal-space failure")

    monkeypatch.setattr(exp19, "projected_normal_linear_summary", fail_normal)
    config = _config()
    config["effective_dynamics"]["rate_pullback_residual_tolerance"] = 1.0
    path = exp19.run_seed(config, 2, tmp_path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    ei_rows = [row for row in _records(path) if row["uses_ei_receiver"]]
    assert all(row["status"] == "complete" for row in ei_rows)
    assert all(not row["normal_stability_eligible"] for row in ei_rows)
    assert all(row["normal_local_decay_ratio"] is None for row in ei_rows)
    assert all(
        "projected_normal_numerical_failure" in str(
            row["normal_stability_ineligibility_reason"]
        )
        for row in ei_rows
    )


def test_exp19_planned_delay_label_is_config_bound() -> None:
    config = _config()
    config["interventions"]["delay_trials"] = 3
    planned = exp19._planned_conditions(config)
    delay = next(row for row in planned if row["condition"] == "md_combined_delay")
    assert delay["belief_intervention"] == "delay_3"
