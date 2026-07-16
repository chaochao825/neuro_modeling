"""Contracts for the standalone Exp21 seed-level snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_exp21 import (
    CONDITION,
    EXPERIMENT,
    _environment_sha256,
    _expected_run_config,
    _holm,
    _require_formal_registration,
    collect_registered_runs,
    summarize_formal_runs,
    write_snapshot_artifacts,
)


def _config(*, n_seeds: int = 30) -> dict[str, object]:
    return {
        "profile": "formal",
        "seeds": list(range(n_seeds)),
        "trajectory_dynamics": {"latent_dim": 4},
        "registered_claim_thresholds": {
            "maximum_rollout_normalized_rmse": 1.0,
            "minimum_controlled_gain": 0.0,
            "minimum_perturbation_eligible_fraction": 0.9,
            "maximum_normal_endpoint_ratio": 1.0,
            "maximum_projected_normal_log_growth_rate": 0.0,
            "maximum_normal_vs_tangent_log_ratio": 0.0,
        },
    }


def _row(
    seed: int,
    *,
    total_gain: float = 0.2,
    state_gain: float = 0.1,
    closure: float = 0.4,
    normal_ratio: float = 0.8,
    normal_growth: float = -0.02,
    normal_relative: float = -0.2,
    separated: bool = True,
) -> dict[str, object]:
    total_full = closure
    raw_common = total_full + total_gain
    routed_state = 0.45
    routed_common = routed_state + state_gain
    row = {
        "run_id": f"run-{seed}",
        "experiment": EXPERIMENT,
        "experiment_protocol_version": "exp21_v2",
        "training_algorithm": (
            "md_filtered_belief_full_substep_controlled_affine_koopman_audit_v2"
        ),
        "seed": seed,
        "condition": CONDITION,
        "status": "complete",
        "statistics_unit": "seed",
        "used_autograd": False,
        "used_bptt": False,
        "recurrent_learning": False,
        "full_trajectory_model": True,
        "full_trajectory_lds": False,
        "preprocessing_fit_train_only": True,
        "operator_fit_train_only": True,
        "gate_fit_accessed_true_context": False,
        "gate_test_accessed_true_context": False,
        "primary_receiver_state_reset_scope": "every_trial_zero_state",
        "trial_reset_trajectory_sequence_scope": "trial_reset_state",
        "trial_reset_paired_models_share_state_pca": True,
        "trial_reset_total_operator_design_full_rank": True,
        "trial_reset_state_affine_operator_design_full_rank": True,
        "total_control_model_input_policy": (
            "raw_receiver_sensory_plus_scalar_control_interactions"
        ),
        "population_state_affine_model_input_policy": (
            "already_routed_sensory_plus_population_gain_belief_"
            "state_and_affine_bias_switch_input_and_epoch_shared"
        ),
        "trial_reset_total_full_rollout_normalized_rmse": total_full,
        "trial_reset_raw_common_rollout_normalized_rmse": raw_common,
        "trial_reset_total_control_rollout_gain_vs_raw_common": total_gain,
        "trial_reset_routed_state_affine_rollout_normalized_rmse": routed_state,
        "trial_reset_routed_common_rollout_normalized_rmse": routed_common,
        "trial_reset_population_state_affine_rollout_gain_vs_routed_common": (
            state_gain
        ),
        "trial_reset_population_state_transition_delta_frobenius": 0.2,
        "trial_reset_population_affine_bias_delta_norm": 0.05,
        "trial_reset_population_exogenous_control_delta_frobenius": 0.0,
        "trial_reset_perturbation_status": "complete",
        "trial_reset_perturbation_eligible_reference_fraction": 1.0,
        "trial_reset_perturbation_normal_endpoint_ratio_maximum": normal_ratio,
        "trial_reset_perturbation_"
        "maximum_projected_finite_time_normal_log_growth_rate": normal_growth,
        "trial_reset_perturbation_normal_vs_tangent_log_ratio_median": (
            normal_relative
        ),
        "trial_reset_perturbation_baseline_replay_max_abs_error": 1e-12,
        "trial_reset_perturbation_baseline_replay_tolerance": 1e-10,
        "trial_reset_perturbation_planned_reference_count": 32,
        "trial_reset_perturbation_sampled_reference_count": 32,
        "trial_reset_perturbation_candidate_reference_count": 100,
        "attractor_anchor_fit_scope": "training_trajectory_only",
        "attractor_population_gain": True,
        "attractor_pathway_gating": True,
        "attractor_both_conditions_contract": separated,
        "attractor_separated_convergence": separated,
        "attractor_centroid_separation_over_initial_dispersion": (
            0.2 if separated else 0.0
        ),
    }
    for prefix in ("trial_reset", "episode_continuous"):
        row.update(
            {
                f"{prefix}_total_operator_mode": "full_shared_neutral_cue",
                f"{prefix}_total_operator_constraint": (
                    "shared_neutral_cue_coefficient"
                ),
                f"{prefix}_total_operator_design_rank": 19,
                f"{prefix}_total_operator_design_columns": 19,
                f"{prefix}_total_operator_unconstrained_columns": 20,
                f"{prefix}_perturbation_geometry": (
                    "joint_state_pca_physical_x_projection_v2"
                ),
                f"{prefix}_perturbation_sampled_reference_fraction": 1.0,
            }
        )
    return row


def _conclusions(summary: pd.DataFrame) -> dict[str, str]:
    return dict(zip(summary["proposition"], summary["conclusion"], strict=True))


def test_exp21_holm_keeps_nan_in_the_planned_family() -> None:
    adjusted = _holm([float("nan"), 0.03])
    assert adjusted.tolist() == pytest.approx([1.0, 0.06])
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        _holm([0.2, 1.01])
    with pytest.raises(ValueError, match="finite or NaN"):
        _holm([0.2, float("inf")])


def test_exp21_missing_first_gain_keeps_two_hypothesis_holm_family() -> None:
    config = _config()
    state_gains = [0.002] * 21 + [-0.001] * 9
    rows = [_row(seed, state_gain=state_gains[seed]) for seed in range(30)]
    for row in rows:
        row.pop("experiment_protocol_version")
        row["training_algorithm"] = (
            "md_filtered_belief_full_substep_controlled_affine_koopman_audit"
        )
        for prefix in ("trial_reset", "episode_continuous"):
            for suffix in (
                "total_operator_mode",
                "total_operator_constraint",
                "total_operator_design_rank",
                "total_operator_design_columns",
                "total_operator_unconstrained_columns",
                "perturbation_geometry",
                "perturbation_sampled_reference_fraction",
            ):
                row.pop(f"{prefix}_{suffix}", None)
    summary = summarize_formal_runs(pd.DataFrame(rows), config, n_bootstrap=500)
    state = summary.loc[
        summary["proposition"].eq(
            "trial_reset_population_state_affine_gain_vs_routed_common"
        )
    ].iloc[0]
    total = summary.loc[
        summary["proposition"].eq("trial_reset_total_control_gain_vs_raw_common")
    ].iloc[0]

    assert state["p_value"] == pytest.approx(0.04277394525706768)
    assert state["holm_adjusted_p"] == pytest.approx(0.08554789051413536)
    assert state["ci_low"] > 0.0
    assert state["n_eligible"] == 30
    assert state["conclusion"] == "inconclusive"
    assert total["holm_adjusted_p"] == pytest.approx(1.0)


def test_exp21_v1_rows_only_enter_definition_compatible_claims() -> None:
    config = _config()
    rows = [_row(seed) for seed in range(30)]
    for row in rows:
        row.pop("experiment_protocol_version")
        row["training_algorithm"] = (
            "md_filtered_belief_full_substep_controlled_affine_koopman_audit"
        )
        for prefix in ("trial_reset", "episode_continuous"):
            for suffix in (
                "total_operator_mode",
                "total_operator_constraint",
                "total_operator_design_rank",
                "total_operator_design_columns",
                "total_operator_unconstrained_columns",
                "perturbation_geometry",
                "perturbation_sampled_reference_fraction",
            ):
                row.pop(f"{prefix}_{suffix}", None)

    summary = summarize_formal_runs(pd.DataFrame(rows), config, n_bootstrap=200)
    by_proposition = summary.set_index("proposition")

    assert set(summary["n_complete"]) == {30}
    for proposition in (
        "trial_reset_total_control_gain_vs_raw_common",
        "trial_reset_full_trajectory_closure",
        "trial_reset_nonlinear_normal_recovery_relative_to_tangent",
    ):
        assert by_proposition.loc[proposition, "n_eligible"] == 0
        assert by_proposition.loc[proposition, "conclusion"] == "inconclusive"
    for proposition in (
        "trial_reset_population_state_affine_gain_vs_routed_common",
        "fixed_drive_separated_endpoint_probe",
    ):
        assert by_proposition.loc[proposition, "n_eligible"] == 30
        assert by_proposition.loc[proposition, "conclusion"] == "support"


def test_exp21_episode_sensitivity_failure_does_not_gate_trial_reset_claims() -> None:
    config = _config()
    rows = [_row(seed) for seed in range(30)]
    for row in rows:
        row["episode_continuous_perturbation_sampled_reference_fraction"] = 0.0
        row["episode_continuous_perturbation_geometry"] = "legacy_rate_pullback_v1"

    summary = summarize_formal_runs(pd.DataFrame(rows), config, n_bootstrap=200)

    assert set(summary["n_eligible"]) == {30}
    assert set(summary["conclusion"]) == {"support"}


def test_exp21_trial_perturbation_failure_only_gates_perturbation_claim() -> None:
    config = _config()
    rows = [_row(seed) for seed in range(30)]
    for row in rows:
        row["trial_reset_perturbation_sampled_reference_fraction"] = 0.5

    summary = summarize_formal_runs(pd.DataFrame(rows), config, n_bootstrap=200)
    perturbation = summary.loc[
        summary["proposition"].eq(
            "trial_reset_nonlinear_normal_recovery_relative_to_tangent"
        )
    ].iloc[0]
    other = summary.loc[
        ~summary["proposition"].eq(
            "trial_reset_nonlinear_normal_recovery_relative_to_tangent"
        )
    ]

    assert perturbation["n_eligible"] == 0
    assert perturbation["conclusion"] == "inconclusive"
    assert set(other["n_eligible"]) == {30}
    assert set(other["conclusion"]) == {"support"}


def test_exp21_summary_supports_and_opposes_registered_seed_claims() -> None:
    config = _config()
    supported = summarize_formal_runs(
        pd.DataFrame([_row(seed) for seed in range(30)]),
        config,
        n_bootstrap=200,
    )
    assert len(supported) == 5
    assert set(supported["inference_unit"]) == {"seed"}
    assert set(supported["conclusion"]) == {"support"}
    assert set(supported["registered_latent_dim"]) == {4}
    assert set(supported["latent_dimension_selection"]) == {
        "fixed_registered_no_nested_cv"
    }
    assert (
        supported["claim_scope"]
        .str.contains("without nested-CV latent-dimension selection")
        .all()
    )
    fixed_drive = supported.loc[
        supported["proposition"].eq("fixed_drive_separated_endpoint_probe")
    ].iloc[0]
    assert "narrow finite-horizon" in fixed_drive["claim_scope"]
    assert "does not support gate causality" in fixed_drive["claim_scope"]
    perturbation = supported.loc[
        supported["proposition"].eq(
            "trial_reset_nonlinear_normal_recovery_relative_to_tangent"
        )
    ].iloc[0]
    assert "physical-x projection" in perturbation["claim_scope"]
    assert "not proof of a joint manifold" in perturbation["claim_scope"]

    opposed_rows = [
        _row(
            seed,
            total_gain=-0.2,
            state_gain=-0.1,
            closure=1.2,
            normal_ratio=1.1,
            normal_growth=0.02,
            normal_relative=0.2,
            separated=False,
        )
        for seed in range(30)
    ]
    opposed = summarize_formal_runs(pd.DataFrame(opposed_rows), config, n_bootstrap=200)
    assert set(opposed["conclusion"]) == {"oppose"}


def test_exp21_failed_seed_is_retained_and_forces_inconclusive() -> None:
    config = _config()
    rows = [_row(seed) for seed in range(30)]
    rows[-1] = {
        "run_id": "failed-run",
        "experiment": EXPERIMENT,
        "seed": 29,
        "condition": CONDITION,
        "status": "failed",
        "error_type": "RuntimeError",
        "error": "planned failure retained",
    }
    raw = pd.DataFrame(rows)
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)
    assert set(summary["conclusion"]) == {"inconclusive"}
    assert set(summary["n_complete"]) == {29}
    assert set(summary["retained_failed_or_invalid_seed_count"]) == {1}


def test_exp21_summary_accepts_legacy_state_switch_raw_aliases() -> None:
    config = _config()
    raw = pd.DataFrame([_row(seed) for seed in range(30)]).rename(
        columns={
            "trial_reset_population_state_affine_rollout_gain_vs_routed_common": (
                "trial_reset_population_state_switch_rollout_gain_vs_routed_common"
            ),
            "trial_reset_routed_state_affine_rollout_normalized_rmse": (
                "trial_reset_routed_state_switch_rollout_normalized_rmse"
            ),
            "trial_reset_state_affine_operator_design_full_rank": (
                "trial_reset_state_switch_operator_design_full_rank"
            ),
            "population_state_affine_model_input_policy": (
                "population_state_switch_model_input_policy"
            ),
            "trial_reset_population_affine_bias_delta_norm": (
                "trial_reset_population_state_affine_bias_delta_norm"
            ),
            "trial_reset_population_state_transition_delta_frobenius": (
                "trial_reset_population_state_affine_transition_delta_frobenius"
            ),
            "trial_reset_population_exogenous_control_delta_frobenius": (
                "trial_reset_population_state_affine_exogenous_control_delta_frobenius"
            ),
        }
    )
    raw["population_state_switch_model_input_policy"] = (
        "already_routed_sensory_plus_population_gain_belief_"
        "state_and_affine_switch_only"
    )
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)
    proposition = "trial_reset_population_state_affine_gain_vs_routed_common"
    assert summary.loc[summary["proposition"].eq(proposition), "conclusion"].item() == (
        "support"
    )


def test_exp21_summary_rejects_conflicting_state_affine_aliases() -> None:
    config = _config()
    raw = pd.DataFrame([_row(seed) for seed in range(30)])
    raw["trial_reset_population_state_switch_rollout_gain_vs_routed_common"] = raw[
        "trial_reset_population_state_affine_rollout_gain_vs_routed_common"
    ]
    raw.loc[
        raw["seed"].eq(0),
        "trial_reset_population_state_switch_rollout_gain_vs_routed_common",
    ] = -1.0
    try:
        summarize_formal_runs(raw, config, n_bootstrap=200)
    except ValueError as error:
        assert "conflicting Exp21 numeric aliases" in str(error)
    else:
        raise AssertionError("conflicting state-affine aliases were accepted")


def test_exp21_real_smoke_row_matches_primary_summary_contract(
    tmp_path: Path,
) -> None:
    from experiments import exp21_belief_ei_full_trajectory as exp21

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "smoke"
        / "exp21_belief_ei_full_trajectory.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_path = exp21.run_seed(config, 0, tmp_path)
    row = json.loads((run_path / "metrics.jsonl").read_text(encoding="utf-8").strip())
    assert row["status"] == "complete"
    assert "trial_reset_population_state_affine_rollout_gain_vs_routed_common" in row
    assert "trial_reset_routed_state_affine_rollout_normalized_rmse" in row
    assert row["trial_reset_state_affine_operator_design_full_rank"]
    assert row["population_state_affine_model_input_policy"] == (
        "already_routed_sensory_plus_population_gain_belief_"
        "state_and_affine_bias_switch_input_and_epoch_shared"
    )

    summary = summarize_formal_runs(pd.DataFrame([row]), config, n_bootstrap=200)
    state_affine = summary.loc[
        summary["proposition"].eq(
            "trial_reset_population_state_affine_gain_vs_routed_common"
        )
    ].iloc[0]
    assert state_affine["n_complete"] == 1
    assert state_affine["n_eligible"] == 1


def test_exp21_all_failed_seeds_remain_publishable_and_inconclusive() -> None:
    config = _config(n_seeds=3)
    raw = pd.DataFrame(
        [
            {
                "run_id": f"failed-{seed}",
                "experiment": EXPERIMENT,
                "seed": seed,
                "condition": CONDITION,
                "status": "failed",
                "error_type": "RuntimeError",
                "error": "retained",
            }
            for seed in range(3)
        ]
    )
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)
    assert set(summary["conclusion"]) == {"inconclusive"}
    assert set(summary["n_complete"]) == {0}
    assert set(summary["retained_failed_or_invalid_seed_count"]) == {3}


def test_exp21_snapshot_writes_deterministic_scoped_artifacts(
    tmp_path: Path,
) -> None:
    config = _config()
    rows = [_row(seed) for seed in range(30)]
    rows[-1] = {
        "run_id": "failed-run",
        "experiment": EXPERIMENT,
        "seed": 29,
        "condition": CONDITION,
        "status": "failed",
        "error_type": "RuntimeError",
        "error": "planned failure retained",
    }
    raw = pd.DataFrame(rows)
    first = write_snapshot_artifacts(
        raw,
        config,
        output_dir=tmp_path / "first",
        prefix="exp21_test",
        n_bootstrap=200,
    )
    second = write_snapshot_artifacts(
        raw,
        config,
        output_dir=tmp_path / "second",
        prefix="exp21_test",
        n_bootstrap=200,
    )

    assert first["raw"].read_bytes()[4:8] == b"\x00\x00\x00\x00"
    published_raw = pd.read_csv(first["raw"])
    assert len(published_raw) == 30
    assert published_raw.loc[published_raw["seed"].eq(29), "status"].item() == "failed"
    published_summary = pd.read_csv(first["summary"])
    assert len(published_summary) == 5
    report = first["report"].read_text(encoding="utf-8")
    assert "Retained failed/invalid seeds: 1" in report
    assert "run-level execution status" in report
    assert "Claim-level scientifically ineligible conclusion rows: 5 of 5" in report
    assert "## Claim-level scientific ineligibility" in report
    assert "does not support gate causality" in report
    assert "Historical v1 rows supplied directly to this audit" in report
    assert "planned failure retained" in report
    assert "does not modify" in report
    for name in ("png", "pdf"):
        assert first[name].stat().st_size > 1_000
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()


def _write_attempt(
    root: Path,
    config: dict[str, object],
    *,
    seed: int,
    row: dict[str, object],
) -> Path:
    attempt = (
        root / "runs" / EXPERIMENT / f"seed_{seed:04d}" / "20260101T000000.000000Z"
    )
    attempt.mkdir(parents=True)
    run_status = "complete" if row["status"] == "complete" else "complete_with_failures"
    run_id = str(row["run_id"])
    (attempt / "config.json").write_text(
        json.dumps(_expected_run_config(config, seed)),
        encoding="utf-8",
    )
    (attempt / "status.json").write_text(
        json.dumps({"status": run_status}),
        encoding="utf-8",
    )
    (attempt / "manifest.json").write_text(
        json.dumps(
            {
                "status": run_status,
                "experiment": EXPERIMENT,
                "seed": seed,
                "run_id": run_id,
                "profile": "formal",
            }
        ),
        encoding="utf-8",
    )
    (attempt / "environment.json").write_text(
        json.dumps(
            {
                "python": "3.11.9 (test)",
                "platform": "Linux-test-x86_64",
                "executable": "/test/python3.11",
                "packages": {
                    "matplotlib": "3.9.0",
                    "numpy": "2.0.0",
                    "pandas": "2.2.0",
                    "scikit-learn": "1.5.0",
                    "scipy": "1.14.0",
                    "statsmodels": "0.14.0",
                    "torch": "2.4.0",
                },
                "git": {"commit": "a" * 40, "dirty": False},
            }
        ),
        encoding="utf-8",
    )
    (attempt / "planned_conditions.json").write_text(
        json.dumps(
            [
                {
                    "condition_index": 0,
                    "condition": CONDITION,
                    "model_family": "frozen_high_rank_dale_ei",
                    "controller_mode": "combined",
                    "belief_intervention": "none",
                    "trajectory_sampling": "euler_substep",
                }
            ]
        ),
        encoding="utf-8",
    )
    (attempt / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return attempt


def test_exp21_collector_keeps_terminal_failed_seed(tmp_path: Path) -> None:
    config = _config()
    expected = _expected_run_config(config, 0)
    assert expected["experiment_protocol_version"] == "exp21_v2"
    assert expected["training_algorithm"].endswith("_audit_v2")
    failed = {
        "run_id": "failed-run",
        "experiment": EXPERIMENT,
        "seed": 29,
        "condition": CONDITION,
        "status": "failed",
        "error_type": "ValueError",
        "error": "retained",
    }
    attempts = [
        _write_attempt(tmp_path, config, seed=seed, row=_row(seed))
        for seed in range(29)
    ]
    _write_attempt(tmp_path, config, seed=29, row=failed)
    raw = collect_registered_runs(tmp_path, config)
    assert raw["seed"].tolist() == list(range(30))
    assert raw["status"].value_counts().to_dict() == {
        "complete": 29,
        "failed": 1,
    }
    assert raw.loc[raw["seed"].eq(29), "error"].item() == "retained"
    assert raw["run_git_commit"].nunique() == 1
    assert raw["environment_sha256"].nunique() == 1

    environment_path = attempts[0] / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["git"]["dirty"] = True
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    with pytest.raises(ValueError, match="clean Git receipt"):
        collect_registered_runs(tmp_path, config)


def test_exp21_formal_provenance_contract_fails_closed(
    tmp_path: Path,
) -> None:
    bad_profile = _config()
    bad_profile["profile"] = "smoke"
    with pytest.raises(ValueError, match="profile=formal"):
        _require_formal_registration(bad_profile)

    too_few = _config(n_seeds=2)
    with pytest.raises(ValueError, match="30 registered"):
        _require_formal_registration(too_few)

    environment = {
        "python": "3.12.1",
        "platform": "Linux-test-x86_64",
        "executable": "/test/python3.12",
        "packages": {
            name: "1.0"
            for name in (
                "matplotlib",
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "statsmodels",
                "torch",
            )
        },
    }
    with pytest.raises(ValueError, match="Python 3.11"):
        _environment_sha256(environment)

    with pytest.raises(ValueError, match="validated publication provenance"):
        write_snapshot_artifacts(
            pd.DataFrame([_row(seed) for seed in range(30)]),
            _config(),
            output_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="valid Git commit"):
        write_snapshot_artifacts(
            pd.DataFrame([_row(seed) for seed in range(30)]),
            _config(),
            output_dir=tmp_path,
            publication_provenance={
                "analysis_git_commit": "forged",
                "analysis_script_sha256": "0" * 64,
                "analysis_python": "3.11.9",
            },
        )

    valid_environment = {
        **environment,
        "python": "3.11.9",
        "executable": "/test/python3.11",
    }
    linux_digest = _environment_sha256(valid_environment)
    windows_digest = _environment_sha256(
        {**valid_environment, "platform": "Windows-test-amd64"}
    )
    assert linux_digest != windows_digest
