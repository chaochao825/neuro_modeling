"""Contracts for the standalone Exp22 off-policy proposal snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_exp22 import (
    EXPERIMENT,
    ORACLE,
    ORTHOGONAL,
    _environment_sha256,
    _expected_run_config,
    _require_formal_registration,
    _validate_publication_provenance,
    collect_registered_runs,
    summarize_formal_runs,
    validate_raw_frame,
    write_snapshot_artifacts,
)
from experiments.exp22_hidden_context_local_gain_axis import (
    _planned_conditions,
)


def _config(*, n_seeds: int = 30) -> dict[str, object]:
    return {
        "profile": "formal",
        "seeds": list(range(n_seeds)),
        "network": {"n_units": 512},
        "gain_axis_learning": {
            "budgets": {"l1": 0.5, "l2": 0.5},
            "budget_tolerance": 1e-9,
        },
        "registered_claim_thresholds": {
            "minimum_aligned_gain_vs_frozen": 0.0,
            "minimum_aligned_gain_vs_random": 0.0,
            "minimum_aligned_gain_vs_shuffled": 0.0,
            "maximum_aligned_oracle_gap": 0.05,
            "minimum_budget_valid_fraction": 0.9,
            "minimum_aligned_absolute_balanced_accuracy": 0.75,
        },
    }


def _condition_accuracy(condition: str, *, positive: bool) -> float:
    feedback = condition.rsplit("_", 1)[0]
    if condition == "frozen_zero":
        return 0.60 if positive else 0.70
    if feedback == "aligned_local":
        return 0.80 if positive else 0.55
    if feedback == "random_signed_feedback":
        return 0.65 if positive else 0.68
    if feedback == "shuffled_feedback":
        return 0.64 if positive else 0.69
    if feedback == ORACLE:
        return 0.82 if positive else 0.80
    if feedback == ORTHOGONAL:
        return 0.63 if positive else 0.66
    raise AssertionError(condition)


def _feedback_receipts(feedback: str) -> tuple[str, str, str]:
    if feedback == ORACLE:
        feedback = "aligned_local"
    return (
        f"local-tape-{feedback}",
        f"feedback-coefficients-{feedback}",
        f"feedback-policy-{feedback}",
    )


def _seed_rows(
    config: dict[str, object],
    seed: int,
    *,
    positive: bool = True,
) -> list[dict[str, object]]:
    common = {
        "experiment": EXPERIMENT,
        "seed": seed,
        "status": "complete",
        "statistics_unit": "seed",
        "base_conditions_share_readout": True,
        "fixed_readout_feedback_coefficients_used": True,
        "gate_fit_train_only": True,
        "readout_fit_train_only": True,
        "gain_axis_fit_dev_only": True,
        "train_dev_test_episode_disjoint": True,
        "proposal_scale_predeclared_in_config": True,
        "off_policy_frozen_trajectory_proposal_audit": True,
        "gain_axis_three_factor_rule_used_for_eligibility": True,
        "feedback_transform_applied_before_local_eligibility": True,
        "budget_preserves_event_relative_magnitude": True,
        "used_autograd": False,
        "used_bptt": False,
        "recurrent_learning": False,
        "homeostasis_learning": False,
        "normalization_learning": False,
        "gate_fit_accessed_true_context": False,
        "gate_test_accessed_true_context": False,
        "axis_test_truth_accessed": False,
        "test_used_for_axis_fit": False,
        "proposal_scale_fit_on_test": False,
        "budget_controller_can_amplify_proposals": False,
        "budget_controller_used": False,
        "budget_matcher_can_amplify_proposals": False,
        "budget_simultaneous_dual_norm_match": False,
        "generic_recurrent_three_factor_claim_eligible": False,
        "gain_axis_local_plasticity_claim_eligible": False,
        "gain_axis_learning_closed_loop": False,
        "dev_trajectory_recomputed_after_each_update": False,
        "proposal_coordinate_permutation_after_eligibility": False,
        "closed_loop_local_plasticity_claim_eligible": False,
        "weight_transport_free_claim": False,
        "test_gain_control_source": "learned_belief_posterior",
        "network_init_id": f"network-{seed}",
        "gate_checkpoint_id": f"gate-{seed}",
        "readout_checkpoint_id": f"readout-{seed}",
        "split_id": f"split-{seed}",
        "random_tape_id": f"random-tape-{seed}",
        "shared_noise_id": f"random-tape-{seed}",
        "dev_neutral_trajectory_id": f"neutral-dev-{seed}",
        "dev_trial_order_id": f"dev-order-{seed}",
        "learned_third_factor_id": f"learned-third-factor-{seed}",
        "planned_condition_grid_id": "grid",
        "experiment_protocol_id": "protocol",
    }
    result: list[dict[str, object]] = []
    for planned in _planned_conditions(config):
        condition = str(planned["condition"])
        row = {
            **common,
            **planned,
            "behavior_balanced_accuracy": _condition_accuracy(
                condition, positive=positive
            ),
            "behavior_accuracy": _condition_accuracy(condition, positive=positive),
        }
        if condition == "frozen_zero":
            row.update(
                gain_axis_learning=False,
                fixed_readout_feedback_coefficients_used=False,
                gain_axis_three_factor_rule_used_for_eligibility=False,
                feedback_transform_applied_before_local_eligibility=False,
                budget_preserves_event_relative_magnitude=False,
                budget_attained=True,
                budget_total=0.0,
                budget_selected_norm="none",
                budget_selected_raw=0.0,
                budget_global_scale_factor=None,
                budget_scaling_policy="none_zero_update_baseline",
                budget_path_application_id=None,
                frozen_zero_update_budget_baseline=True,
                dev_truth_accessed_for_axis=False,
                gain_axis_off_policy_proposal_claim_eligible=False,
                condition_dev_local_tape_id=None,
                condition_third_factor_id=None,
                feedback_coefficients_id=None,
                feedback_policy=None,
                feedback_coefficient_l2=0.0,
                feedback_angle_to_aligned_degrees=None,
            )
        else:
            feedback = str(planned["feedback_condition"])
            panel = str(planned["budget_norm"])
            tape, coefficients, policy = _feedback_receipts(feedback)
            row.update(
                gain_axis_learning=True,
                budget_attained=True,
                budget_total=0.5,
                budget_selected_raw=1.0,
                budget_selected_applied=0.5,
                budget_global_scale_factor=0.5,
                budget_scaling_policy=(
                    "single_global_downscale_preserves_event_relative_magnitude"
                ),
                budget_path_application_id=f"budget-path-{feedback}-{panel}",
                frozen_zero_update_budget_baseline=False,
                budget_selected_norm=panel,
                budget_secondary_norm_is_diagnostic_only=True,
                recurrent_weights_bitwise_frozen=True,
                dev_truth_accessed_for_axis=feedback == ORACLE,
                gain_axis_off_policy_proposal_claim_eligible=(
                    feedback == "aligned_local"
                ),
                condition_dev_local_tape_id=tape,
                condition_third_factor_id=(
                    f"oracle-third-factor-{seed}"
                    if feedback == ORACLE
                    else f"learned-third-factor-{seed}"
                ),
                feedback_coefficients_id=coefficients,
                feedback_policy=policy,
                feedback_coefficient_l2=2.0,
                feedback_angle_to_aligned_degrees=(
                    90.0
                    if feedback == ORTHOGONAL
                    else 0.0
                    if feedback in {"aligned_local", ORACLE}
                    else 60.0
                ),
                feedback_coefficient_angle_to_aligned_degrees=(
                    90.0
                    if feedback == ORTHOGONAL
                    else 0.0
                    if feedback in {"aligned_local", ORACLE}
                    else 60.0
                ),
                feedback_sensory_angle_to_aligned_degrees=(
                    90.0
                    if feedback == ORTHOGONAL
                    else 0.0
                    if feedback in {"aligned_local", ORACLE}
                    else 60.0
                ),
                feedback_delay_angle_to_aligned_degrees=(
                    90.0
                    if feedback == ORTHOGONAL
                    else 0.0
                    if feedback in {"aligned_local", ORACLE}
                    else 60.0
                ),
                feedback_response_angle_to_aligned_degrees=(
                    90.0
                    if feedback == ORTHOGONAL
                    else 0.0
                    if feedback in {"aligned_local", ORACLE}
                    else 60.0
                ),
            )
        result.append(row)
    return result


def _raw(config: dict[str, object], *, positive: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            row
            for seed in config["seeds"]
            for row in _seed_rows(config, int(seed), positive=positive)
        ]
    )


def _joint(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.loc[
        summary["proposition"].eq("joint_off_policy_proposal_alignment_specificity")
    ]


def test_exp22_positive_and_negative_seed_panels_are_fail_closed() -> None:
    config = _config()
    supported = summarize_formal_runs(_raw(config), config, n_bootstrap=200)
    assert len(supported) == 16
    assert set(supported["inference_unit"]) == {"seed"}
    assert set(_joint(supported)["panel"]) == {"l1", "l2"}
    assert set(_joint(supported)["conclusion"]) == {"support"}
    primary = supported.loc[supported["control_role"].eq("primary")]
    assert len(primary) == 6
    assert set(primary["conclusion"]) == {"support"}
    oracle = supported.loc[supported["control_role"].eq("upper_bound")]
    assert set(oracle["conclusion"]) == {"support"}
    orthogonal = supported.loc[supported["control_role"].eq("negative_control")]
    assert set(orthogonal["conclusion"]) == {"support"}
    absolute_descriptive = supported.loc[
        supported["control_role"].eq("absolute_descriptive")
    ]
    assert len(absolute_descriptive) == 2
    assert set(absolute_descriptive["conclusion"]) == {"inconclusive"}
    assert set(absolute_descriptive["effect_definition"]) == {
        "aligned_heldout_behavior_accuracy"
    }
    absolute_registered = supported.loc[
        supported["control_role"].eq("absolute_registered")
    ]
    assert len(absolute_registered) == 2
    assert set(absolute_registered["conclusion"]) == {"support"}
    assert set(absolute_registered["effect_definition"]) == {
        "aligned_heldout_behavior_balanced_accuracy"
    }

    opposed = summarize_formal_runs(
        _raw(config, positive=False), config, n_bootstrap=200
    )
    assert set(opposed.loc[opposed["control_role"].eq("primary"), "conclusion"]) == {
        "oppose"
    }
    assert set(_joint(opposed)["conclusion"]) == {"oppose"}
    assert set(
        opposed.loc[opposed["control_role"].eq("absolute_registered"), "conclusion"]
    ) == {"oppose"}
    assert set(
        opposed.loc[opposed["control_role"].eq("upper_bound"), "conclusion"]
    ) == {"oppose"}


def test_exp22_joint_never_supports_without_frozen_and_shuffled_gain() -> None:
    config = _config()
    raw = _raw(config)
    mask = raw["condition"].isin(["aligned_local_l1", "aligned_local_l2"])
    raw.loc[mask, "behavior_balanced_accuracy"] = 0.60
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)
    assert not _joint(summary)["conclusion"].eq("support").any()


def test_exp22_unattained_budget_invalidates_only_affected_panel() -> None:
    config = _config()
    raw = _raw(config)
    affected = raw["seed"].isin([0, 1, 2, 3]) & raw["condition"].eq("aligned_local_l1")
    raw.loc[affected, "budget_attained"] = False
    raw.loc[affected, "budget_selected_applied"] = 0.25
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)

    l1 = summary.loc[summary["panel"].eq("l1")]
    l2 = summary.loc[summary["panel"].eq("l2")]
    assert set(l1["n_eligible"]) == {26}
    assert set(l1["conclusion"]) == {"inconclusive"}
    assert set(l2["n_eligible"]) == {30}
    assert set(_joint(l2)["conclusion"]) == {"support"}


def test_exp22_failed_l2_comparator_does_not_invalidate_l1_panel() -> None:
    config = _config()
    raw = _raw(config)
    failed = raw["condition"].eq("random_signed_feedback_l2") & raw["seed"].eq(0)
    raw.loc[failed, "status"] = "failed"
    raw.loc[failed, "error_type"] = "RuntimeError"
    raw.loc[failed, "error"] = "l2-only comparator failure"
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)

    l1_primary = summary.loc[
        summary["panel"].eq("l1") & summary["control_role"].eq("primary")
    ]
    l2_primary = summary.loc[
        summary["panel"].eq("l2") & summary["control_role"].eq("primary")
    ]
    assert set(l1_primary["n_eligible"]) == {30}
    assert set(l1_primary["conclusion"]) == {"support"}
    assert set(l2_primary["n_eligible"]) == {29}


def test_exp22_oracle_failure_keeps_primary_claims_but_invalidates_joint() -> None:
    config = _config()
    raw = _raw(config)
    failed = raw["feedback_condition"].eq(ORACLE)
    raw.loc[failed, "status"] = "failed"
    raw.loc[failed, "error_type"] = "RuntimeError"
    raw.loc[failed, "error"] = "oracle-only failure retained"
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)

    assert set(_joint(summary)["conclusion"]) == {"inconclusive"}
    primary = summary.loc[summary["control_role"].eq("primary")]
    assert set(primary["n_eligible"]) == {30}
    assert set(primary["conclusion"]) == {"support"}
    oracle = summary.loc[summary["control_role"].eq("upper_bound")]
    assert set(oracle["n_eligible"]) == {0}
    assert set(oracle["conclusion"]) == {"inconclusive"}
    assert set(summary["retained_failed_or_invalid_cell_count"]) == {60}


def test_exp22_oracle_margin_failure_opposes_joint_only() -> None:
    config = _config()
    raw = _raw(config)
    oracle = raw["feedback_condition"].eq(ORACLE)
    raw.loc[oracle, "behavior_balanced_accuracy"] = 0.90
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)

    assert set(_joint(summary)["conclusion"]) == {"oppose"}
    primary = summary.loc[summary["control_role"].eq("primary")]
    assert set(primary["conclusion"]) == {"support"}
    oracle_rows = summary.loc[summary["control_role"].eq("upper_bound")]
    assert set(oracle_rows["conclusion"]) == {"oppose"}


def test_exp22_orthogonal_epoch_angle_receipt_is_fail_closed() -> None:
    config = _config()
    raw = _raw(config)
    orthogonal = raw["feedback_condition"].eq(ORTHOGONAL)
    raw.loc[orthogonal, "feedback_delay_angle_to_aligned_degrees"] = 80.0
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)

    primary = summary.loc[summary["control_role"].eq("primary")]
    negative = summary.loc[summary["control_role"].eq("negative_control")]
    assert set(primary["n_eligible"]) == {30}
    assert set(negative["n_eligible"]) == {0}
    assert set(negative["conclusion"]) == {"inconclusive"}


def test_exp22_failed_primary_grid_is_retained_and_inconclusive() -> None:
    config = _config(n_seeds=3)
    raw = _raw(config)
    failed = raw["seed"].eq(2)
    raw.loc[failed, "status"] = "failed"
    raw.loc[failed, "error_type"] = "RuntimeError"
    raw.loc[failed, "error"] = "planned seed failure retained"
    summary = summarize_formal_runs(raw, config, n_bootstrap=200)
    assert set(summary["conclusion"]) == {"inconclusive"}
    assert set(summary.loc[summary["control_role"].eq("primary"), "n_eligible"]) == {2}
    assert set(summary["retained_failed_or_invalid_seed_count"]) == {1}


def test_exp22_duplicate_seed_condition_is_rejected() -> None:
    config = _config(n_seeds=2)
    raw = _raw(config)
    duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate seed-condition"):
        validate_raw_frame(duplicated, config)


def test_exp22_formal_publication_requires_registered_provenance(
    tmp_path: Path,
) -> None:
    config = _config()
    _require_formal_registration(config)
    with pytest.raises(ValueError, match="30 registered"):
        _require_formal_registration(_config(n_seeds=29))
    wrong_network = _config()
    wrong_network["network"] = {"n_units": 256}
    with pytest.raises(ValueError, match="N=512"):
        _require_formal_registration(wrong_network)
    with pytest.raises(ValueError, match="publication provenance"):
        write_snapshot_artifacts(
            _raw(config),
            config,
            output_dir=tmp_path,
            n_bootstrap=200,
        )

    environment = {
        "python": "3.11.15 test",
        "platform": "test",
        "executable": "/test/python",
        "packages": {
            "matplotlib": "3.10",
            "numpy": "2.0",
            "pandas": "2.0",
            "scikit-learn": "1.0",
            "scipy": "1.0",
            "statsmodels": "0.14",
            "torch": "2.0",
        },
    }
    assert len(_environment_sha256(environment)) == 64
    with pytest.raises(ValueError, match="Python 3.11"):
        _environment_sha256({**environment, "python": "3.10.0"})
    validated = _validate_publication_provenance(
        {
            "analysis_git_commit": "a" * 40,
            "analysis_script_sha256": "b" * 64,
            "analysis_python": "3.11.15 test",
        }
    )
    assert validated["analysis_git_commit"] == "a" * 40


def test_exp22_snapshot_is_deterministic_and_never_claims_local_learning(
    tmp_path: Path,
) -> None:
    config = _config(n_seeds=3)
    raw = _raw(config)
    failed = raw["seed"].eq(2)
    raw.loc[failed, "status"] = "failed"
    raw.loc[failed, "error_type"] = "RuntimeError"
    raw.loc[failed, "error"] = "retained failure"
    first = write_snapshot_artifacts(
        raw,
        config,
        output_dir=tmp_path / "first",
        prefix="exp22_test",
        n_bootstrap=200,
    )
    second = write_snapshot_artifacts(
        raw,
        config,
        output_dir=tmp_path / "second",
        prefix="exp22_test",
        n_bootstrap=200,
    )

    assert all(path.name.startswith("exp22_test") for path in first.values())
    assert first["raw"].read_bytes()[4:8] == b"\x00\x00\x00\x00"
    published_raw = pd.read_csv(first["raw"])
    assert len(published_raw) == 33
    assert published_raw.loc[published_raw["seed"].eq(2), "status"].eq("failed").all()
    report = first["report"].read_text(encoding="utf-8")
    assert "off-policy" in report
    assert "no conclusion is a claim of online local plasticity" in report
    assert "retained failure" in report
    for name in ("png", "pdf"):
        assert first[name].stat().st_size > 1_000
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()


def _write_attempt(
    root: Path,
    config: dict[str, object],
    *,
    seed: int,
    rows: list[dict[str, object]],
) -> None:
    attempt = (
        root / "runs" / EXPERIMENT / f"seed_{seed:04d}" / "20260101T000000.000000Z"
    )
    attempt.mkdir(parents=True)
    run_id = f"run-{seed}"
    for row in rows:
        row["run_id"] = run_id
    failures = sum(row["status"] == "failed" for row in rows)
    invalid = sum(row["status"] == "invalid" for row in rows)
    run_status = "complete_with_failures" if failures or invalid else "complete"
    (attempt / "config.json").write_text(
        json.dumps(_expected_run_config(config, seed)), encoding="utf-8"
    )
    (attempt / "status.json").write_text(
        json.dumps(
            {
                "status": run_status,
                "condition_failures": failures,
                "condition_invalid": invalid,
            }
        ),
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
                "python": "3.11.15 test",
                "platform": "test-platform",
                "executable": "/test/python",
                "packages": {
                    "matplotlib": "3.10",
                    "numpy": "2.0",
                    "pandas": "2.0",
                    "scikit-learn": "1.0",
                    "scipy": "1.0",
                    "statsmodels": "0.14",
                    "torch": "2.0",
                },
                "git": {"commit": "a" * 40, "dirty": False},
            }
        ),
        encoding="utf-8",
    )
    planned = [
        {"condition_index": index, **item}
        for index, item in enumerate(_planned_conditions(config))
    ]
    (attempt / "planned_conditions.json").write_text(
        json.dumps(planned), encoding="utf-8"
    )
    (attempt / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_exp22_collector_retains_failed_oracle_cells(tmp_path: Path) -> None:
    config = _config()
    assert _expected_run_config(config, 0)["training_algorithm"] == (
        "dev_frozen_trajectory_three_factor_gain_axis_proposal_audit"
    )
    for seed in range(30):
        rows = _seed_rows(config, seed)
        if seed == 1:
            for row in rows:
                if row["feedback_condition"] == ORACLE:
                    row["status"] = "failed"
                    row["error_type"] = "RuntimeError"
                    row["error"] = "oracle retained"
        _write_attempt(tmp_path, config, seed=seed, rows=rows)
    raw = collect_registered_runs(tmp_path, config)
    assert len(raw) == 330
    failed = raw.loc[raw["status"].eq("failed")]
    assert len(failed) == 2
    assert set(failed["feedback_condition"]) == {ORACLE}


def test_exp22_real_smoke_rows_match_off_policy_summary_contract(
    tmp_path: Path,
) -> None:
    from experiments import exp22_hidden_context_local_gain_axis as exp22

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "smoke"
        / "exp22_hidden_context_local_gain_axis.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_path = exp22.run_seed(config, 0, tmp_path)
    rows = [
        json.loads(line)
        for line in (run_path / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 11
    summary = summarize_formal_runs(pd.DataFrame(rows), config, n_bootstrap=200)
    primary = summary.loc[summary["control_role"].eq("primary")]
    assert set(primary["n_eligible"]) == {1}
    assert all(not row["gain_axis_local_plasticity_claim_eligible"] for row in rows)
    assert all(not row["closed_loop_local_plasticity_claim_eligible"] for row in rows)
    assert all(row["off_policy_frozen_trajectory_proposal_audit"] for row in rows)
