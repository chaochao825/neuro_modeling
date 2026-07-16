"""Contracts for the joint Exp23--25 fail-closed summary layer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.summarize_exp23_exp25 import (
    EXP23,
    EXP24,
    EXP25,
    EXP23_CONDITIONS,
    EXP23_TASKS,
    EXP24_MODES,
    EXP24_TASKS,
    EXP25_FAMILIES,
    EXP25_PROTOCOLS,
    PROJECT_ROOT,
    _expected_plans,
    _holm_adjust,
    collect_planned_rows,
    summarize_claims,
    write_summary_artifacts,
)


CONFIG_ROOT = PROJECT_ROOT / "configs" / "formal"
CONFIG_FILES = {
    EXP23: "exp23_closed_loop_local_controller.json",
    EXP24: "exp24_factorized_control_benchmark.json",
    EXP25: "exp25_compositional_tasks_real.json",
}


def _write_attempt(
    root: Path,
    *,
    experiment: str,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    profile: str = "formal",
    stamp: str = "20260101T000000.000000Z",
) -> None:
    attempt = root / "runs" / experiment / f"seed_{seed:04d}" / stamp
    attempt.mkdir(parents=True)
    failures = sum(row.get("status") == "failed" for row in rows)
    invalid = sum(row.get("status") == "invalid" for row in rows)
    run_status = "complete_with_failures" if failures or invalid else "complete"
    config = json.loads(
        (CONFIG_ROOT / CONFIG_FILES[experiment]).read_text(encoding="utf-8")
    )
    config.update(experiment=experiment, seed=seed, profile=profile)
    (attempt / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (attempt / "status.json").write_text(
        json.dumps(
            {
                "status": run_status,
                "started_at": stamp,
                "condition_failures": failures,
                "condition_invalid": invalid,
            }
        ),
        encoding="utf-8",
    )
    (attempt / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": f"{experiment}-{seed}-{stamp}",
                "experiment": experiment,
                "seed": seed,
                "status": run_status,
            }
        ),
        encoding="utf-8",
    )
    planned = [
        {"condition_index": index, **condition}
        for index, condition in enumerate(_expected_plans(experiment))
    ]
    (attempt / "planned_conditions.json").write_text(
        json.dumps(planned), encoding="utf-8"
    )
    materialized = []
    for index, row in enumerate(rows):
        materialized.append(
            {
                "run_id": f"{experiment}-{seed}-{stamp}",
                "experiment": experiment,
                "seed": seed,
                "recorded_at": f"2026-01-01T00:00:{index:02d}Z",
                **dict(row),
            }
        )
    (attempt / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in materialized),
        encoding="utf-8",
    )


def _exp23_rows(seed: int, *, failed_local: bool = False) -> list[dict[str, Any]]:
    accuracy = {
        "frozen": 0.60,
        "current_off_policy": 0.64,
        "random_update": 0.61,
        "exact_forward_sensitivity": 0.73,
        "bptt_axis_only": 0.75,
        "local_eprop": 0.70,
    }
    rows: list[dict[str, Any]] = []
    for task in EXP23_TASKS:
        pairing_token = f"seed-{seed}:{task}"
        recurrent_hash = f"recurrent:{pairing_token}"
        for condition in EXP23_CONDITIONS:
            status = (
                "failed"
                if failed_local and task == "delayed" and condition == "local_eprop"
                else "complete"
            )
            row = {
                "condition": condition,
                "task_variant": task,
                "controller_parameterization": "population_gain_axis",
                "status": status,
            }
            if status == "complete":
                row.update(
                    behavior_balanced_accuracy=accuracy[condition],
                    functional_budget_satisfied=True,
                    random_tape_id=f"tape:{pairing_token}",
                    split_id=f"split:{pairing_token}",
                    network_init_id=f"network:{pairing_token}",
                    gate_checkpoint_id=f"gate:{pairing_token}",
                    readout_checkpoint_id=f"readout:{pairing_token}",
                    readout_fit_data_id=f"readout-data:{pairing_token}",
                    pairing_bundle_id=f"bundle:{pairing_token}",
                    paired_network_gate_readout_split_tape=True,
                    recurrent_frozen_hash=recurrent_hash,
                    recurrent_weights_hash_before_condition=recurrent_hash,
                    recurrent_weights_hash_after_condition=recurrent_hash,
                    recurrent_weights_snapshot_is_copy=True,
                    recurrent_copy_isolation_audit_passed=True,
                    recurrent_hash_audit_passed=True,
                    recurrent_weights_bitwise_frozen=True,
                    recurrent_weights_initial_id=f"training:{recurrent_hash}",
                    recurrent_weights_final_id=f"training:{recurrent_hash}",
                    recurrent_weights_audit="independent_copy_and_sha256",
                    recurrent_learning=False,
                    readout_fit_train_only=True,
                    readout_fit_scope="training_split_only",
                    gate_fit_train_only=True,
                    gate_fit_scope="training_split_only",
                    axis_fit_dev_only=True,
                    axis_fit_scope="development_split_only",
                    test_used_for_axis_fit=False,
                    axis_selection_accessed_test=False,
                    gate_test_accessed_true_context=False,
                    third_factor_accessed_true_context=False,
                    hidden_context_access_audit_passed=True,
                    local_learning=condition == "local_eprop",
                    used_autograd=condition == "bptt_axis_only",
                    used_bptt=condition == "bptt_axis_only",
                    local_rule_autograd_free=condition == "local_eprop",
                    local_rule_bptt_free=condition == "local_eprop",
                    gate_moment_anchor_identifiable=True,
                    gate_mean_absolute_signed_belief_dev=0.75,
                    median_update_cosine_to_exact=(
                        0.50 if condition == "local_eprop" else None
                    ),
                )
            else:
                row.update(error="retained local condition failure")
            rows.append(row)
    return rows


def _exp24_rows() -> list[dict[str, Any]]:
    accuracy = {
        "routing_dominant": {
            "frozen": 0.55,
            "routing": 0.82,
            "gain": 0.78,
            "low_rank": 0.60,
            "rgl": 0.84,
        },
        "dynamics_dominant": {
            "frozen": 0.55,
            "routing": 0.60,
            "gain": 0.65,
            "low_rank": 0.81,
            "rgl": 0.86,
        },
    }
    rows: list[dict[str, Any]] = []
    for task in EXP24_TASKS:
        for mode in EXP24_MODES:
            row = {
                "task": task,
                "condition": mode,
                "actuator_mode": mode,
                "controller_source": "oracle_true_context_actuator_isolation",
                "control_dim": 2,
                "status": "complete",
                "test_balanced_accuracy": accuracy[task][mode],
            }
            if mode != "frozen":
                row.update(
                    functional_budget_converged=True,
                    functional_budget_relative_error=0.0,
                    functional_budget_fit_scope="training_blocks_only",
                    parameter_norm_budget_used=False,
                    shared_control_dim_across_actuators=True,
                )
            rows.append(row)
    return rows


def _exp25_rows(*, invalid_cross_session: bool = False) -> list[dict[str, Any]]:
    mean_likelihood = {
        "common": -2.00,
        "input-gated": -1.60,
        "state-gated": -1.80,
        "fully-gated": -1.40,
        "separate-task": -1.35,
    }
    parameter_count = {
        "common": 80,
        "input-gated": 90,
        "state-gated": 95,
        "fully-gated": 100,
        "separate-task": 200,
    }
    rows: list[dict[str, Any]] = []
    for protocol in EXP25_PROTOCOLS:
        for family in EXP25_FAMILIES:
            dimensions = {
                "condition": f"{protocol}:{family}",
                "protocol": protocol,
                "model_family": family,
                "evaluation_level": "animal_session",
            }
            if invalid_cross_session and protocol == "cross-session-transfer":
                rows.append(
                    {
                        **dimensions,
                        "status": "invalid",
                        "reason": "hierarchical unseen-session observation map absent",
                    }
                )
                continue
            model_mean = mean_likelihood[family]
            if (
                protocol == "unseen-stimulus-action-composition"
                and family == "separate-task"
            ):
                model_mean = -1.45
            per_session = [
                {
                    "session_id": f"session-{index}",
                    "animal_id": f"animal-{index}",
                    "log_likelihood": model_mean * 100.0,
                    "null_log_likelihood": -250.0,
                    "n_observations": 100,
                    "n_spikes": 50,
                    "bits_per_spike": 0.1,
                }
                for index in range(2)
            ]
            rows.append(
                {
                    **dimensions,
                    "record_type": "outer_fold",
                    "fold_id": "fold-0",
                    "status": "complete",
                    "heldout_mean_log_likelihood": model_mean,
                    "parameter_count": parameter_count[family],
                    "per_session": per_session,
                }
            )
            rows.append(
                {
                    **dimensions,
                    "record_type": "protocol_aggregate",
                    "status": "complete",
                    "outer_folds_planned": 1,
                    "outer_folds_complete": 1,
                    "outer_folds_failed": 0,
                }
            )
    return rows


def _write_positive_formal_bundle(root: Path) -> None:
    for seed in range(30):
        _write_attempt(
            root,
            experiment=EXP23,
            seed=seed,
            rows=_exp23_rows(seed),
        )
        _write_attempt(
            root,
            experiment=EXP24,
            seed=seed,
            rows=_exp24_rows(),
        )
    _write_attempt(root, experiment=EXP25, seed=0, rows=_exp25_rows())


def test_joint_summarizer_cli_imports_outside_project(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "summarize_exp23_exp25.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--results-root" in completed.stdout


def test_holm_adjustment_is_monotone_and_retains_invalid_entries() -> None:
    adjusted = _holm_adjust([0.01, 0.02, 0.04, None])
    assert adjusted == [0.04, 0.06, 0.08, None]


def test_smoke_attempt_is_never_promoted_and_all_planned_cells_materialize(
    tmp_path: Path,
) -> None:
    _write_attempt(
        tmp_path,
        experiment=EXP23,
        seed=0,
        rows=_exp23_rows(0),
        profile="smoke",
    )
    bundle = collect_planned_rows(tmp_path, config_root=CONFIG_ROOT)

    assert len(bundle.coverage) == 30 * 12 + 30 * 10 + 20
    exp23 = bundle.coverage.loc[bundle.coverage["experiment"].eq(EXP23)]
    assert exp23["condition_status"].eq("missing").all()
    assert bundle.raw.empty
    summary = summarize_claims(bundle, n_bootstrap=100)
    joint = summary.loc[summary["claim_id"].eq("exp23_joint_both_tasks")].iloc[0]
    assert joint["conclusion"] == "inconclusive"


def test_positive_formal_bundle_supports_all_registered_joint_claims_and_writes(
    tmp_path: Path,
) -> None:
    _write_positive_formal_bundle(tmp_path)
    bundle = collect_planned_rows(tmp_path, config_root=CONFIG_ROOT)
    summary = summarize_claims(bundle, n_bootstrap=200)
    joints = summary.loc[
        summary["claim_id"].isin(
            [
                "exp23_joint_both_tasks",
                "exp24_joint_task_dependent_actuator_specialization",
                "exp25_joint_reusable_shared_belief_dynamics",
            ]
        )
    ]
    assert set(joints["conclusion"]) == {"support"}
    assert set(
        summary.loc[
            summary["row_kind"].eq("condition_coverage"), "status"
        ]
    ) == {"complete"}
    assert (
        summary["row_kind"].eq("condition_coverage").sum()
        == 30 * 12 + 30 * 10 + 20
    )
    assert set(
        summary.loc[summary["row_kind"].eq("claim"), "stats_unit"]
    ) <= {"seed", "animal (sessions nested)"}
    exp23_components = summary.loc[
        summary["claim_id"].str.match(
            r"exp23_(current|delayed)_(gain_vs_|fraction_|median_)",
            na=False,
        )
    ]
    assert len(exp23_components) == 8
    assert exp23_components["p_value"].notna().all()
    assert exp23_components["p_adjusted"].notna().all()
    assert (pd.to_numeric(exp23_components["p_adjusted"]) <= 0.05).all()
    assert set(exp23_components["multiplicity_method"]) == {
        "Holm within the four registered components for this task"
    }
    exp24_components = summary.loc[
        summary["claim_id"].str.match(r"exp24_(?!joint_)", na=False)
    ]
    assert len(exp24_components) == 4
    assert exp24_components["p_adjusted"].notna().all()
    assert (pd.to_numeric(exp24_components["p_adjusted"]) <= 0.05).all()
    assert set(exp24_components["multiplicity_method"]) == {
        "Holm across the four registered Exp24 actuator comparisons"
    }

    paths = write_summary_artifacts(
        bundle,
        output_dir=tmp_path / "published",
        n_bootstrap=200,
    )
    assert paths["summary"].name == "summary.csv"
    assert paths["report"].name == "report.md"
    assert paths["figure"].stat().st_size > 1_000
    assert paths["figure_pdf"].stat().st_size > 1_000
    published = pd.read_csv(paths["summary"])
    assert {"condition_coverage", "claim"} <= set(published["row_kind"])
    report = paths["report"].read_text(encoding="utf-8")
    assert "smoke and pilot attempts" in report
    assert "AND, never OR" in report
    assert "Holm correction" in report
    assert "frozen-recurrent hash/copy receipts" in report


def test_one_failed_exp23_cell_is_retained_and_blocks_formal_support(
    tmp_path: Path,
) -> None:
    for seed in range(30):
        _write_attempt(
            tmp_path,
            experiment=EXP23,
            seed=seed,
            rows=_exp23_rows(seed, failed_local=seed == 29),
        )
    bundle = collect_planned_rows(tmp_path, config_root=CONFIG_ROOT)
    failed = bundle.coverage.loc[
        bundle.coverage["experiment"].eq(EXP23)
        & bundle.coverage["seed"].eq(29)
        & bundle.coverage["task_variant"].eq("delayed")
        & bundle.coverage["condition"].eq("local_eprop")
    ].iloc[0]
    assert failed["condition_status"] == "failed"
    assert "retained local condition failure" in failed["failure_detail"]

    summary = summarize_claims(bundle, n_bootstrap=100)
    delayed = summary.loc[
        summary["claim_id"].eq("exp23_delayed_joint_closed_loop_local_controller")
    ].iloc[0]
    overall = summary.loc[
        summary["claim_id"].eq("exp23_joint_both_tasks")
    ].iloc[0]
    retained = summary.loc[
        summary["row_kind"].eq("condition_coverage")
        & summary["experiment"].eq(EXP23)
        & summary["scope"].eq("delayed")
        & summary["condition"].eq("local_eprop")
        & summary["unit_id"].eq("seed:29")
    ].iloc[0]
    assert retained["status"] == "failed"
    assert delayed["conclusion"] == "inconclusive"
    assert overall["conclusion"] == "inconclusive"


def test_exp23_missing_local_mechanism_receipt_blocks_formal_support(
    tmp_path: Path,
) -> None:
    for seed in range(30):
        rows = _exp23_rows(seed)
        if seed == 29:
            local = next(
                row
                for row in rows
                if row["task_variant"] == "current"
                and row["condition"] == "local_eprop"
            )
            local["local_rule_bptt_free"] = False
        _write_attempt(
            tmp_path,
            experiment=EXP23,
            seed=seed,
            rows=rows,
        )
    bundle = collect_planned_rows(tmp_path, config_root=CONFIG_ROOT)
    summary = summarize_claims(bundle, n_bootstrap=100)

    coverage = summary.loc[
        summary["row_kind"].eq("condition_coverage")
        & summary["experiment"].eq(EXP23)
    ]
    component = summary.loc[
        summary["claim_id"].eq("exp23_current_gain_vs_frozen")
    ].iloc[0]
    joint = summary.loc[
        summary["claim_id"].eq("exp23_current_joint_closed_loop_local_controller")
    ].iloc[0]
    assert coverage["status"].eq("complete").all()
    assert component["n_invalid"] == 1
    assert component["conclusion"] == "inconclusive"
    assert joint["n_invalid"] == 1
    assert joint["conclusion"] == "inconclusive"


def test_invalid_cross_session_never_supports_exp25_joint(tmp_path: Path) -> None:
    _write_attempt(
        tmp_path,
        experiment=EXP25,
        seed=0,
        rows=_exp25_rows(invalid_cross_session=True),
    )
    bundle = collect_planned_rows(tmp_path, config_root=CONFIG_ROOT)
    summary = summarize_claims(bundle, n_bootstrap=100)

    implemented = summary.loc[
        summary["claim_id"].eq("exp25_fully_gated_vs_common")
    ].iloc[0]
    cross = summary.loc[
        summary["claim_id"].eq("exp25_cross_session_fully_vs_common")
    ].iloc[0]
    joint = summary.loc[
        summary["claim_id"].eq("exp25_joint_reusable_shared_belief_dynamics")
    ].iloc[0]
    assert implemented["conclusion"] == "support"
    assert cross["n_invalid"] == 2
    assert cross["conclusion"] == "inconclusive"
    assert joint["conclusion"] == "inconclusive"


def test_exp25_retention_cannot_support_a_nonpositive_separate_gain(
    tmp_path: Path,
) -> None:
    rows = _exp25_rows()
    for row in rows:
        if (
            row.get("record_type") == "outer_fold"
            and row.get("model_family") == "separate-task"
            and row.get("protocol") != "cross-session-transfer"
        ):
            for session in row["per_session"]:
                session["log_likelihood"] = -210.0
    _write_attempt(tmp_path, experiment=EXP25, seed=0, rows=rows)
    bundle = collect_planned_rows(tmp_path, config_root=CONFIG_ROOT)
    summary = summarize_claims(bundle, n_bootstrap=100)

    retention = summary.loc[
        summary["claim_id"].eq("exp25_fully_retains_90pct_separate_gain")
    ].iloc[0]
    assert retention["conclusion"] == "inconclusive"
    assert "separate-task > common" in retention["note"]
