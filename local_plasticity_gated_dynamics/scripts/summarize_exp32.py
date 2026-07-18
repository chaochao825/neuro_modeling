"""Fail-closed seed-level summary for Exp32 persistent sparse feedback."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp32_persistent_sparse_feedback import (
    EVIDENCE_SCHEMA_VERSION,
    EXPERIMENT,
    MODES,
    PROTOCOL_VERSION,
    PROJECT_ROOT,
    SUPPORTED_EVIDENCE_SCHEMA_VERSIONS,
    _expected_holm_family,
    _formal_authorization_digest,
    _git_checkout_identity,
    _planned_conditions,
    _task_config,
    _validate_config,
)
from src.analysis.hidden_selector_metrics import (
    holm_adjust,
    paired_bootstrap_interval,
    sign_flip_pvalue,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


CHECKOUT_POLICIES = frozenset({"live", "archived"})


def _validate_checkout_binding(
    raw: pd.DataFrame,
    provenance: Mapping[str, Any],
    *,
    profile: str,
    checkout_policy: str,
) -> dict[str, object]:
    """Bind live formal inference to its run checkout.

    Archived reanalysis is an explicit exception: it still requires the exact
    config, authorization receipt, and critical-file hashes, but does not claim
    that the analyst's current HEAD is the historical execution checkout.
    """

    if checkout_policy not in CHECKOUT_POLICIES:
        raise ValueError(f"checkout_policy must be one of {sorted(CHECKOUT_POLICIES)}")
    run_git = provenance.get("git")
    if not isinstance(run_git, Mapping):
        raise RuntimeError("Exp32 source provenance has no Git identity")
    for key, column in (
        ("commit", "run_git_commit"),
        ("tree", "run_git_tree"),
        ("dirty", "run_git_dirty"),
    ):
        if column not in raw:
            raise RuntimeError(f"Exp32 rows lack provenance column: {column}")
        expected = run_git.get(key)
        observed = raw[column]
        if expected is None:
            if observed.notna().any():
                raise RuntimeError(
                    f"Exp32 row {column} disagrees with source provenance"
                )
        elif key == "dirty":
            if not observed.eq(bool(expected)).all():
                raise RuntimeError(
                    f"Exp32 row {column} disagrees with source provenance"
                )
        elif (
            set(observed.dropna().astype(str)) != {str(expected)}
            or observed.isna().any()
        ):
            raise RuntimeError(f"Exp32 row {column} disagrees with source provenance")

    run_commit = run_git.get("commit")
    run_tree = run_git.get("tree")
    if profile == "formal" and (
        not isinstance(run_commit, str)
        or len(run_commit) != 40
        or not isinstance(run_tree, str)
        or len(run_tree) != 40
    ):
        raise RuntimeError("formal Exp32 provenance lacks bound Git objects")
    current = _git_checkout_identity()
    checkout_matches = bool(
        isinstance(run_commit, str)
        and isinstance(run_tree, str)
        and current.get("commit") == run_commit
        and current.get("tree") == run_tree
    )
    if profile == "formal" and checkout_policy == "live" and not checkout_matches:
        raise RuntimeError(
            "formal Exp32 live summary current checkout commit/tree mismatch"
        )
    return {
        "checkout_policy": checkout_policy,
        "archived_reanalysis": checkout_policy == "archived",
        "run_git_commit": run_commit,
        "run_git_tree": run_tree,
        "current_git_commit": current.get("commit"),
        "current_git_tree": current.get("tree"),
        "current_checkout_matches_run": checkout_matches,
        "current_checkout_reproducibility_claimed": bool(
            checkout_policy == "live" and checkout_matches
        ),
    }


def _checkout_summary_fields(raw: pd.DataFrame) -> dict[str, object]:
    policy = str(raw.attrs.get("checkout_policy", "direct_records_unbound"))
    matches = raw.attrs.get("current_checkout_matches_run")
    return {
        "checkout_policy": policy,
        "archived_reanalysis": policy == "archived",
        "current_checkout_matches_run": matches,
        "current_checkout_reproducibility_claimed": bool(
            policy == "live" and matches is True
        ),
    }


def _eligible_run_dirs(
    root: Path, *, seeds: list[int], profile: str, run_label: str
) -> list[Path]:
    selected: list[Path] = []
    for seed in seeds:
        seed_root = root / "runs" / EXPERIMENT / f"seed_{seed:04d}"
        candidates: list[Path] = []
        if seed_root.exists():
            for path in sorted(item for item in seed_root.iterdir() if item.is_dir()):
                if (
                    not (path / "status.json").exists()
                    or not (path / "manifest.json").exists()
                ):
                    continue
                status = _read_json(path / "status.json")
                manifest = _read_json(path / "manifest.json")
                if (
                    manifest.get("profile") == profile
                    and manifest.get("run_label") == run_label
                    and status.get("status") in {"complete", "complete_with_failures"}
                ):
                    candidates.append(path)
        if len(candidates) != 1:
            raise RuntimeError(
                f"seed {seed} has {len(candidates)} eligible Exp32 runs for "
                f"label {run_label!r}; expected exactly one"
            )
        selected.append(candidates[0])
    return selected


def load_panel(
    root: Path,
    config: dict[str, Any],
    *,
    run_label: str,
    checkout_policy: str = "live",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_config(config)
    seeds = seed_list(config["seeds"])
    run_dirs = _eligible_run_dirs(
        root, seeds=seeds, profile=str(config["profile"]), run_label=run_label
    )
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    common_provenance: dict[str, Any] | None = None
    expected_rows = len(_planned_conditions(config))
    for path in run_dirs:
        observed_config = _read_json(path / "config.json")
        observed_static = {
            key: value
            for key, value in observed_config.items()
            if key not in {"experiment", "seed", "run_label", "evidence_provenance"}
        }
        expected_static = {
            key: value for key, value in config.items() if key != "config_path"
        }
        if observed_static != expected_static:
            raise RuntimeError(f"{path} does not match the selected Exp32 config")
        provenance = observed_config.get("evidence_provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError(f"{path} has no evidence provenance")
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            raise RuntimeError("Exp32 panel mixes source provenance")
        rows = pd.read_json(path / "metrics.jsonl", lines=True)
        if len(rows) != expected_rows:
            raise RuntimeError(f"{path} has {len(rows)} rows; expected {expected_rows}")
        frames.append(rows)
        status = _read_json(path / "status.json")
        manifest_rows.append(
            {
                "seed": int(status["seed"]),
                "run_path": str(path.resolve()),
                "run_status": str(status["status"]),
                "condition_failures": int(status["condition_failures"]),
                "condition_invalid": int(status["condition_invalid"]),
            }
        )
    raw = pd.concat(frames, ignore_index=True)
    if common_provenance is None:
        raise RuntimeError("Exp32 panel has no common source provenance")
    critical = common_provenance.get("critical_file_sha256")
    if not isinstance(critical, dict) or not critical:
        raise RuntimeError("Exp32 source provenance has no critical file hashes")
    for relative, expected_sha in critical.items():
        source = (PROJECT_ROOT / str(relative)).resolve()
        if not source.is_relative_to(PROJECT_ROOT.resolve()) or not source.is_file():
            raise RuntimeError(f"Exp32 critical source is missing: {relative}")
        if _sha256(source) != str(expected_sha):
            raise RuntimeError(
                f"Exp32 critical source changed after execution: {relative}"
            )
    config_path = Path(str(config.get("config_path", "")))
    if not config_path.is_file() or _sha256(config_path) != common_provenance.get(
        "source_config_sha256"
    ):
        raise RuntimeError("Exp32 selected config changed after execution")
    if str(config["profile"]) == "formal":
        authorization_digest = _formal_authorization_digest(config)
        if (
            common_provenance.get("formal_authorization_receipt_sha256")
            != authorization_digest
        ):
            raise RuntimeError("Exp32 formal authorization provenance mismatch")
    validate_panel_contract(raw, config)
    checkout_binding = _validate_checkout_binding(
        raw,
        common_provenance,
        profile=str(config["profile"]),
        checkout_policy=checkout_policy,
    )
    raw.attrs.update(checkout_binding)
    manifest = pd.DataFrame(manifest_rows).sort_values("seed").reset_index(drop=True)
    for key, value in checkout_binding.items():
        manifest[key] = value
    return raw, manifest


def validate_panel_contract(raw: pd.DataFrame, config: dict[str, Any]) -> None:
    base_required = {
        "seed",
        "hazard",
        "feedback_fraction",
        "feedback_delay",
        "selector_mode",
        "status",
        "protocol_version",
        "evidence_schema_version",
        "run_git_commit",
        "run_git_tree",
        "run_git_dirty",
    }
    missing = base_required - set(raw.columns)
    if missing:
        raise ValueError(f"missing Exp32 contract columns: {sorted(missing)}")
    seeds = seed_list(config["seeds"])
    planned = _planned_conditions(config)
    expected_cells = {
        (
            float(row["hazard"]),
            float(row["feedback_fraction"]),
            int(row["feedback_delay"]),
            str(row["selector_mode"]),
        )
        for row in planned
    }
    observed = list(
        zip(
            raw["seed"].astype(int),
            raw["hazard"].astype(float),
            raw["feedback_fraction"].astype(float),
            raw["feedback_delay"].astype(int),
            raw["selector_mode"].astype(str),
            strict=True,
        )
    )
    if len(observed) != len(set(observed)):
        raise RuntimeError("Exp32 contains duplicate seed/cell/mode rows")
    expected = {
        (seed, hazard, fraction, delay, mode)
        for seed in seeds
        for hazard, fraction, delay, mode in expected_cells
    }
    if set(observed) != expected:
        raise RuntimeError("Exp32 panel is missing or adds planned rows")
    if set(raw["protocol_version"].astype(str)) != {PROTOCOL_VERSION}:
        raise RuntimeError("Exp32 panel mixes protocol versions")
    schemas = set(raw["evidence_schema_version"].dropna().astype(str))
    if len(schemas) != 1 or not schemas <= SUPPORTED_EVIDENCE_SCHEMA_VERSIONS:
        raise RuntimeError("Exp32 panel mixes evidence schemas")
    formal = str(config["profile"]) == "formal"
    if formal and schemas != {EVIDENCE_SCHEMA_VERSION}:
        raise RuntimeError("formal Exp32 requires the current evidence schema")
    if formal and not raw["run_git_dirty"].eq(False).all():
        raise RuntimeError("publishable formal Exp32 requires clean Git runs")
    if formal:
        for column in ("run_git_commit", "run_git_tree"):
            values = raw[column].astype(str)
            if values.nunique() != 1 or len(values.iloc[0]) != 40:
                raise RuntimeError(f"Exp32 lacks one bound {column}")
    statuses = set(raw["status"].astype(str))
    allowed_statuses = {"complete", "failed", "invalid"}
    if not statuses <= allowed_statuses:
        raise RuntimeError(f"Exp32 contains unknown row statuses: {sorted(statuses)}")
    if statuses != {"complete"}:
        # Setup and condition failures intentionally lack downstream pairing
        # fingerprints.  The exact planned grid, schema, and Git provenance above
        # remain mandatory, but no directional audit may be inferred from a
        # scientifically incomplete panel.
        return
    fingerprint_columns = {
        "train_pool_fingerprint",
        "test_pool_fingerprint",
        "selected_potential_outcome_fingerprint",
        "state_tape_fingerprint",
        "feedback_tape_fingerprint",
        "action_uniform_tape_fingerprint",
    }
    missing = fingerprint_columns - set(raw.columns)
    if missing:
        raise ValueError(f"missing Exp32 contract columns: {sorted(missing)}")
    group = ["seed", "hazard", "feedback_fraction", "feedback_delay"]
    for column in (
        "train_pool_fingerprint",
        "test_pool_fingerprint",
        "selected_potential_outcome_fingerprint",
        "state_tape_fingerprint",
        "feedback_tape_fingerprint",
        "action_uniform_tape_fingerprint",
    ):
        if raw.groupby(group)[column].nunique().max() != 1:
            raise RuntimeError(f"Exp32 controls are not paired for {column}")
    delay_group = ["seed", "hazard", "feedback_fraction"]
    for column in (
        "train_pool_fingerprint",
        "test_pool_fingerprint",
        "selected_potential_outcome_fingerprint",
        "state_tape_fingerprint",
        "feedback_tape_fingerprint",
        "action_uniform_tape_fingerprint",
    ):
        if raw.groupby(delay_group)[column].nunique().max() != 1:
            raise RuntimeError(f"Exp32 delay cells changed the paired tape {column}")
    for column in (
        "train_pool_fingerprint",
        "test_pool_fingerprint",
        "action_uniform_tape_fingerprint",
    ):
        if raw.groupby("seed")[column].nunique().max() != 1:
            raise RuntimeError(f"Exp32 seed-level paired tape changed: {column}")
    for column in (
        "selected_potential_outcome_fingerprint",
        "state_tape_fingerprint",
    ):
        if raw.groupby(["seed", "hazard"])[column].nunique().max() != 1:
            raise RuntimeError(f"Exp32 hazard-level paired tape changed: {column}")
    if (
        raw.groupby(["seed", "feedback_fraction"])["feedback_tape_fingerprint"]
        .nunique()
        .max()
        != 1
    ):
        raise RuntimeError("Exp32 feedback tape changed across hazard or delay")
    if schemas == {EVIDENCE_SCHEMA_VERSION}:
        for column in (
            "feedback_schedule_nested_audit",
            "no_feedback_matches_random_action_tape",
        ):
            if column not in raw or not raw[column].eq(True).all():
                raise RuntimeError(f"Exp32 current-schema audit failed: {column}")


def _primary(raw: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    analysis = dict(config["analysis"])
    return raw[
        np.isclose(raw["hazard"], float(analysis["primary_hazard"]))
        & np.isclose(
            raw["feedback_fraction"],
            float(analysis["primary_feedback_fraction"]),
        )
        & raw["feedback_delay"].eq(int(analysis["primary_feedback_delay"]))
    ]


def _phase_seed_metrics(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
    seed_order: np.ndarray,
) -> pd.DataFrame:
    """Fit the frozen two-timescale response model once within each seed."""

    analysis = dict(config["analysis"])
    selector = dict(config["selector"])
    retention = float(selector["retention"])
    tau_q = -1.0 / np.log(retention)
    pivot = raw.pivot(
        index=["seed", "hazard", "feedback_fraction", "feedback_delay"],
        columns="selector_mode",
        values="full_stream_accuracy",
    ).reset_index()
    pivot["local_minus_fixed"] = (
        pivot["persistent_rpe_local"] - pivot["train_fixed_best"]
    )

    def effect(
        frame: pd.DataFrame, *, hazard: float, fraction: float, delay: int
    ) -> float:
        selected = frame[
            np.isclose(frame["hazard"], hazard)
            & np.isclose(frame["feedback_fraction"], fraction)
            & frame["feedback_delay"].eq(delay)
        ]
        if len(selected) != 1:
            raise RuntimeError(
                "Exp32 phase contrast does not identify exactly one registered cell"
            )
        return float(selected["local_minus_fixed"].iloc[0])

    rows: list[dict[str, float | int]] = []
    for seed in seed_order:
        frame = pivot[pivot["seed"].eq(int(seed))].copy()
        hazard = frame["hazard"].to_numpy(float)
        fraction = frame["feedback_fraction"].to_numpy(float)
        delay = frame["feedback_delay"].to_numpy(float)
        log_lambda = np.log2(fraction / hazard)
        log_chi = np.log2(hazard * tau_q)
        kappa = np.power(1.0 - 2.0 * hazard, delay + 1.0)
        if np.any(kappa <= 0.0):
            raise RuntimeError("Exp32 delay attenuation is not positive")
        delay_burden = -np.log2(kappa)
        design = np.column_stack(
            [
                np.ones(len(frame)),
                log_lambda,
                log_chi,
                delay_burden,
                log_lambda * log_chi,
            ]
        )
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise RuntimeError("Exp32 phase-response design is rank deficient")
        target = frame["local_minus_fixed"].to_numpy(float)
        coefficient, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficient
        slow_hazard = float(analysis["iso_lambda_slow_hazard"])
        fast_hazard = float(analysis["iso_lambda_fast_hazard"])
        iso_delay = int(analysis["primary_feedback_delay"])
        iso_differences = []
        for lambda_value in analysis["iso_lambda_values"]:
            value = float(lambda_value)
            iso_differences.append(
                effect(
                    frame,
                    hazard=slow_hazard,
                    fraction=value * slow_hazard,
                    delay=iso_delay,
                )
                - effect(
                    frame,
                    hazard=fast_hazard,
                    fraction=value * fast_hazard,
                    delay=iso_delay,
                )
            )
        delay_differences = []
        for probe_fraction in analysis["delay_probe_feedback_fractions"]:
            delay_differences.append(
                effect(
                    frame,
                    hazard=float(analysis["delay_probe_hazard"]),
                    fraction=float(probe_fraction),
                    delay=int(analysis["delay_probe_short"]),
                )
                - effect(
                    frame,
                    hazard=float(analysis["delay_probe_hazard"]),
                    fraction=float(probe_fraction),
                    delay=int(analysis["delay_probe_long"]),
                )
            )
        rows.append(
            {
                "seed": int(seed),
                "evidence_response_slope": float(coefficient[1]),
                "memory_timescale_slope": float(coefficient[2]),
                "delay_burden_slope": float(coefficient[3]),
                "evidence_timescale_interaction": float(coefficient[4]),
                "phase_model_rmse": float(np.sqrt(np.mean(residual * residual))),
                "iso_lambda_slow_minus_fast": float(np.mean(iso_differences)),
                "short_minus_long_delay": float(np.mean(delay_differences)),
            }
        )
    return pd.DataFrame(rows)


def summarize_records(
    raw: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    checkout_summary = _checkout_summary_fields(raw)
    base_required = {
        "seed",
        "status",
        "hazard",
        "feedback_fraction",
        "feedback_delay",
        "selector_mode",
    }
    missing = base_required - set(raw.columns)
    if missing:
        raise ValueError(f"missing Exp32 summary columns: {sorted(missing)}")
    if not raw["status"].eq("complete").all():
        conditions = (
            raw.groupby(
                [
                    "hazard",
                    "feedback_fraction",
                    "feedback_delay",
                    "selector_mode",
                    "status",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(n_rows=("seed", "size"), n_seeds=("seed", "nunique"))
            .sort_values(
                [
                    "hazard",
                    "feedback_fraction",
                    "feedback_delay",
                    "selector_mode",
                    "status",
                ]
            )
            .reset_index(drop=True)
        )
        return (
            conditions,
            pd.DataFrame(),
            {
                "experiment": EXPERIMENT,
                "profile": str(config["profile"]),
                **checkout_summary,
                "n_seeds": int(raw["seed"].nunique()),
                "n_planned_rows": int(len(raw)),
                "n_complete_rows": int(raw["status"].eq("complete").sum()),
                "n_failed_rows": int(raw["status"].eq("failed").sum()),
                "n_invalid_rows": int(raw["status"].eq("invalid").sum()),
                "panel_complete": False,
                "access_and_pairing_audit_passed": False,
                "scale_decision": "scale-not-authorized",
                "claim_classification": "inconclusive",
                "failure_reason": "one or more planned cells failed",
            },
        )
    required = {
        "full_stream_accuracy",
        "dynamic_regret_to_hidden_oracle",
        "reward_only_interface_audit",
        "selector_received_true_context",
        "selector_received_unexecuted_reward",
        "both_actuator_winners_present",
        "feedback_pending_count",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing Exp32 summary columns: {sorted(missing)}")
    raw = raw.copy()
    for column in (
        "full_stream_accuracy",
        "dynamic_regret_to_hidden_oracle",
        "median_switch_latency",
        "false_switch_rate",
        "context_nll",
        "context_brier",
        "selector_update_l1",
    ):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    conditions = (
        raw.groupby(
            ["hazard", "feedback_fraction", "feedback_delay", "selector_mode"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_accuracy=("full_stream_accuracy", "mean"),
            sd_accuracy=("full_stream_accuracy", "std"),
            mean_dynamic_regret=("dynamic_regret_to_hidden_oracle", "mean"),
            mean_switch_latency=("median_switch_latency", "mean"),
            mean_false_switch_rate=("false_switch_rate", "mean"),
            mean_context_nll=("context_nll", "mean"),
            mean_context_brier=("context_brier", "mean"),
        )
        .sort_values(["hazard", "feedback_fraction", "feedback_delay", "selector_mode"])
        .reset_index(drop=True)
    )
    conditions["expected_feedback_per_dwell"] = (
        conditions["feedback_fraction"] / conditions["hazard"]
    )
    conditions["delay_fraction_of_expected_dwell"] = (
        conditions["feedback_delay"] * conditions["hazard"]
    )
    primary = _primary(raw, config)
    if primary["seed"].nunique() != raw["seed"].nunique():
        raise RuntimeError("Exp32 primary cell is incomplete")
    pivot = primary.pivot(
        index="seed", columns="selector_mode", values="full_stream_accuracy"
    )
    if tuple(sorted(pivot.columns)) != tuple(sorted(MODES)):
        raise RuntimeError("Exp32 primary cell lacks registered modes")
    seeds = pd.DataFrame(
        {
            "seed": pivot.index.astype(int),
            **{f"{mode}_accuracy": pivot[mode].to_numpy(float) for mode in MODES},
        }
    )
    seeds["persistent_minus_fixed"] = (
        seeds["persistent_rpe_local_accuracy"] - seeds["train_fixed_best_accuracy"]
    )
    seeds["persistent_minus_credit_shuffled"] = (
        seeds["persistent_rpe_local_accuracy"] - seeds["credit_shuffled_local_accuracy"]
    )
    seeds["persistent_minus_opposite_eligibility"] = seeds[
        "persistent_minus_credit_shuffled"
    ]
    seeds["persistent_minus_cumulative"] = (
        seeds["persistent_rpe_local_accuracy"]
        - seeds["cumulative_sample_average_accuracy"]
    )
    seeds["oracle_opportunity"] = (
        seeds["oracle_hidden_state_accuracy"] - seeds["train_fixed_best_accuracy"]
    )
    seeds["oracle_gain_retained"] = np.divide(
        seeds["persistent_minus_fixed"],
        seeds["oracle_opportunity"],
        out=np.zeros(len(seeds), dtype=np.float64),
        where=seeds["oracle_opportunity"].to_numpy(float) > 0.0,
    )
    seeds = seeds.sort_values("seed").reset_index(drop=True)

    adaptive = raw[
        raw["selector_mode"].isin(
            [
                "cumulative_sample_average",
                "persistent_rpe_local",
                "credit_shuffled_local",
                "bayes_reward_filter",
            ]
        )
    ]
    nonoracle = raw[raw["selector_mode"] != "oracle_hidden_state"]
    audit_ok = bool(
        raw["reward_only_interface_audit"].eq(True).all()
        and nonoracle["selector_received_true_context"].eq(False).all()
        and raw["selector_received_unexecuted_reward"].eq(False).all()
        and raw["both_actuator_winners_present"].eq(True).all()
        and raw["feedback_pending_count"].eq(0).all()
        and adaptive["feedback_delivered_count"]
        .eq(adaptive["feedback_available_count"])
        .all()
        and raw["time_points_randomly_split"].eq(False).all()
        and raw["controller_reset_at_switch"].eq(False).all()
        and raw["switch_times_exposed_to_selector"].eq(False).all()
        and (
            "feedback_schedule_nested_audit" not in raw
            or raw["feedback_schedule_nested_audit"].eq(True).all()
        )
        and (
            "no_feedback_matches_random_action_tape" not in raw
            or raw["no_feedback_matches_random_action_tape"].eq(True).all()
        )
    )
    analysis = dict(config["analysis"])
    claim_family = str(analysis.get("claim_family", "original_primary"))
    if claim_family == "evidence_per_dwell_boundary":
        reference = raw[
            np.isclose(raw["hazard"], float(analysis["boundary_reference_hazard"]))
            & np.isclose(
                raw["feedback_fraction"],
                float(analysis["primary_feedback_fraction"]),
            )
            & raw["feedback_delay"].eq(int(analysis["primary_feedback_delay"]))
        ].pivot(index="seed", columns="selector_mode", values="full_stream_accuracy")
        if set(reference.index.astype(int)) != set(seeds["seed"].astype(int)):
            raise RuntimeError("Exp32 boundary reference cell is incomplete")
        reference = reference.reindex(seeds["seed"].to_numpy(int))
        seeds["reference_persistent_minus_fixed"] = reference[
            "persistent_rpe_local"
        ].to_numpy(float) - reference["train_fixed_best"].to_numpy(float)
        seeds["evidence_per_dwell_boundary_interaction"] = (
            seeds["persistent_minus_fixed"] - seeds["reference_persistent_minus_fixed"]
        )
    elif claim_family == "feedback_memory_timescale_phase":
        phase = _phase_seed_metrics(
            raw,
            config,
            seeds["seed"].to_numpy(int),
        )
        seeds = seeds.merge(phase, on="seed", how="left", validate="one_to_one")
        if (
            seeds[
                [
                    "evidence_response_slope",
                    "iso_lambda_slow_minus_fast",
                    "short_minus_long_delay",
                ]
            ]
            .isna()
            .any()
            .any()
        ):
            raise RuntimeError("Exp32 phase metrics are incomplete")
        primary_updates = (
            _primary(raw, config)
            .pivot(index="seed", columns="selector_mode", values="selector_update_l1")
            .reindex(seeds["seed"].to_numpy(int))
        )
        seeds["opposite_over_executed_reward_update_l1"] = np.divide(
            primary_updates["credit_shuffled_local"].to_numpy(float),
            primary_updates["persistent_rpe_local"].to_numpy(float),
            out=np.full(len(seeds), np.nan, dtype=np.float64),
            where=primary_updates["persistent_rpe_local"].to_numpy(float) > 0.0,
        )
    positive_fraction = float(np.mean(seeds["persistent_minus_fixed"] > 0.0))
    mean_retention = float(seeds["oracle_gain_retained"].mean())
    smoke_gate = bool(
        audit_ok
        and len(seeds) == 5
        and float(seeds["persistent_minus_fixed"].mean())
        >= float(analysis["primary_mcid"])
        and positive_fraction >= float(analysis["minimum_positive_seed_fraction"])
        and float(seeds["persistent_minus_credit_shuffled"].mean())
        >= float(analysis["credit_specificity_mcid"])
        and mean_retention >= float(analysis["oracle_gain_retention_fraction"])
    )
    summary: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "profile": str(config["profile"]),
        **checkout_summary,
        "n_seeds": int(len(seeds)),
        "panel_complete": True,
        "access_and_pairing_audit_passed": audit_ok,
        "scale_decision": "scale-authorized" if smoke_gate else "scale-not-authorized",
        "claim_classification": "inconclusive",
        "claim_family": claim_family,
        "primary_hazard": float(analysis["primary_hazard"]),
        "primary_feedback_fraction": float(analysis["primary_feedback_fraction"]),
        "primary_feedback_delay": int(analysis["primary_feedback_delay"]),
        "mean_persistent_minus_fixed": float(seeds["persistent_minus_fixed"].mean()),
        "mean_persistent_minus_credit_shuffled": float(
            seeds["persistent_minus_credit_shuffled"].mean()
        ),
        "mean_persistent_minus_cumulative": float(
            seeds["persistent_minus_cumulative"].mean()
        ),
        "mean_oracle_opportunity": float(seeds["oracle_opportunity"].mean()),
        "mean_oracle_gain_retained": mean_retention,
        "positive_seed_fraction": positive_fraction,
        "primary_expected_feedback_per_dwell": float(
            analysis["primary_feedback_fraction"] / analysis["primary_hazard"]
        ),
        "high_rank_carrier_claimed": False,
        "real_data_claimed": False,
    }
    if claim_family == "evidence_per_dwell_boundary":
        summary.update(
            {
                "boundary_reference_hazard": float(
                    analysis["boundary_reference_hazard"]
                ),
                "boundary_reference_expected_feedback_per_dwell": float(
                    analysis["primary_feedback_fraction"]
                    / analysis["boundary_reference_hazard"]
                ),
                "mean_reference_persistent_minus_fixed": float(
                    seeds["reference_persistent_minus_fixed"].mean()
                ),
                "mean_evidence_per_dwell_boundary_interaction": float(
                    seeds["evidence_per_dwell_boundary_interaction"].mean()
                ),
            }
        )
    elif claim_family == "feedback_memory_timescale_phase":
        summary.update(
            {
                "controller_memory_time_constant_trials": float(
                    -1.0 / np.log(float(config["selector"]["retention"]))
                ),
                "mean_evidence_response_slope": float(
                    seeds["evidence_response_slope"].mean()
                ),
                "mean_iso_lambda_slow_minus_fast": float(
                    seeds["iso_lambda_slow_minus_fast"].mean()
                ),
                "mean_short_minus_long_delay": float(
                    seeds["short_minus_long_delay"].mean()
                ),
                "mean_opposite_over_executed_reward_update_l1": float(
                    seeds["opposite_over_executed_reward_update_l1"].mean()
                ),
                "credit_control_semantics": "opposite_action_eligibility_not_random_shuffle",
                "credit_budget_matched_claimed": False,
                "local_context_metrics_semantics": "action_policy_proxy_not_calibrated_context_posterior",
            }
        )
    for fraction in _task_config(config).feedback_fractions:
        selected = raw[
            np.isclose(raw["hazard"], float(analysis["primary_hazard"]))
            & np.isclose(raw["feedback_fraction"], fraction)
            & raw["feedback_delay"].eq(int(analysis["primary_feedback_delay"]))
        ].pivot(index="seed", columns="selector_mode", values="full_stream_accuracy")
        summary[f"local_minus_fixed_feedback_{fraction:g}"] = float(
            (selected["persistent_rpe_local"] - selected["train_fixed_best"]).mean()
        )
    if config["profile"] == "formal":
        contrasts = {
            "persistent_over_train_fixed": (
                seeds["persistent_minus_fixed"].to_numpy(float),
                float(analysis["primary_mcid"]),
            ),
            "persistent_over_opposite_eligibility": (
                seeds["persistent_minus_opposite_eligibility"].to_numpy(float),
                float(analysis["credit_specificity_mcid"]),
            ),
        }
        if claim_family == "evidence_per_dwell_boundary":
            contrasts["evidence_per_dwell_boundary_interaction"] = (
                seeds["evidence_per_dwell_boundary_interaction"].to_numpy(float),
                float(analysis["boundary_interaction_mcid"]),
            )
        elif claim_family == "feedback_memory_timescale_phase":
            contrasts["evidence_response_slope"] = (
                seeds["evidence_response_slope"].to_numpy(float),
                float(analysis["evidence_slope_mcid"]),
            )
        configured_holm_family = tuple(
            str(value) for value in analysis.get("holm_family", ())
        )
        expected_holm_family = _expected_holm_family(claim_family)
        if tuple(contrasts) != expected_holm_family:
            raise RuntimeError("Exp32 code contrasts differ from the registered family")
        if configured_holm_family != expected_holm_family:
            raise RuntimeError(
                "Exp32 configured Holm family differs from the inferred contrasts"
            )
        raw_p: dict[str, float] = {}
        lower_pass: dict[str, bool] = {}
        for index, (name, (values, threshold)) in enumerate(contrasts.items()):
            lower, upper = paired_bootstrap_interval(
                values,
                n_resamples=int(analysis["bootstrap_samples"]),
                seed=int(analysis["statistics_seed"]) + index,
            )
            pvalue = sign_flip_pvalue(
                values - threshold,
                n_resamples=int(analysis["permutation_samples"]),
                seed=int(analysis["statistics_seed"]) + 100 + index,
            )
            raw_p[name] = pvalue
            lower_pass[name] = lower > threshold
            summary[f"{name}_ci_lower"] = lower
            summary[f"{name}_ci_upper"] = upper
            summary[f"{name}_raw_p"] = pvalue
            summary[f"{name}_threshold"] = threshold
        adjusted = holm_adjust(raw_p)
        for name, value in adjusted.items():
            summary[f"{name}_holm_p"] = value
        support = bool(
            audit_ok
            and len(seeds) == 30
            and all(lower_pass.values())
            and all(value < 0.05 for value in adjusted.values())
            and mean_retention >= float(analysis["oracle_gain_retention_fraction"])
            and positive_fraction >= float(analysis["minimum_positive_seed_fraction"])
        )
        main_classification = "inconclusive"
        if support:
            main_classification = "support"
        elif audit_ok and any(
            float(summary[f"{name}_ci_upper"]) <= 0.0 for name in contrasts
        ):
            main_classification = "oppose"
        summary["main_controller_claim_classification"] = main_classification
        if claim_family == "feedback_memory_timescale_phase":
            structure_values = seeds["iso_lambda_slow_minus_fast"].to_numpy(float)
            structure_threshold = float(analysis["timescale_structure_mcid"])
            lower, upper = paired_bootstrap_interval(
                structure_values,
                n_resamples=int(analysis["bootstrap_samples"]),
                seed=int(analysis["statistics_seed"]) + 50,
            )
            pvalue = sign_flip_pvalue(
                structure_values - structure_threshold,
                n_resamples=int(analysis["permutation_samples"]),
                seed=int(analysis["statistics_seed"]) + 150,
            )
            summary.update(
                {
                    "iso_lambda_slow_minus_fast_ci_lower": lower,
                    "iso_lambda_slow_minus_fast_ci_upper": upper,
                    "iso_lambda_slow_minus_fast_raw_p": pvalue,
                    "iso_lambda_slow_minus_fast_threshold": structure_threshold,
                }
            )
            structure_classification = (
                "support"
                if audit_ok
                and len(seeds) == 30
                and lower > structure_threshold
                and pvalue < 0.05
                else "oppose"
                if audit_ok and upper <= 0.0
                else "inconclusive"
            )
            summary["timescale_structure_claim_classification"] = (
                structure_classification
            )
            summary["claim_classification"] = (
                "support"
                if main_classification == structure_classification == "support"
                else "oppose"
                if "oppose" in {main_classification, structure_classification}
                else "inconclusive"
            )
        else:
            summary["claim_classification"] = main_classification
        summary["scale_decision"] = "formal-complete"
    return conditions, seeds, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_lf(path: Path, text: str) -> None:
    """Write portable, hash-stable UTF-8 text without OS newline translation."""

    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _write_csv_lf(
    frame: pd.DataFrame, path: Path, *, gzip_compressed: bool = False
) -> None:
    if gzip_compressed:
        with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                frame.to_csv(stream, index=False, lineterminator="\n")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")


def package_run_receipts(manifest: pd.DataFrame, output: Path) -> pd.DataFrame:
    required = (
        ("config.json", "config.json"),
        ("environment.json", "environment.json"),
        ("status.json", "status.json"),
        ("manifest.json", "manifest.json"),
        ("planned_conditions.json", "planned_conditions.json"),
        ("metrics.jsonl", "metrics.jsonl"),
        ("run.log", "run_log.txt"),
    )
    receipt_root = output / "run_receipts"
    receipt_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for record in manifest.itertuples(index=False):
        source = Path(str(record.run_path))
        destination = receipt_root / f"seed_{int(record.seed):04d}"
        destination.mkdir(parents=True, exist_ok=False)
        for source_name, destination_name in required:
            source_file = source / source_name
            if not source_file.exists():
                raise FileNotFoundError(f"missing Exp32 receipt: {source_file}")
            destination_file = destination / destination_name
            shutil.copy2(source_file, destination_file)
            rows.append(
                {
                    "seed": int(record.seed),
                    "relative_path": destination_file.relative_to(output).as_posix(),
                    "size_bytes": destination_file.stat().st_size,
                    "sha256": _sha256(destination_file),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "relative_path"])


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Exp32 persistent sparse-feedback reward belief",
        "",
        f"- Profile: `{summary['profile']}`; seeds: {summary['n_seeds']}.",
        f"- Checkout policy: `{summary.get('checkout_policy', 'unknown')}`; "
        f"current-checkout reproducibility claimed: "
        f"`{summary.get('current_checkout_reproducibility_claimed', False)}`.",
        f"- Scale decision: **{summary['scale_decision']}**.",
        f"- Claim classification: **{summary['claim_classification']}**.",
        f"- Primary local minus train-fixed: {summary.get('mean_persistent_minus_fixed', float('nan')):+.4f}.",
        f"- Local minus opposite eligibility: {summary.get('mean_persistent_minus_credit_shuffled', float('nan')):+.4f}.",
        f"- Local minus no-forgetting: {summary.get('mean_persistent_minus_cumulative', float('nan')):+.4f}.",
        f"- Oracle opportunity retained: {summary.get('mean_oracle_gain_retained', float('nan')):.3f}.",
    ]
    if summary.get("claim_family") == "evidence_per_dwell_boundary":
        lines.extend(
            [
                f"- Faster-hazard reference gain: {summary.get('mean_reference_persistent_minus_fixed', float('nan')):+.4f}.",
                f"- Slow-minus-fast adaptation interaction: {summary.get('mean_evidence_per_dwell_boundary_interaction', float('nan')):+.4f}.",
                "",
                "This confirmatory analysis was frozen after the original primary",
                "cell failed its smoke gate.  It tests the independently replicated",
                "evidence-per-dwell boundary without changing the controller.",
            ]
        )
    elif summary.get("claim_family") == "feedback_memory_timescale_phase":
        lines.extend(
            [
                f"- Main controller claim: **{summary.get('main_controller_claim_classification', 'inconclusive')}**.",
                f"- Timescale-structure claim: **{summary.get('timescale_structure_claim_classification', 'inconclusive')}**.",
                f"- Evidence-response slope: {summary.get('mean_evidence_response_slope', float('nan')):+.4f} accuracy / doubling.",
                f"- Iso-lambda slow-minus-fast effect: {summary.get('mean_iso_lambda_slow_minus_fast', float('nan')):+.4f}.",
                f"- Short-minus-long delay effect: {summary.get('mean_short_minus_long_delay', float('nan')):+.4f}.",
                f"- Opposite/executed reward-update L1 ratio: {summary.get('mean_opposite_over_executed_reward_update_l1', float('nan')):.3f}.",
                "",
                "This independent confirmation was frozen after the original",
                "primary smoke cell failed. It tests a two-timescale phase claim",
                "without retuning the local controller.",
            ]
        )
    lines.extend(
        [
            "",
            "The primary cell is a continuous HMM stream with no reset,",
            f"hazard {summary.get('primary_hazard', float('nan')):g}, reward fraction "
            f"{summary.get('primary_feedback_fraction', float('nan')):g}, and delay "
            f"{summary.get('primary_feedback_delay', 0)} trials.  The local method receives",
            "only executed scalar reward. It has two internal action values and",
            "one action-control coordinate; its context scores are policy proxies.",
            "The Bayesian comparator knows the registered hazard and uses supervised",
            "train-state emissions, but not the test state. The opposite-credit",
            "condition is an eligibility-location intervention, not a random shuffle.",
            "The true-state oracle is",
            "not deployable.  This experiment contains no participating E/I",
            "carrier and makes no real-data or biological-identity claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/smoke/exp32_persistent_sparse_feedback.json"
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--archived-reanalysis",
        action="store_true",
        help=(
            "explicitly reanalyse an immutable historical panel without claiming "
            "that the current checkout is its execution checkout"
        ),
    )
    parser.add_argument(
        "--output-dir", default="results/exp32_persistent_sparse_feedback"
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    raw, manifest = load_panel(
        Path(args.results_root),
        config,
        run_label=args.run_label,
        checkout_policy="archived" if args.archived_reanalysis else "live",
    )
    conditions, seeds, summary = summarize_records(raw, config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv_lf(raw, output / "raw_metrics.csv.gz", gzip_compressed=True)
    manifest["run_path_semantics"] = "informational_source_location_not_portable"
    _write_csv_lf(manifest, output / "run_manifest.csv")
    receipts = package_run_receipts(manifest, output)
    _write_csv_lf(receipts, output / "receipt_manifest.csv")
    _write_csv_lf(conditions, output / "conditions.csv")
    _write_csv_lf(seeds, output / "seed_summary.csv")
    _write_csv_lf(pd.DataFrame([summary]), output / "summary.csv")
    _write_text_lf(
        output / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_text_lf(output / "report.md", _report(summary))
    print(output.resolve())


if __name__ == "__main__":
    main()
