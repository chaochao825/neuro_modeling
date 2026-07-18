"""Fail-closed seed-level summary for the Exp31 hidden-demand panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp31_hidden_reliability_reward_selector import (
    EVIDENCE_SCHEMA_VERSION,
    EXPERIMENT,
    MODES,
    PROTOCOL_VERSION,
    _planned_conditions,
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


def _eligible_run_dirs(
    root: Path,
    *,
    seeds: list[int],
    profile: str,
    run_label: str,
) -> list[Path]:
    selected: list[Path] = []
    for seed in seeds:
        seed_root = root / "runs" / EXPERIMENT / f"seed_{seed:04d}"
        candidates: list[Path] = []
        if seed_root.exists():
            for path in sorted(item for item in seed_root.iterdir() if item.is_dir()):
                if not (path / "status.json").exists() or not (path / "manifest.json").exists():
                    continue
                status = _read_json(path / "status.json")
                manifest = _read_json(path / "manifest.json")
                if (
                    manifest.get("profile") == profile
                    and manifest.get("run_label") == run_label
                    and status.get("status")
                    in {"complete", "complete_with_failures"}
                ):
                    candidates.append(path)
        if len(candidates) != 1:
            raise RuntimeError(
                f"seed {seed} has {len(candidates)} eligible Exp31 runs for "
                f"label {run_label!r}; expected exactly one"
            )
        selected.append(candidates[0])
    return selected


def load_panel(
    root: Path,
    config: dict[str, Any],
    *,
    run_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = seed_list(config["seeds"])
    run_dirs = _eligible_run_dirs(
        root,
        seeds=seeds,
        profile=str(config["profile"]),
        run_label=run_label,
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
            raise RuntimeError(f"{path} does not match the selected Exp31 config")
        provenance = observed_config.get("evidence_provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError(f"{path} has no evidence provenance")
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            raise RuntimeError("Exp31 panel mixes source provenance")
        rows = pd.read_json(path / "metrics.jsonl", lines=True)
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"{path} has {len(rows)} rows; expected {expected_rows}"
            )
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
    validate_panel_contract(raw, config)
    manifest = pd.DataFrame(manifest_rows).sort_values("seed").reset_index(drop=True)
    return raw, manifest


def validate_panel_contract(raw: pd.DataFrame, config: dict[str, Any]) -> None:
    required = {
        "seed",
        "block_id",
        "actuator_mode",
        "status",
        "protocol_version",
        "evidence_schema_version",
        "run_git_commit",
        "run_git_tree",
        "run_git_dirty",
        "test_block_fingerprint",
        "train_split_fingerprint",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing Exp31 contract columns: {sorted(missing)}")
    seeds = seed_list(config["seeds"])
    planned = _planned_conditions(config)
    expected_block_modes = {
        (int(row["block_id"]), str(row["actuator_mode"])) for row in planned
    }
    observed = list(
        zip(
            raw["seed"].astype(int),
            raw["block_id"].astype(int),
            raw["actuator_mode"].astype(str),
            strict=True,
        )
    )
    if len(observed) != len(set(observed)):
        raise RuntimeError("Exp31 panel contains duplicate seed/block/mode rows")
    expected = {
        (seed, block, mode)
        for seed in seeds
        for block, mode in expected_block_modes
    }
    if set(observed) != expected:
        raise RuntimeError("Exp31 panel is missing or adds seed/block/mode rows")
    if set(raw["protocol_version"].astype(str)) != {PROTOCOL_VERSION}:
        raise RuntimeError("Exp31 panel mixes protocol versions")
    if set(raw["evidence_schema_version"].dropna().astype(str)) != {
        EVIDENCE_SCHEMA_VERSION
    }:
        raise RuntimeError("Exp31 panel mixes evidence schemas")
    if not raw["run_git_dirty"].eq(False).all():
        raise RuntimeError("publishable Exp31 panels require clean Git runs")
    for column in ("run_git_commit", "run_git_tree"):
        values = raw[column].astype(str)
        if values.nunique() != 1 or len(values.iloc[0]) != 40:
            raise RuntimeError(f"Exp31 panel lacks one bound {column}")
        try:
            int(values.iloc[0], 16)
        except ValueError as error:
            raise RuntimeError(f"Exp31 {column} is not hexadecimal") from error
    if raw.groupby(["seed", "block_id"])["test_block_fingerprint"].nunique().max() != 1:
        raise RuntimeError("Exp31 test tapes are not paired across controls")
    if raw.groupby("seed")["train_split_fingerprint"].nunique().max() != 1:
        raise RuntimeError("Exp31 training dictionary is not paired within seed")


def _mode_macro(frame: pd.DataFrame) -> dict[str, float]:
    return {
        mode: float(frame.loc[frame["actuator_mode"] == mode, "full_block_accuracy"].mean())
        for mode in MODES
    }


def summarize_records(
    raw: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        "seed",
        "status",
        "block_id",
        "direct_reliability",
        "association_load",
        "distractor_writes",
        "interference_pressure",
        "actuator_mode",
        "full_block_accuracy",
        "deployment_accuracy",
        "selected_posthoc_best",
        "reward_only_interface_audit",
        "selector_received_true_context",
        "selector_received_unexecuted_reward",
        "associative_query_shuffled_write_budget_exact",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing Exp31 summary columns: {sorted(missing)}")
    complete = raw["status"].eq("complete").all()
    if not complete:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "experiment": EXPERIMENT,
                "profile": str(config["profile"]),
                "n_seeds": int(raw["seed"].nunique()),
                "scale_decision": "scale-not-authorized",
                "claim_classification": "inconclusive",
                "panel_complete": False,
                "failure_reason": "one or more planned conditions failed",
            },
        )
    conditions = (
        raw.groupby(
            [
                "direct_reliability",
                "association_load",
                "distractor_writes",
                "interference_pressure",
                "actuator_mode",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            n_blocks=("block_id", "count"),
            mean_full_accuracy=("full_block_accuracy", "mean"),
            sd_full_accuracy=("full_block_accuracy", "std"),
            mean_deployment_accuracy=("deployment_accuracy", "mean"),
        )
        .sort_values(
            [
                "direct_reliability",
                "association_load",
                "distractor_writes",
                "actuator_mode",
            ]
        )
        .reset_index(drop=True)
    )
    analysis = dict(config["analysis"])
    retention_fraction = float(analysis["oracle_gain_retention_fraction"])
    low_reliability = float(min(raw["direct_reliability"]))
    high_reliability = float(max(raw["direct_reliability"]))
    low_load = int(min(raw["association_load"]))
    low_distractor = int(min(raw["distractor_writes"]))
    rows: list[dict[str, Any]] = []
    for seed, frame in raw.groupby("seed", sort=True):
        macro = _mode_macro(frame)
        local_rows = frame[frame["actuator_mode"] == "reward_only_local"]
        low_stress = frame[
            (frame["association_load"] == low_load)
            & (frame["distractor_writes"] == low_distractor)
        ]

        def advantage(reliability: float) -> float:
            selected = low_stress[
                low_stress["direct_reliability"] == reliability
            ]
            return float(
                selected.loc[
                    selected["actuator_mode"] == "fixed_associative",
                    "full_block_accuracy",
                ].mean()
                - selected.loc[
                    selected["actuator_mode"] == "fixed_routing",
                    "full_block_accuracy",
                ].mean()
            )

        low_advantage = advantage(low_reliability)
        high_advantage = advantage(high_reliability)
        specificity_rows = low_stress[
            low_stress["direct_reliability"] == low_reliability
        ]
        specificity = float(
            specificity_rows.loc[
                specificity_rows["actuator_mode"] == "fixed_associative",
                "full_block_accuracy",
            ].mean()
            - specificity_rows.loc[
                specificity_rows["actuator_mode"]
                == "associative_query_shuffled",
                "full_block_accuracy",
            ].mean()
        )
        cell_advantage = (
            frame[
                frame["actuator_mode"].isin(
                    ["fixed_routing", "fixed_associative"]
                )
            ]
            .groupby(
                ["direct_reliability", "association_load", "distractor_writes", "actuator_mode"]
            )["full_block_accuracy"]
            .mean()
            .unstack("actuator_mode")
        )
        differences = (
            cell_advantage["fixed_associative"]
            - cell_advantage["fixed_routing"]
        )
        memory_pressure = (
            frame[frame["actuator_mode"] == "fixed_associative"]
            .groupby("interference_pressure", as_index=False)["full_block_accuracy"]
            .mean()
            .sort_values("interference_pressure")
        )
        pressure_rho = float(
            spearmanr(
                memory_pressure["interference_pressure"],
                memory_pressure["full_block_accuracy"],
            ).statistic
        )
        primary = macro["reward_only_local"] - macro["train_fixed_best"]
        opportunity = (
            macro["oracle_hidden_train_map"] - macro["train_fixed_best"]
        )
        rows.append(
            {
                "seed": int(seed),
                **{f"{mode}_macro_accuracy": value for mode, value in macro.items()},
                "reward_only_minus_fixed": primary,
                "reward_only_minus_random": macro["reward_only_local"]
                - macro["matched_probe_random"],
                "oracle_opportunity": opportunity,
                "oracle_gain_retained": (
                    primary / opportunity if opportunity > 0.0 else 0.0
                ),
                "oracle_retention_margin": primary
                - retention_fraction * opportunity,
                "reliability_crossover": low_advantage - high_advantage,
                "associative_query_specificity": specificity,
                "reward_selector_posthoc_choice_accuracy": float(
                    local_rows["selected_posthoc_best"].mean()
                ),
                "mean_local_regret": float(
                    local_rows["full_block_regret_to_posthoc_fixed_oracle"].mean()
                ),
                "memory_pressure_spearman": pressure_rho,
                "n_memory_better_cells": int(np.sum(differences > 0.0)),
                "n_routing_better_cells": int(np.sum(differences < 0.0)),
                "both_actuator_winners_present": bool(
                    np.any(differences > 0.0) and np.any(differences < 0.0)
                ),
            }
        )
    seeds = pd.DataFrame(rows)
    audit_ok = bool(
        raw.loc[
            raw["actuator_mode"] == "reward_only_local",
            "reward_only_interface_audit",
        ].eq(True).all()
        and raw.loc[
            raw["actuator_mode"] == "reward_only_local",
            "selector_received_true_context",
        ].eq(False).all()
        and raw["selector_received_unexecuted_reward"].eq(False).all()
        and raw["associative_query_shuffled_write_budget_exact"].eq(True).all()
    )
    fixed_mean = float(seeds["train_fixed_best_macro_accuracy"].mean())
    shuffled_mean = float(
        seeds["associative_query_shuffled_macro_accuracy"].mean()
    )
    positive_fraction = float(np.mean(seeds["reward_only_minus_fixed"] > 0.0))
    crossover_fraction = float(np.mean(seeds["reliability_crossover"] > 0.0))
    specificity_fraction = float(
        np.mean(seeds["associative_query_specificity"] > 0.0)
    )
    both_winners_fraction = float(
        np.mean(seeds["both_actuator_winners_present"])
    )
    smoke_gate = bool(
        audit_ok
        and positive_fraction >= 0.8
        and float(seeds["reward_only_minus_fixed"].mean())
        >= float(analysis["smoke_primary_mean_gate"])
        and crossover_fraction >= 0.8
        and float(seeds["reliability_crossover"].mean())
        >= float(analysis["crossover_mcid"])
        and specificity_fraction >= 0.8
        and float(seeds["associative_query_specificity"].mean())
        >= float(analysis["specificity_mcid"])
        and float(seeds["oracle_opportunity"].mean()) >= 0.05
        and float(seeds["oracle_gain_retained"].mean()) >= retention_fraction
        and float(analysis["fixed_best_floor"]) <= fixed_mean
        <= float(analysis["fixed_best_ceiling"])
        and 0.45 <= shuffled_mean <= 0.55
        and both_winners_fraction >= 0.8
        and float(seeds["memory_pressure_spearman"].mean()) <= -0.7
    )
    summary: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "profile": str(config["profile"]),
        "n_seeds": int(seeds["seed"].nunique()),
        "panel_complete": True,
        "feedback_access_audit_passed": audit_ok,
        "scale_decision": (
            "scale-authorized" if smoke_gate else "scale-not-authorized"
        ),
        "claim_classification": "inconclusive",
        "mean_reward_only_minus_fixed": float(
            seeds["reward_only_minus_fixed"].mean()
        ),
        "mean_reward_only_minus_random": float(
            seeds["reward_only_minus_random"].mean()
        ),
        "mean_oracle_opportunity": float(seeds["oracle_opportunity"].mean()),
        "mean_oracle_gain_retained": float(
            seeds["oracle_gain_retained"].mean()
        ),
        "mean_reliability_crossover": float(
            seeds["reliability_crossover"].mean()
        ),
        "mean_associative_query_specificity": float(
            seeds["associative_query_specificity"].mean()
        ),
        "mean_selector_choice_accuracy": float(
            seeds["reward_selector_posthoc_choice_accuracy"].mean()
        ),
        "mean_memory_pressure_spearman": float(
            seeds["memory_pressure_spearman"].mean()
        ),
        "positive_seed_fraction": positive_fraction,
        "both_winners_seed_fraction": both_winners_fraction,
        "fixed_best_mean_accuracy": fixed_mean,
        "query_shuffled_mean_accuracy": shuffled_mean,
        "high_rank_carrier_claimed": False,
        "strong_baseline_claimed": False,
    }
    if config["profile"] == "formal":
        n_bootstrap = int(analysis["bootstrap_samples"])
        n_permutation = int(analysis["permutation_samples"])
        statistics_seed = int(analysis["statistics_seed"])
        contrasts = {
            "reward_only_over_train_fixed_best": (
                seeds["reward_only_minus_fixed"].to_numpy(float),
                float(analysis["primary_mcid"]),
            ),
            "reliability_crossover": (
                seeds["reliability_crossover"].to_numpy(float),
                float(analysis["crossover_mcid"]),
            ),
            "associative_query_specificity": (
                seeds["associative_query_specificity"].to_numpy(float),
                float(analysis["specificity_mcid"]),
            ),
            "oracle_gain_retention": (
                seeds["oracle_retention_margin"].to_numpy(float),
                0.0,
            ),
        }
        raw_p: dict[str, float] = {}
        lower_pass: dict[str, bool] = {}
        for index, (name, (values, threshold)) in enumerate(contrasts.items()):
            lower, upper = paired_bootstrap_interval(
                values,
                n_resamples=n_bootstrap,
                seed=statistics_seed + index,
            )
            pvalue = sign_flip_pvalue(
                values - threshold,
                n_resamples=n_permutation,
                seed=statistics_seed + 100 + index,
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
        )
        primary_upper = float(
            summary["reward_only_over_train_fixed_best_ci_upper"]
        )
        if support:
            summary["claim_classification"] = "support"
        elif primary_upper <= 0.0:
            summary["claim_classification"] = "oppose"
        else:
            summary["claim_classification"] = "inconclusive"
        summary["scale_decision"] = "formal-complete"
    return conditions, seeds, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_run_receipts(manifest: pd.DataFrame, output: Path) -> pd.DataFrame:
    required = (
        "config.json",
        "environment.json",
        "status.json",
        "manifest.json",
        "planned_conditions.json",
        "metrics.jsonl",
        "run.log",
    )
    receipt_root = output / "run_receipts"
    receipt_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for record in manifest.itertuples(index=False):
        source = Path(str(record.run_path))
        destination = receipt_root / f"seed_{int(record.seed):04d}"
        destination.mkdir(parents=True, exist_ok=False)
        for name in required:
            source_file = source / name
            if not source_file.exists():
                raise FileNotFoundError(f"missing Exp31 receipt: {source_file}")
            destination_file = destination / name
            shutil.copy2(source_file, destination_file)
            rows.append(
                {
                    "seed": int(record.seed),
                    "relative_path": str(destination_file.relative_to(output)),
                    "size_bytes": destination_file.stat().st_size,
                    "sha256": _sha256(destination_file),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "relative_path"])


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Exp31 hidden-demand reward-only selector",
            "",
            f"- Profile: `{summary['profile']}`; seeds: {summary['n_seeds']}.",
            f"- Scale decision: **{summary['scale_decision']}**.",
            f"- Claim classification: **{summary['claim_classification']}**.",
            f"- Reward-only minus train-fixed: {summary.get('mean_reward_only_minus_fixed', float('nan')):+.4f}.",
            f"- Reward-only minus matched random: {summary.get('mean_reward_only_minus_random', float('nan')):+.4f}.",
            f"- Oracle gain retained: {summary.get('mean_oracle_gain_retained', float('nan')):.3f}.",
            f"- Reliability crossover: {summary.get('mean_reliability_crossover', float('nan')):+.4f}.",
            f"- Associative minus query-shuffled: {summary.get('mean_associative_query_specificity', float('nan')):+.4f}.",
            f"- Memory accuracy/pressure Spearman: {summary.get('mean_memory_pressure_spearman', float('nan')):+.3f}.",
            "",
            "The primary score includes the forced-exploration prefix. The local",
            "selector never receives true reliability, task descriptors, unexecuted",
            "rewards, or a candidate-utility vector. The oracle is an explicitly",
            "labelled train-map upper bound. This panel isolates controller",
            "identifiability; it does not establish high-rank E/I carrier dynamics",
            "or real-data validity.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/smoke/exp31_hidden_reliability_reward_selector.json",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--output-dir", default="results/exp31_hidden_reliability_reward_selector"
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    raw, manifest = load_panel(Path(args.results_root), config, run_label=args.run_label)
    conditions, seeds, summary = summarize_records(raw, config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    raw.to_csv(output / "raw_metrics.csv.gz", index=False, compression="gzip")
    manifest.to_csv(output / "run_manifest.csv", index=False)
    receipts = package_run_receipts(manifest, output)
    receipts.to_csv(output / "receipt_manifest.csv", index=False)
    conditions.to_csv(output / "conditions.csv", index=False)
    seeds.to_csv(output / "seed_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
