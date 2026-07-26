#!/usr/bin/env python3
"""Compositional h/Q/R generalization with a three-state causal filter."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.factorized_uncertainty_filter import (
    AdaptationRates,
    FilterTrace,
    JumpFilterParameters,
    run_ema_filter,
    run_factorized_filter,
    run_fixed_jump_filter,
    run_imm_filter,
    run_oracle_filter,
    run_window_filter,
)
from src.tasks.factorized_uncertainty import (
    FactorialStreamConfig,
    UncertaintyLevels,
    UncertaintyTape,
    all_factorial_cells,
    generate_uncertainty_tape,
    heldout_composition_cells,
    parse_cell,
    single_factor_training_cells,
)
from src.utils.reproducibility import set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp39_factorized_uncertainty"
METHODS = (
    "selected_fixed",
    "seen_mode_imm",
    "oracle_factorial_imm",
    "factorized",
    "clamp_h",
    "clamp_q",
    "clamp_r",
    "oracle_dynamic",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_config(config: Mapping[str, Any], *, formal: bool) -> None:
    required = {
        "protocol_version",
        "profile",
        "claim_upgrade_allowed",
        "seeds",
        "levels",
        "stream",
        "partitions",
        "selection",
        "analysis",
        "used_autograd",
        "used_bptt",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration is missing {sorted(missing)}")
    seeds = tuple(config["seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if any(isinstance(seed, bool) or int(seed) != seed or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    if bool(config["used_autograd"]) or bool(config["used_bptt"]):
        raise ValueError("Exp39 prohibits autograd and BPTT")
    partitions = config["partitions"]
    fit_cells = tuple(partitions["fit_cells"])
    heldout_cells = tuple(partitions["heldout_composition_cells"])
    if fit_cells != single_factor_training_cells():
        raise ValueError("fit cells must be baseline plus single-factor elevations")
    if heldout_cells != heldout_composition_cells():
        raise ValueError("heldout cells must be pairwise and triple compositions")
    if set(fit_cells) | set(heldout_cells) != set(all_factorial_cells()):
        raise ValueError("fit and heldout cells must partition the factorial")
    calibration = set(map(int, partitions["calibration_sequence_ids"]))
    selection = set(map(int, partitions["selection_sequence_ids"]))
    if not calibration or not selection or calibration & selection:
        raise ValueError("calibration and selection sequences must be disjoint")
    n_sequences = int(config["stream"]["n_sequences"])
    if calibration | selection != set(range(n_sequences)):
        raise ValueError("fit sequence partition must be complete")
    if formal:
        if len(seeds) != 30 or not bool(config["claim_upgrade_allowed"]):
            raise ValueError("formal Exp39 requires 30 seeds and claim eligibility")
        if not config.get("protocol_frozen_at"):
            raise ValueError("formal Exp39 requires a frozen timestamp")
        if not config.get("implementation_receipt_path"):
            raise ValueError("formal Exp39 requires an implementation receipt")


def validate_implementation_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    receipt_path = PROJECT_ROOT / str(config["implementation_receipt_path"])
    receipt = _read_json(receipt_path)
    if receipt.get("protocol_version") != config["protocol_version"]:
        raise RuntimeError("implementation receipt protocol mismatch")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("implementation receipt has no file hashes")
    mismatches: list[str] = []
    for relative, expected in files.items():
        path = PROJECT_ROOT / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(str(relative))
    if mismatches:
        raise RuntimeError(f"implementation receipt mismatch: {mismatches}")
    return receipt


def _levels(config: Mapping[str, Any]) -> UncertaintyLevels:
    values = config["levels"]
    return UncertaintyLevels(
        hazard=tuple(map(float, values["hazard"])),
        process_variance=tuple(map(float, values["process_variance"])),
        observation_variance=tuple(map(float, values["observation_variance"])),
    )


def _stream_config(config: Mapping[str, Any]) -> FactorialStreamConfig:
    values = config["stream"]
    return FactorialStreamConfig(
        block_length=int(values["block_length"]),
        blocks_per_sequence=int(values["blocks_per_sequence"]),
        n_sequences=int(values["n_sequences"]),
        jump_variance=float(values["jump_variance"]),
    )


def _subset(
    tape: UncertaintyTape, sequence_ids: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(tape.sequence_ids, np.asarray(sequence_ids))
    return tape.observations[mask], tape.sequence_ids[mask]


def _initial(
    levels: UncertaintyLevels, jump_variance: float
) -> JumpFilterParameters:
    return JumpFilterParameters(
        hazard=float(np.sqrt(np.prod(levels.hazard))),
        process_variance=float(np.sqrt(np.prod(levels.process_variance))),
        observation_variance=float(np.sqrt(np.prod(levels.observation_variance))),
        jump_variance=jump_variance,
    )


def _modes(
    cells: Sequence[str], levels: UncertaintyLevels, jump_variance: float
) -> tuple[JumpFilterParameters, ...]:
    return tuple(
        JumpFilterParameters(*levels.values(cell), jump_variance) for cell in cells
    )


def _calibrated_normal_candidate(
    calibration: tuple[np.ndarray, np.ndarray],
    selection: tuple[np.ndarray, np.ndarray],
    *,
    family: str,
    value: float | int,
    variance_floor: float,
) -> tuple[float, float]:
    runner: Callable[..., FilterTrace]
    kwargs: dict[str, Any]
    if family == "ema":
        runner = run_ema_filter
        kwargs = {"alpha": float(value)}
    elif family == "window":
        runner = run_window_filter
        kwargs = {"window": int(value)}
    else:
        raise ValueError(f"unknown normal family {family}")
    calibration_trace = runner(
        calibration[0],
        sequence_ids=calibration[1],
        predictive_variance=1.0,
        **kwargs,
    )
    residual = calibration[0] - calibration_trace.predictive_mean
    variance = float(max(np.mean(residual**2), variance_floor))
    selection_trace = runner(
        selection[0],
        sequence_ids=selection[1],
        predictive_variance=variance,
        **kwargs,
    )
    return float(np.mean(selection_trace.predictive_nll)), variance


def select_models(
    fit_tape: UncertaintyTape,
    *,
    config: Mapping[str, Any],
    levels: UncertaintyLevels,
    stream: FactorialStreamConfig,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selection_config = config["selection"]
    partitions = config["partitions"]
    calibration = _subset(fit_tape, partitions["calibration_sequence_ids"])
    selection = _subset(fit_tape, partitions["selection_sequence_ids"])
    audit: list[dict[str, Any]] = []
    variance_floor = float(selection_config["predictive_variance_floor"])

    for alpha in selection_config["ema_alpha_grid"]:
        score, variance = _calibrated_normal_candidate(
            calibration,
            selection,
            family="ema",
            value=float(alpha),
            variance_floor=variance_floor,
        )
        audit.append(
            {
                "seed": seed,
                "selection_family": "fixed",
                "candidate_family": "ema",
                "candidate": float(alpha),
                "predictive_variance": variance,
                "selection_nll": score,
            }
        )
    for window in selection_config["window_grid"]:
        score, variance = _calibrated_normal_candidate(
            calibration,
            selection,
            family="window",
            value=int(window),
            variance_floor=variance_floor,
        )
        audit.append(
            {
                "seed": seed,
                "selection_family": "fixed",
                "candidate_family": "window",
                "candidate": int(window),
                "predictive_variance": variance,
                "selection_nll": score,
            }
        )
    jump_grid = selection_config["fixed_jump_grid"]
    for h_value in jump_grid["hazard"]:
        for q_value in jump_grid["process_variance"]:
            for r_value in jump_grid["observation_variance"]:
                trace = run_fixed_jump_filter(
                    selection[0],
                    sequence_ids=selection[1],
                    parameters=JumpFilterParameters(
                        float(h_value),
                        float(q_value),
                        float(r_value),
                        stream.jump_variance,
                    ),
                )
                audit.append(
                    {
                        "seed": seed,
                        "selection_family": "fixed",
                        "candidate_family": "fixed_jump",
                        "candidate": f"{h_value}:{q_value}:{r_value}",
                        "candidate_h": float(h_value),
                        "candidate_q": float(q_value),
                        "candidate_r": float(r_value),
                        "selection_nll": float(np.mean(trace.predictive_nll)),
                    }
                )

    initial = _initial(levels, stream.jump_variance)
    shared_rates = selection_config.get("adaptation_rate_grid")
    hazard_rates = tuple(
        map(
            float,
            shared_rates
            if shared_rates is not None
            else selection_config["hazard_adaptation_rate_grid"],
        )
    )
    process_rates = tuple(
        map(
            float,
            shared_rates
            if shared_rates is not None
            else selection_config["process_adaptation_rate_grid"],
        )
    )
    observation_rates = tuple(
        map(
            float,
            shared_rates
            if shared_rates is not None
            else selection_config["observation_adaptation_rate_grid"],
        )
    )
    for beta_h in hazard_rates:
        for beta_q in process_rates:
            for beta_r in observation_rates:
                trace = run_factorized_filter(
                    selection[0],
                    sequence_ids=selection[1],
                    initial=initial,
                    adaptation=AdaptationRates(beta_h, beta_q, beta_r),
                )
                audit.append(
                    {
                        "seed": seed,
                        "selection_family": "factorized",
                        "candidate_family": "online_em",
                        "candidate": f"{beta_h}:{beta_q}:{beta_r}",
                        "candidate_h": beta_h,
                        "candidate_q": beta_q,
                        "candidate_r": beta_r,
                        "selection_nll": float(np.mean(trace.predictive_nll)),
                    }
                )

    seen_modes = _modes(
        config["partitions"]["fit_cells"], levels, stream.jump_variance
    )
    all_modes = _modes(all_factorial_cells(), levels, stream.jump_variance)
    for family, modes in (("seen_imm", seen_modes), ("oracle_imm", all_modes)):
        for switch in selection_config["imm_switch_grid"]:
            trace = run_imm_filter(
                selection[0],
                sequence_ids=selection[1],
                modes=modes,
                mode_switch_probability=float(switch),
            )
            audit.append(
                {
                    "seed": seed,
                    "selection_family": family,
                    "candidate_family": family,
                    "candidate": float(switch),
                    "selection_nll": float(np.mean(trace.predictive_nll)),
                }
            )
    frame = pd.DataFrame(audit)
    frame["selected"] = False
    selected: dict[str, Any] = {}
    for family in ("fixed", "factorized", "seen_imm", "oracle_imm"):
        candidates = frame.loc[frame["selection_family"] == family]
        index = candidates.sort_values(
            ["selection_nll", "candidate_family", "candidate"], kind="stable"
        ).index[0]
        frame.loc[index, "selected"] = True
        selected[family] = frame.loc[index].dropna().to_dict()
    selected["fit_tape_digest"] = fit_tape.digest
    return selected, frame


def _run_selected_fixed(
    tape: UncertaintyTape, selected: Mapping[str, Any], *, jump_variance: float
) -> FilterTrace:
    family = str(selected["candidate_family"])
    if family == "ema":
        return run_ema_filter(
            tape.observations,
            sequence_ids=tape.sequence_ids,
            alpha=float(selected["candidate"]),
            predictive_variance=float(selected["predictive_variance"]),
        )
    if family == "window":
        return run_window_filter(
            tape.observations,
            sequence_ids=tape.sequence_ids,
            window=int(selected["candidate"]),
            predictive_variance=float(selected["predictive_variance"]),
        )
    if family != "fixed_jump":
        raise ValueError(f"unknown selected fixed family {family}")
    return run_fixed_jump_filter(
        tape.observations,
        sequence_ids=tape.sequence_ids,
        parameters=JumpFilterParameters(
            float(selected["candidate_h"]),
            float(selected["candidate_q"]),
            float(selected["candidate_r"]),
            jump_variance,
        ),
    )


def run_seed(
    config: Mapping[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    set_global_seed(seed)
    levels = _levels(config)
    stream = _stream_config(config)
    fit_tape = generate_uncertainty_tape(
        seed=seed,
        split="fit_single_factor",
        cells=tuple(config["partitions"]["fit_cells"]),
        levels=levels,
        config=stream,
    )
    test_tape = generate_uncertainty_tape(
        seed=seed,
        split="test_full_factorial",
        cells=all_factorial_cells(),
        levels=levels,
        config=stream,
    )
    selected, audit = select_models(
        fit_tape, config=config, levels=levels, stream=stream, seed=seed
    )
    initial = _initial(levels, stream.jump_variance)
    factorized_selection = selected["factorized"]
    adaptation = AdaptationRates(
        float(factorized_selection["candidate_h"]),
        float(factorized_selection["candidate_q"]),
        float(factorized_selection["candidate_r"]),
    )
    seen_modes = _modes(
        tuple(config["partitions"]["fit_cells"]), levels, stream.jump_variance
    )
    full_modes = _modes(all_factorial_cells(), levels, stream.jump_variance)
    traces = {
        "selected_fixed": _run_selected_fixed(
            test_tape, selected["fixed"], jump_variance=stream.jump_variance
        ),
        "seen_mode_imm": run_imm_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            modes=seen_modes,
            mode_switch_probability=float(selected["seen_imm"]["candidate"]),
        ),
        "oracle_factorial_imm": run_imm_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            modes=full_modes,
            mode_switch_probability=float(selected["oracle_imm"]["candidate"]),
        ),
        "factorized": run_factorized_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            initial=initial,
            adaptation=adaptation,
        ),
        "clamp_h": run_factorized_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            initial=initial,
            adaptation=adaptation,
            clamp="h",
        ),
        "clamp_q": run_factorized_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            initial=initial,
            adaptation=adaptation,
            clamp="q",
        ),
        "clamp_r": run_factorized_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            initial=initial,
            adaptation=adaptation,
            clamp="r",
        ),
        "oracle_dynamic": run_oracle_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            hazard=test_tape.hazard,
            process_variance=test_tape.process_variance,
            observation_variance=test_tape.observation_variance,
            jump_variance=stream.jump_variance,
        ),
    }
    rows: list[dict[str, Any]] = []
    recovery_window = int(config["analysis"]["recovery_window"])
    for method, trace in traces.items():
        for block in np.unique(test_tape.block_ids):
            indices = np.flatnonzero(test_tape.block_ids == block)
            cell = str(test_tape.cells[indices[0]])
            bits = parse_cell(cell)
            early = indices[:recovery_window]
            late = indices[-recovery_window:]
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "sequence_id": int(test_tape.sequence_ids[indices[0]]),
                    "block_id": int(block),
                    "cell": cell,
                    "h_high": bits[0],
                    "q_high": bits[1],
                    "r_high": bits[2],
                    "heldout_composition": cell in heldout_composition_cells(),
                    "n_steps": int(len(indices)),
                    "mean_nll": float(np.mean(trace.predictive_nll[indices])),
                    "latent_mse": float(
                        np.mean(
                            (
                                trace.filtered_mean[indices]
                                - test_tape.latent[indices]
                            )
                            ** 2
                        )
                    ),
                    "early_nll": float(np.mean(trace.predictive_nll[early])),
                    "late_nll": float(np.mean(trace.predictive_nll[late])),
                    "mean_h_estimate": float(np.mean(trace.hazard[indices])),
                    "mean_q_estimate": float(
                        np.mean(trace.process_variance[indices])
                    ),
                    "mean_r_estimate": float(
                        np.mean(trace.observation_variance[indices])
                    ),
                    "true_h": float(test_tape.hazard[indices[0]]),
                    "true_q": float(test_tape.process_variance[indices[0]]),
                    "true_r": float(test_tape.observation_variance[indices[0]]),
                    "jump_count": int(np.sum(test_tape.jump_flags[indices])),
                    "test_tape_digest": test_tape.digest,
                }
            )
    metadata = {
        "seed": seed,
        "fit_tape_digest": fit_tape.digest,
        "test_tape_digest": test_tape.digest,
        "selected": selected,
        "controller_state_dimension": 3,
        "seen_imm_modes": len(seen_modes),
        "oracle_factorial_imm_modes": len(full_modes),
    }
    return pd.DataFrame(rows), audit, metadata


def summarize(
    block_metrics: pd.DataFrame, *, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if block_metrics.empty or set(block_metrics["method"]) != set(METHODS):
        raise RuntimeError("formal block metrics are incomplete")
    primary = block_metrics.loc[block_metrics["heldout_composition"]].copy()
    seed_metrics = (
        primary.groupby(["seed", "method"], as_index=False)
        .agg(
            heldout_nll=("mean_nll", "mean"),
            heldout_latent_mse=("latent_mse", "mean"),
            heldout_early_nll=("early_nll", "mean"),
            heldout_late_nll=("late_nll", "mean"),
        )
        .sort_values(["seed", "method"])
    )
    pivot = seed_metrics.pivot(
        index="seed", columns="method", values="heldout_nll"
    )
    comparisons: list[dict[str, Any]] = []
    effect_vectors: dict[str, np.ndarray] = {}
    for baseline in (
        "selected_fixed",
        "seen_mode_imm",
        "oracle_factorial_imm",
        "oracle_dynamic",
    ):
        gain = pivot[baseline] - pivot["factorized"]
        effect_vectors[f"factorized_over_{baseline}"] = gain.to_numpy(float)
        comparisons.append(
            {
                "comparison": f"factorized_over_{baseline}",
                "mean_nll_gain": float(gain.mean()),
                "median_nll_gain": float(gain.median()),
                "positive_seeds": int(np.sum(gain > 0.0)),
                "n_seeds": int(len(gain)),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    clamp_rows: list[dict[str, Any]] = []
    factor_columns = {"h": "h_high", "q": "q_high", "r": "r_high"}
    for factor, column in factor_columns.items():
        paired = block_metrics.loc[
            block_metrics["method"].isin(("factorized", f"clamp_{factor}"))
        ].pivot(
            index=["seed", "block_id", column],
            columns="method",
            values="mean_nll",
        )
        paired["penalty"] = paired[f"clamp_{factor}"] - paired["factorized"]
        by_level = paired.groupby(["seed", column])["penalty"].mean().unstack()
        selectivity = by_level[1] - by_level[0]
        for seed in selectivity.index:
            clamp_rows.append(
                {
                    "seed": int(seed),
                    "factor": factor,
                    "high_penalty": float(by_level.loc[seed, 1]),
                    "low_penalty": float(by_level.loc[seed, 0]),
                    "selectivity": float(selectivity.loc[seed]),
                }
            )
    clamp_frame = pd.DataFrame(clamp_rows)
    tracking_rows: list[dict[str, Any]] = []
    factorized_blocks = block_metrics.loc[
        block_metrics["method"] == "factorized"
    ]
    for seed, seed_frame in factorized_blocks.groupby("seed"):
        for factor in ("h", "q", "r"):
            estimate = np.log(seed_frame[f"mean_{factor}_estimate"].to_numpy(float))
            truth = np.log(seed_frame[f"true_{factor}"].to_numpy(float))
            tracking_rows.append(
                {
                    "seed": int(seed),
                    "factor": factor,
                    "log_parameter_correlation": float(
                        np.corrcoef(estimate, truth)[0, 1]
                    ),
                    "mean_absolute_log_error": float(
                        np.mean(np.abs(estimate - truth))
                    ),
                }
            )
    tracking_frame = pd.DataFrame(tracking_rows)
    thresholds = config["analysis"]["acceptance"]
    for factor in factor_columns:
        effect_vectors[f"clamp_{factor}_selectivity"] = clamp_frame.loc[
            clamp_frame["factor"] == factor, "selectivity"
        ].to_numpy(float)
    confirmatory_names = (
        "factorized_over_selected_fixed",
        "factorized_over_seen_mode_imm",
        "clamp_h_selectivity",
        "clamp_q_selectivity",
        "clamp_r_selectivity",
    )
    raw_p = {
        name: float(
            binomtest(
                int(np.sum(effect_vectors[name] > 0.0)),
                int(np.sum(effect_vectors[name] != 0.0)),
                0.5,
                alternative="greater",
            ).pvalue
        )
        for name in confirmatory_names
    }
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted_p: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (family_size - rank) * raw_p[name])
        running = max(running, value)
        adjusted_p[name] = running
    alpha = float(config["analysis"]["holm_alpha"])
    bootstrap_samples = int(config["analysis"]["bootstrap_samples"])
    statistics_seed = int(config["analysis"]["statistics_seed"])
    bootstrap_ci: dict[str, tuple[float, float]] = {}
    for offset, name in enumerate(effect_vectors):
        values = effect_vectors[name]
        rng = np.random.default_rng(statistics_seed + offset)
        sampled = rng.choice(
            values, size=(bootstrap_samples, len(values)), replace=True
        ).mean(axis=1)
        bootstrap_ci[name] = tuple(
            map(float, np.quantile(sampled, (0.025, 0.975)))
        )

    def comparison_gate(
        name: str, min_gain: float, min_positive: int
    ) -> dict[str, Any]:
        row = comparison_frame.set_index("comparison").loc[name]
        return {
            "mean_nll_gain": float(row["mean_nll_gain"]),
            "positive_seeds": int(row["positive_seeds"]),
            "minimum_gain": float(min_gain),
            "minimum_positive_seeds": int(min_positive),
            "one_sided_sign_p": raw_p[name],
            "holm_adjusted_p": adjusted_p[name],
            "holm_alpha": alpha,
            "bootstrap_mean_ci95": list(bootstrap_ci[name]),
            "passed": bool(
                row["mean_nll_gain"] >= min_gain
                and row["positive_seeds"] >= min_positive
                and adjusted_p[name] <= alpha
            ),
        }

    utility = {
        "best_fixed": comparison_gate(
            "factorized_over_selected_fixed",
            float(thresholds["min_fixed_nll_gain"]),
            int(thresholds["min_fixed_positive_seeds"]),
        ),
        "seen_mode_imm": comparison_gate(
            "factorized_over_seen_mode_imm",
            float(thresholds["min_seen_imm_nll_gain"]),
            int(thresholds["min_seen_imm_positive_seeds"]),
        ),
    }
    clamp_summary: dict[str, Any] = {}
    for factor in factor_columns:
        values = clamp_frame.loc[clamp_frame["factor"] == factor]
        test_name = f"clamp_{factor}_selectivity"
        clamp_summary[factor] = {
            "mean_high_penalty": float(values["high_penalty"].mean()),
            "mean_low_penalty": float(values["low_penalty"].mean()),
            "mean_selectivity": float(values["selectivity"].mean()),
            "positive_selectivity_seeds": int(
                np.sum(values["selectivity"] > 0.0)
            ),
            "one_sided_sign_p": raw_p[test_name],
            "holm_adjusted_p": adjusted_p[test_name],
            "holm_alpha": alpha,
            "bootstrap_mean_selectivity_ci95": list(bootstrap_ci[test_name]),
            "passed": bool(
                values["high_penalty"].mean()
                >= float(thresholds["min_target_clamp_penalty"])
                and values["selectivity"].mean()
                >= float(thresholds["min_clamp_selectivity"])
                and np.sum(values["selectivity"] > 0.0)
                >= int(thresholds["min_clamp_positive_seeds"])
                and adjusted_p[test_name] <= alpha
            ),
        }
    all_passed = all(value["passed"] for value in utility.values()) and all(
        value["passed"] for value in clamp_summary.values()
    )
    summary = {
        "protocol_version": config["protocol_version"],
        "claim_eligible": bool(config["claim_upgrade_allowed"]),
        "n_complete_seeds": int(block_metrics["seed"].nunique()),
        "primary_panel": "heldout_pairwise_and_triple_h_q_r_compositions",
        "utility_gates": utility,
        "selective_clamp_gates": clamp_summary,
        "joint_gate_passed": bool(all_passed),
        "verdict": "support" if all_passed else "oppose",
        "oracle_baselines_are_upper_bounds": True,
        "statistics_unit": "seed",
        "multiplicity": {
            "method": "Holm",
            "alpha": alpha,
            "family": list(confirmatory_names),
            "bootstrap_samples": bootstrap_samples,
            "statistics_seed": statistics_seed,
        },
        "parameter_tracking": {
            factor: {
                "mean_log_parameter_correlation": float(
                    tracking_frame.loc[
                        tracking_frame["factor"] == factor,
                        "log_parameter_correlation",
                    ].mean()
                ),
                "mean_absolute_log_error": float(
                    tracking_frame.loc[
                        tracking_frame["factor"] == factor,
                        "mean_absolute_log_error",
                    ].mean()
                ),
            }
            for factor in ("h", "q", "r")
        },
        "real_data_unlocked": bool(all_passed),
    }
    combined = pd.concat([comparison_frame, clamp_frame], ignore_index=True)
    return seed_metrics, combined, tracking_frame, summary


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Exp39 Factorized-Uncertainty Result",
        "",
        f"Verdict: **{str(summary['verdict']).upper()}**.",
        (
            "Joint preregistered gate: "
            f"**{'PASS' if summary['joint_gate_passed'] else 'FAIL'}**."
        ),
        "",
        (
            "The primary panel contains only pairwise and triple h/Q/R "
            "combinations absent from fitting."
        ),
        (
            "The eight-mode factorial IMM and time-varying oracle receive "
            "privileged generator support and are upper bounds, not baselines "
            "the method is required to beat."
        ),
        "",
        "## Utility",
        "",
    ]
    for name, value in summary["utility_gates"].items():
        lines.append(
            f"- {name}: mean NLL gain {value['mean_nll_gain']:+.6f}; "
            f"positive in {value['positive_seeds']}/"
            f"{summary['n_complete_seeds']} seeds; "
            f"Holm p={value['holm_adjusted_p']:.6g}; "
            f"{'PASS' if value['passed'] else 'FAIL'}."
        )
    lines.extend(["", "## Selective clamps", ""])
    for factor, value in summary["selective_clamp_gates"].items():
        lines.append(
            f"- {factor}: high-factor penalty "
            f"{value['mean_high_penalty']:+.6f}; low-factor penalty "
            f"{value['mean_low_penalty']:+.6f}; selectivity "
            f"{value['mean_selectivity']:+.6f}; "
            f"Holm p={value['holm_adjusted_p']:.6g}; "
            f"{'PASS' if value['passed'] else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            (
                "Passing would support only compositional synthetic "
                "identifiability. Failure keeps IBL behavior and neural "
                "analysis locked. Low state dimension alone is never counted "
                "as support."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    formal = bool(config["claim_upgrade_allowed"])
    validate_config(config, formal=formal)
    if formal:
        validate_implementation_receipt(config)
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.json")
    _write_json(
        output / "planned_conditions.json",
        {
            "methods": list(METHODS),
            "fit_cells": list(config["partitions"]["fit_cells"]),
            "heldout_composition_cells": list(
                config["partitions"]["heldout_composition_cells"]
            ),
            "seeds": list(map(int, config["seeds"])),
        },
    )
    logger = logging.getLogger(f"{EXPERIMENT}.{output.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    logger.addHandler(handler)
    blocks: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    try:
        for seed in map(int, config["seeds"]):
            seed_dir = output / f"seed_{seed}"
            seed_dir.mkdir()
            try:
                block, audit, metadata = run_seed(config, seed)
                block.to_csv(seed_dir / "block_metrics.csv", index=False)
                audit.to_csv(seed_dir / "selection_audit.csv", index=False)
                _write_json(seed_dir / "metadata.json", metadata)
                _write_json(
                    seed_dir / "status.json",
                    {"seed": seed, "status": "complete"},
                )
                blocks.append(block)
                audits.append(audit)
                logger.info("seed %s complete", seed)
            except Exception as error:  # retain every failed formal seed
                failure = {
                    "seed": seed,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                _write_json(seed_dir / "status.json", failure)
                logger.exception("seed %s failed", seed)
        _write_json(output / "failures.json", failures)
        if failures or len(blocks) != len(config["seeds"]):
            raise RuntimeError(
                "one or more Exp39 seeds failed; summary is claim-ineligible"
            )
        block_metrics = pd.concat(blocks, ignore_index=True)
        selection_audit = pd.concat(audits, ignore_index=True)
        block_metrics.to_csv(output / "block_metrics.csv", index=False)
        selection_audit.to_csv(output / "selection_audit.csv", index=False)
        seed_metrics, comparisons, parameter_tracking, summary = summarize(
            block_metrics, config=config
        )
        seed_metrics.to_csv(output / "seed_metrics.csv", index=False)
        comparisons.to_csv(
            output / "comparisons_and_clamps.csv", index=False
        )
        parameter_tracking.to_csv(
            output / "parameter_tracking.csv", index=False
        )
        _write_json(output / "summary.json", summary)
        (output / "report.md").write_text(_report(summary), encoding="utf-8")
        _write_json(output / "status.json", {"status": "complete", **summary})
        return summary
    except Exception as error:
        _write_json(
            output / "status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = execute(args.config.resolve(), args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
