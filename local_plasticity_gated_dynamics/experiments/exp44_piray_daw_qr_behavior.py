"""Prospective Experiment-1 development audit on Piray--Daw behavior."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ttest_1samp
import statsmodels
from statsmodels.stats.multitest import multipletests

from src.data.piray_daw import (
    DATA_ARCHIVE_MD5,
    EXPECTED_BLOCKS,
    EXPECTED_PARTICIPANTS,
    EXPECTED_TRIALS,
    PirayDawDataset,
    load_piray_daw,
)
from src.models.piray_daw_qr_controller import (
    QRControllerTrace,
    VarianceBounds,
    average_traces,
    run_autocovariance_qr,
    run_factorized_local_em,
    run_fixed_gain,
    run_hierarchical_particle,
    run_kalman_schedule,
    run_total_uncertainty,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp44_piray_daw_qr_behavior"
PROFILE = "development_piray_daw_qr_behavior"
PROTOCOL_VERSION = "exp44_piray_daw_qr_behavior_v1"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "exp44_piray_daw_qr_behavior_protocol_20260730.md"

FIXED = "fixed_gain"
TOTAL = "total_uncertainty"
FACTORIZED = "factorized_local_em"
AUTOCOV = "autocovariance_qr"
PARTICLE = "hierarchical_particle"
ORACLE = "oracle_qr"
METHODS = (FIXED, TOTAL, FACTORIZED, AUTOCOV, PARTICLE, ORACLE)
DEPLOYABLE_METHODS = METHODS[:-1]
SOURCE_FILES = (
    "provenance/exp44_development_implementation_lock_20260730.json",
    "provenance/piray_daw_zenodo_v1.json",
    "src/data/piray_daw.py",
    "src/models/piray_daw_qr_controller.py",
    "experiments/exp44_piray_daw_qr_behavior.py",
    "scripts/fetch_piray_daw.py",
    "tests/test_piray_daw_loader.py",
    "tests/test_piray_daw_qr_controller.py",
    "tests/test_exp44_piray_daw_qr_behavior.py",
)


@dataclass(frozen=True)
class Candidate:
    method: str
    candidate_id: str
    parameters: dict[str, Any]
    trace: QRControllerTrace


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be one JSON object")
    return value


def _finite_grid(config: Mapping[str, Any], name: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in config["selection"][name])
    if not values or not all(np.isfinite(value) for value in values):
        raise ValueError(f"selection grid {name} must be finite and non-empty")
    return values


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("profile") != PROFILE:
        raise ValueError(f"profile must be {PROFILE}")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
    if config.get("stage") != "development":
        raise ValueError("this entry point authorizes development only")
    if config.get("claim_upgrade_allowed") is not False:
        raise ValueError("development results cannot directly upgrade claims")
    if config.get("used_autograd") is not False or config.get("used_bptt") is not False:
        raise ValueError("local Exp44 methods forbid autograd and BPTT")
    data = config["data"]
    if int(data["experiment"]) != 1:
        raise ValueError("development must use Experiment 1")
    if int(data["expected_participants"]) != EXPECTED_PARTICIPANTS[1]:
        raise ValueError("Experiment-1 participant count changed")
    if int(data["expected_blocks"]) != EXPECTED_BLOCKS:
        raise ValueError("block count changed")
    if int(data["expected_trials"]) != EXPECTED_TRIALS:
        raise ValueError("trial count changed")
    if data["data_archive_md5"] != DATA_ARCHIVE_MD5:
        raise ValueError("data archive checksum changed")
    lock = config["confirmation_lock"]
    if int(lock["experiment"]) != 2 or bool(lock["allow_confirmation"]):
        raise ValueError("Experiment 2 must remain locked in development")
    if not bool(lock["same_stimulus_tape"]):
        raise ValueError("same-stimulus confirmation limitation must remain explicit")
    folds = int(config["cross_validation"]["participant_folds"])
    if folds < 2 or folds >= EXPECTED_PARTICIPANTS[1]:
        raise ValueError("participant_folds must define non-trivial held-out folds")
    filter_config = config["filter"]
    VarianceBounds(
        float(filter_config["minimum_variance"]),
        float(filter_config["maximum_variance"]),
    )
    if not np.isfinite(float(filter_config["initial_mean"])):
        raise ValueError("initial_mean must be finite")
    if float(filter_config["initial_state_variance"]) <= 0.0:
        raise ValueError("initial_state_variance must be positive")
    for name in (
        "fixed_gain_grid",
        "initial_q_grid",
        "initial_r_grid",
        "factorized_q_rate_grid",
        "factorized_r_rate_grid",
        "initial_total_variance_grid",
        "total_rate_grid",
        "total_q_fraction_grid",
        "autocovariance_decay_grid",
        "autocovariance_prior_mass_grid",
        "particle_mu_q_grid",
        "particle_mu_r_grid",
        "particle_log_step_grid",
    ):
        _finite_grid(config, name)
    if int(config["selection"]["particle_count"]) < 32:
        raise ValueError("particle_count must be >= 32")
    particle_seeds = tuple(int(value) for value in config["selection"]["particle_seeds"])
    if not particle_seeds or len(set(particle_seeds)) != len(particle_seeds):
        raise ValueError("particle seeds must be unique and non-empty")
    analysis = config["analysis"]
    if analysis["primary_method"] != FACTORIZED:
        raise ValueError("the registered primary method must remain factorized_local_em")
    if analysis["primary_metric"] != "conditional_update_nll":
        raise ValueError("the registered primary metric changed")
    if int(analysis["early_trials"]) <= 0 or int(analysis["late_trials"]) <= 0:
        raise ValueError("analysis windows must be positive")
    if int(analysis["bootstrap_resamples"]) < 100:
        raise ValueError("bootstrap_resamples must be >= 100")
    for name in ("minimum_nll_gain", "minimum_mse_gain"):
        if not np.isfinite(float(analysis[name])) or float(analysis[name]) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


def _candidate_id(method: str, parameters: Mapping[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{method}:{suffix}"


def _candidate(method: str, parameters: dict[str, Any], trace: QRControllerTrace) -> Candidate:
    return Candidate(method, _candidate_id(method, parameters), parameters, trace)


def _build_candidates(
    dataset: PirayDawDataset,
    config: Mapping[str, Any],
) -> dict[str, list[Candidate]]:
    selection = config["selection"]
    filter_config = config["filter"]
    observations = dataset.bag
    initial_mean = float(filter_config["initial_mean"])
    initial_state_variance = float(filter_config["initial_state_variance"])
    bounds = VarianceBounds(
        float(filter_config["minimum_variance"]),
        float(filter_config["maximum_variance"]),
    )
    common = {
        "initial_mean": initial_mean,
        "initial_state_variance": initial_state_variance,
        "bounds": bounds,
    }
    output: dict[str, list[Candidate]] = {method: [] for method in METHODS}

    for gain in _finite_grid(config, "fixed_gain_grid"):
        parameters = {"gain": gain, "initial_mean": initial_mean}
        output[FIXED].append(
            _candidate(
                FIXED,
                parameters,
                run_fixed_gain(observations, gain=gain, initial_mean=initial_mean),
            )
        )

    initial_q = _finite_grid(config, "initial_q_grid")
    initial_r = _finite_grid(config, "initial_r_grid")
    q_rates = _finite_grid(config, "factorized_q_rate_grid")
    r_rates = _finite_grid(config, "factorized_r_rate_grid")
    for q0, r0, beta_q, beta_r in product(initial_q, initial_r, q_rates, r_rates):
        parameters = {
            "initial_process_variance": q0,
            "initial_observation_variance": r0,
            "process_rate": beta_q,
            "observation_rate": beta_r,
        }
        output[FACTORIZED].append(
            _candidate(
                FACTORIZED,
                parameters,
                run_factorized_local_em(observations, **parameters, **common),
            )
        )

    for total0, rate, fraction in product(
        _finite_grid(config, "initial_total_variance_grid"),
        _finite_grid(config, "total_rate_grid"),
        _finite_grid(config, "total_q_fraction_grid"),
    ):
        parameters = {
            "initial_total_variance": total0,
            "adaptation_rate": rate,
            "q_fraction": fraction,
        }
        output[TOTAL].append(
            _candidate(
                TOTAL,
                parameters,
                run_total_uncertainty(observations, **parameters, **common),
            )
        )

    for q0, r0, decay, prior_mass in product(
        initial_q,
        initial_r,
        _finite_grid(config, "autocovariance_decay_grid"),
        _finite_grid(config, "autocovariance_prior_mass_grid"),
    ):
        parameters = {
            "initial_process_variance": q0,
            "initial_observation_variance": r0,
            "statistic_decay": decay,
            "prior_mass": prior_mass,
        }
        output[AUTOCOV].append(
            _candidate(
                AUTOCOV,
                parameters,
                run_autocovariance_qr(observations, **parameters, **common),
            )
        )

    particle_seeds = tuple(int(value) for value in selection["particle_seeds"])
    particle_count = int(selection["particle_count"])
    particle_common = {
        **common,
        "particle_count": particle_count,
    }
    for mu_q, mu_r, log_step in product(
        _finite_grid(config, "particle_mu_q_grid"),
        _finite_grid(config, "particle_mu_r_grid"),
        _finite_grid(config, "particle_log_step_grid"),
    ):
        parameters = {
            "change_probability_q": mu_q,
            "change_probability_r": mu_r,
            "log_step_scale": log_step,
            "particle_count": particle_count,
            "particle_seeds": list(particle_seeds),
        }
        traces = [
            run_hierarchical_particle(
                observations,
                change_probability_q=mu_q,
                change_probability_r=mu_r,
                log_step_scale=log_step,
                seed=seed,
                **particle_common,
            )
            for seed in particle_seeds
        ]
        output[PARTICLE].append(
            _candidate(PARTICLE, parameters, average_traces(traces))
        )

    q_schedule = np.broadcast_to(
        dataset.true_process_variance[None, :], observations.shape
    )
    r_schedule = np.broadcast_to(
        dataset.true_observation_variance[None, :], observations.shape
    )
    oracle_parameters = {"privileged": True, "source": "released_true_block_qr"}
    output[ORACLE].append(
        _candidate(
            ORACLE,
            oracle_parameters,
            run_kalman_schedule(
                observations,
                process_variance=q_schedule,
                observation_variance=r_schedule,
                **common,
            ),
        )
    )
    if any(not values for values in output.values()):
        raise RuntimeError("every registered method must have at least one candidate")
    return output


def _fold_assignment(n_participants: int, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_participants)
    assignment = np.empty(n_participants, dtype=np.int64)
    assignment[order] = np.arange(n_participants) % n_folds
    return assignment


def _residuals(dataset: PirayDawDataset, trace: QRControllerTrace) -> np.ndarray:
    update = dataset.bucket[:, 1:, :] - dataset.bucket[:, :-1, :]
    prediction_error = dataset.bag[None, :-1, :] - dataset.bucket[:, :-1, :]
    predicted_update = trace.gain[None, :-1, :] * prediction_error
    return update - predicted_update


def _free_run_residuals(dataset: PirayDawDataset, trace: QRControllerTrace) -> np.ndarray:
    return dataset.bucket[:, 1:, :] - trace.posterior_mean[None, :-1, :]


def _gaussian_nll(residual: np.ndarray, sigma: float) -> np.ndarray:
    variance = max(float(sigma) ** 2, 1e-12)
    return 0.5 * (np.log(2.0 * np.pi * variance) + residual**2 / variance)


def _select_candidate(
    candidates: Sequence[Candidate],
    dataset: PirayDawDataset,
    train_mask: np.ndarray,
) -> tuple[Candidate, float, list[dict[str, Any]]]:
    scores: list[tuple[float, str, Candidate, float]] = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        residual = _residuals(dataset, candidate.trace)[train_mask]
        mse = float(np.mean(residual**2))
        sigma = float(max(np.sqrt(mse), 1e-6))
        scores.append((mse, candidate.candidate_id, candidate, sigma))
        rows.append(
            {
                "method": candidate.method,
                "candidate_id": candidate.candidate_id,
                "parameters_json": json.dumps(candidate.parameters, sort_keys=True),
                "train_mse": mse,
                "train_sigma": sigma,
            }
        )
    _, _, selected, sigma = min(scores, key=lambda value: (value[0], value[1]))
    for row in rows:
        row["selected"] = row["candidate_id"] == selected.candidate_id
    return selected, sigma, rows


def cross_validated_behavior(
    dataset: PirayDawDataset,
    candidates: Mapping[str, Sequence[Candidate]],
    config: Mapping[str, Any],
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]],
    pd.DataFrame,
]:
    n_folds = int(config["cross_validation"]["participant_folds"])
    split_seed = int(config["cross_validation"]["split_seed"])
    early = int(config["analysis"]["early_trials"])
    late = int(config["analysis"]["late_trials"])
    folds = _fold_assignment(dataset.n_participants, n_folds, split_seed)
    participant_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for method in METHODS:
        for fold in range(n_folds):
            train_mask = folds != fold
            test_indices = np.flatnonzero(folds == fold)
            selected, sigma, score_rows = _select_candidate(
                candidates[method], dataset, train_mask
            )
            for row in score_rows:
                row["selection_scope"] = f"outer_fold_{fold}_train"
                row["fold"] = fold
            candidate_rows.extend(score_rows)
            residual = _residuals(dataset, selected.trace)[test_indices]
            free_residual = _free_run_residuals(dataset, selected.trace)[test_indices]
            nll = _gaussian_nll(residual, sigma)
            for local_index, participant_id in enumerate(test_indices):
                participant_rows.append(
                    {
                        "participant_id": int(participant_id),
                        "fold": fold,
                        "method": method,
                        "candidate_id": selected.candidate_id,
                        "response_sigma": sigma,
                        "conditional_update_nll": float(nll[local_index].mean()),
                        "conditional_update_mse": float(
                            np.mean(residual[local_index] ** 2)
                        ),
                        "free_run_bucket_mse": float(
                            np.mean(free_residual[local_index] ** 2)
                        ),
                        "early_update_nll": float(nll[local_index, :early].mean()),
                        "late_update_nll": float(nll[local_index, -late:].mean()),
                    }
                )
                for block in range(EXPECTED_BLOCKS):
                    cell_rows.append(
                        {
                            "participant_id": int(participant_id),
                            "fold": fold,
                            "method": method,
                            "candidate_id": selected.candidate_id,
                            "block_id": block,
                            "true_process_variance": float(
                                dataset.true_process_variance[block]
                            ),
                            "true_observation_variance": float(
                                dataset.true_observation_variance[block]
                            ),
                            "conditional_update_nll": float(
                                nll[local_index, :, block].mean()
                            ),
                            "conditional_update_mse": float(
                                np.mean(residual[local_index, :, block] ** 2)
                            ),
                        }
                    )

    full_mask = np.ones(dataset.n_participants, dtype=bool)
    selected_full: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        selected, sigma, score_rows = _select_candidate(
            candidates[method], dataset, full_mask
        )
        for row in score_rows:
            row["selection_scope"] = "full_experiment1"
            row["fold"] = -1
        candidate_rows.extend(score_rows)
        selected_full[method] = {
            "candidate_id": selected.candidate_id,
            "parameters": selected.parameters,
            "response_sigma": sigma,
            "trace": selected.trace,
        }

    folds_frame = pd.DataFrame(
        {"participant_id": np.arange(dataset.n_participants), "fold": folds}
    )
    return (
        pd.DataFrame(participant_rows),
        pd.DataFrame(cell_rows),
        pd.DataFrame(candidate_rows),
        selected_full,
    ), folds_frame


def _bootstrap_interval(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError("bootstrap values must be a finite participant vector")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 500):
        stop = min(start + 500, resamples)
        indices = rng.integers(0, len(data), size=(stop - start, len(data)))
        means[start:stop] = data[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def compare_methods(
    participant_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    resamples = int(config["analysis"]["bootstrap_resamples"])
    seed = int(config["analysis"]["bootstrap_seed"])
    baselines = (FIXED, TOTAL, AUTOCOV, PARTICLE, ORACLE)
    for metric_index, metric in enumerate(
        ("conditional_update_nll", "conditional_update_mse")
    ):
        pivot = participant_metrics.pivot(
            index="participant_id", columns="method", values=metric
        )
        if tuple(sorted(pivot.columns)) != tuple(sorted(METHODS)):
            raise RuntimeError("participant metrics are incomplete")
        for baseline_index, baseline in enumerate(baselines):
            difference = (pivot[baseline] - pivot[FACTORIZED]).to_numpy()
            ci_low, ci_high = _bootstrap_interval(
                difference,
                resamples=resamples,
                seed=seed + metric_index * len(baselines) + baseline_index,
            )
            if np.ptp(difference) <= np.finfo(np.float64).eps:
                p_value = 1.0 if difference[0] == 0.0 else 0.0
            else:
                p_value = float(ttest_1samp(difference, 0.0).pvalue)
            rows.append(
                {
                    "candidate": FACTORIZED,
                    "baseline": baseline,
                    "metric": metric,
                    "contrast": "baseline_minus_candidate",
                    "n_participants": len(difference),
                    "mean_gain": float(difference.mean()),
                    "median_gain": float(np.median(difference)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "positive_participants": int(np.sum(difference > 0.0)),
                    "p_raw": p_value,
                }
            )
    adjusted = multipletests(
        [row["p_raw"] for row in rows], alpha=0.05, method="holm"
    )[1]
    for row, value in zip(rows, adjusted, strict=True):
        row["p_holm"] = float(value)
    return pd.DataFrame(rows)


def trace_diagnostics(
    dataset: PirayDawDataset,
    selected_full: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        trace: QRControllerTrace = selected_full[method]["trace"]
        for block in range(EXPECTED_BLOCKS):
            row: dict[str, Any] = {
                "method": method,
                "candidate_id": selected_full[method]["candidate_id"],
                "block_id": block,
                "true_process_variance": float(dataset.true_process_variance[block]),
                "true_observation_variance": float(
                    dataset.true_observation_variance[block]
                ),
                "mean_gain": float(trace.gain[:-1, block].mean()),
                "early_gain": float(trace.gain[:10, block].mean()),
                "late_gain": float(trace.gain[-20:, block].mean()),
                "bird_tracking_mse": float(
                    np.mean((trace.posterior_mean[:, block] - dataset.bird[:, block]) ** 2)
                ),
                "bag_predictive_nll": (
                    float(trace.predictive_nll[:, block].mean())
                    if trace.predictive_nll is not None
                    else np.nan
                ),
                "mean_estimated_q": (
                    float(trace.process_variance[:, block].mean())
                    if trace.process_variance is not None
                    else np.nan
                ),
                "mean_estimated_r": (
                    float(trace.observation_variance[:, block].mean())
                    if trace.observation_variance is not None
                    else np.nan
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def development_decision(
    comparisons: pd.DataFrame,
    cell_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = config["analysis"]
    minimum_gain = float(analysis["minimum_nll_gain"])
    minimum_mse_gain = float(analysis["minimum_mse_gain"])
    nll_comparisons = comparisons.loc[
        comparisons["metric"] == "conditional_update_nll"
    ]
    mse_comparisons = comparisons.loc[
        comparisons["metric"] == "conditional_update_mse"
    ]
    primary: dict[str, dict[str, Any]] = {}
    for baseline in (FIXED, TOTAL):
        row = nll_comparisons.loc[nll_comparisons["baseline"] == baseline].iloc[0]
        mse_row = mse_comparisons.loc[
            mse_comparisons["baseline"] == baseline
        ].iloc[0]
        primary[baseline] = {
            "nll_mean_gain": float(row["mean_gain"]),
            "nll_ci_low": float(row["ci_low"]),
            "nll_passes": bool(
                row["mean_gain"] >= minimum_gain and row["ci_low"] > 0.0
            ),
            "mse_mean_gain": float(mse_row["mean_gain"]),
            "mse_ci_low": float(mse_row["ci_low"]),
            "mse_passes": bool(
                mse_row["mean_gain"] > minimum_mse_gain
                and mse_row["ci_low"] > 0.0
            ),
        }

    factor_diag = diagnostics.loc[diagnostics["method"] == FACTORIZED]
    q_high = factor_diag["true_process_variance"].max()
    q_low = factor_diag["true_process_variance"].min()
    r_high = factor_diag["true_observation_variance"].max()
    r_low = factor_diag["true_observation_variance"].min()
    q_effect = float(
        factor_diag.loc[factor_diag["true_process_variance"] == q_high, "mean_gain"].mean()
        - factor_diag.loc[
            factor_diag["true_process_variance"] == q_low, "mean_gain"
        ].mean()
    )
    r_effect = float(
        factor_diag.loc[
            factor_diag["true_observation_variance"] == r_low, "mean_gain"
        ].mean()
        - factor_diag.loc[
            factor_diag["true_observation_variance"] == r_high, "mean_gain"
        ].mean()
    )
    directional_pass = bool(q_effect > 0.0 and r_effect > 0.0)

    pivot = cell_metrics.pivot_table(
        index=["participant_id", "block_id"],
        columns="method",
        values="conditional_update_nll",
    )
    cell_gain = (pivot[TOTAL] - pivot[FACTORIZED]).groupby("block_id").mean()
    cell_margin = float(analysis["cell_noninferiority_margin"])
    cell_pass = bool(np.all(cell_gain.to_numpy() >= cell_margin))

    comparison_map = nll_comparisons.set_index("baseline")["mean_gain"].to_dict()
    factor_over_fixed = float(comparison_map[FIXED])
    # comparison_map[PARTICLE] is particle NLL minus factorized NLL.
    particle_over_fixed = factor_over_fixed - float(comparison_map[PARTICLE])
    if particle_over_fixed > 0.0:
        retention = factor_over_fixed / particle_over_fixed
        retention_pass = bool(retention >= float(analysis["particle_gain_retention"]))
        retention_applicable = True
    else:
        retention = None
        retention_pass = True
        retention_applicable = False

    clauses = {
        "nll_gain_vs_fixed": primary[FIXED]["nll_passes"],
        "nll_gain_vs_total_uncertainty": primary[TOTAL]["nll_passes"],
        "mse_gain_vs_fixed": primary[FIXED]["mse_passes"],
        "mse_gain_vs_total_uncertainty": primary[TOTAL]["mse_passes"],
        "directional_qr_effects": directional_pass,
        "cellwise_noninferiority_vs_total": cell_pass,
        "particle_gain_retention": retention_pass,
    }
    passed = bool(all(clauses.values()))
    return {
        "experiment": EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "development",
        "development_gate_passed": passed,
        "conclusion": "support" if passed else "oppose",
        "claim_upgrade_allowed": False,
        "confirmation_unlocked": passed,
        "popgym_unlocked": False,
        "clauses": clauses,
        "primary_comparisons": primary,
        "q_gain_effect": q_effect,
        "r_gain_effect": r_effect,
        "cell_gain_total_minus_factorized": {
            str(int(index)): float(value) for index, value in cell_gain.items()
        },
        "particle_gain_retention": retention,
        "particle_gain_retention_applicable": retention_applicable,
        "same_stimulus_tape_across_experiments": True,
        "independent_statistical_unit": "participant",
    }


def _selected_json(
    selected_full: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        method: {
            "candidate_id": value["candidate_id"],
            "parameters": value["parameters"],
            "response_sigma": value["response_sigma"],
        }
        for method, value in selected_full.items()
    }


def _plot(
    path: Path,
    comparisons: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panel = comparisons.loc[comparisons["baseline"].isin((FIXED, TOTAL, PARTICLE))]
    panel = panel.loc[panel["metric"] == "conditional_update_nll"]
    x = np.arange(len(panel))
    mean = panel["mean_gain"].to_numpy()
    low = mean - panel["ci_low"].to_numpy()
    high = panel["ci_high"].to_numpy() - mean
    axes[0].bar(x, mean, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].errorbar(x, mean, yerr=np.vstack((low, high)), fmt="none", color="black")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, panel["baseline"], rotation=20, ha="right")
    axes[0].set_ylabel("NLL gain (baseline - factorized)")
    axes[0].set_title("Held-out participant updates")

    factor = diagnostics.loc[diagnostics["method"] == FACTORIZED].copy()
    labels = [
        f"Q={row.true_process_variance:g}\nR={row.true_observation_variance:g}"
        for row in factor.itertuples()
    ]
    axes[1].bar(np.arange(len(factor)), factor["mean_gain"], color="#B279A2")
    axes[1].set_xticks(np.arange(len(factor)), labels)
    axes[1].set_ylabel("Mean executed gain")
    axes[1].set_title("Factorized gain by held-out label")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _report(
    summary: Mapping[str, Any],
    comparisons: pd.DataFrame,
    selected: Mapping[str, Any],
) -> str:
    verdict = "passes" if summary["development_gate_passed"] else "fails"
    lines = [
        "# Exp44 Piray--Daw Q/R behavioral-utility development report",
        "",
        f"The registered Experiment-1 development conjunction **{verdict}**. ",
        "This is development evidence only and cannot directly upgrade a claim.",
        "",
        "## Primary held-out participant contrasts",
        "",
        "| Baseline | NLL gain (baseline - factorized) | 95% bootstrap CI | Holm p |",
        "|---|---:|---:|---:|",
    ]
    for baseline in (FIXED, TOTAL, PARTICLE, AUTOCOV, ORACLE):
        row = comparisons.loc[
            (comparisons["baseline"] == baseline)
            & (comparisons["metric"] == "conditional_update_nll")
        ].iloc[0]
        lines.append(
            f"| {baseline} | {row['mean_gain']:+.6f} | "
            f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
            f"{row['p_holm']:.6g} |"
        )
    lines.extend(
        [
            "",
            "| Baseline | MSE gain (baseline - factorized) | 95% bootstrap CI | Holm p |",
            "|---|---:|---:|---:|",
        ]
    )
    for baseline in (FIXED, TOTAL, PARTICLE, AUTOCOV, ORACLE):
        row = comparisons.loc[
            (comparisons["baseline"] == baseline)
            & (comparisons["metric"] == "conditional_update_mse")
        ].iloc[0]
        lines.append(
            f"| {baseline} | {row['mean_gain']:+.6f} | "
            f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
            f"{row['p_holm']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Gate clauses",
            "",
        ]
    )
    for name, value in summary["clauses"].items():
        lines.append(f"- `{name}`: {'pass' if value else 'fail'}")
    lines.extend(
        [
            "",
            f"Executed gain Q effect: {summary['q_gain_effect']:+.6f}.",
            f"Executed gain R effect (low R - high R): {summary['r_gain_effect']:+.6f}.",
            "",
            "## Frozen full-Experiment-1 selections",
            "",
        ]
    )
    for method, value in selected.items():
        lines.append(
            f"- `{method}`: `{value['candidate_id']}`, response sigma "
            f"{value['response_sigma']:.6f}."
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "Experiment 1 and Experiment 2 use the same bag/bird stimulus tape. "
            "This run tests held-out participant behavior only. It does not test "
            "unseen streams, hidden changes, POPGym control, neural activity, or "
            "a participating E/I mechanism.",
            "",
            (
                "A passing development gate authorizes freezing a separate "
                "Experiment-2 configuration; it does not authorize POPGym."
                if summary["development_gate_passed"]
                else "The stop rule keeps Experiment 2 and POPGym unexecuted."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _environment() -> dict[str, Any]:
    git: dict[str, Any] = {"commit": None, "tree": None, "dirty": None}
    try:
        git["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        git["tree"] = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        git["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "git": git,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _artifact_manifest(output: Path, *, status: str) -> dict[str, Any]:
    artifacts = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    sources = {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in SOURCE_FILES
        if (PROJECT_ROOT / relative).is_file()
    }
    return {
        "experiment": EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "source_sha256": sources,
        "artifacts": artifacts,
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_config(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    # Capture provenance before creating the untracked result directory.  This
    # distinguishes a genuinely dirty starting tree from artifacts produced by
    # the current run.
    environment = _environment()
    output.mkdir(parents=True)
    _atomic_text(output / "config.json", config_path.read_text(encoding="utf-8"))
    _atomic_json(output / "environment.json", environment)
    log_path = output / "run.log"
    logger = logging.getLogger(EXPERIMENT)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    status = "failed"
    try:
        if environment.get("git", {}).get("dirty") is not False:
            raise RuntimeError("Exp44 must start from a clean git worktree")
        data_root = PROJECT_ROOT / str(config["data"]["root"])
        dataset = load_piray_daw(
            data_root,
            experiment=1,
            allow_confirmation=False,
            verify_hashes=True,
        )
        logger.info("loaded %d participants from %s", dataset.n_participants, dataset.source_path)
        candidates = _build_candidates(dataset, config)
        logger.info(
            "built candidate counts %s",
            {method: len(values) for method, values in candidates.items()},
        )
        cv_outputs, folds = cross_validated_behavior(dataset, candidates, config)
        participant_metrics, cell_metrics, candidate_scores, selected_full = cv_outputs
        comparisons = compare_methods(participant_metrics, config)
        diagnostics = trace_diagnostics(dataset, selected_full)
        summary = development_decision(
            comparisons, cell_metrics, diagnostics, config
        )
        selected_json = _selected_json(selected_full)

        _atomic_csv(output / "participant_folds.csv", folds)
        _atomic_csv(output / "participant_metrics.csv", participant_metrics)
        _atomic_csv(output / "cell_metrics.csv", cell_metrics)
        _atomic_csv(output / "candidate_scores.csv", candidate_scores)
        _atomic_csv(output / "comparisons.csv", comparisons)
        _atomic_csv(output / "trace_diagnostics.csv", diagnostics)
        _atomic_json(output / "selected_candidates.json", selected_json)
        _atomic_json(output / "summary.json", summary)
        _atomic_text(output / "report.md", _report(summary, comparisons, selected_json))
        _plot(output / "exp44_piray_daw_qr_behavior", comparisons, diagnostics)
        status = "complete"
        logger.info("development gate passed=%s", summary["development_gate_passed"])
        return summary
    except Exception as error:
        logger.exception("Exp44 failed")
        _atomic_json(
            output / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
        _atomic_json(output / "manifest.json", _artifact_manifest(output, status=status))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "development"
        / "exp44_piray_daw_qr_behavior_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "exp44_piray_daw_qr_behavior_development_v1",
    )
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
