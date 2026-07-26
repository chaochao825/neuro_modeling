#!/usr/bin/env python3
"""Development-only matched-Q/R identifiability probe.

This diagnostic cannot upgrade claims.  It compares causal filters on H=0
random-walk tapes whose Q/R regimes are paired to have equal ``Q + 2R``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import scipy

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.autocovariance_uncertainty_filter import (
    AutocovarianceFilterTrace,
    AutocovarianceUpdateConfig,
    TotalVarianceFilterTrace,
    run_autocovariance_filter,
    run_total_variance_filter,
)
from src.models.factorized_uncertainty_filter import (
    AdaptationRates,
    FilterTrace,
    JumpFilterParameters,
    ParameterBounds,
    run_factorized_filter,
    run_fixed_jump_filter,
    run_imm_filter,
    run_oracle_filter,
)
from src.tasks.matched_uncertainty import (
    MATCHED_QR_REGIMES,
    MatchedUncertaintyConfig,
    MatchedUncertaintyTape,
    generate_matched_uncertainty_tape,
)
from src.utils.reproducibility import set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp41_matched_identifiability"
PROFILE = "development_matched_identifiability_probe"
PROTOCOL_VERSION = "exp41_matched_identifiability_probe_v1"
DEVELOPMENT_SEEDS = tuple(range(41000, 41008))
METHODS = (
    "selected_fixed_jump",
    "current_online_em",
    "h_plus_total_variance",
    "autocov_factorized",
    "generator_supported_seen_regime_imm",
    "dynamic_qr_oracle",
)
PRIMARY_METHOD = "autocov_factorized"
FROZEN_EXP39_DEPENDENCY = "src/models/factorized_uncertainty_filter.py"

Trace = FilterTrace | AutocovarianceFilterTrace | TotalVarianceFilterTrace


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a JSON object")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(
        _jsonable(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    _atomic_text(path, encoded + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_provenance() -> dict[str, Any]:
    values: dict[str, Any] = {"commit": None, "tree": None, "dirty": None}
    try:
        values["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        values["tree"] = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        values["dirty"] = bool(
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
    return values


def _development_environment() -> dict[str, Any]:
    return {
        "development_run_environment": True,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "git": _git_provenance(),
    }


def _nonempty_numeric_grid(
    values: object,
    *,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty numeric grid")
    result = tuple(map(float, values))
    if not result or not np.all(np.isfinite(result)) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be finite, non-empty, and unique")
    lower_ok = (
        all(value >= lower for value in result)
        if lower_inclusive
        else all(value > lower for value in result)
    )
    if not lower_ok or not all(value <= upper for value in result):
        bracket = "[" if lower_inclusive else "("
        raise ValueError(f"{name} values must lie in {bracket}{lower}, {upper}]")
    return result


def _strict_integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer and cannot be boolean")
    result = int(value)
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail closed: Exp41 is executable only as the registered dev probe."""

    required = {
        "profile",
        "protocol_version",
        "claim_upgrade_allowed",
        "seeds",
        "used_autograd",
        "used_bptt",
        "generator_hazard",
        "filter_hazard_floor",
        "stream",
        "filter",
        "selection",
        "analysis",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration is missing {sorted(missing)}")
    if config["profile"] != PROFILE:
        raise ValueError("Exp41 is development-only and rejects formal profiles")
    if config["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Exp41 protocol_version does not match the development probe")
    if config["claim_upgrade_allowed"] is not False:
        raise ValueError("Exp41 rejects every claim upgrade")
    raw_seeds = config["seeds"]
    if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
        raise ValueError("Exp41 development seeds must be a sequence")
    seeds = tuple(
        _strict_integer(value, name="development seed") for value in raw_seeds
    )
    if seeds != DEVELOPMENT_SEEDS:
        raise ValueError("Exp41 requires the registered development seeds")
    if config["used_autograd"] is not False:
        raise ValueError("Exp41 prohibits autograd")
    if config["used_bptt"] is not False:
        raise ValueError("Exp41 prohibits BPTT")
    if float(config["generator_hazard"]) != 0.0:
        raise ValueError("Exp41 generator semantics require H=0")
    hazard_floor = float(config["filter_hazard_floor"])
    if not np.isfinite(hazard_floor) or not 0.0 < hazard_floor < 0.5:
        raise ValueError("filter_hazard_floor must lie strictly between 0 and 0.5")

    stream_values = config["stream"]
    for name in ("block_length", "blocks_per_sequence", "n_sequences"):
        _strict_integer(stream_values[name], name=f"stream {name}", positive=True)
    stream = _stream_config(config)
    if stream.blocks_per_sequence % len(MATCHED_QR_REGIMES):
        raise ValueError("blocks_per_sequence must balance all registered regimes")
    analysis = config["analysis"]
    windows = tuple(
        _strict_integer(value, name="transition window", positive=True)
        for value in analysis["transition_windows"]
    )
    if windows != (1, 4, 8, 16):
        raise ValueError("transition_windows must remain [1, 4, 8, 16]")
    if stream.block_length < max(windows):
        raise ValueError("block_length must cover every transition endpoint")
    late_window = _strict_integer(
        analysis["late_window"], name="late_window", positive=True
    )
    if late_window <= 0 or late_window > stream.block_length:
        raise ValueError("late_window must be positive and fit within a block")
    if analysis["primary_method"] != PRIMARY_METHOD:
        raise ValueError("primary_method must remain autocov_factorized")
    _strict_integer(
        analysis["bootstrap_samples"], name="bootstrap_samples", positive=True
    )
    _strict_integer(analysis["statistics_seed"], name="statistics_seed")

    filter_config = config["filter"]
    for name in (
        "initial_process_variance",
        "initial_observation_variance",
        "jump_variance",
    ):
        value = float(filter_config[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"filter {name} must be positive")
    if float(filter_config["hazard_adaptation_rate"]) != 0.0:
        raise ValueError("H=0 probe keeps the numerical hazard floor fixed")
    continuation_floor = float(filter_config["minimum_continuation_weight"])
    if not 0.0 <= continuation_floor <= 1.0:
        raise ValueError("minimum_continuation_weight must lie in [0, 1]")

    selection = config["selection"]
    fixed = selection["fixed_jump_grid"]
    _nonempty_numeric_grid(
        fixed["process_variance"],
        name="fixed process grid",
        lower=0.0,
        upper=0.5,
        lower_inclusive=False,
    )
    _nonempty_numeric_grid(
        fixed["observation_variance"],
        name="fixed observation grid",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    for name in (
        "online_em_process_rate_grid",
        "online_em_observation_rate_grid",
    ):
        _nonempty_numeric_grid(selection[name], name=name, lower=0.0, upper=1.0)
    _nonempty_numeric_grid(
        selection["autocovariance_decay_grid"],
        name="autocovariance_decay_grid",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    _nonempty_numeric_grid(
        selection["autocovariance_prior_mass_grid"],
        name="autocovariance_prior_mass_grid",
        lower=0.0,
        upper=float("inf"),
        lower_inclusive=False,
    )
    _nonempty_numeric_grid(
        selection["total_variance_decay_grid"],
        name="total_variance_decay_grid",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    _nonempty_numeric_grid(
        selection["total_variance_prior_mass_grid"],
        name="total_variance_prior_mass_grid",
        lower=0.0,
        upper=float("inf"),
        lower_inclusive=False,
    )
    _nonempty_numeric_grid(
        selection["total_variance_q_fraction_grid"],
        name="total_variance_q_fraction_grid",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    if any(
        float(value) >= 1.0 for value in selection["total_variance_q_fraction_grid"]
    ):
        raise ValueError("total_variance_q_fraction_grid must remain below one")
    _nonempty_numeric_grid(
        selection["imm_switch_grid"],
        name="imm_switch_grid",
        lower=0.0,
        upper=1.0,
    )


def _stream_config(config: Mapping[str, Any]) -> MatchedUncertaintyConfig:
    values = config["stream"]
    return MatchedUncertaintyConfig(
        block_length=int(values["block_length"]),
        blocks_per_sequence=int(values["blocks_per_sequence"]),
        n_sequences=int(values["n_sequences"]),
        initial_state_variance=float(values["initial_state_variance"]),
    )


def _initial(config: Mapping[str, Any]) -> JumpFilterParameters:
    values = config["filter"]
    return JumpFilterParameters(
        hazard=float(config["filter_hazard_floor"]),
        process_variance=float(values["initial_process_variance"]),
        observation_variance=float(values["initial_observation_variance"]),
        jump_variance=float(values["jump_variance"]),
    )


def _seen_modes(config: Mapping[str, Any]) -> tuple[JumpFilterParameters, ...]:
    initial = _initial(config)
    return tuple(
        JumpFilterParameters(
            initial.hazard,
            regime.process_variance,
            regime.observation_variance,
            initial.jump_variance,
        )
        for regime in MATCHED_QR_REGIMES
    )


def _dynamic_oracle(
    tape: MatchedUncertaintyTape, config: Mapping[str, Any]
) -> FilterTrace:
    """Privileged Q/R oracle using the disclosed numerical hazard floor."""

    initial = _initial(config)
    return run_oracle_filter(
        tape.observations,
        sequence_ids=tape.sequence_ids,
        hazard=np.full(len(tape.observations), initial.hazard),
        process_variance=tape.process_variance,
        observation_variance=tape.observation_variance,
        jump_variance=initial.jump_variance,
    )


def _audit_row(
    *,
    seed: int,
    family: str,
    candidate: str,
    nll: float,
    fit_digest: str,
    privileged: bool = False,
    **parameters: Any,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "selection_family": family,
        "candidate": candidate,
        "selection_nll": nll,
        "data_split": "fit",
        "fit_tape_digest": fit_digest,
        "uses_true_parameters": privileged,
        "selected": False,
        **parameters,
    }


def select_models(
    fit_tape: MatchedUncertaintyTape,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Select every non-oracle hyperparameter using the fit tape only."""

    values, groups = fit_tape.method_inputs()
    initial = _initial(config)
    selection = config["selection"]
    rows: list[dict[str, Any]] = []

    fixed_grid = selection["fixed_jump_grid"]
    for q_value in map(float, fixed_grid["process_variance"]):
        for r_value in map(float, fixed_grid["observation_variance"]):
            trace = run_fixed_jump_filter(
                values,
                sequence_ids=groups,
                parameters=JumpFilterParameters(
                    initial.hazard,
                    q_value,
                    r_value,
                    initial.jump_variance,
                ),
            )
            rows.append(
                _audit_row(
                    seed=seed,
                    family="selected_fixed_jump",
                    candidate=f"q={q_value}:r={r_value}",
                    nll=float(np.mean(trace.predictive_nll)),
                    fit_digest=fit_tape.digest,
                    candidate_q=q_value,
                    candidate_r=r_value,
                )
            )

    # Frozen comparison dependency: this calls the existing Exp39 online-EM
    # implementation verbatim and never modifies its algorithm or trace.
    for q_rate in map(float, selection["online_em_process_rate_grid"]):
        for r_rate in map(float, selection["online_em_observation_rate_grid"]):
            trace = run_factorized_filter(
                values,
                sequence_ids=groups,
                initial=initial,
                adaptation=AdaptationRates(0.0, q_rate, r_rate),
            )
            rows.append(
                _audit_row(
                    seed=seed,
                    family="current_online_em",
                    candidate=f"beta_q={q_rate}:beta_r={r_rate}",
                    nll=float(np.mean(trace.predictive_nll)),
                    fit_digest=fit_tape.digest,
                    candidate_q_rate=q_rate,
                    candidate_r_rate=r_rate,
                )
            )

    filter_config = config["filter"]
    for decay in map(float, selection["total_variance_decay_grid"]):
        for prior_mass in map(float, selection["total_variance_prior_mass_grid"]):
            for q_fraction in map(float, selection["total_variance_q_fraction_grid"]):
                trace = run_total_variance_filter(
                    values,
                    sequence_ids=groups,
                    initial=initial,
                    q_fraction=q_fraction,
                    update=AutocovarianceUpdateConfig(
                        statistic_decay=decay,
                        prior_mass=prior_mass,
                        hazard_rate=float(filter_config["hazard_adaptation_rate"]),
                        minimum_continuation_weight=float(
                            filter_config["minimum_continuation_weight"]
                        ),
                    ),
                )
                rows.append(
                    _audit_row(
                        seed=seed,
                        family="h_plus_total_variance",
                        candidate=(
                            f"decay={decay}:prior_mass={prior_mass}:"
                            f"q_fraction={q_fraction}"
                        ),
                        nll=float(np.mean(trace.predictive_nll)),
                        fit_digest=fit_tape.digest,
                        candidate_decay=decay,
                        candidate_prior_mass=prior_mass,
                        candidate_q_fraction=q_fraction,
                    )
                )

    for decay in map(float, selection["autocovariance_decay_grid"]):
        for prior_mass in map(float, selection["autocovariance_prior_mass_grid"]):
            trace = run_autocovariance_filter(
                values,
                sequence_ids=groups,
                initial=initial,
                update=AutocovarianceUpdateConfig(
                    statistic_decay=decay,
                    prior_mass=prior_mass,
                    hazard_rate=float(filter_config["hazard_adaptation_rate"]),
                    minimum_continuation_weight=float(
                        filter_config["minimum_continuation_weight"]
                    ),
                ),
            )
            rows.append(
                _audit_row(
                    seed=seed,
                    family="autocov_factorized",
                    candidate=f"decay={decay}:prior_mass={prior_mass}",
                    nll=float(np.mean(trace.predictive_nll)),
                    fit_digest=fit_tape.digest,
                    candidate_decay=decay,
                    candidate_prior_mass=prior_mass,
                )
            )

    modes = _seen_modes(config)
    for switch in map(float, selection["imm_switch_grid"]):
        trace = run_imm_filter(
            values,
            sequence_ids=groups,
            modes=modes,
            mode_switch_probability=switch,
        )
        rows.append(
            _audit_row(
                seed=seed,
                family="generator_supported_seen_regime_imm",
                candidate=f"switch={switch}",
                nll=float(np.mean(trace.predictive_nll)),
                fit_digest=fit_tape.digest,
                privileged=True,
                candidate_switch=switch,
            )
        )

    oracle = _dynamic_oracle(fit_tape, config)
    rows.append(
        _audit_row(
            seed=seed,
            family="dynamic_qr_oracle",
            candidate="privileged_time_varying_qr",
            nll=float(np.mean(oracle.predictive_nll)),
            fit_digest=fit_tape.digest,
            privileged=True,
        )
    )

    frame = pd.DataFrame(rows)
    selected: dict[str, dict[str, Any]] = {}
    for family in METHODS:
        candidates = frame.loc[frame["selection_family"].eq(family)]
        index = candidates.sort_values(
            ["selection_nll", "candidate"], kind="stable"
        ).index[0]
        frame.loc[index, "selected"] = True
        selected[family] = frame.loc[index].dropna().to_dict()
    return selected, frame


def _run_selected_methods(
    tape: MatchedUncertaintyTape,
    *,
    config: Mapping[str, Any],
    selected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Trace]:
    values, groups = tape.method_inputs()
    initial = _initial(config)
    fixed = selected["selected_fixed_jump"]
    online = selected["current_online_em"]
    total_variance = selected["h_plus_total_variance"]
    autocov = selected["autocov_factorized"]
    imm = selected["generator_supported_seen_regime_imm"]
    filter_config = config["filter"]
    return {
        "selected_fixed_jump": run_fixed_jump_filter(
            values,
            sequence_ids=groups,
            parameters=JumpFilterParameters(
                initial.hazard,
                float(fixed["candidate_q"]),
                float(fixed["candidate_r"]),
                initial.jump_variance,
            ),
        ),
        "current_online_em": run_factorized_filter(
            values,
            sequence_ids=groups,
            initial=initial,
            adaptation=AdaptationRates(
                0.0,
                float(online["candidate_q_rate"]),
                float(online["candidate_r_rate"]),
            ),
        ),
        "h_plus_total_variance": run_total_variance_filter(
            values,
            sequence_ids=groups,
            initial=initial,
            q_fraction=float(total_variance["candidate_q_fraction"]),
            update=AutocovarianceUpdateConfig(
                statistic_decay=float(total_variance["candidate_decay"]),
                prior_mass=float(total_variance["candidate_prior_mass"]),
                hazard_rate=float(filter_config["hazard_adaptation_rate"]),
                minimum_continuation_weight=float(
                    filter_config["minimum_continuation_weight"]
                ),
            ),
        ),
        "autocov_factorized": run_autocovariance_filter(
            values,
            sequence_ids=groups,
            initial=initial,
            update=AutocovarianceUpdateConfig(
                statistic_decay=float(autocov["candidate_decay"]),
                prior_mass=float(autocov["candidate_prior_mass"]),
                hazard_rate=float(filter_config["hazard_adaptation_rate"]),
                minimum_continuation_weight=float(
                    filter_config["minimum_continuation_weight"]
                ),
            ),
        ),
        "generator_supported_seen_regime_imm": run_imm_filter(
            values,
            sequence_ids=groups,
            modes=_seen_modes(config),
            mode_switch_probability=float(imm["candidate_switch"]),
        ),
        "dynamic_qr_oracle": _dynamic_oracle(tape, config),
    }


def _regime_metadata(name: str) -> tuple[str, bool]:
    if name.endswith("q_dominant"):
        return name.split("_", maxsplit=1)[0], True
    if name.endswith("r_dominant"):
        return name.split("_", maxsplit=1)[0], False
    raise ValueError(f"unregistered matched regime {name}")


def _block_rows(
    tape: MatchedUncertaintyTape,
    traces: Mapping[str, Trace],
    *,
    seed: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    windows = tuple(map(int, config["analysis"]["transition_windows"]))
    late_window = int(config["analysis"]["late_window"])
    bounds = ParameterBounds()
    rows: list[dict[str, Any]] = []
    for sequence_id in np.unique(tape.sequence_ids):
        sequence_mask = tape.sequence_ids == sequence_id
        sequence_blocks = np.unique(tape.block_ids[sequence_mask])
        for block_within_sequence, block_id in enumerate(sequence_blocks):
            indices = np.flatnonzero(tape.block_ids == block_id)
            regime = str(tape.regimes[indices[0]])
            pair_id, q_dominant = _regime_metadata(regime)
            eligible = block_within_sequence > 0
            for method, trace in traces.items():
                q_estimate = float(np.mean(trace.process_variance[indices]))
                r_estimate = float(np.mean(trace.observation_variance[indices]))
                true_q = float(tape.process_variance[indices[0]])
                true_r = float(tape.observation_variance[indices[0]])
                true_gamma0 = true_q + 2.0 * true_r
                true_gamma1 = -true_r
                implied_gamma0 = float(
                    np.mean(
                        trace.process_variance[indices]
                        + 2.0 * trace.observation_variance[indices]
                    )
                )
                implied_gamma1 = float(np.mean(-trace.observation_variance[indices]))
                direct_gamma0: float | None = None
                direct_gamma1: float | None = None
                covariance_diagnostic = "not_applicable"
                if isinstance(trace, AutocovarianceFilterTrace):
                    direct_gamma0 = float(np.mean(trace.gamma0[indices]))
                    direct_gamma1 = float(np.mean(trace.gamma1[indices]))
                    covariance_diagnostic = "lag0_and_lag1"
                elif isinstance(trace, TotalVarianceFilterTrace):
                    direct_gamma0 = float(np.mean(trace.gamma0[indices]))
                    covariance_diagnostic = "lag0_only"

                update_indices = indices
                if block_within_sequence > 0:
                    update_indices = np.r_[indices[0] - 1, indices]
                parameter_matrix = np.column_stack(
                    (
                        trace.hazard[update_indices],
                        trace.process_variance[update_indices],
                        trace.observation_variance[update_indices],
                    )
                )
                parameter_delta = np.diff(parameter_matrix, axis=0)
                update_norm = np.linalg.norm(parameter_delta, axis=1)
                update_l1 = float(np.sum(np.abs(parameter_delta)))
                update_l2 = float(np.sqrt(np.sum(parameter_delta**2)))
                update_count = int(np.sum(update_norm > 1e-15))
                q_values = trace.process_variance[indices]
                r_values = trace.observation_variance[indices]
                q_clipped = np.isclose(
                    q_values,
                    bounds.process_variance[0],
                    rtol=0.0,
                    atol=0.0,
                ) | np.isclose(
                    q_values,
                    bounds.process_variance[1],
                    rtol=0.0,
                    atol=0.0,
                )
                r_clipped = np.isclose(
                    r_values,
                    bounds.observation_variance[0],
                    rtol=0.0,
                    atol=0.0,
                ) | np.isclose(
                    r_values,
                    bounds.observation_variance[1],
                    rtol=0.0,
                    atol=0.0,
                )
                invalid = ~(
                    np.isfinite(trace.predictive_nll[indices])
                    & np.isfinite(trace.filtered_mean[indices])
                    & np.isfinite(q_values)
                    & np.isfinite(r_values)
                    & (q_values > 0.0)
                    & (r_values > 0.0)
                )
                row: dict[str, Any] = {
                    "seed": seed,
                    "method": method,
                    "sequence_id": int(sequence_id),
                    "block_id": int(block_id),
                    "block_within_sequence": block_within_sequence,
                    "regime": regime,
                    "pair_id": pair_id,
                    "q_dominant": q_dominant,
                    "transition_eligible": eligible,
                    "n_steps": len(indices),
                    "mean_nll": float(np.mean(trace.predictive_nll[indices])),
                    "latent_mse": float(
                        np.mean(
                            (trace.filtered_mean[indices] - tape.latent[indices]) ** 2
                        )
                    ),
                    "late_nll": float(
                        np.mean(trace.predictive_nll[indices[-late_window:]])
                    ),
                    "mean_q_estimate": q_estimate,
                    "mean_r_estimate": r_estimate,
                    "true_q": true_q,
                    "true_r": true_r,
                    "q_bias": q_estimate - true_q,
                    "r_bias": r_estimate - true_r,
                    "q_log_error": float(np.log(q_estimate) - np.log(true_q)),
                    "r_log_error": float(np.log(r_estimate) - np.log(true_r)),
                    "q_absolute_log_error": float(
                        abs(np.log(q_estimate) - np.log(true_q))
                    ),
                    "r_absolute_log_error": float(
                        abs(np.log(r_estimate) - np.log(true_r))
                    ),
                    "true_increment_variance": true_gamma0,
                    "true_gamma0": true_gamma0,
                    "true_gamma1": true_gamma1,
                    "mean_implied_gamma0_estimate": implied_gamma0,
                    "mean_implied_gamma1_estimate": implied_gamma1,
                    "implied_gamma0_bias": implied_gamma0 - true_gamma0,
                    "implied_gamma1_bias": implied_gamma1 - true_gamma1,
                    "direct_covariance_diagnostic": covariance_diagnostic,
                    "mean_direct_gamma0_estimate": direct_gamma0,
                    "mean_direct_gamma1_estimate": direct_gamma1,
                    "direct_gamma0_absolute_error": (
                        abs(direct_gamma0 - true_gamma0)
                        if direct_gamma0 is not None
                        else None
                    ),
                    "direct_gamma1_absolute_error": (
                        abs(direct_gamma1 - true_gamma1)
                        if direct_gamma1 is not None
                        else None
                    ),
                    "q_clipping_fraction": float(np.mean(q_clipped)),
                    "r_clipping_fraction": float(np.mean(r_clipped)),
                    "any_clipping_fraction": float(np.mean(q_clipped | r_clipped)),
                    "parameter_saturation_fraction": float(
                        np.mean(q_clipped | r_clipped)
                    ),
                    "invalid_rows": int(np.sum(invalid)),
                    "parameter_update_l1": update_l1,
                    "parameter_update_l2": update_l2,
                    "parameter_update_squared_sum": float(np.sum(parameter_delta**2)),
                    "parameter_update_count": update_count,
                    "generator_hazard": 0.0,
                    "filter_hazard_floor": float(config["filter_hazard_floor"]),
                    "test_tape_digest": tape.digest,
                }
                for window in windows:
                    row[f"transition_nll_{window}"] = (
                        float(np.mean(trace.predictive_nll[indices[:window]]))
                        if eligible
                        else None
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def run_seed(
    config: Mapping[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    set_global_seed(seed)
    stream = _stream_config(config)
    fit_tape = generate_matched_uncertainty_tape(
        seed=seed,
        split="fit_matched_qr",
        config=stream,
    )
    test_tape = generate_matched_uncertainty_tape(
        seed=seed,
        split="test_matched_qr",
        config=stream,
    )
    if fit_tape.digest == test_tape.digest:
        raise RuntimeError("fit and test tapes must be independent")
    selected, audit = select_models(fit_tape, config=config, seed=seed)
    traces = _run_selected_methods(test_tape, config=config, selected=selected)
    if tuple(traces) != METHODS:
        raise RuntimeError("method panel is incomplete or reordered")
    blocks = _block_rows(test_tape, traces, seed=seed, config=config)
    metadata = {
        "seed": seed,
        "fit_tape_digest": fit_tape.digest,
        "test_tape_digest": test_tape.digest,
        "fit_test_tapes_independent": True,
        "all_methods_share_test_tape": True,
        "selection_data_split": "fit_only",
        "selected": selected,
        "generator_hazard": 0.0,
        "filter_hazard_floor": float(config["filter_hazard_floor"]),
        "frozen_exp39_dependency": FROZEN_EXP39_DEPENDENCY,
        "generator_supported_privileged_methods": [
            "generator_supported_seen_regime_imm",
            "dynamic_qr_oracle",
        ],
        "tied_qr_executed_separately": False,
        "tied_qr_equivalent_to": "h_plus_total_variance",
    }
    return blocks, audit, metadata


def _bootstrap_ci(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(map(float, np.quantile(sampled, (0.025, 0.975))))


def _matched_pair_separation(block_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, method, pair_id), frame in block_metrics.groupby(
        ["seed", "method", "pair_id"], sort=True
    ):
        by_dominance = frame.groupby("q_dominant").agg(
            mean_q_estimate=("mean_q_estimate", "mean"),
            mean_r_estimate=("mean_r_estimate", "mean"),
        )
        if set(by_dominance.index) != {False, True}:
            raise RuntimeError("matched-pair panel lacks one dominance regime")
        rows.append(
            {
                "seed": int(seed),
                "method": str(method),
                "pair_id": str(pair_id),
                "q_estimate_q_dominant_minus_r_dominant": float(
                    by_dominance.loc[True, "mean_q_estimate"]
                    - by_dominance.loc[False, "mean_q_estimate"]
                ),
                "r_estimate_r_dominant_minus_q_dominant": float(
                    by_dominance.loc[False, "mean_r_estimate"]
                    - by_dominance.loc[True, "mean_r_estimate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    block_metrics: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if block_metrics.empty or set(block_metrics["method"]) != set(METHODS):
        raise RuntimeError("Exp41 block metrics are incomplete")
    windows = tuple(map(int, config["analysis"]["transition_windows"]))
    aggregations: dict[str, tuple[str, str]] = {
        "mean_nll": ("mean_nll", "mean"),
        "latent_mse": ("latent_mse", "mean"),
        "late_nll": ("late_nll", "mean"),
        "mean_q_bias": ("q_bias", "mean"),
        "mean_r_bias": ("r_bias", "mean"),
        "mean_q_absolute_log_error": ("q_absolute_log_error", "mean"),
        "mean_r_absolute_log_error": ("r_absolute_log_error", "mean"),
        "mean_implied_gamma0_bias": ("implied_gamma0_bias", "mean"),
        "mean_implied_gamma1_bias": ("implied_gamma1_bias", "mean"),
        "mean_direct_gamma0_estimate": (
            "mean_direct_gamma0_estimate",
            "mean",
        ),
        "mean_direct_gamma1_estimate": (
            "mean_direct_gamma1_estimate",
            "mean",
        ),
        "mean_direct_gamma0_absolute_error": (
            "direct_gamma0_absolute_error",
            "mean",
        ),
        "mean_direct_gamma1_absolute_error": (
            "direct_gamma1_absolute_error",
            "mean",
        ),
        "mean_q_clipping_fraction": ("q_clipping_fraction", "mean"),
        "mean_r_clipping_fraction": ("r_clipping_fraction", "mean"),
        "mean_parameter_saturation_fraction": (
            "parameter_saturation_fraction",
            "mean",
        ),
        "invalid_rows": ("invalid_rows", "sum"),
        "parameter_update_l1": ("parameter_update_l1", "sum"),
        "parameter_update_squared_sum": (
            "parameter_update_squared_sum",
            "sum",
        ),
        "parameter_update_count": ("parameter_update_count", "sum"),
    }
    for window in windows:
        aggregations[f"transition_nll_{window}"] = (
            f"transition_nll_{window}",
            "mean",
        )
    seed_metrics = (
        block_metrics.groupby(["seed", "method"], as_index=False)
        .agg(**aggregations)
        .sort_values(["seed", "method"])
    )
    seed_metrics["parameter_update_l2"] = np.sqrt(
        seed_metrics.pop("parameter_update_squared_sum")
    )

    comparison_metrics = (
        "mean_nll",
        "late_nll",
        *(f"transition_nll_{window}" for window in windows),
    )
    metric_pivots = {
        metric: seed_metrics.pivot(index="seed", columns="method", values=metric)
        for metric in comparison_metrics
    }
    comparisons: list[dict[str, Any]] = []
    bootstrap_samples = int(config["analysis"]["bootstrap_samples"])
    statistics_seed = int(config["analysis"]["statistics_seed"])
    for offset, baseline in enumerate(
        method for method in METHODS if method != PRIMARY_METHOD
    ):
        nll_gain = (
            metric_pivots["mean_nll"][baseline]
            - metric_pivots["mean_nll"][PRIMARY_METHOD]
        )
        row: dict[str, Any] = {
            "baseline": baseline,
            "mean_nll_gain_baseline_minus_autocov": float(nll_gain.mean()),
            "median_nll_gain_baseline_minus_autocov": float(nll_gain.median()),
            "positive_nll_gain_seeds": int(np.sum(nll_gain > 0.0)),
            "n_seeds": int(len(nll_gain)),
            "bootstrap_nll_gain_ci95_low": _bootstrap_ci(
                nll_gain.to_numpy(float),
                samples=bootstrap_samples,
                seed=statistics_seed + offset,
            )[0],
            "bootstrap_nll_gain_ci95_high": _bootstrap_ci(
                nll_gain.to_numpy(float),
                samples=bootstrap_samples,
                seed=statistics_seed + offset,
            )[1],
            "mean_late_nll_gain": float(
                (
                    metric_pivots["late_nll"][baseline]
                    - metric_pivots["late_nll"][PRIMARY_METHOD]
                ).mean()
            ),
        }
        for window in windows:
            metric = f"transition_nll_{window}"
            gain = (
                metric_pivots[metric][baseline] - metric_pivots[metric][PRIMARY_METHOD]
            )
            row[f"mean_transition_nll_gain_{window}"] = float(gain.mean())
        comparisons.append(row)
    comparison_frame = pd.DataFrame(comparisons)

    separation = _matched_pair_separation(block_metrics)
    separation_by_method: dict[str, Any] = {}
    for (method, pair_id), frame in separation.groupby(
        ["method", "pair_id"], sort=True
    ):
        q_values = frame["q_estimate_q_dominant_minus_r_dominant"].to_numpy(float)
        r_values = frame["r_estimate_r_dominant_minus_q_dominant"].to_numpy(float)
        separation_by_method.setdefault(str(method), {})[str(pair_id)] = {
            "mean_q_separation": float(np.mean(q_values)),
            "positive_q_separation_seeds": int(np.sum(q_values > 0.0)),
            "mean_r_separation": float(np.mean(r_values)),
            "positive_r_separation_seeds": int(np.sum(r_values > 0.0)),
            "n_seeds": len(frame),
        }
    separation_summary = separation_by_method[PRIMARY_METHOD]

    diagnostic_summary: dict[str, Any] = {}
    for method, frame in seed_metrics.groupby("method", sort=True):
        direct_gamma0 = frame["mean_direct_gamma0_estimate"].to_numpy(float)
        direct_gamma1 = frame["mean_direct_gamma1_estimate"].to_numpy(float)
        finite_gamma0 = direct_gamma0[np.isfinite(direct_gamma0)]
        finite_gamma1 = direct_gamma1[np.isfinite(direct_gamma1)]
        diagnostic_summary[str(method)] = {
            "mean_q_bias": float(frame["mean_q_bias"].mean()),
            "mean_r_bias": float(frame["mean_r_bias"].mean()),
            "mean_q_absolute_log_error": float(
                frame["mean_q_absolute_log_error"].mean()
            ),
            "mean_r_absolute_log_error": float(
                frame["mean_r_absolute_log_error"].mean()
            ),
            "mean_implied_gamma0_bias": float(frame["mean_implied_gamma0_bias"].mean()),
            "mean_implied_gamma1_bias": float(frame["mean_implied_gamma1_bias"].mean()),
            "mean_direct_gamma0_estimate": (
                float(np.mean(finite_gamma0)) if len(finite_gamma0) else None
            ),
            "mean_direct_gamma1_estimate": (
                float(np.mean(finite_gamma1)) if len(finite_gamma1) else None
            ),
            "mean_direct_gamma0_absolute_error": (
                float(frame["mean_direct_gamma0_absolute_error"].mean())
                if frame["mean_direct_gamma0_absolute_error"].notna().any()
                else None
            ),
            "mean_direct_gamma1_absolute_error": (
                float(frame["mean_direct_gamma1_absolute_error"].mean())
                if frame["mean_direct_gamma1_absolute_error"].notna().any()
                else None
            ),
            "mean_q_clipping_fraction": float(frame["mean_q_clipping_fraction"].mean()),
            "mean_r_clipping_fraction": float(frame["mean_r_clipping_fraction"].mean()),
            "mean_parameter_saturation_fraction": float(
                frame["mean_parameter_saturation_fraction"].mean()
            ),
            "invalid_rows": int(frame["invalid_rows"].sum()),
            "mean_parameter_update_l1": float(frame["parameter_update_l1"].mean()),
            "mean_parameter_update_l2": float(frame["parameter_update_l2"].mean()),
            "mean_parameter_update_count": float(
                frame["parameter_update_count"].mean()
            ),
        }

    summary = {
        "protocol_version": config["protocol_version"],
        "profile": config["profile"],
        "development_only": True,
        "claim_eligible": False,
        "claim_upgrade_allowed": False,
        "verdict": "inconclusive",
        "statistics_unit": "seed",
        "n_complete_seeds": int(block_metrics["seed"].nunique()),
        "generator_hazard": 0.0,
        "filter_hazard_floor": float(config["filter_hazard_floor"]),
        "fit_selection_only": True,
        "all_methods_paired_on_test_tape": True,
        "transition_first_block_excluded": True,
        "transition_windows": list(windows),
        "late_window": int(config["analysis"]["late_window"]),
        "matched_pair_separation": separation_summary,
        "matched_pair_separation_by_method": separation_by_method,
        "matched_pair_go_diagnostic_method": PRIMARY_METHOD,
        "mandatory_diagnostics": diagnostic_summary,
        "budget_matched": False,
        "budget_note": (
            "Parameter-update L1/L2/count are measured, but no functional or "
            "update budget has yet been matched across adaptive methods."
        ),
        "development_go_gate_satisfied": False,
        "tied_qr_executed_separately": False,
        "tied_qr_equivalence_note": (
            "At H=0 with a fixed Q/R allocation, tied Q/R and h-plus-total-"
            "variance are the same one-scalar parameterization; only "
            "h_plus_total_variance is executed."
        ),
        "generator_supported_privileged_methods": [
            "generator_supported_seen_regime_imm",
            "dynamic_qr_oracle",
        ],
        "method_roles": {
            "selected_fixed_jump": "deployable_fit_selected",
            "current_online_em": "deployable_frozen_exp39",
            "h_plus_total_variance": "deployable_reduced_one_coordinate",
            "autocov_factorized": "deployable_full_qr_covariance",
            "generator_supported_seen_regime_imm": (
                "privileged_generator_supported_regime_modes"
            ),
            "dynamic_qr_oracle": "privileged_dynamic_truth_oracle",
        },
        "cross_loading_diagonal_dominance_claimed": False,
        "cross_loading_note": (
            "The matched panel intentionally anticorrelates Q and R within each "
            "equal-marginal pair; it tests separation, not a diagonal-dominance "
            "cross-loading claim."
        ),
        "interpretation": (
            "Post-hoc development diagnostic only; results can motivate a frozen "
            "future protocol but cannot modify Exp39 conclusions."
        ),
    }
    return seed_metrics, comparison_frame, separation, summary


def _report(summary: Mapping[str, Any], comparisons: pd.DataFrame) -> str:
    lines = [
        "# Exp41 Matched-Q/R Identifiability Probe",
        "",
        "Verdict: **INCONCLUSIVE (development-only diagnostic)**.",
        "",
        (
            "The generator has exactly H=0. All numerical jump filters use the "
            f"disclosed hazard floor `{summary['filter_hazard_floor']}` because "
            "the frozen Exp39 jump step requires a strictly positive hazard."
        ),
        (
            "Fit and test tapes are independently generated. Hyperparameters see "
            "only the fit tape, and every method is evaluated on the same test tape."
        ),
        (
            "`generator_supported_seen_regime_imm` and `dynamic_qr_oracle` are "
            "privileged generator-supported references, not deployable or fair "
            "baselines."
        ),
        "",
        "## Descriptive utility",
        "",
    ]
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"- {row.baseline}: baseline-minus-autocov NLL "
            f"{row.mean_nll_gain_baseline_minus_autocov:+.6f}; positive in "
            f"{row.positive_nll_gain_seeds}/{row.n_seeds} seeds."
        )
    lines.extend(["", "## Matched-pair separation", ""])
    for pair_id, values in summary["matched_pair_separation"].items():
        lines.append(
            f"- {pair_id}: Q separation {values['mean_q_separation']:+.6f} "
            f"({values['positive_q_separation_seeds']}/{values['n_seeds']} positive); "
            f"R separation {values['mean_r_separation']:+.6f} "
            f"({values['positive_r_separation_seeds']}/{values['n_seeds']} positive)."
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(summary["cross_loading_note"]),
            str(summary["tied_qr_equivalence_note"]),
            (
                "Functional/update budgets are **not matched**. L1, L2, and "
                "update counts are diagnostics only, so the development go gate "
                "is forced to FAIL and inference remains inconclusive."
            ),
            (
                "Transition endpoints use the first 1/4/8/16 samples after a block "
                "transition and exclude the first block of every sequence."
            ),
            (
                "This artifact cannot upgrade claims, cannot change Exp39, and "
                "does not constitute a formal-seed result."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_manifest(
    output: Path,
    *,
    status: str,
    run_start_environment: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if (
            not path.is_file()
            or path.name == "manifest.json"
            or path.name.endswith(".tmp")
        ):
            continue
        artifacts[path.relative_to(output).as_posix()] = _sha256(path)
    source_files = (
        FROZEN_EXP39_DEPENDENCY,
        "src/models/autocovariance_uncertainty_filter.py",
        "src/tasks/matched_uncertainty.py",
        "src/utils/reproducibility.py",
        "experiments/exp41_matched_identifiability.py",
    )
    source_sha256 = {
        relative: _sha256(PROJECT_ROOT / relative) for relative in source_files
    }
    implementation_sha256 = hashlib.sha256(
        json.dumps(source_sha256, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "profile": PROFILE,
        "status": status,
        "development_only": True,
        "claim_upgrade_allowed": False,
        "artifacts": artifacts,
        "source_sha256": source_sha256,
        "implementation_sha256": implementation_sha256,
        "git": dict(run_start_environment["git"]),
        "git_snapshot_role": "run_start_before_output_mutation",
        "finalization_git": _git_provenance(),
        "finalization_git_note": (
            "May include run artifacts when the output directory is not ignored; "
            "clean-run evidence is the run-start snapshot."
        ),
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_config(config)
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    run_start_environment = _development_environment()
    output.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(output / "config.json", config_path.read_bytes())
    _atomic_json(output / "environment.json", run_start_environment)
    _atomic_json(
        output / "planned_conditions.json",
        {
            "development_only": True,
            "claim_upgrade_allowed": False,
            "methods": list(METHODS),
            "regimes": [regime.name for regime in MATCHED_QR_REGIMES],
            "seeds": list(map(int, config["seeds"])),
            "generator_hazard": 0.0,
            "filter_hazard_floor": float(config["filter_hazard_floor"]),
            "transition_windows": list(config["analysis"]["transition_windows"]),
            "late_window": int(config["analysis"]["late_window"]),
            "budget_matched": False,
            "tied_qr_executed_separately": False,
            "tied_qr_equivalent_to": "h_plus_total_variance",
            "generator_supported_privileged_methods": [
                "generator_supported_seen_regime_imm",
                "dynamic_qr_oracle",
            ],
        },
    )

    logger = logging.getLogger(f"{EXPERIMENT}.{output.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_temporary = output / f".run.log.{uuid.uuid4().hex}.tmp"
    handler = logging.FileHandler(log_temporary, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    blocks: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    fatal_error: Exception | None = None

    try:
        for seed in map(int, config["seeds"]):
            seed_dir = output / f"seed_{seed}"
            seed_dir.mkdir()
            try:
                block, audit, metadata = run_seed(config, seed)
                _atomic_csv(seed_dir / "block_metrics.csv", block)
                _atomic_csv(seed_dir / "selection_audit.csv", audit)
                _atomic_json(seed_dir / "metadata.json", metadata)
                _atomic_json(
                    seed_dir / "status.json", {"seed": seed, "status": "complete"}
                )
                blocks.append(block)
                audits.append(audit)
                logger.info("seed %s complete", seed)
            except Exception as error:  # preserve every development failure
                failure = {
                    "seed": seed,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                _atomic_json(seed_dir / "status.json", failure)
                logger.exception("seed %s failed", seed)

        failure_columns = ["seed", "status", "error_type", "error"]
        _atomic_json(output / "failures.json", failures)
        _atomic_csv(
            output / "failed_seeds.csv",
            pd.DataFrame(failures, columns=failure_columns),
        )
        if blocks:
            block_metrics = pd.concat(blocks, ignore_index=True)
            selection_audit = pd.concat(audits, ignore_index=True)
            _atomic_csv(output / "block_metrics.csv", block_metrics)
            _atomic_csv(output / "selection_audit.csv", selection_audit)
        else:
            block_metrics = pd.DataFrame()

        if failures:
            fatal_error = RuntimeError("one or more Exp41 development seed failed")
        elif len(blocks) != len(config["seeds"]):
            fatal_error = RuntimeError("Exp41 completed seed count is inconsistent")
        else:
            seed_metrics, comparisons, separation, summary = summarize(
                block_metrics, config=config
            )
            _atomic_csv(output / "seed_metrics.csv", seed_metrics)
            _atomic_csv(output / "comparisons.csv", comparisons)
            _atomic_csv(output / "matched_pair_separation.csv", separation)
            _atomic_json(output / "summary.json", summary)
            _atomic_text(output / "report.md", _report(summary, comparisons))
    except Exception as error:
        fatal_error = error
        logger.exception("Exp41 execution failed")
    finally:
        status = "failed" if fatal_error is not None else "complete"
        status_payload: dict[str, Any] = {
            "status": status,
            "development_only": True,
            "claim_upgrade_allowed": False,
            "n_planned_seeds": len(config["seeds"]),
            "n_complete_seeds": len(blocks),
            "n_failed_seeds": len(failures),
        }
        if fatal_error is not None:
            status_payload.update(
                error_type=type(fatal_error).__name__, error=str(fatal_error)
            )
        elif summary is not None:
            status_payload["verdict"] = summary["verdict"]
        _atomic_json(output / "status.json", status_payload)
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
        log_temporary.replace(output / "run.log")
        _atomic_json(
            output / "manifest.json",
            _artifact_manifest(
                output,
                status=status,
                run_start_environment=run_start_environment,
            ),
        )

    if fatal_error is not None:
        raise fatal_error
    assert summary is not None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = execute(args.config.resolve(), args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
