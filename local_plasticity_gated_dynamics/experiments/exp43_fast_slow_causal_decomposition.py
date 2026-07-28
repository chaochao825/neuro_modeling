"""Development-only causal exchange audit for fast event and slow Q/R paths."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.autocovariance_uncertainty_filter import (
    AutocovarianceUpdateConfig,
    run_total_variance_filter,
)
from src.models.factorized_uncertainty_filter import (
    AdaptationRates,
    JumpFilterParameters,
    ParameterBounds,
    run_fixed_jump_filter,
    run_imm_filter,
    run_oracle_filter,
)
from src.models.fast_slow_uncertainty_audit import (
    run_fast_slow_exchange_filter,
)
from src.tasks.factorized_uncertainty import (
    UncertaintyLevels,
    heldout_composition_cells,
    single_factor_training_cells,
)
from src.tasks.fast_slow_uncertainty import (
    FastSlowStreamConfig,
    generate_fast_slow_tape,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp43_fast_slow_causal_decomposition"
PROFILE = "development_fast_slow_causal_decomposition_probe"
PROTOCOL_VERSION = "exp43_fast_slow_causal_decomposition_probe_v1"

LEARNED = "learned_event_learned_qr"
ORACLE_EVENT = "oracle_event_learned_qr"
ORACLE_QR = "learned_event_oracle_qr"
ORACLE_BOTH = "oracle_event_oracle_qr"
TOTAL_VARIANCE = "h_plus_total_variance"
SEEN_IMM = "generator_supported_seen_mode_imm"
FIXED = "selected_fixed_jump"
DYNAMIC_ORACLE = "dynamic_parameter_oracle"
METHODS = (
    LEARNED,
    ORACLE_EVENT,
    ORACLE_QR,
    ORACLE_BOTH,
    TOTAL_VARIANCE,
    SEEN_IMM,
    FIXED,
    DYNAMIC_ORACLE,
)
ORACLE_METHODS = frozenset(
    {ORACLE_EVENT, ORACLE_QR, ORACLE_BOTH, DYNAMIC_ORACLE, SEEN_IMM}
)
SOURCE_FILES = (
    "src/models/factorized_uncertainty_filter.py",
    "src/models/autocovariance_uncertainty_filter.py",
    "src/models/fast_slow_uncertainty_audit.py",
    "src/tasks/factorized_uncertainty.py",
    "src/tasks/fast_slow_uncertainty.py",
    "experiments/exp43_fast_slow_causal_decomposition.py",
    "scripts/validate_exp43_development_result.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must contain one JSON object")
    return payload


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


def _git_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "tree": None, "dirty": None}
    try:
        result["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        result["tree"] = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        result["dirty"] = bool(
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
    return result


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "git": _git_provenance(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("profile") != PROFILE:
        raise ValueError(f"profile must be {PROFILE}")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
    if config.get("claim_upgrade_allowed") is not False:
        raise ValueError("Exp43 is development-only and cannot upgrade claims")
    if config.get("used_autograd") is not False or config.get("used_bptt") is not False:
        raise ValueError("Exp43 local arms forbid autograd and BPTT")
    seeds = tuple(map(int, config["seeds"]))
    reserved = tuple(map(int, config["reserved_formal_seeds"]))
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("development seeds must be unique non-negative integers")
    if not reserved or len(set(reserved)) != len(reserved):
        raise ValueError("reserved formal seeds must be unique")
    if reserved != tuple(range(43100, 43130)):
        raise ValueError("reserved formal seeds must be exactly 43100--43129")
    if set(seeds) & set(reserved):
        raise ValueError("development seeds must not overlap reserved formal seeds")
    if any(43100 <= seed <= 43129 for seed in seeds):
        raise ValueError("reserved formal seeds must not be used as development seeds")

    levels = config["levels"]
    UncertaintyLevels(
        hazard=tuple(map(float, levels["hazard"])),
        process_variance=tuple(map(float, levels["process_variance"])),
        observation_variance=tuple(map(float, levels["observation_variance"])),
    )
    stream = config["stream"]
    stream_config = FastSlowStreamConfig(
        block_lengths=tuple(map(int, stream["block_lengths"])),
        blocks_per_sequence=int(stream["blocks_per_sequence"]),
        n_sequences=int(stream["n_sequences"]),
        jump_variance=float(stream["jump_variance"]),
    )
    pair_count = len(single_factor_training_cells()) * len(
        stream_config.block_lengths
    )
    if stream_config.blocks_per_sequence % pair_count:
        raise ValueError("stream is not balanced over training cell-duration pairs")

    filter_config = config["filter"]
    JumpFilterParameters(
        float(filter_config["initial_hazard"]),
        float(filter_config["initial_process_variance"]),
        float(filter_config["initial_observation_variance"]),
        float(filter_config["jump_variance"]),
    )
    bounds = filter_config["bounds"]
    ParameterBounds(
        hazard=tuple(map(float, bounds["hazard"])),
        process_variance=tuple(map(float, bounds["process_variance"])),
        observation_variance=tuple(map(float, bounds["observation_variance"])),
    )
    for key in (
        "hazard_rate_grid",
        "process_rate_grid",
        "observation_rate_grid",
        "fixed_process_grid",
        "fixed_observation_grid",
        "total_variance_decay_grid",
        "total_variance_prior_mass_grid",
        "total_variance_q_fraction_grid",
        "imm_switch_grid",
    ):
        values = tuple(map(float, config["selection"][key]))
        if not values or not all(np.isfinite(value) for value in values):
            raise ValueError(f"selection grid {key} must be finite and non-empty")
    windows = tuple(map(int, config["analysis"]["transition_windows"]))
    if windows != tuple(sorted(set(windows))) or any(value <= 0 for value in windows):
        raise ValueError("transition windows must be increasing positive integers")
    if int(config["analysis"]["required_positive_seeds"]) > len(seeds):
        raise ValueError("required positive seeds cannot exceed development seeds")


def _objects(
    config: Mapping[str, Any],
) -> tuple[UncertaintyLevels, FastSlowStreamConfig, JumpFilterParameters, ParameterBounds]:
    level_values = config["levels"]
    levels = UncertaintyLevels(
        hazard=tuple(map(float, level_values["hazard"])),
        process_variance=tuple(map(float, level_values["process_variance"])),
        observation_variance=tuple(
            map(float, level_values["observation_variance"])
        ),
    )
    stream_values = config["stream"]
    stream = FastSlowStreamConfig(
        block_lengths=tuple(map(int, stream_values["block_lengths"])),
        blocks_per_sequence=int(stream_values["blocks_per_sequence"]),
        n_sequences=int(stream_values["n_sequences"]),
        jump_variance=float(stream_values["jump_variance"]),
    )
    filter_values = config["filter"]
    initial = JumpFilterParameters(
        float(filter_values["initial_hazard"]),
        float(filter_values["initial_process_variance"]),
        float(filter_values["initial_observation_variance"]),
        float(filter_values["jump_variance"]),
    )
    bound_values = filter_values["bounds"]
    bounds = ParameterBounds(
        hazard=tuple(map(float, bound_values["hazard"])),
        process_variance=tuple(map(float, bound_values["process_variance"])),
        observation_variance=tuple(
            map(float, bound_values["observation_variance"])
        ),
    )
    return levels, stream, initial, bounds


def _select_adaptation(
    tape: Any,
    *,
    initial: JumpFilterParameters,
    bounds: ParameterBounds,
    selection: Mapping[str, Any],
    seed: int,
) -> tuple[AdaptationRates, list[dict[str, Any]]]:
    candidates: list[tuple[float, tuple[float, float, float], AdaptationRates]] = []
    for h_rate in map(float, selection["hazard_rate_grid"]):
        for q_rate in map(float, selection["process_rate_grid"]):
            for r_rate in map(float, selection["observation_rate_grid"]):
                rates = AdaptationRates(h_rate, q_rate, r_rate)
                trace = run_fast_slow_exchange_filter(
                    tape.observations,
                    sequence_ids=tape.sequence_ids,
                    initial=initial,
                    adaptation=rates,
                    bounds=bounds,
                )
                candidates.append(
                    (float(np.mean(trace.predictive_nll)), (h_rate, q_rate, r_rate), rates)
                )
    score, key, selected = min(candidates, key=lambda item: (item[0], item[1]))
    audit = [
        {
            "seed": seed,
            "selection_family": LEARNED,
            "hazard_rate": candidate[1][0],
            "process_rate": candidate[1][1],
            "observation_rate": candidate[1][2],
            "fit_nll": candidate[0],
            "selected": candidate[1] == key and candidate[0] == score,
        }
        for candidate in candidates
    ]
    return selected, audit


def _select_fixed(
    tape: Any,
    *,
    initial: JumpFilterParameters,
    selection: Mapping[str, Any],
    seed: int,
) -> tuple[JumpFilterParameters, list[dict[str, Any]]]:
    candidates: list[tuple[float, tuple[float, float], JumpFilterParameters]] = []
    for q_value in map(float, selection["fixed_process_grid"]):
        for r_value in map(float, selection["fixed_observation_grid"]):
            parameters = JumpFilterParameters(
                initial.hazard, q_value, r_value, initial.jump_variance
            )
            trace = run_fixed_jump_filter(
                tape.observations,
                sequence_ids=tape.sequence_ids,
                parameters=parameters,
            )
            candidates.append(
                (float(np.mean(trace.predictive_nll)), (q_value, r_value), parameters)
            )
    score, key, selected = min(candidates, key=lambda item: (item[0], item[1]))
    audit = [
        {
            "seed": seed,
            "selection_family": FIXED,
            "process_variance": candidate[1][0],
            "observation_variance": candidate[1][1],
            "fit_nll": candidate[0],
            "selected": candidate[1] == key and candidate[0] == score,
        }
        for candidate in candidates
    ]
    return selected, audit


def _select_total_variance(
    tape: Any,
    *,
    initial: JumpFilterParameters,
    bounds: ParameterBounds,
    selection: Mapping[str, Any],
    seed: int,
) -> tuple[float, AutocovarianceUpdateConfig, list[dict[str, Any]]]:
    candidates: list[
        tuple[float, tuple[float, float, float, float], float, AutocovarianceUpdateConfig]
    ] = []
    for decay in map(float, selection["total_variance_decay_grid"]):
        for prior in map(float, selection["total_variance_prior_mass_grid"]):
            for q_fraction in map(float, selection["total_variance_q_fraction_grid"]):
                for h_rate in map(float, selection["hazard_rate_grid"]):
                    update = AutocovarianceUpdateConfig(
                        statistic_decay=decay,
                        prior_mass=prior,
                        hazard_rate=h_rate,
                    )
                    trace = run_total_variance_filter(
                        tape.observations,
                        sequence_ids=tape.sequence_ids,
                        initial=initial,
                        q_fraction=q_fraction,
                        update=update,
                        bounds=bounds,
                    )
                    key = (decay, prior, q_fraction, h_rate)
                    candidates.append(
                        (float(np.mean(trace.predictive_nll)), key, q_fraction, update)
                    )
    score, key, selected_fraction, selected_update = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    audit = [
        {
            "seed": seed,
            "selection_family": TOTAL_VARIANCE,
            "statistic_decay": candidate[1][0],
            "prior_mass": candidate[1][1],
            "q_fraction": candidate[1][2],
            "hazard_rate": candidate[1][3],
            "fit_nll": candidate[0],
            "selected": candidate[1] == key and candidate[0] == score,
        }
        for candidate in candidates
    ]
    return selected_fraction, selected_update, audit


def _select_imm(
    tape: Any,
    *,
    modes: Sequence[JumpFilterParameters],
    selection: Mapping[str, Any],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    candidates: list[tuple[float, float]] = []
    for switch in map(float, selection["imm_switch_grid"]):
        trace = run_imm_filter(
            tape.observations,
            sequence_ids=tape.sequence_ids,
            modes=modes,
            mode_switch_probability=switch,
        )
        candidates.append((float(np.mean(trace.predictive_nll)), switch))
    score, selected = min(candidates, key=lambda item: (item[0], item[1]))
    audit = [
        {
            "seed": seed,
            "selection_family": SEEN_IMM,
            "mode_switch_probability": switch,
            "fit_nll": fit_nll,
            "selected": switch == selected and fit_nll == score,
        }
        for fit_nll, switch in candidates
    ]
    return selected, audit


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _trace_metrics(
    method: str,
    trace: Any,
    tape: Any,
    *,
    seed: int,
    bounds: ParameterBounds,
) -> dict[str, Any]:
    release = np.asarray(
        getattr(trace, "release_probability", trace.jump_probability),
        dtype=np.float64,
    )
    write = np.asarray(
        getattr(trace, "write_gain", np.full(len(release), np.nan)),
        dtype=np.float64,
    )
    non_jump = ~np.asarray(tape.jump_flags, dtype=bool)
    jump = ~non_jump
    return {
        "seed": seed,
        "method": method,
        "privileged": method in ORACLE_METHODS,
        "overall_nll": float(np.mean(trace.predictive_nll)),
        "latent_mse": float(np.mean((trace.filtered_mean - tape.latent) ** 2)),
        "jump_auc": _auc(tape.jump_flags, trace.jump_probability),
        "jump_recall_at_half": float(np.mean(release[jump] >= 0.5))
        if np.any(jump)
        else float("nan"),
        "false_release_rate_at_half": float(np.mean(release[non_jump] >= 0.5)),
        "release_mass": float(np.sum(release)),
        "mean_write_gain": float(np.nanmean(write))
        if np.any(np.isfinite(write))
        else float("nan"),
        "q_lower_clip_fraction": float(
            np.mean(trace.process_variance <= bounds.process_variance[0] + 1e-12)
        ),
        "q_upper_clip_fraction": float(
            np.mean(trace.process_variance >= bounds.process_variance[1] - 1e-12)
        ),
        "r_lower_clip_fraction": float(
            np.mean(
                trace.observation_variance <= bounds.observation_variance[0] + 1e-12
            )
        ),
        "r_upper_clip_fraction": float(
            np.mean(
                trace.observation_variance >= bounds.observation_variance[1] - 1e-12
            )
        ),
    }


def _block_rows(method: str, trace: Any, tape: Any, *, seed: int, late: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    release = np.asarray(
        getattr(trace, "release_probability", trace.jump_probability),
        dtype=np.float64,
    )
    for block in np.unique(tape.block_ids):
        indices = np.flatnonzero(tape.block_ids == block)
        late_indices = indices[-min(late, len(indices)) :]
        rows.append(
            {
                "seed": seed,
                "method": method,
                "block_id": int(block),
                "sequence_id": int(tape.sequence_ids[indices[0]]),
                "cell": str(tape.cells[indices[0]]),
                "block_length": int(len(indices)),
                "n_jumps": int(np.sum(tape.jump_flags[indices])),
                "nll": float(np.mean(trace.predictive_nll[indices])),
                "latent_mse": float(
                    np.mean((trace.filtered_mean[indices] - tape.latent[indices]) ** 2)
                ),
                "late_nll": float(np.mean(trace.predictive_nll[late_indices])),
                "late_latent_mse": float(
                    np.mean(
                        (trace.filtered_mean[late_indices] - tape.latent[late_indices])
                        ** 2
                    )
                ),
                "mean_h": float(np.mean(trace.hazard[indices])),
                "mean_q": float(np.mean(trace.process_variance[indices])),
                "mean_r": float(np.mean(trace.observation_variance[indices])),
                "mean_jump_probability": float(
                    np.mean(trace.jump_probability[indices])
                ),
                "mean_release_probability": float(np.mean(release[indices])),
            }
        )
    return rows


def _window_indices_after_event(
    event_index: int,
    *,
    window: int,
    sequence_ids: np.ndarray,
    jump_flags: np.ndarray,
) -> np.ndarray:
    sequence = sequence_ids[event_index]
    stop = min(event_index + 1 + window, len(sequence_ids))
    candidates = np.arange(event_index + 1, stop)
    candidates = candidates[sequence_ids[candidates] == sequence]
    future_events = candidates[jump_flags[candidates]]
    if len(future_events):
        candidates = candidates[candidates < future_events[0]]
    return candidates


def _event_rows(
    method: str,
    trace: Any,
    tape: Any,
    *,
    seed: int,
    windows: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_index in np.flatnonzero(tape.jump_flags):
        for window in windows:
            indices = _window_indices_after_event(
                int(event_index),
                window=int(window),
                sequence_ids=tape.sequence_ids,
                jump_flags=tape.jump_flags,
            )
            if not len(indices):
                continue
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "event_index": int(event_index),
                    "sequence_id": int(tape.sequence_ids[event_index]),
                    "cell": str(tape.cells[event_index]),
                    "window": int(window),
                    "n_samples": int(len(indices)),
                    "nll": float(np.mean(trace.predictive_nll[indices])),
                    "latent_mse": float(
                        np.mean(
                            (trace.filtered_mean[indices] - tape.latent[indices]) ** 2
                        )
                    ),
                }
            )
    return rows


def _regime_rows(
    method: str,
    trace: Any,
    tape: Any,
    *,
    seed: int,
    windows: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = np.unique(tape.block_ids)
    for block in blocks:
        indices = np.flatnonzero(tape.block_ids == block)
        start = int(indices[0])
        if start == 0 or tape.sequence_ids[start] != tape.sequence_ids[start - 1]:
            continue
        previous_cell = str(tape.cells[start - 1])
        current_cell = str(tape.cells[start])
        for window in windows:
            selected = indices[: min(int(window), len(indices))]
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "block_id": int(block),
                    "sequence_id": int(tape.sequence_ids[start]),
                    "previous_cell": previous_cell,
                    "cell": current_cell,
                    "window": int(window),
                    "n_samples": int(len(selected)),
                    "nll": float(np.mean(trace.predictive_nll[selected])),
                    "latent_mse": float(
                        np.mean(
                            (trace.filtered_mean[selected] - tape.latent[selected]) ** 2
                        )
                    ),
                }
            )
    return rows


def run_seed(
    config: Mapping[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_config(config)
    if int(seed) not in set(map(int, config["seeds"])):
        raise ValueError("seed is not registered in the development config")
    levels, stream, initial, bounds = _objects(config)
    fit_tape = generate_fast_slow_tape(
        seed=int(seed),
        split="fit",
        cells=single_factor_training_cells(),
        levels=levels,
        config=stream,
    )
    test_tape = generate_fast_slow_tape(
        seed=int(seed),
        split="test",
        cells=heldout_composition_cells(),
        levels=levels,
        config=stream,
    )
    if fit_tape.digest == test_tape.digest:
        raise RuntimeError("fit and test tape digests unexpectedly match")

    selection = config["selection"]
    adaptation, adaptation_audit = _select_adaptation(
        fit_tape,
        initial=initial,
        bounds=bounds,
        selection=selection,
        seed=int(seed),
    )
    fixed_parameters, fixed_audit = _select_fixed(
        fit_tape,
        initial=initial,
        selection=selection,
        seed=int(seed),
    )
    q_fraction, total_update, total_audit = _select_total_variance(
        fit_tape,
        initial=initial,
        bounds=bounds,
        selection=selection,
        seed=int(seed),
    )
    seen_modes = tuple(
        JumpFilterParameters(*levels.values(cell), stream.jump_variance)
        for cell in single_factor_training_cells()
    )
    imm_switch, imm_audit = _select_imm(
        fit_tape,
        modes=seen_modes,
        selection=selection,
        seed=int(seed),
    )
    selection_audit = pd.DataFrame(
        adaptation_audit + fixed_audit + total_audit + imm_audit
    )
    if "test" in " ".join(selection_audit.columns).lower():
        raise RuntimeError("selection audit unexpectedly contains a test field")

    release_truth = np.asarray(test_tape.jump_flags, dtype=np.float64)
    traces: dict[str, Any] = {
        LEARNED: run_fast_slow_exchange_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            initial=initial,
            adaptation=adaptation,
            bounds=bounds,
        ),
        ORACLE_EVENT: run_fast_slow_exchange_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            initial=initial,
            adaptation=adaptation,
            bounds=bounds,
            release_override=release_truth,
        ),
        ORACLE_QR: run_fast_slow_exchange_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            initial=initial,
            adaptation=adaptation,
            bounds=bounds,
            process_override=test_tape.process_variance,
            observation_override=test_tape.observation_variance,
        ),
        ORACLE_BOTH: run_fast_slow_exchange_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            initial=initial,
            adaptation=adaptation,
            bounds=bounds,
            release_override=release_truth,
            process_override=test_tape.process_variance,
            observation_override=test_tape.observation_variance,
        ),
        TOTAL_VARIANCE: run_total_variance_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            initial=initial,
            q_fraction=q_fraction,
            update=total_update,
            bounds=bounds,
        ),
        SEEN_IMM: run_imm_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            modes=seen_modes,
            mode_switch_probability=imm_switch,
        ),
        FIXED: run_fixed_jump_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            parameters=fixed_parameters,
        ),
        DYNAMIC_ORACLE: run_oracle_filter(
            test_tape.observations,
            sequence_ids=test_tape.sequence_ids,
            hazard=test_tape.hazard,
            process_variance=test_tape.process_variance,
            observation_variance=test_tape.observation_variance,
            jump_variance=stream.jump_variance,
        ),
    }
    if tuple(traces) != METHODS:
        raise RuntimeError("executed method order differs from the registered panel")

    analysis = config["analysis"]
    windows = tuple(map(int, analysis["transition_windows"]))
    late = int(analysis["late_window"])
    seed_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for method, trace in traces.items():
        seed_rows.append(
            _trace_metrics(method, trace, test_tape, seed=int(seed), bounds=bounds)
        )
        block_rows.extend(
            _block_rows(method, trace, test_tape, seed=int(seed), late=late)
        )
        event_rows.extend(
            _event_rows(
                method,
                trace,
                test_tape,
                seed=int(seed),
                windows=windows,
            )
        )
        regime_rows.extend(
            _regime_rows(
                method,
                trace,
                test_tape,
                seed=int(seed),
                windows=windows,
            )
        )

    metadata = {
        "seed": int(seed),
        "fit_tape_digest": fit_tape.digest,
        "test_tape_digest": test_tape.digest,
        "fit_cells": list(single_factor_training_cells()),
        "test_cells": list(heldout_composition_cells()),
        "selected_adaptation": asdict(adaptation),
        "selected_fixed": asdict(fixed_parameters),
        "selected_total_variance": {
            "q_fraction": q_fraction,
            "update": asdict(total_update),
        },
        "selected_imm_switch": imm_switch,
        "n_fit_samples": int(len(fit_tape.observations)),
        "n_test_samples": int(len(test_tape.observations)),
        "n_test_jumps": int(np.sum(test_tape.jump_flags)),
        "oracle_event_timing": "post_observation_action_only",
        "claim_upgrade_allowed": False,
    }
    return (
        pd.DataFrame(seed_rows),
        pd.DataFrame(block_rows),
        pd.DataFrame(event_rows),
        pd.DataFrame(regime_rows),
        selection_audit,
        metadata,
    )


def _paired_gain(
    frame: pd.DataFrame,
    *,
    baseline: str,
    comparator: str,
    metric: str,
) -> pd.Series:
    pivot = frame.pivot(index="seed", columns="method", values=metric)
    return pivot[baseline] - pivot[comparator]


def _window_seed_means(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    selected = frame[frame["window"] == int(window)]
    return (
        selected.groupby(["seed", "method"], as_index=False)[["nll", "latent_mse"]]
        .mean()
        .rename(columns={"nll": "window_nll", "latent_mse": "window_latent_mse"})
    )


def _comparison_payload(values: pd.Series) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "positive_seeds": int(np.sum(values > 0.0)),
        "n_seeds": int(len(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "values_by_seed": {str(int(seed)): float(value) for seed, value in values.items()},
    }


def summarize(
    seed_metrics: pd.DataFrame,
    block_metrics: pd.DataFrame,
    event_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected = {(int(seed), method) for seed in config["seeds"] for method in METHODS}
    observed = set(
        zip(seed_metrics["seed"].astype(int), seed_metrics["method"], strict=True)
    )
    if observed != expected:
        raise ValueError("seed metrics do not contain the complete paired method panel")
    comparisons: list[dict[str, Any]] = []
    for comparator in (ORACLE_EVENT, ORACLE_QR, ORACLE_BOTH):
        for metric in ("overall_nll", "latent_mse"):
            values = _paired_gain(
                seed_metrics,
                baseline=LEARNED,
                comparator=comparator,
                metric=metric,
            )
            comparisons.append(
                {
                    "contrast": f"{LEARNED}_minus_{comparator}",
                    "endpoint": metric,
                    **_comparison_payload(values),
                }
            )
    for baseline in (TOTAL_VARIANCE, SEEN_IMM, FIXED):
        for metric in ("overall_nll", "latent_mse"):
            values = _paired_gain(
                seed_metrics,
                baseline=baseline,
                comparator=LEARNED,
                metric=metric,
            )
            comparisons.append(
                {
                    "contrast": f"{baseline}_minus_{LEARNED}",
                    "endpoint": metric,
                    **_comparison_payload(values),
                }
            )

    comparison_frame = pd.DataFrame(comparisons).drop(columns="values_by_seed")
    required = int(config["analysis"]["required_positive_seeds"])
    event_window = max(
        value
        for value in map(int, config["analysis"]["transition_windows"])
        if value <= 8
    )
    event_seed = _window_seed_means(event_metrics, window=event_window)
    event_nll_gain = _paired_gain(
        event_seed,
        baseline=LEARNED,
        comparator=ORACLE_EVENT,
        metric="window_nll",
    )
    event_mse_gain = _paired_gain(
        event_seed,
        baseline=LEARNED,
        comparator=ORACLE_EVENT,
        metric="window_latent_mse",
    )
    learned_minus_oracle_qr = _paired_gain(
        seed_metrics,
        baseline=LEARNED,
        comparator=ORACLE_QR,
        metric="overall_nll",
    )
    learned_minus_oracle_qr_mse = _paired_gain(
        seed_metrics,
        baseline=LEARNED,
        comparator=ORACLE_QR,
        metric="latent_mse",
    )
    late = (
        block_metrics.groupby(["seed", "method"], as_index=False)["late_nll"]
        .mean()
        .rename(columns={"late_nll": "overall_late_nll"})
    )
    late_qr_gain = _paired_gain(
        late,
        baseline=LEARNED,
        comparator=ORACLE_QR,
        metric="overall_late_nll",
    )
    cell_110 = (
        block_metrics[block_metrics["cell"] == "110"]
        .groupby(["seed", "method"], as_index=False)["nll"]
        .mean()
    )
    cell_110_qr_gain = _paired_gain(
        cell_110,
        baseline=LEARNED,
        comparator=ORACLE_QR,
        metric="nll",
    )
    both_nll_gain = _paired_gain(
        seed_metrics,
        baseline=LEARNED,
        comparator=ORACLE_BOTH,
        metric="overall_nll",
    )
    both_mse_gain = _paired_gain(
        seed_metrics,
        baseline=LEARNED,
        comparator=ORACLE_BOTH,
        metric="latent_mse",
    )

    actuator_gate = bool(
        both_nll_gain.mean()
        >= float(config["analysis"]["actuator_headroom_mcid_nll"])
        and int(np.sum(both_mse_gain > 0.0)) >= required
    )
    event_gate = bool(
        event_nll_gain.mean()
        >= float(config["analysis"]["event_path_mcid_nll"])
        and int(np.sum(event_nll_gain > 0.0)) >= required
        and int(np.sum(event_mse_gain > 0.0)) >= required
    )
    qr_gate = bool(
        max(float(learned_minus_oracle_qr.mean()), float(late_qr_gain.mean()))
        >= float(config["analysis"]["qr_path_mcid_nll"])
        and int(np.sum(learned_minus_oracle_qr_mse > 0.0)) >= required
        and int(np.sum(cell_110_qr_gain >= 0.0)) >= required
    )

    total_nll_gain = _paired_gain(
        seed_metrics,
        baseline=TOTAL_VARIANCE,
        comparator=LEARNED,
        metric="overall_nll",
    )
    imm_nll_gain = _paired_gain(
        seed_metrics,
        baseline=SEEN_IMM,
        comparator=LEARNED,
        metric="overall_nll",
    )
    total_mse_gain = _paired_gain(
        seed_metrics,
        baseline=TOTAL_VARIANCE,
        comparator=LEARNED,
        metric="latent_mse",
    )
    imm_mse_gain = _paired_gain(
        seed_metrics,
        baseline=SEEN_IMM,
        comparator=LEARNED,
        metric="latent_mse",
    )
    cell_means = block_metrics.groupby(["cell", "method"])["nll"].mean().unstack()
    no_negative_cell = bool(
        np.all(cell_means[TOTAL_VARIANCE] - cell_means[LEARNED] >= 0.0)
        and np.all(cell_means[SEEN_IMM] - cell_means[LEARNED] >= 0.0)
    )
    early_gains: dict[str, dict[str, float]] = {}
    any_early_gain = False
    for window in map(int, config["analysis"]["transition_windows"]):
        window_frame = _window_seed_means(regime_metrics, window=window)
        total_gain = _paired_gain(
            window_frame,
            baseline=TOTAL_VARIANCE,
            comparator=LEARNED,
            metric="window_nll",
        )
        imm_gain = _paired_gain(
            window_frame,
            baseline=SEEN_IMM,
            comparator=LEARNED,
            metric="window_nll",
        )
        early_gains[str(window)] = {
            "total_variance_minus_learned": float(total_gain.mean()),
            "seen_imm_minus_learned": float(imm_gain.mean()),
        }
        any_early_gain |= bool(total_gain.mean() > 0.0 and imm_gain.mean() > 0.0)
    deployable_gate = bool(
        total_nll_gain.mean() > 0.0
        and imm_nll_gain.mean() > 0.0
        and total_mse_gain.mean() > 0.0
        and imm_mse_gain.mean() > 0.0
        and int(np.sum(total_nll_gain > 0.0)) >= required
        and int(np.sum(imm_nll_gain > 0.0)) >= required
        and int(np.sum(total_mse_gain > 0.0)) >= required
        and int(np.sum(imm_mse_gain > 0.0)) >= required
        and no_negative_cell
        and any_early_gain
    )

    if not actuator_gate:
        localization = "actuator_task_pair_lacks_registered_oracle_headroom"
    elif event_gate and qr_gate:
        localization = "both_event_and_qr_inference_paths_have_headroom"
    elif event_gate:
        localization = "event_inference_path_has_headroom"
    elif qr_gate:
        localization = "qr_inference_path_has_headroom"
    else:
        localization = "oracle_interaction_or_unlocalized_headroom"

    summary = {
        "experiment": EXPERIMENT,
        "profile": PROFILE,
        "development_only": True,
        "claim_upgrade_allowed": False,
        "verdict": "inconclusive_development_only",
        "n_seeds": int(seed_metrics["seed"].nunique()),
        "methods": list(METHODS),
        "gates": {
            "actuator_headroom": actuator_gate,
            "event_path": event_gate,
            "qr_path": qr_gate,
            "deployable_advance": deployable_gate,
        },
        "localization": localization,
        "oracle_headroom": {
            "both_nll": _comparison_payload(both_nll_gain),
            "both_latent_mse": _comparison_payload(both_mse_gain),
            "event_8step_nll": _comparison_payload(event_nll_gain),
            "event_8step_latent_mse": _comparison_payload(event_mse_gain),
            "qr_overall_nll": _comparison_payload(learned_minus_oracle_qr),
            "qr_late_nll": _comparison_payload(late_qr_gain),
            "qr_cell_110_nll": _comparison_payload(cell_110_qr_gain),
        },
        "deployable_comparisons": {
            "total_variance_minus_learned_nll": _comparison_payload(total_nll_gain),
            "seen_imm_minus_learned_nll": _comparison_payload(imm_nll_gain),
            "total_variance_minus_learned_mse": _comparison_payload(total_mse_gain),
            "seen_imm_minus_learned_mse": _comparison_payload(imm_mse_gain),
            "no_negative_cell": no_negative_cell,
            "early_regime_gains": early_gains,
        },
        "interpretation_boundary": (
            "Oracle arms localize inference/action headroom only. No development "
            "outcome can upgrade Exp39, unlock Exp42, or support neural/EI claims."
        ),
    }
    return comparison_frame, summary


def _report(summary: Mapping[str, Any]) -> str:
    gates = summary["gates"]
    headroom = summary["oracle_headroom"]
    deployable = summary["deployable_comparisons"]
    lines = [
        "# Exp43 fast/slow causal decomposition probe",
        "",
        "Verdict: **INCONCLUSIVE (development-only mechanism localization)**.",
        "",
        f"Localization: `{summary['localization']}`.",
        "",
        "## Registered gates",
        "",
        f"- Actuator oracle headroom: **{'PASS' if gates['actuator_headroom'] else 'FAIL'}**.",
        f"- Event-path headroom: **{'PASS' if gates['event_path'] else 'FAIL'}**.",
        f"- Q/R-path headroom: **{'PASS' if gates['qr_path'] else 'FAIL'}**.",
        f"- Deployable advance: **{'PASS' if gates['deployable_advance'] else 'FAIL'}**.",
        "",
        "## Paired effects (positive means the named oracle/comparator is better)",
        "",
        (
            "- Learned minus oracle-both NLL: "
            f"{headroom['both_nll']['mean']:+.6f} "
            f"({headroom['both_nll']['positive_seeds']}/{headroom['both_nll']['n_seeds']} positive)."
        ),
        (
            "- Learned minus oracle-event 8-step post-jump NLL: "
            f"{headroom['event_8step_nll']['mean']:+.6f} "
            f"({headroom['event_8step_nll']['positive_seeds']}/{headroom['event_8step_nll']['n_seeds']} positive)."
        ),
        (
            "- Learned minus oracle-Q/R overall NLL: "
            f"{headroom['qr_overall_nll']['mean']:+.6f} "
            f"({headroom['qr_overall_nll']['positive_seeds']}/{headroom['qr_overall_nll']['n_seeds']} positive)."
        ),
        (
            "- Total variance minus learned NLL: "
            f"{deployable['total_variance_minus_learned_nll']['mean']:+.6f}."
        ),
        (
            "- Seen IMM minus learned NLL: "
            f"{deployable['seen_imm_minus_learned_nll']['mean']:+.6f}."
        ),
        "",
        "## Boundary",
        "",
        str(summary["interpretation_boundary"]),
        (
            "The true event is injected only after scoring its observation, so "
            "post-jump endpoints begin at the next prediction. Dynamic Q/R, event "
            "truth, and generator-supported modes are privileged diagnostics."
        ),
        "Reserved formal seeds 43100--43129 were not accessed.",
        "",
    ]
    return "\n".join(lines)


def _method_budget() -> pd.DataFrame:
    rows = [
        (LEARNED, 5, 1, 3, False, "local h/Q/R updates; learned release"),
        (
            ORACLE_EVENT,
            5,
            1,
            3,
            True,
            "true event drives post-observation release",
        ),
        (ORACLE_QR, 5, 1, 3, True, "dynamic generating Q/R drives gain"),
        (ORACLE_BOTH, 5, 1, 3, True, "both privileged paths"),
        (
            TOTAL_VARIANCE,
            10,
            1,
            2,
            False,
            "mean/variance/h/Q/R plus five local sufficient-statistic states",
        ),
        (SEEN_IMM, 12, 4, 0, True, "four generator-supported fit modes"),
        (FIXED, 2, 1, 0, False, "mean/variance states; fixed h/Q/R"),
        (DYNAMIC_ORACLE, 2, 1, 0, True, "dynamic generating h/Q/R"),
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "method",
            "persistent_scalar_state_count",
            "mode_count",
            "adaptive_scalar_updates_per_sample",
            "uses_generator_truth",
            "budget_note",
        ),
    ).assign(
        used_autograd=False,
        used_bptt=False,
        online_gradient_updates=0,
        functional_budget_matched=False,
    )


def _artifact_manifest(output: Path, *, status: str, environment: Mapping[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.name.endswith(".tmp"):
            continue
        artifacts[path.relative_to(output).as_posix()] = _sha256(path)
    source_sha256 = {relative: _sha256(PROJECT_ROOT / relative) for relative in SOURCE_FILES}
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
        "run_start_git": dict(environment["git"]),
        "finalization_git": _git_provenance(),
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_config(config)
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    environment = _environment()
    output.mkdir(parents=True, exist_ok=False)
    _atomic_text(output / "config.json", config_path.read_text(encoding="utf-8"))
    _atomic_json(output / "environment.json", environment)
    _atomic_json(
        output / "planned_conditions.json",
        {
            "methods": list(METHODS),
            "development_seeds": list(map(int, config["seeds"])),
            "reserved_formal_seeds": list(map(int, config["reserved_formal_seeds"])),
            "fit_cells": list(single_factor_training_cells()),
            "test_cells": list(heldout_composition_cells()),
            "claim_upgrade_allowed": False,
            "oracle_event_timing": "post_observation_action_only",
        },
    )
    _atomic_csv(output / "method_budget.csv", _method_budget())

    logger = logging.getLogger(f"{EXPERIMENT}.{output.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    temporary_log = output / f".run.log.{uuid.uuid4().hex}.tmp"
    handler = logging.FileHandler(temporary_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    seed_frames: list[pd.DataFrame] = []
    block_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    regime_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    fatal_error: Exception | None = None
    try:
        for seed in map(int, config["seeds"]):
            seed_dir = output / f"seed_{seed}"
            seed_dir.mkdir()
            try:
                seed_frame, block_frame, event_frame, regime_frame, audit, metadata = run_seed(
                    config, seed
                )
                _atomic_csv(seed_dir / "seed_metrics.csv", seed_frame)
                _atomic_csv(seed_dir / "block_metrics.csv", block_frame)
                _atomic_csv(seed_dir / "event_window_metrics.csv", event_frame)
                _atomic_csv(seed_dir / "regime_window_metrics.csv", regime_frame)
                _atomic_csv(seed_dir / "selection_audit.csv", audit)
                _atomic_json(seed_dir / "metadata.json", metadata)
                _atomic_json(seed_dir / "status.json", {"seed": seed, "status": "complete"})
                seed_frames.append(seed_frame)
                block_frames.append(block_frame)
                event_frames.append(event_frame)
                regime_frames.append(regime_frame)
                selection_frames.append(audit)
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

        _atomic_json(output / "failures.json", failures)
        _atomic_csv(
            output / "failed_seeds.csv",
            pd.DataFrame(failures, columns=("seed", "status", "error_type", "error")),
        )
        if failures or len(seed_frames) != len(config["seeds"]):
            fatal_error = RuntimeError("one or more Exp43 development seeds failed")
        else:
            seed_metrics = pd.concat(seed_frames, ignore_index=True)
            block_metrics = pd.concat(block_frames, ignore_index=True)
            event_metrics = pd.concat(event_frames, ignore_index=True)
            regime_metrics = pd.concat(regime_frames, ignore_index=True)
            selection_audit = pd.concat(selection_frames, ignore_index=True)
            comparisons, summary = summarize(
                seed_metrics,
                block_metrics,
                event_metrics,
                regime_metrics,
                config=config,
            )
            _atomic_csv(output / "seed_metrics.csv", seed_metrics)
            _atomic_csv(output / "block_metrics.csv", block_metrics)
            _atomic_csv(output / "event_window_metrics.csv", event_metrics)
            _atomic_csv(output / "regime_window_metrics.csv", regime_metrics)
            _atomic_csv(output / "selection_audit.csv", selection_audit)
            _atomic_csv(output / "comparisons.csv", comparisons)
            _atomic_json(output / "summary.json", summary)
            _atomic_text(output / "report.md", _report(summary))
    except Exception as error:
        fatal_error = error
        logger.exception("Exp43 execution failed")
    finally:
        status = "failed" if fatal_error is not None else "complete"
        status_payload: dict[str, Any] = {
            "status": status,
            "development_only": True,
            "claim_upgrade_allowed": False,
            "n_planned_seeds": len(config["seeds"]),
            "n_complete_seeds": len(seed_frames),
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
        temporary_log.replace(output / "run.log")
        _atomic_json(
            output / "manifest.json",
            _artifact_manifest(output, status=status, environment=environment),
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
