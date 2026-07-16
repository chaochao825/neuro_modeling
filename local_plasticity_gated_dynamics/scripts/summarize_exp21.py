"""Publish the scoped Exp21 full-trajectory seed audit.

Exp21 has one registered raw row per independent seed.  This summarizer keeps
failed and invalid seeds in the raw snapshot, uses only the trial-reset
receiver for primary claims, and writes standalone artifacts without editing
the historical project-wide summary or report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp21_belief_ei_full_trajectory"
CONDITION = "md_combined_intact_full_trajectory"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/formal/exp21_belief_ei_full_trajectory.json"
DEFAULT_PREFIX = "exp21_belief_ei_full_trajectory_formal"
TERMINAL_RUN_STATUS = {"complete", "complete_with_failures"}
ROW_STATUS = {"complete", "failed", "invalid"}
GAIN_FAMILY = "Holm(exp21_trial_reset_seed_sign_family)"
PROTOCOL_VERSION = "exp21_v2"
TRAINING_ALGORITHM = (
    "md_filtered_belief_full_substep_controlled_affine_koopman_audit_v2"
)
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ENVIRONMENT_PACKAGES = (
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "torch",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain a non-empty JSONL object stream")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_sha256(environment: Mapping[str, Any]) -> str:
    """Validate and hash the declared Python 3.11 scientific stack."""

    if not isinstance(environment, Mapping):
        raise ValueError("formal run environment must be a JSON object")
    python = environment.get("python")
    platform = environment.get("platform")
    executable = environment.get("executable")
    packages = environment.get("packages")
    if not isinstance(python, str) or re.match(r"^3\.11\.", python) is None:
        raise ValueError("formal run environment must use Python 3.11")
    if not isinstance(platform, str) or not platform.strip():
        raise ValueError("formal run environment lacks platform provenance")
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("formal run environment lacks executable provenance")
    if not isinstance(packages, Mapping):
        raise ValueError("formal run environment lacks package provenance")
    missing = [name for name in _ENVIRONMENT_PACKAGES if not packages.get(name)]
    if missing:
        raise ValueError(f"formal run environment lacks packages: {missing}")
    payload = {
        "python": python,
        "platform": platform,
        "executable": executable,
        "packages": {name: str(packages[name]) for name in _ENVIRONMENT_PACKAGES},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_publication_provenance(
    provenance: Mapping[str, object],
) -> dict[str, str]:
    if not isinstance(provenance, Mapping):
        raise ValueError("Exp21 publication provenance must be a mapping")
    commit = str(provenance.get("analysis_git_commit", ""))
    script_sha256 = str(provenance.get("analysis_script_sha256", ""))
    python = provenance.get("analysis_python")
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("Exp21 publication provenance lacks a valid Git commit")
    if _DIGEST.fullmatch(script_sha256) is None:
        raise ValueError("Exp21 publication provenance lacks a valid script digest")
    if not isinstance(python, str) or re.match(r"^3\.11\.", python) is None:
        raise ValueError("Exp21 publication analysis must use Python 3.11")
    return {
        "analysis_git_commit": commit,
        "analysis_script_sha256": script_sha256,
        "analysis_python": python,
    }


def _analysis_provenance() -> dict[str, object]:
    """Bind a formal snapshot to clean Python 3.11 analysis code."""

    if sys.version_info[:2] != (3, 11):
        raise ValueError("formal snapshot analysis must use Python 3.11")
    repository = PROJECT_ROOT.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("formal snapshot lacks Git analysis provenance") from error
    if _COMMIT.fullmatch(commit) is None or status:
        raise ValueError("formal snapshot analysis requires a clean Git commit")
    return {
        "analysis_git_commit": commit,
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "analysis_python": sys.version,
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    payload = _json(config_path)
    if not isinstance(payload, dict):
        raise ValueError("Exp21 config must be a JSON object")
    payload["config_path"] = str(config_path.resolve())
    return payload


def _planned_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    raw = config.get("seeds")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("Exp21 config seeds must be a sequence")
    seeds: list[int] = []
    for value in raw:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError("Exp21 seeds must be non-negative integers")
        seeds.append(int(value))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Exp21 seeds must be unique and non-empty")
    return tuple(seeds)


def _registered_latent_dim(config: Mapping[str, Any]) -> int:
    dynamics = config.get("trajectory_dynamics")
    if not isinstance(dynamics, Mapping):
        raise ValueError("Exp21 config lacks trajectory_dynamics")
    value = dynamics.get("latent_dim")
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError("Exp21 latent_dim must be a positive integer")
    return int(value)


def _require_formal_registration(config: Mapping[str, Any]) -> None:
    """Reject publication inputs outside the frozen Exp21 formal design."""

    if config.get("profile") != "formal":
        raise ValueError("Exp21 publication requires profile=formal")
    if _planned_seeds(config) != tuple(range(30)):
        raise ValueError(
            "Exp21 publication requires the 30 registered independent seeds"
        )
    if _registered_latent_dim(config) != 4:
        raise ValueError(
            "Exp21 publication requires the registered latent dimension d=4"
        )


def _thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    raw = config.get("registered_claim_thresholds")
    required = {
        "maximum_rollout_normalized_rmse",
        "minimum_controlled_gain",
        "minimum_perturbation_eligible_fraction",
        "maximum_normal_endpoint_ratio",
        "maximum_projected_normal_log_growth_rate",
        "maximum_normal_vs_tangent_log_ratio",
    }
    if not isinstance(raw, Mapping) or not required <= set(raw):
        raise ValueError("Exp21 config lacks registered claim thresholds")
    result: dict[str, float] = {}
    for name in required:
        value = raw[name]
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError(f"{name} must be a finite scalar")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name} must be finite")
        result[name] = number
    return result


def _expected_run_config(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "seed": int(seed),
        **dict(config),
        "experiment_protocol_version": PROTOCOL_VERSION,
        "training_algorithm": TRAINING_ALGORITHM,
        "used_autograd": False,
        "used_bptt": False,
        "recurrent_learning": False,
        "full_trajectory_model": True,
        "full_trajectory_lds": False,
    }


def _configs_match(observed: object, expected: Mapping[str, Any]) -> bool:
    if not isinstance(observed, Mapping):
        return False
    left, right = dict(observed), dict(expected)
    left_path = left.pop("config_path", None)
    right_path = right.pop("config_path", None)
    if left != right:
        return False
    if left_path is None or right_path is None:
        return left_path == right_path
    return Path(str(left_path)).name == Path(str(right_path)).name


def _latest_terminal_attempt(
    results_root: Path,
    *,
    seed: int,
    expected_config: Mapping[str, Any],
) -> Path:
    seed_root = results_root / "runs" / EXPERIMENT / f"seed_{seed:04d}"
    for attempt in sorted(seed_root.glob("*"), reverse=True):
        config_path = attempt / "config.json"
        status_path = attempt / "status.json"
        if (
            not config_path.is_file()
            or not status_path.is_file()
            or not _configs_match(_json(config_path), expected_config)
        ):
            continue
        status = _json(status_path)
        if isinstance(status, Mapping) and status.get("status") in TERMINAL_RUN_STATUS:
            return attempt
    raise FileNotFoundError(
        f"no terminal exact-config Exp21 attempt exists for seed={seed}"
    )


def _validate_attempt(
    attempt: Path,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    required = (
        "config.json",
        "environment.json",
        "manifest.json",
        "metrics.jsonl",
        "planned_conditions.json",
        "status.json",
    )
    missing = [name for name in required if not (attempt / name).is_file()]
    if missing:
        raise FileNotFoundError(f"seed {seed} attempt lacks artifacts: {missing}")
    if not _configs_match(
        _json(attempt / "config.json"), _expected_run_config(config, seed)
    ):
        raise ValueError(f"seed {seed} run config differs from Exp21 config")

    status = _json(attempt / "status.json")
    manifest = _json(attempt / "manifest.json")
    environment = _json(attempt / "environment.json")
    environment_sha256 = _environment_sha256(environment)
    git = environment.get("git")
    if (
        not isinstance(status, Mapping)
        or status.get("status") not in TERMINAL_RUN_STATUS
        or not isinstance(manifest, Mapping)
        or manifest.get("status") != status.get("status")
        or manifest.get("experiment") != EXPERIMENT
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("profile") != "formal"
    ):
        raise ValueError(f"seed {seed} status/manifest contract is malformed")
    if (
        not isinstance(git, Mapping)
        or _COMMIT.fullmatch(str(git.get("commit", ""))) is None
        or git.get("dirty") is not False
    ):
        raise ValueError(f"seed {seed} formal attempt must have a clean Git receipt")

    planned = _json(attempt / "planned_conditions.json")
    if not isinstance(planned, list) or planned != [
        {
            "condition_index": 0,
            "condition": CONDITION,
            "model_family": "frozen_high_rank_dale_ei",
            "controller_mode": "combined",
            "belief_intervention": "none",
            "trajectory_sampling": "euler_substep",
        }
    ]:
        raise ValueError(f"seed {seed} planned Exp21 cell is malformed")
    rows = _records(attempt / "metrics.jsonl")
    if len(rows) != 1:
        raise ValueError(f"seed {seed} must retain exactly one Exp21 row")
    row = rows[0]
    if (
        row.get("experiment") != EXPERIMENT
        or int(row.get("seed", -1)) != seed
        or row.get("condition") != CONDITION
        or row.get("status") not in ROW_STATUS
        or row.get("run_id") != manifest.get("run_id")
    ):
        raise ValueError(f"seed {seed} raw Exp21 row identity is malformed")
    expected_run_status = (
        "complete" if row["status"] == "complete" else "complete_with_failures"
    )
    if status.get("status") != expected_run_status:
        raise ValueError(f"seed {seed} run and row status disagree")
    return {
        **row,
        "attempt_path": str(attempt.resolve()),
        "run_status": str(status["status"]),
        "run_git_commit": str(git["commit"]),
        "environment_sha256": environment_sha256,
    }


def collect_registered_runs(
    results_root: str | Path, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Collect one terminal, exact-config row for every registered seed."""

    _require_formal_registration(config)
    seeds = _planned_seeds(config)
    root = Path(results_root)
    rows = []
    for seed in seeds:
        attempt = _latest_terminal_attempt(
            root,
            seed=seed,
            expected_config=_expected_run_config(config, seed),
        )
        rows.append(_validate_attempt(attempt, config=config, seed=seed))
    raw = pd.DataFrame(rows).sort_values("seed", kind="stable").reset_index(drop=True)
    validate_raw_frame(raw, config)
    if raw["run_git_commit"].astype(str).nunique(dropna=False) != 1:
        raise ValueError("Exp21 formal seeds use mixed Git commits")
    if raw["environment_sha256"].astype(str).nunique(dropna=False) != 1:
        raise ValueError("Exp21 formal seeds use mixed software environments")
    return raw


def _strict_bool(value: object, expected: bool) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value) is expected


def validate_raw_frame(raw: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Fail closed on identity, capability, and seed-retention contracts."""

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Exp21 raw table must be a non-empty DataFrame")
    required = {"experiment", "seed", "condition", "status"}
    if not required <= set(raw):
        missing = sorted(required - set(raw))
        raise ValueError(f"Exp21 raw table lacks columns: {missing}")
    seeds = pd.to_numeric(raw["seed"], errors="coerce")
    if not np.isfinite(seeds).all() or not np.equal(seeds, np.floor(seeds)).all():
        raise ValueError("Exp21 raw table must contain one integer row per seed")
    normalized_seeds = seeds.astype(np.int64)
    if normalized_seeds.duplicated().any():
        raise ValueError("Exp21 raw table must contain one integer row per seed")
    observed = set(seeds.astype(int).tolist())
    planned = set(_planned_seeds(config))
    if observed != planned:
        raise ValueError("Exp21 raw table does not retain every registered seed")
    if not raw["experiment"].eq(EXPERIMENT).all():
        raise ValueError("Exp21 raw table contains another experiment")
    if not raw["condition"].eq(CONDITION).all():
        raise ValueError("Exp21 raw table contains another registered cell")
    if not raw["status"].isin(ROW_STATUS).all():
        raise ValueError("Exp21 raw table contains an unknown row status")

    complete = raw.loc[raw["status"].eq("complete")]
    required_complete = {
        "statistics_unit",
        "used_autograd",
        "used_bptt",
        "recurrent_learning",
        "full_trajectory_model",
        "full_trajectory_lds",
        "preprocessing_fit_train_only",
        "operator_fit_train_only",
        "gate_fit_accessed_true_context",
        "gate_test_accessed_true_context",
        "primary_receiver_state_reset_scope",
        "trial_reset_trajectory_sequence_scope",
    }
    if len(complete) and not required_complete <= set(raw):
        raise ValueError(
            "Exp21 complete-row capability columns are missing: "
            f"{sorted(required_complete - set(raw))}"
        )
    for row in complete.to_dict(orient="records"):
        if (
            row["statistics_unit"] != "seed"
            or not _strict_bool(row["used_autograd"], False)
            or not _strict_bool(row["used_bptt"], False)
            or not _strict_bool(row["recurrent_learning"], False)
            or not _strict_bool(row["full_trajectory_model"], True)
            or not _strict_bool(row["full_trajectory_lds"], False)
            or not _strict_bool(row["preprocessing_fit_train_only"], True)
            or not _strict_bool(row["operator_fit_train_only"], True)
            or not _strict_bool(row["gate_fit_accessed_true_context"], False)
            or not _strict_bool(row["gate_test_accessed_true_context"], False)
            or row["primary_receiver_state_reset_scope"] != "every_trial_zero_state"
            or row["trial_reset_trajectory_sequence_scope"] != "trial_reset_state"
        ):
            raise ValueError("Exp21 complete row violates the primary claim scope")


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[field], errors="coerce")


def _numeric_alias(frame: pd.DataFrame, *fields: str) -> pd.Series:
    existing = [field for field in fields if field in frame]
    if not existing:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    converted: list[tuple[str, pd.Series]] = []
    for field in existing:
        values = pd.to_numeric(frame[field], errors="coerce")
        if bool((frame[field].notna() & values.isna()).any()):
            raise ValueError(f"numeric alias {field} contains non-numeric values")
        converted.append((field, values))
    result = converted[0][1].copy()
    for field, values in converted[1:]:
        overlap = result.notna() & values.notna()
        if bool(
            (
                overlap
                & ~np.isclose(
                    result,
                    values,
                    atol=1e-12,
                    rtol=1e-9,
                    equal_nan=True,
                )
            ).any()
        ):
            raise ValueError(
                f"conflicting Exp21 numeric aliases: {existing[0]} and {field}"
            )
        result = result.where(result.notna(), values)
    return result.astype(float)


def _series_alias(frame: pd.DataFrame, *fields: str, default: object = "") -> pd.Series:
    existing = [field for field in fields if field in frame]
    if not existing:
        return pd.Series(default, index=frame.index)
    result = frame[existing[0]].copy()
    for field in existing[1:]:
        values = frame[field]
        overlap = result.notna() & values.notna()
        if bool((overlap & result.ne(values)).any()):
            raise ValueError(f"conflicting Exp21 aliases: {existing[0]} and {field}")
        result = result.where(result.notna(), values)
    return result


def _true(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[field].map(lambda value: _strict_bool(value, True))


def _false(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[field].map(lambda value: _strict_bool(value, False))


def _bootstrap_interval(
    values: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    if n_bootstrap < 100:
        raise ValueError("Exp21 seed bootstrap requires at least 100 draws")
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True)
    low, high = np.quantile(np.mean(draws, axis=1), [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_p(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > np.finfo(np.float64).eps]
    if nonzero.size == 0:
        return 1.0
    positives = int(np.count_nonzero(nonzero > 0.0))
    smaller = min(positives, int(nonzero.size) - positives)
    tail = sum(math.comb(int(nonzero.size), k) for k in range(smaller + 1))
    return float(min(1.0, 2.0 * tail / (2 ** int(nonzero.size))))


def _holm(values: Sequence[float]) -> np.ndarray:
    """Holm-adjust a fixed planned family, treating missing tests as p=1."""

    p = np.asarray(values, dtype=float)
    if p.ndim != 1:
        raise ValueError("Holm p-values must be a one-dimensional planned family")
    if bool(np.isinf(p).any()):
        raise ValueError("Holm p-values must be finite or NaN")
    finite = np.isfinite(p)
    if bool(((p[finite] < 0.0) | (p[finite] > 1.0)).any()):
        raise ValueError("finite Holm p-values must lie in [0, 1]")
    planned = np.where(np.isnan(p), 1.0, p)
    adjusted = np.empty_like(planned)
    running = 0.0
    ordered = np.argsort(planned, kind="stable")
    for rank, index in enumerate(ordered):
        running = max(
            running,
            min(1.0, (len(planned) - rank) * planned[index]),
        )
        adjusted[index] = running
    return adjusted


def _v2_protocol_receipt(frame: pd.DataFrame) -> pd.Series:
    """Identify exact v2 collection provenance without defining claim validity."""

    return frame.get(
        "experiment_protocol_version",
        pd.Series("", index=frame.index),
    ).eq(PROTOCOL_VERSION) & frame.get(
        "training_algorithm",
        pd.Series("", index=frame.index),
    ).eq(TRAINING_ALGORITHM)


def _total_operator_v2_receipt(
    frame: pd.DataFrame,
    *,
    prefix: str,
) -> pd.Series:
    """Require the identifiable 19-column neutral-cue operator definition."""

    eligible = frame.get(
        f"{prefix}_total_operator_mode",
        pd.Series("", index=frame.index),
    ).eq("full_shared_neutral_cue")
    eligible &= frame.get(
        f"{prefix}_total_operator_constraint",
        pd.Series("", index=frame.index),
    ).eq("shared_neutral_cue_coefficient")
    design_rank = _numeric(frame, f"{prefix}_total_operator_design_rank")
    design_columns = _numeric(frame, f"{prefix}_total_operator_design_columns")
    unconstrained_columns = _numeric(
        frame,
        f"{prefix}_total_operator_unconstrained_columns",
    )
    eligible &= (
        design_rank.eq(19.0) & design_columns.eq(19.0) & unconstrained_columns.eq(20.0)
    )
    if "total_control_operator_mode" in frame:
        eligible &= frame["total_control_operator_mode"].eq("full_shared_neutral_cue")
    return eligible


def _perturbation_v2_receipt(
    frame: pd.DataFrame,
    *,
    prefix: str,
) -> pd.Series:
    """Require the physical-x perturbation definition and planned coverage."""

    geometry = frame.get(
        f"{prefix}_perturbation_geometry",
        pd.Series("", index=frame.index),
    ).eq("joint_state_pca_physical_x_projection_v2")
    sampled_fraction = _numeric(
        frame,
        f"{prefix}_perturbation_sampled_reference_fraction",
    )
    return geometry & np.isfinite(sampled_fraction) & (sampled_fraction >= 1.0 - 1e-12)


def _gain_rows(
    complete: pd.DataFrame,
    *,
    n_planned: int,
    thresholds: Mapping[str, float],
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    minimum = float(thresholds["minimum_controlled_gain"])
    trial_total_v2 = _total_operator_v2_receipt(complete, prefix="trial_reset")
    shared = _true(complete, "trial_reset_paired_models_share_state_pca")
    primary_scope = complete.get(
        "trial_reset_trajectory_sequence_scope",
        pd.Series("", index=complete.index),
    ).eq("trial_reset_state")

    total_gain = _numeric(
        complete, "trial_reset_total_control_rollout_gain_vs_raw_common"
    )
    total_controlled = _numeric(
        complete, "trial_reset_total_full_rollout_normalized_rmse"
    )
    total_comparator = _numeric(
        complete, "trial_reset_raw_common_rollout_normalized_rmse"
    )
    total_consistent = np.isclose(
        total_gain,
        total_comparator - total_controlled,
        atol=1e-10,
        rtol=1e-8,
    )
    total_valid = (
        trial_total_v2
        & shared
        & primary_scope
        & _true(complete, "trial_reset_total_operator_design_full_rank")
        & complete.get(
            "total_control_model_input_policy",
            pd.Series("", index=complete.index),
        ).eq("raw_receiver_sensory_plus_scalar_control_interactions")
        & np.isfinite(total_gain)
        & np.isfinite(total_controlled)
        & np.isfinite(total_comparator)
        & total_consistent
    )

    state_gain = _numeric_alias(
        complete,
        "trial_reset_population_state_affine_rollout_gain_vs_routed_common",
        "trial_reset_population_state_affine_switch_rollout_gain_vs_routed_common",
        "trial_reset_population_state_switch_rollout_gain_vs_routed_common",
    )
    state_controlled = _numeric_alias(
        complete,
        "trial_reset_routed_state_affine_rollout_normalized_rmse",
        "trial_reset_routed_state_affine_switch_rollout_normalized_rmse",
        "trial_reset_routed_state_switch_rollout_normalized_rmse",
    )
    state_comparator = _numeric(
        complete, "trial_reset_routed_common_rollout_normalized_rmse"
    )
    state_delta = _numeric_alias(
        complete,
        "trial_reset_population_state_affine_transition_delta_frobenius",
        "trial_reset_population_state_transition_delta_frobenius",
    )
    affine_bias_delta = _numeric_alias(
        complete,
        "trial_reset_population_state_affine_bias_delta_norm",
        "trial_reset_population_affine_bias_delta_norm",
    )
    exogenous_delta = _numeric_alias(
        complete,
        "trial_reset_population_state_affine_exogenous_control_delta_frobenius",
        "trial_reset_population_exogenous_control_delta_frobenius",
    )
    state_consistent = np.isclose(
        state_gain,
        state_comparator - state_controlled,
        atol=1e-10,
        rtol=1e-8,
    )
    state_valid = (
        shared
        & primary_scope
        & _series_alias(
            complete,
            "trial_reset_state_affine_operator_design_full_rank",
            "trial_reset_state_affine_switch_operator_design_full_rank",
            "trial_reset_state_switch_operator_design_full_rank",
            default=False,
        ).map(lambda value: _strict_bool(value, True))
        & _series_alias(
            complete,
            "population_state_affine_model_input_policy",
            "population_state_affine_switch_model_input_policy",
            "population_state_switch_model_input_policy",
        ).isin(
            {
                "already_routed_sensory_plus_population_gain_belief_"
                "state_and_affine_bias_switch_input_and_epoch_shared",
                "already_routed_sensory_plus_population_gain_belief_"
                "state_and_affine_switch_only",
                "already_routed_sensory_plus_population_gain_belief_"
                "state_affine_switch_only",
            }
        )
        & np.isfinite(state_gain)
        & np.isfinite(state_controlled)
        & np.isfinite(state_comparator)
        & np.isfinite(state_delta)
        & np.isfinite(affine_bias_delta)
        & np.isfinite(exogenous_delta)
        & (state_delta > 0.0)
        & np.isclose(exogenous_delta, 0.0, atol=1e-12, rtol=0.0)
        & state_consistent
    )

    specifications = (
        (
            "trial_reset_total_control_gain_vs_raw_common",
            "trial_reset_total_full_vs_raw_common",
            total_gain,
            total_valid,
            (
                "raw receiver sensory input is shared; positive gain is raw-common "
                "rollout RMSE minus total scalar-control predictor RMSE"
            ),
        ),
        (
            "trial_reset_population_state_affine_gain_vs_routed_common",
            "trial_reset_routed_state_affine_vs_routed_common",
            state_gain,
            state_valid,
            (
                "routed sensory input and train PCA are shared; state transition "
                "A and affine bias may depend on population-gain belief, while "
                "exogenous input and epoch coefficients remain shared"
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for index, (proposition, comparison, values, valid, scope) in enumerate(
        specifications
    ):
        selected = values.loc[valid].to_numpy(dtype=float)
        if selected.size:
            estimate = float(np.mean(selected))
            low, high = _bootstrap_interval(
                selected,
                n_bootstrap=n_bootstrap,
                seed=210_021 + index * 104_729,
            )
            p_value = _exact_sign_p(selected - minimum)
        else:
            estimate = low = high = p_value = float("nan")
        p_values.append(p_value)
        rows.append(
            {
                "experiment": EXPERIMENT,
                "proposition": proposition,
                "comparison": comparison,
                "effect_definition": "comparator_minus_controlled_rollout_rmse",
                "inference_unit": "seed",
                "multiplicity_family": GAIN_FAMILY,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "p_value": p_value,
                "holm_adjusted_p": float("nan"),
                "n_complete": int(len(complete)),
                "n_eligible": int(selected.size),
                "n_planned": int(n_planned),
                "threshold": f"mean gain CI > {minimum}",
                "conclusion": "inconclusive",
                "claim_scope": scope,
            }
        )
    for row, adjusted in zip(rows, _holm(p_values), strict=True):
        row["holm_adjusted_p"] = float(adjusted)
        full = (
            row["n_complete"] == row["n_planned"]
            and row["n_eligible"] == row["n_planned"]
        )
        if full and row["ci_low"] > minimum and adjusted < 0.05:
            row["conclusion"] = "support"
        elif full and row["ci_high"] < minimum and adjusted < 0.05:
            row["conclusion"] = "oppose"
    return rows


def _audit_row(
    *,
    proposition: str,
    comparison: str,
    complete: pd.DataFrame,
    eligible: pd.Series,
    passed: pd.Series,
    estimate: float,
    threshold: str,
    scope: str,
    n_planned: int,
) -> dict[str, Any]:
    n_eligible = int(np.count_nonzero(eligible))
    full = len(complete) == n_planned and n_eligible == n_planned
    conclusion = (
        "support"
        if full and bool(passed.loc[eligible].all())
        else "oppose"
        if full
        else "inconclusive"
    )
    return {
        "experiment": EXPERIMENT,
        "proposition": proposition,
        "comparison": comparison,
        "effect_definition": scope,
        "inference_unit": "seed",
        "multiplicity_family": "none_registered_threshold",
        "estimate": estimate,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float("nan"),
        "holm_adjusted_p": float("nan"),
        "n_complete": int(len(complete)),
        "n_eligible": n_eligible,
        "n_planned": int(n_planned),
        "threshold": threshold,
        "conclusion": conclusion,
        "claim_scope": scope,
    }


def _audit_rows(
    complete: pd.DataFrame,
    *,
    n_planned: int,
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    trial_total_v2 = _total_operator_v2_receipt(complete, prefix="trial_reset")
    trial_perturbation_v2 = _perturbation_v2_receipt(
        complete,
        prefix="trial_reset",
    )
    closure = _numeric(complete, "trial_reset_total_full_rollout_normalized_rmse")
    closure_limit = float(thresholds["maximum_rollout_normalized_rmse"])
    closure_eligible = (
        trial_total_v2
        & np.isfinite(closure)
        & _true(complete, "trial_reset_total_operator_design_full_rank")
        & _true(complete, "trial_reset_paired_models_share_state_pca")
    )
    closure_pass = closure <= closure_limit
    closure_estimate = (
        float(np.max(closure.loc[closure_eligible]))
        if bool(closure_eligible.any())
        else float("nan")
    )

    perturbation_complete = complete.get(
        "trial_reset_perturbation_status",
        pd.Series("", index=complete.index),
    ).eq("complete")
    eligible_fraction = _numeric(
        complete, "trial_reset_perturbation_eligible_reference_fraction"
    )
    endpoint_ratio = _numeric(
        complete, "trial_reset_perturbation_normal_endpoint_ratio_maximum"
    )
    growth = _numeric(
        complete,
        "trial_reset_perturbation_maximum_projected_finite_time_normal_log_growth_rate",
    )
    relative = _numeric(
        complete,
        "trial_reset_perturbation_normal_vs_tangent_log_ratio_median",
    )
    replay = _numeric(
        complete, "trial_reset_perturbation_baseline_replay_max_abs_error"
    )
    replay_tolerance = _numeric(
        complete, "trial_reset_perturbation_baseline_replay_tolerance"
    )
    planned_references = _numeric(
        complete, "trial_reset_perturbation_planned_reference_count"
    )
    sampled_references = _numeric(
        complete, "trial_reset_perturbation_sampled_reference_count"
    )
    candidate_references = _numeric(
        complete, "trial_reset_perturbation_candidate_reference_count"
    )
    perturbation_eligible = (
        trial_total_v2
        & trial_perturbation_v2
        & perturbation_complete
        & np.isfinite(eligible_fraction)
        & np.isfinite(endpoint_ratio)
        & np.isfinite(growth)
        & np.isfinite(relative)
        & np.isfinite(replay)
        & np.isfinite(replay_tolerance)
        & np.isfinite(planned_references)
        & np.isfinite(sampled_references)
        & np.isfinite(candidate_references)
        & (planned_references > 0)
    )
    perturbation_pass = (
        (eligible_fraction >= thresholds["minimum_perturbation_eligible_fraction"])
        & (endpoint_ratio < thresholds["maximum_normal_endpoint_ratio"])
        & (growth < thresholds["maximum_projected_normal_log_growth_rate"])
        & (relative < thresholds["maximum_normal_vs_tangent_log_ratio"])
        & (replay <= replay_tolerance)
        & (sampled_references == planned_references)
        & (candidate_references >= planned_references)
    )
    perturbation_estimate = (
        float(np.max(relative.loc[perturbation_eligible]))
        if bool(perturbation_eligible.any())
        else float("nan")
    )

    fixed_scope = complete.get(
        "attractor_anchor_fit_scope", pd.Series("", index=complete.index)
    ).eq("training_trajectory_only")
    separated = _true(complete, "attractor_separated_convergence")
    contracted = _true(complete, "attractor_both_conditions_contract")
    combined = _true(complete, "attractor_population_gain") & _true(
        complete, "attractor_pathway_gating"
    )
    scaled_separation = _numeric(
        complete, "attractor_centroid_separation_over_initial_dispersion"
    )
    attractor_eligible = fixed_scope & combined & np.isfinite(scaled_separation)
    attractor_pass = separated & contracted
    attractor_estimate = (
        float(np.min(scaled_separation.loc[attractor_eligible]))
        if bool(attractor_eligible.any())
        else float("nan")
    )

    return [
        _audit_row(
            proposition="trial_reset_full_trajectory_closure",
            comparison="trial_reset_total_full_registered_threshold",
            complete=complete,
            eligible=closure_eligible,
            passed=closure_pass,
            estimate=closure_estimate,
            threshold=f"rollout_normalized_rmse <= {closure_limit} every seed",
            scope=(
                "controlled affine predictor rollout conditioned on observed future "
                "exogenous controls; not an autonomous or probabilistic LDS score"
            ),
            n_planned=n_planned,
        ),
        _audit_row(
            proposition=("trial_reset_nonlinear_normal_recovery_relative_to_tangent"),
            comparison="frozen_physical_network_perturbation_audit",
            complete=complete,
            eligible=perturbation_eligible,
            passed=perturbation_pass,
            estimate=perturbation_estimate,
            threshold=(
                "eligible_fraction>="
                f"{thresholds['minimum_perturbation_eligible_fraction']}; "
                f"normal_endpoint_ratio<{thresholds['maximum_normal_endpoint_ratio']}; "
                "projected_normal_log_growth_rate<"
                f"{thresholds['maximum_projected_normal_log_growth_rate']}; "
                "normal_vs_tangent_log_ratio<"
                f"{thresholds['maximum_normal_vs_tangent_log_ratio']}; "
                "exact baseline replay and full planned reference coverage"
            ),
            scope=(
                "finite-amplitude recovery normal to the physical-x projection "
                "of the train-fitted joint latent subspace; not proof of a joint "
                "manifold, global or asymptotic Lyapunov stability, or an attractor"
            ),
            n_planned=n_planned,
        ),
        _audit_row(
            proposition="fixed_drive_separated_endpoint_probe",
            comparison="combined_actuator_fixed_drive_training_anchor_probe",
            complete=complete,
            eligible=attractor_eligible,
            passed=attractor_pass,
            estimate=attractor_estimate,
            threshold=(
                "both_conditions_contract=true and separated_convergence=true "
                "for every seed"
            ),
            scope=(
                "narrow finite-horizon combined-actuator endpoint-separation "
                "sanity probe on training-derived anchors; does not support gate "
                "causality, low-dimensional or shared-manifold dynamics, or "
                "attractor claims"
            ),
            n_planned=n_planned,
        ),
    ]


def summarize_formal_runs(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Classify the five registered Exp21 propositions at the seed level."""

    validate_raw_frame(raw, config)
    latent_dim = _registered_latent_dim(config)
    thresholds = _thresholds(config)
    planned = _planned_seeds(config)
    complete = (
        raw.loc[raw["status"].eq("complete")]
        .sort_values("seed", kind="stable")
        .reset_index(drop=True)
    )
    rows = [
        *_gain_rows(
            complete,
            n_planned=len(planned),
            thresholds=thresholds,
            n_bootstrap=n_bootstrap,
        ),
        *_audit_rows(
            complete,
            n_planned=len(planned),
            thresholds=thresholds,
        ),
    ]
    summary = pd.DataFrame(rows)
    summary["registered_latent_dim"] = latent_dim
    summary["latent_dimension_selection"] = "fixed_registered_no_nested_cv"
    summary["claim_scope"] = summary["claim_scope"].map(
        lambda value: (
            f"{value}; registered d={latent_dim} mechanism audit without "
            "nested-CV latent-dimension selection"
        )
    )
    failure_count = int(np.count_nonzero(~raw["status"].eq("complete")))
    summary["retained_failed_or_invalid_seed_count"] = failure_count
    return summary


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].map(
            lambda value: (
                json.dumps(value, sort_keys=True, ensure_ascii=False)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        )
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            return ""
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, rule, *body))


def _plot(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    png: Path,
    pdf: Path,
) -> None:
    effect = summary.loc[summary["multiplicity_family"].eq(GAIN_FAMILY)]
    audits = summary.loc[summary["multiplicity_family"].eq("none_registered_threshold")]
    colors = {
        "support": "#0072B2",
        "oppose": "#D55E00",
        "inconclusive": "#7F7F7F",
    }
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(10.2, 4.2),
            gridspec_kw={"width_ratios": (1.25, 1.0)},
            constrained_layout=True,
        )
        axes[0].text(
            -0.12,
            1.02,
            "a",
            transform=axes[0].transAxes,
            fontweight="bold",
        )
        if len(effect):
            y = np.arange(len(effect))
            labels = [
                "Total control vs raw common",
                "State + affine switch vs routed common",
            ][: len(effect)]
            for index, row in enumerate(effect.itertuples()):
                center = float(row.estimate)
                low = float(row.ci_low)
                high = float(row.ci_high)
                if np.isfinite([center, low, high]).all():
                    axes[0].errorbar(
                        center,
                        y[index],
                        xerr=np.asarray(
                            [[max(0.0, center - low)], [max(0.0, high - center)]]
                        ),
                        fmt="o",
                        color=colors[str(row.conclusion)],
                        capsize=3,
                        markersize=5,
                    )
            axes[0].set_yticks(y, labels)
            axes[0].set_ylim(len(effect) - 0.5, -0.5)
        else:
            axes[0].text(
                0.5,
                0.5,
                "No eligible seed-level gains",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
        axes[0].axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
        axes[0].set_xlabel("Comparator - controlled rollout RMSE (95% seed CI)")
        axes[0].spines[["top", "right"]].set_visible(False)

        axes[1].text(
            -0.12,
            1.02,
            "b",
            transform=axes[1].transAxes,
            fontweight="bold",
        )
        audit_labels = {
            "trial_reset_full_trajectory_closure": "Closure",
            "trial_reset_nonlinear_normal_recovery_relative_to_tangent": (
                "Normal vs tangent"
            ),
            "fixed_drive_separated_endpoint_probe": "Fixed-drive probe",
        }
        if len(audits):
            y = np.arange(len(audits))
            x_map = {"oppose": -1.0, "inconclusive": 0.0, "support": 1.0}
            for index, row in enumerate(audits.itertuples()):
                conclusion = str(row.conclusion)
                axes[1].scatter(
                    x_map[conclusion],
                    y[index],
                    marker="s",
                    s=45,
                    color=colors[conclusion],
                    zorder=3,
                )
            axes[1].set_yticks(
                y,
                [
                    audit_labels.get(str(item), str(item))
                    for item in audits["proposition"]
                ],
            )
            axes[1].set_ylim(len(audits) - 0.5, -0.5)
        axes[1].set_xticks([-1.0, 0.0, 1.0], ["Oppose", "Inconclusive", "Support"])
        axes[1].set_xlim(-1.35, 1.35)
        axes[1].grid(axis="x", color="0.9", linewidth=0.7)
        axes[1].set_xlabel("Registered seed-level conclusion")
        axes[1].spines[["top", "right", "left"]].set_visible(False)
        axes[1].tick_params(axis="y", length=0)
        axes[1].text(
            0.0,
            -0.22,
            f"Retained failed/invalid seeds: "
            f"{int(np.count_nonzero(~raw['status'].eq('complete')))}",
            transform=axes[1].transAxes,
            fontsize=8,
        )
        figure.savefig(
            png,
            dpi=300,
            bbox_inches="tight",
            metadata={"Software": "Exp21 deterministic matplotlib summary"},
        )
        figure.savefig(
            pdf,
            bbox_inches="tight",
            metadata={
                "Creator": "Exp21 deterministic matplotlib summary",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)


def write_snapshot_artifacts(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    prefix: str = DEFAULT_PREFIX,
    n_bootstrap: int = 5000,
    publication_provenance: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write deterministic, standalone Exp21 raw/summary/report/figure files."""

    if prefix == DEFAULT_PREFIX:
        _require_formal_registration(config)
        if publication_provenance is None:
            raise ValueError(
                "the formal Exp21 prefix requires validated publication provenance"
            )
    validate_raw_frame(raw, config)
    raw_sorted = raw.sort_values("seed", kind="stable").reset_index(drop=True)
    summary = summarize_formal_runs(raw_sorted, config, n_bootstrap=n_bootstrap)
    provenance_lines = [
        "- Publication provenance: unvalidated non-formal helper snapshot."
    ]
    if publication_provenance is not None:
        validated_provenance = _validate_publication_provenance(publication_provenance)
        if (
            "run_git_commit" not in raw_sorted
            or "environment_sha256" not in raw_sorted
            or raw_sorted["run_git_commit"].astype(str).nunique(dropna=False) != 1
            or raw_sorted["environment_sha256"].astype(str).nunique(dropna=False) != 1
        ):
            raise ValueError("Exp21 formal raw provenance is incomplete")
        for field, value in validated_provenance.items():
            summary[field] = value
        provenance_lines = [
            (
                "- Raw-run commit: "
                f"`{raw_sorted['run_git_commit'].astype(str).iloc[0]}`."
            ),
            (
                "- Raw software-environment SHA-256: "
                f"`{raw_sorted['environment_sha256'].astype(str).iloc[0]}`."
            ),
            (
                "- Analysis commit: "
                f"`{validated_provenance['analysis_git_commit']}`; "
                "analysis-script SHA-256: "
                f"`{validated_provenance['analysis_script_sha256']}`."
            ),
            f"- Analysis Python: `{validated_provenance['analysis_python']}`.",
        ]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": target / f"{prefix}_raw.csv.gz",
        "summary": target / f"{prefix}_summary.csv",
        "report": target / f"{prefix}_report.md",
        "png": target / f"{prefix}.png",
        "pdf": target / f"{prefix}.pdf",
    }
    _csv_safe(raw_sorted).to_csv(
        paths["raw"],
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    _csv_safe(summary).to_csv(paths["summary"], index=False)

    failures = raw_sorted.loc[~raw_sorted["status"].eq("complete")].copy()
    failure_columns = [
        column
        for column in ("seed", "status", "error_type", "error", "reason")
        if column in failures
    ]
    claim_ineligible = summary.loc[
        summary["n_eligible"].astype(int) < summary["n_planned"].astype(int),
        [
            "proposition",
            "comparison",
            "n_complete",
            "n_eligible",
            "n_planned",
            "conclusion",
        ],
    ]
    complete_rows = raw_sorted.loc[raw_sorted["status"].eq("complete")]
    v2_protocol_count = int(np.count_nonzero(_v2_protocol_receipt(complete_rows)))
    trial_operator_count = int(
        np.count_nonzero(
            _total_operator_v2_receipt(complete_rows, prefix="trial_reset")
        )
    )
    trial_perturbation_count = int(
        np.count_nonzero(_perturbation_v2_receipt(complete_rows, prefix="trial_reset"))
    )
    episode_sensitivity_count = int(
        np.count_nonzero(
            _total_operator_v2_receipt(
                complete_rows,
                prefix="episode_continuous",
            )
            & _perturbation_v2_receipt(
                complete_rows,
                prefix="episode_continuous",
            )
        )
    )
    report = [
        "# Exp21: belief-controlled high-rank E/I full-trajectory audit",
        "",
        (
            "This is a standalone snapshot. It does not modify the historical "
            "project-wide `results/summary.csv` or `results/report.md`."
        ),
        "",
        f"- Registered independent seeds: {len(_planned_seeds(config))}",
        f"- Complete seeds: {int(raw_sorted['status'].eq('complete').sum())}",
        (
            f"- Retained failed/invalid seeds: {len(failures)} "
            "(run-level execution status)"
        ),
        (
            "- Claim-level scientifically ineligible conclusion rows: "
            f"{len(claim_ineligible)} of {len(summary)}"
        ),
        (
            "- Exact v2 collection-provenance seeds: "
            f"{v2_protocol_count} of {len(_planned_seeds(config))}"
        ),
        (
            "- Trial-reset identifiable total-operator receipts: "
            f"{trial_operator_count} of {len(_planned_seeds(config))}"
        ),
        (
            "- Trial-reset physical-x perturbation receipts: "
            f"{trial_perturbation_count} of {len(_planned_seeds(config))}"
        ),
        (
            "- Episode-continuous v2 sensitivity receipts: "
            f"{episode_sensitivity_count} of {len(_planned_seeds(config))} "
            "(reported only; never gates trial-reset conclusions)"
        ),
        *provenance_lines,
        (
            f"- Registered latent dimension: d={_registered_latent_dim(config)} "
            "(fixed mechanism audit; no nested-CV dimension selection)."
        ),
        "- Primary state policy: every trial starts from the zero receiver state.",
        (
            "- The episode-continuous receiver is a sensitivity analysis and is "
            "not used for these five primary conclusions."
        ),
        (
            "- Rollouts are conditioned on observed future exogenous controls; "
            "they are not autonomous forecasts or probabilistic LDS likelihoods."
        ),
        (
            "- The nonlinear endpoint metric is finite-amplitude and finite-time; "
            "the fixed-drive result is only a narrow combined-actuator endpoint-"
            "separation sanity probe. It does not support gate causality, low-"
            "dimensional or shared-manifold dynamics, or attractor claims."
        ),
        (
            "- Claim-level eligibility is definition-specific. Total-control gain "
            "and closure require the trial-reset 19-column "
            "full_shared_neutral_cue operator; perturbation additionally requires "
            "joint_state_pca_physical_x_projection_v2 with full sampled-reference "
            "coverage. State-affine and fixed-drive measurements are not gated by "
            "unrelated total-operator, perturbation, or episode-sensitivity "
            "receipts. Historical v1 rows supplied directly to this audit remain "
            "readable for claims whose scientific definition did not change, "
            "while the formal collector selects exact v2 run configurations."
        ),
        "",
        "## Conclusions",
        "",
        _markdown_table(
            summary[
                [
                    "proposition",
                    "comparison",
                    "estimate",
                    "ci_low",
                    "ci_high",
                    "n_eligible",
                    "n_planned",
                    "conclusion",
                    "claim_scope",
                ]
            ]
        ),
        "",
        "## Retained failed or invalid seeds",
        "",
        (_markdown_table(failures[failure_columns]) if len(failures) else "None."),
        "",
        "## Claim-level scientific ineligibility",
        "",
        (
            _markdown_table(claim_ineligible)
            if len(claim_ineligible)
            else "None; every conclusion row had all planned seed-level measurements."
        ),
        "",
        "## Classification rule",
        "",
        textwrap.fill(
            "The two rollout-gain claims use the independent seed as the paired "
            "unit, deterministic bootstrap confidence intervals, exact sign "
            "tests, and one fixed two-hypothesis Holm family. A missing or "
            "scientifically ineligible hypothesis is entered as p=1 and still "
            "occupies its planned family slot. Registered mechanism audits require "
            "complete, identifiable measurements for every planned seed. Run-level "
            "failures and claim-level scientific ineligibility are reported "
            "separately; either prevents support for the affected claim. A "
            "completely absent registered attempt aborts collection.",
            width=96,
        ),
        "",
    ]
    paths["report"].write_text("\n".join(report), encoding="utf-8")
    _plot(summary, raw_sorted, png=paths["png"], pdf=paths["pdf"])
    return paths


def publish_snapshot(
    results_root: str | Path,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    prefix: str = DEFAULT_PREFIX,
    n_bootstrap: int = 5000,
) -> dict[str, Path]:
    _require_formal_registration(config)
    provenance = _analysis_provenance()
    raw = collect_registered_runs(results_root, config)
    return write_snapshot_artifacts(
        raw,
        config,
        output_dir=output_dir,
        prefix=prefix,
        n_bootstrap=n_bootstrap,
        publication_provenance=provenance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    config = _load_config(args.config)
    paths = publish_snapshot(
        args.results_root,
        config,
        output_dir=args.output_dir,
        prefix=args.prefix,
        n_bootstrap=args.n_bootstrap,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
