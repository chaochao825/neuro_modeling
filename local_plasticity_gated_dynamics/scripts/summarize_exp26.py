"""Fail-closed seed-level summary for the Exp26 actuator phase diagram.

The collector never chooses a favourable attempt.  A profile may contain at
most one attempt per seed unless ``--run-label`` explicitly selects a labelled
attempt.  Generator cells are averaged within seed before descriptive means
and standard deviations are computed, and confirmatory inference is delegated
to :func:`src.analysis.actuator_phase_statistics.summarize_phase_diagram`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.actuator_phase_statistics import (  # noqa: E402
    PRIMARY_MODES,
    summarize_phase_diagram,
)


EXPERIMENT = "exp26_actuator_phase_diagram"
EXPECTED_SEEDS = {
    "formal": tuple(range(30)),
    "smoke": (9000, 9001),
}
TERMINAL_RUN_STATUSES = {"complete", "complete_with_failures"}
ROW_STATUSES = {"complete", "failed", "invalid"}
SEED_PATTERN = re.compile(r"^seed_(\d+)$")
TIMESTAMP_LABEL_PATTERN = re.compile(
    r"^\d{8}T\d{6}(?:\.\d+)?Z(?:_(?P<label>.+))?$"
)

REQUIRED_INFERENCE_COLUMNS = (
    "seed",
    "generator_id",
    "generator_split",
    "actuator_mode",
    "chi",
    "alpha",
    "validation_balanced_accuracy",
    "test_balanced_accuracy",
    "status",
    "functional_budget_valid",
)

GROUP_DIMENSION_CANDIDATES = (
    "generator_split",
    "actuator_mode",
    "alpha",
    "rank_a",
    "rank_b",
    "transition_rank",
    "delta_a_rank",
    "delta_b_rank",
    "dynamics_rank",
    "input_rank",
    "delay_steps",
    "delay",
    "control_delay_steps",
    "noise_std",
    "observation_noise_std",
    "switch_hazard",
    "rotation_split",
)

PREFERRED_METRICS = (
    "validation_balanced_accuracy",
    "test_balanced_accuracy",
    "train_balanced_accuracy",
    "heldout_balanced_accuracy",
    "behavior_balanced_accuracy",
    "controlled_rollout_rmse",
    "controlled_rollout_normalized_rmse",
    "rollout_rmse",
    "rollout_normalized_rmse",
    "functional_state_displacement",
    "functional_budget_state_displacement",
    "energy_proxy",
    "plasticity_cost",
    "chi",
)


@dataclass(frozen=True)
class Attempt:
    """One explicitly selected immutable run directory."""

    seed: int
    path: Path
    run_status: str
    run_label: str | None
    planned_condition_count: int | None
    observed_metric_count: int
    planned_coverage_valid: bool
    planned_fingerprint: str | None


@dataclass(frozen=True)
class Collection:
    """Selected attempts and their unfiltered raw metric rows."""

    raw: pd.DataFrame
    attempts: tuple[Attempt, ...]
    profile: str
    expected_seeds: tuple[int, ...]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}:{line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _seed_from_metrics_path(path: Path) -> int | None:
    experiment_seen = False
    seed: int | None = None
    for parent in path.parents:
        if parent.name == EXPERIMENT:
            experiment_seen = True
            break
        match = SEED_PATTERN.fullmatch(parent.name)
        if match is not None:
            seed = int(match.group(1))
    return seed if experiment_seen else None


def _declared_run_label(path: Path, config: Mapping[str, Any]) -> str | None:
    configured = config.get("run_label")
    if isinstance(configured, str) and configured:
        return configured
    match = TIMESTAMP_LABEL_PATTERN.fullmatch(path.name)
    if match is not None:
        return match.group("label")
    return None


def _label_matches(path: Path, declared: str | None, requested: str) -> bool:
    return bool(
        declared == requested
        or path.name == requested
        or path.name.endswith(f"_{requested}")
    )


def _strict_true(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _planned_contract(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int | None, bool, str | None]:
    planned_path = path / "planned_conditions.json"
    if not planned_path.is_file():
        return None, False, None
    value = _read_json(planned_path)
    if not isinstance(value, list):
        return None, False, None
    normalized: list[dict[str, Any]] = []
    keys: list[tuple[str, str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            return len(value), False, None
        planned = dict(item)
        if planned.pop("condition_index", None) != index:
            return len(value), False, None
        try:
            key = (
                str(planned["generator_id"]),
                str(planned["generator_split"]),
                str(planned["actuator_mode"]),
            )
        except KeyError:
            return len(value), False, None
        normalized.append(planned)
        keys.append(key)
    row_keys = [
        (
            str(row.get("generator_id")),
            str(row.get("generator_split")),
            str(row.get("actuator_mode")),
        )
        for row in rows
    ]
    valid = bool(
        keys
        and len(keys) == len(set(keys))
        and len(row_keys) == len(set(row_keys))
        and sorted(keys) == sorted(row_keys)
    )
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return len(value), valid, hashlib.sha256(encoded).hexdigest()


def _candidate_attempts(
    results_root: Path,
    *,
    profile: str,
    run_label: str | None,
) -> dict[int, list[tuple[Path, Mapping[str, Any], str | None]]]:
    if not results_root.exists():
        return {}
    candidates: dict[int, list[tuple[Path, Mapping[str, Any], str | None]]] = {}
    for metrics_path in sorted(results_root.rglob("metrics.jsonl")):
        seed = _seed_from_metrics_path(metrics_path)
        if seed is None:
            continue
        attempt = metrics_path.parent
        config_path = attempt / "config.json"
        if not config_path.is_file():
            raise ValueError(
                f"Exp26 attempt {attempt} has metrics but no config.json; "
                "profile cannot be verified"
            )
        config = _read_json(config_path)
        if not isinstance(config, Mapping):
            raise ValueError(f"{config_path} must contain a JSON object")
        if config.get("profile") != profile:
            continue
        if config.get("experiment") != EXPERIMENT:
            raise ValueError(f"{config_path} has the wrong experiment identity")
        if config.get("seeds") != list(EXPECTED_SEEDS[profile]):
            raise ValueError(
                f"{config_path} does not register the exact {profile} seed panel"
            )
        expected_dev_only = profile == "smoke"
        if config.get("dev_only") is not expected_dev_only:
            raise ValueError(
                f"{config_path} must set dev_only={expected_dev_only!r}"
            )
        try:
            configured_seed = int(config.get("seed", -1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{config_path} has an invalid seed") from error
        if configured_seed != seed:
            raise ValueError(f"{config_path} seed disagrees with its path")
        declared = _declared_run_label(attempt, config)
        if run_label is not None and not _label_matches(
            attempt, declared, run_label
        ):
            continue
        candidates.setdefault(seed, []).append((attempt, config, declared))
    return candidates


def collect_metrics(
    results_root: str | Path,
    *,
    profile: str,
    run_label: str | None = None,
) -> Collection:
    """Collect one explicitly identifiable Exp26 attempt per seed.

    Multiple attempts for any seed are rejected instead of selecting the newest,
    most complete, or numerically best run.
    """

    if profile not in EXPECTED_SEEDS:
        raise ValueError(f"profile must be one of {sorted(EXPECTED_SEEDS)}")
    if run_label is not None and (not run_label or Path(run_label).name != run_label):
        raise ValueError("run_label must be a non-empty path-safe component")
    candidate_map = _candidate_attempts(
        Path(results_root), profile=profile, run_label=run_label
    )
    duplicate_seeds = {
        seed: [str(item[0]) for item in values]
        for seed, values in candidate_map.items()
        if len(values) > 1
    }
    if duplicate_seeds:
        detail = "; ".join(
            f"seed {seed}: {paths}" for seed, paths in sorted(duplicate_seeds.items())
        )
        selector = "matching --run-label" if run_label else "without --run-label"
        raise ValueError(
            f"multiple Exp26 attempts found {selector}; refusing to choose: {detail}"
        )

    attempts: list[Attempt] = []
    raw_rows: list[dict[str, Any]] = []
    for seed, entries in sorted(candidate_map.items()):
        attempt, _config, declared = entries[0]
        status_path = attempt / "status.json"
        status_payload = _read_json(status_path) if status_path.is_file() else {}
        run_status = (
            str(status_payload.get("status", "missing"))
            if isinstance(status_payload, Mapping)
            else "malformed"
        )
        rows = _read_jsonl(attempt / "metrics.jsonl")
        plan_count, plan_valid, plan_fingerprint = _planned_contract(
            attempt, rows
        )
        attempts.append(
            Attempt(
                seed=seed,
                path=attempt.resolve(),
                run_status=run_status,
                run_label=declared,
                planned_condition_count=plan_count,
                observed_metric_count=len(rows),
                planned_coverage_valid=plan_valid,
                planned_fingerprint=plan_fingerprint,
            )
        )
        for row in rows:
            if row.get("experiment") != EXPERIMENT:
                raise ValueError(f"{attempt}/metrics.jsonl has wrong experiment row")
            try:
                row_seed = int(row.get("seed", -1))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{attempt}/metrics.jsonl has invalid row seed") from error
            if row_seed != seed:
                raise ValueError(f"{attempt}/metrics.jsonl row seed disagrees with path")
            row_status = str(row.get("status", "missing"))
            terminal = run_status in TERMINAL_RUN_STATUSES
            effective_status = (
                row_status
                if terminal and row_status in ROW_STATUSES
                else "failed"
            )
            budget_valid = _strict_true(row.get("functional_budget_valid"))
            raw_rows.append(
                {
                    **row,
                    "_profile": profile,
                    "_attempt_path": str(attempt.resolve()),
                    "_run_status": run_status,
                    "_run_label": declared,
                    "_run_terminal": terminal,
                    "_effective_status": effective_status,
                    "_effective_functional_budget_valid": bool(
                        terminal and effective_status == "complete" and budget_valid
                    ),
                }
            )
    base_columns = [
        *REQUIRED_INFERENCE_COLUMNS,
        "_profile",
        "_attempt_path",
        "_run_status",
        "_run_label",
        "_run_terminal",
        "_effective_status",
        "_effective_functional_budget_valid",
    ]
    raw = pd.DataFrame(raw_rows)
    for column in base_columns:
        if column not in raw:
            raw[column] = pd.Series(dtype=object)
    return Collection(
        raw=raw,
        attempts=tuple(attempts),
        profile=profile,
        expected_seeds=EXPECTED_SEEDS[profile],
    )


def _analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["status"] = frame["_effective_status"]
    frame["functional_budget_valid"] = frame[
        "_effective_functional_budget_valid"
    ]
    return frame


def _metric_columns(frame: pd.DataFrame, group_columns: Sequence[str]) -> list[str]:
    available = [name for name in PREFERRED_METRICS if name in frame]
    excluded = {
        *group_columns,
        "seed",
        "alpha",
        "rank_a",
        "rank_b",
        "transition_rank",
        "delta_a_rank",
        "delta_b_rank",
        "dynamics_rank",
        "input_rank",
        "delay_steps",
        "delay",
        "control_delay_steps",
        "noise_std",
        "observation_noise_std",
        "switch_hazard",
    }
    tokens = (
        "accuracy",
        "rmse",
        "energy",
        "cost",
        "displacement",
        "latency",
        "correlation",
        "advantage",
        "overlap",
    )
    for column in frame.columns:
        if column in available or column in excluded or column.startswith("_"):
            continue
        if not any(token in column.lower() for token in tokens):
            continue
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            continue
        converted = pd.to_numeric(values, errors="coerce")
        if converted.notna().any():
            available.append(column)
    return available


def _seed_status(values: pd.Series) -> str:
    statuses = set(values.astype(str))
    if "failed" in statuses:
        return "failed"
    if "invalid" in statuses:
        return "invalid"
    if statuses == {"complete"}:
        return "complete"
    return "unknown"


def descriptive_summary(collection: Collection) -> pd.DataFrame:
    """Return tidy descriptive statistics with seed as the only replicate."""

    frame = _analysis_frame(collection.raw)
    columns = [
        "generator_split",
        "actuator_mode",
        "metric",
        "mean",
        "sd",
        "n_seed",
        "statistics_unit",
        "expected_seed_count",
        "n_seed_observed",
        "n_seed_complete",
        "n_seed_failed",
        "n_seed_invalid",
        "n_seed_unknown",
        "n_seed_budget_invalid",
        "n_seed_missing",
        "n_rows",
        "coverage_fraction",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    group_columns = [
        column for column in GROUP_DIMENSION_CANDIDATES if column in frame
    ]
    for required in ("generator_split", "actuator_mode"):
        if required not in group_columns:
            group_columns.append(required)
    metric_columns = _metric_columns(frame, group_columns)
    if not metric_columns:
        return pd.DataFrame(columns=[*group_columns, *columns[2:]])
    expected = set(collection.expected_seeds)
    expected_frame = frame[frame["seed"].astype(int).isin(expected)].copy()
    output: list[dict[str, Any]] = []
    grouped = expected_frame.groupby(group_columns, dropna=False, sort=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        dimensions = dict(zip(group_columns, group_key, strict=True))
        statuses = group.groupby("seed", sort=False)["_effective_status"].apply(
            _seed_status
        )
        observed_seeds = set(int(seed) for seed in statuses.index)
        eligible = group[
            group["_effective_status"].eq("complete")
            & group["_effective_functional_budget_valid"].astype(bool)
        ]
        budget_invalid_seeds = group.loc[
            group["_effective_status"].eq("complete")
            & ~group["_effective_functional_budget_valid"].astype(bool),
            "seed",
        ].nunique()
        for metric in metric_columns:
            numeric = pd.to_numeric(eligible[metric], errors="coerce")
            seed_means = (
                eligible.assign(_numeric_metric=numeric)
                .dropna(subset=["_numeric_metric"])
                .groupby("seed", sort=True)["_numeric_metric"]
                .mean()
            )
            output.append(
                {
                    **dimensions,
                    "metric": metric,
                    "mean": (
                        float(seed_means.mean()) if not seed_means.empty else math.nan
                    ),
                    "sd": (
                        float(seed_means.std(ddof=1))
                        if len(seed_means) > 1
                        else math.nan
                    ),
                    "n_seed": int(seed_means.size),
                    "statistics_unit": "seed",
                    "expected_seed_count": len(expected),
                    "n_seed_observed": len(observed_seeds),
                    "n_seed_complete": int(statuses.eq("complete").sum()),
                    "n_seed_failed": int(statuses.eq("failed").sum()),
                    "n_seed_invalid": int(statuses.eq("invalid").sum()),
                    "n_seed_unknown": int(statuses.eq("unknown").sum()),
                    "n_seed_budget_invalid": int(budget_invalid_seeds),
                    "n_seed_missing": len(expected - observed_seeds),
                    "n_rows": int(group.shape[0]),
                    "coverage_fraction": len(observed_seeds) / len(expected),
                }
            )
    return pd.DataFrame(output).sort_values(
        [*group_columns, "metric"], kind="stable"
    ).reset_index(drop=True)


def _coverage(collection: Collection) -> dict[str, Any]:
    raw = collection.raw
    expected = set(collection.expected_seeds)
    observed = {attempt.seed for attempt in collection.attempts}
    effective = raw["_effective_status"].astype(str)
    primary = raw[raw["actuator_mode"].isin(PRIMARY_MODES)]
    duplicate_columns = [
        "seed",
        "generator_id",
        "generator_split",
        "actuator_mode",
    ]
    duplicate_primary = (
        int(primary.duplicated(duplicate_columns, keep=False).sum())
        if not primary.empty and all(column in primary for column in duplicate_columns)
        else 0
    )
    plan_fingerprints = {
        attempt.planned_fingerprint
        for attempt in collection.attempts
        if attempt.planned_fingerprint is not None
    }
    return {
        "expected_seed_count": len(expected),
        "expected_seeds": sorted(expected),
        "observed_attempt_seed_count": len(observed & expected),
        "observed_attempt_seeds": sorted(observed & expected),
        "missing_seed_count": len(expected - observed),
        "missing_seeds": sorted(expected - observed),
        "unexpected_seed_count": len(observed - expected),
        "unexpected_seeds": sorted(observed - expected),
        "selected_attempt_count": len(collection.attempts),
        "terminal_attempt_count": sum(
            attempt.run_status in TERMINAL_RUN_STATUSES
            for attempt in collection.attempts
        ),
        "failed_or_nonterminal_attempt_count": sum(
            attempt.run_status not in TERMINAL_RUN_STATUSES
            for attempt in collection.attempts
        ),
        "planned_coverage_valid_attempt_count": sum(
            attempt.planned_coverage_valid for attempt in collection.attempts
        ),
        "planned_coverage_invalid_attempt_count": sum(
            not attempt.planned_coverage_valid for attempt in collection.attempts
        ),
        "distinct_planned_fingerprint_count": len(plan_fingerprints),
        "raw_row_count": int(raw.shape[0]),
        "complete_row_count": int(effective.eq("complete").sum()),
        "failed_row_count": int(effective.eq("failed").sum()),
        "invalid_row_count": int(effective.eq("invalid").sum()),
        "unknown_original_row_status_count": int(
            (~raw["status"].astype(str).isin(ROW_STATUSES)).sum()
        ),
        "budget_invalid_complete_row_count": int(
            (
                effective.eq("complete")
                & ~raw["_effective_functional_budget_valid"].astype(bool)
            ).sum()
        ),
        "primary_row_count": int(primary.shape[0]),
        "rgl_ceiling_row_count": int(raw["actuator_mode"].eq("rgl").sum()),
        "duplicate_primary_cell_row_count": duplicate_primary,
    }


def _empty_conclusion(reason: str) -> dict[str, Any]:
    return {
        "conclusion": "inconclusive",
        "statistics_unit": "seed",
        "n_seeds": 0,
        "complete_primary_coverage": False,
        "endpoint_summaries": [],
        "incremental_auc_summary": {
            "name": "chi_minus_alpha_auroc",
            "null_value": 0.0,
            "mean": None,
            "lower_confidence": None,
            "upper_confidence": None,
            "p_value": 1.0,
            "p_value_holm": 1.0,
        },
        "gramian_predictor_beats_alpha": False,
        "seed_endpoints": [],
        "reason": reason,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def statistical_conclusion(
    collection: Collection,
    *,
    tie_margin: float = 0.01,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 100_000,
) -> dict[str, Any]:
    """Run the registered statistics, converting malformed coverage to no claim."""

    frame = _analysis_frame(collection.raw)
    try:
        missing = set(REQUIRED_INFERENCE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"missing inference columns: {sorted(missing)}")
        result = summarize_phase_diagram(
            frame,
            expected_seeds=collection.expected_seeds,
            tie_margin=tie_margin,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
        )
    except (KeyError, TypeError, ValueError) as error:
        payload = _empty_conclusion(f"statistical audit failed closed: {error}")
    else:
        payload = result.to_dict()
    coverage = _coverage(collection)
    plan_gate = bool(
        coverage["observed_attempt_seeds"] == coverage["expected_seeds"]
        and coverage["unexpected_seed_count"] == 0
        and coverage["failed_or_nonterminal_attempt_count"] == 0
        and coverage["planned_coverage_invalid_attempt_count"] == 0
        and coverage["distinct_planned_fingerprint_count"] == 1
    )
    if not plan_gate:
        payload["complete_primary_coverage"] = False
        payload["conclusion"] = "inconclusive"
        payload["reason"] = (
            "registered seed/run/planned-condition coverage is incomplete or "
            "inconsistent"
        )
    payload.update(
        profile=collection.profile,
        evidence_scope=(
            "formal_confirmatory"
            if collection.profile == "formal"
            else "development_only"
        ),
        dev_only=collection.profile == "smoke",
        confirmatory_eligible=bool(
            collection.profile == "formal"
            and payload.get("complete_primary_coverage") is True
            and plan_gate
        ),
        coverage=coverage,
        rgl_role=(
            "descriptive composite ceiling only; excluded from all co-primary "
            "actuator-family inference"
        ),
    )
    if collection.profile == "smoke":
        original = str(payload.get("conclusion", "inconclusive"))
        payload["development_result_before_scope_gate"] = original
        payload["conclusion"] = "inconclusive"
        payload["confirmatory_eligible"] = False
        payload["reason"] = (
            "smoke profile is development-only and cannot support or oppose the "
            "registered claim"
        )
    return _json_safe(payload)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No estimable rows."

    def cell(value: object) -> str:
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, rule, *rows))


def _write_plots(
    raw: pd.DataFrame,
    seed_endpoints: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, ...]:
    try:
        from figures.exp26_actuator_phase_diagram_plot import plot_exp26
    except ImportError:
        return ()
    try:
        return tuple(
            plot_exp26(
                _analysis_frame(raw),
                seed_endpoints,
                output_dir,
            )
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        # A figure is optional and must never turn incomplete data into a claim.
        return ()


def _report(
    collection: Collection,
    summary: pd.DataFrame,
    conclusion: Mapping[str, Any],
    *,
    plots_skipped: bool,
    plot_written: bool,
) -> str:
    coverage = conclusion["coverage"]
    endpoint_frame = pd.DataFrame(conclusion.get("endpoint_summaries", []))
    if not endpoint_frame.empty:
        endpoint_frame = endpoint_frame[
            [
                "name",
                "null_value",
                "mean",
                "lower_confidence",
                "upper_confidence",
                "p_value",
                "p_value_holm",
            ]
        ]
    incremental = pd.DataFrame([conclusion["incremental_auc_summary"]])
    analysis = _analysis_frame(collection.raw)
    eligible = analysis[
        analysis["status"].eq("complete")
        & analysis["functional_budget_valid"].astype(bool)
    ]
    preview_rows: list[dict[str, object]] = []
    for metric in ("validation_balanced_accuracy", "test_balanced_accuracy"):
        if metric not in eligible:
            continue
        seed_values = (
            eligible.assign(_value=pd.to_numeric(eligible[metric], errors="coerce"))
            .dropna(subset=["_value"])
            .groupby(
                ["seed", "generator_split", "actuator_mode"],
                sort=True,
            )["_value"]
            .mean()
            .reset_index()
        )
        for (split, mode), values in seed_values.groupby(
            ["generator_split", "actuator_mode"], sort=True
        ):
            preview_rows.append(
                {
                    "generator_split": split,
                    "actuator_mode": mode,
                    "metric": metric,
                    "mean": float(values["_value"].mean()),
                    "sd": (
                        float(values["_value"].std(ddof=1))
                        if values.shape[0] > 1
                        else math.nan
                    ),
                    "n_seed": int(values["seed"].nunique()),
                }
            )
    summary_preview = pd.DataFrame(preview_rows)
    preview_columns = list(summary_preview.columns)
    scope_warning = (
        "**DEVELOPMENT ONLY:** smoke seeds 9000 and 9001 are permanently scoped "
        "to pipeline validation. Their numerical outcome is forced to "
        "`inconclusive`."
        if collection.profile == "smoke"
        else (
            "Formal scope requires exactly seeds 0--29 and complete paired primary "
            "coverage. Missing or failed cells cannot be dropped."
        )
    )
    plot_text = (
        "skipped by `--skip-plots`"
        if plots_skipped
        else "written" if plot_written else "not estimable from retained rows"
    )
    return "\n".join(
        [
            "# Exp26 actuator phase-diagram summary",
            "",
            f"**Conclusion: {conclusion['conclusion']}**",
            "",
            scope_warning,
            "",
            textwrap.fill(str(conclusion.get("reason", "")), width=96),
            "",
            "## Confirmatory endpoints",
            "",
            (
                "The three co-primary endpoints are seed-level held-out Spearman "
                "rho, threshold-classifier balanced accuracy, and AUROC. Their "
                "one-sided tests form one Holm-corrected family and the joint claim "
                "uses an intersection-union AND gate. Generator, neuron, and time "
                "point are not replicates."
            ),
            "",
            _markdown_table(endpoint_frame),
            "",
            "## Gramian χ versus raw α incremental gate",
            "",
            (
                "Support additionally requires the held-out χ AUROC to exceed the "
                "raw-α AUROC with a positive seed-level confidence bound and "
                "one-sided p < 0.05. This gate is reported separately from the "
                "three-member Holm family."
            ),
            "",
            _markdown_table(incremental),
            "",
            "## Coverage and retained failures",
            "",
            f"- Expected seeds: {coverage['expected_seed_count']}",
            f"- Observed expected seeds: {coverage['observed_attempt_seed_count']}",
            f"- Missing seeds: {coverage['missing_seed_count']}",
            f"- Unexpected seeds: {coverage['unexpected_seed_count']}",
            (
                "- Failed or non-terminal attempts: "
                f"{coverage['failed_or_nonterminal_attempt_count']}"
            ),
            f"- Failed rows retained: {coverage['failed_row_count']}",
            f"- Invalid rows retained: {coverage['invalid_row_count']}",
            (
                "- Complete rows failing the functional-budget gate: "
                f"{coverage['budget_invalid_complete_row_count']}"
            ),
            (
                "- Attempts with incomplete/malformed planned-cell coverage: "
                f"{coverage['planned_coverage_invalid_attempt_count']}"
            ),
            (
                "- Duplicate primary cell rows (automatically non-confirmatory): "
                f"{coverage['duplicate_primary_cell_row_count']}"
            ),
            "",
            "## Seed-level descriptive metrics",
            "",
            _markdown_table(summary_preview[preview_columns]),
            "",
            "## RGL interpretation boundary",
            "",
            (
                "RGL is a descriptive composite ceiling. It is not an additional "
                "primary actuator family, is excluded from χ threshold fitting and "
                "all three co-primary tests, and cannot rescue failed routing, gain, "
                "or low-rank cells."
            ),
            "",
            f"Plot status: {plot_text}.",
            "",
            "All raw rows, including scientific failures, remain in `raw_metrics.csv`.",
            "",
        ]
    )


def write_summary_artifacts(
    results_root: str | Path,
    *,
    output_dir: str | Path,
    profile: str,
    run_label: str | None = None,
    skip_plots: bool = False,
    tie_margin: float = 0.01,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 100_000,
) -> dict[str, Path]:
    """Collect Exp26 attempts and write the complete fail-closed snapshot."""

    collection = collect_metrics(
        results_root, profile=profile, run_label=run_label
    )
    summary = descriptive_summary(collection)
    conclusion = statistical_conclusion(
        collection,
        tie_margin=tie_margin,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": output / "raw_metrics.csv",
        "summary": output / "summary.csv",
        "seed_endpoints": output / "seed_endpoints.csv",
        "conclusion": output / "conclusion.json",
        "report": output / "report.md",
    }
    collection.raw.to_csv(paths["raw"], index=False, lineterminator="\n")
    summary.to_csv(paths["summary"], index=False, lineterminator="\n")
    endpoint_columns = [
        "seed",
        "discovery_threshold",
        "discovery_alpha_threshold",
        "heldout_generators",
        "heldout_ties",
        "spearman_rho",
        "classifier_balanced_accuracy",
        "classifier_auroc",
        "alpha_classifier_balanced_accuracy",
        "alpha_classifier_auroc",
        "chi_minus_alpha_auroc",
    ]
    endpoints = pd.DataFrame(
        conclusion.get("seed_endpoints", []), columns=endpoint_columns
    )
    endpoints.to_csv(paths["seed_endpoints"], index=False, lineterminator="\n")
    paths["conclusion"].write_text(
        json.dumps(conclusion, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_paths: tuple[Path, ...] = ()
    if not skip_plots:
        plot_paths = _write_plots(collection.raw, endpoints, output)
        for plot_path in plot_paths:
            paths[f"plot_{plot_path.suffix.lstrip('.')}"] = plot_path
    paths["report"].write_text(
        _report(
            collection,
            summary,
            conclusion,
            plots_skipped=skip_plots,
            plot_written=bool(plot_paths),
        ),
        encoding="utf-8",
    )
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=sorted(EXPECTED_SEEDS), required=True)
    parser.add_argument(
        "--run-label",
        help=(
            "explicitly select one labelled attempt per seed; without this, "
            "multiple same-profile attempts are an error"
        ),
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--tie-margin", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--permutation-samples", type=int, default=100_000)
    args = parser.parse_args(argv)
    paths = write_summary_artifacts(
        args.results_root,
        output_dir=args.output_dir,
        profile=args.profile,
        run_label=args.run_label,
        skip_plots=args.skip_plots,
        tie_margin=args.tie_margin,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
