"""Fail-closed formal summary for Exp23, Exp24, and Exp25.

The collector reads immutable :class:`src.utils.artifacts.ExperimentRun`
directories, selects the latest *formal* attempt for every registered seed,
and materializes every planned condition even when it is missing, failed, or
invalid.  Smoke and pilot attempts are never promoted to formal evidence.

The output contains condition-coverage rows and claim rows in ``summary.csv``.
Core claims use seed-level inference for simulations and animal-level
inference with sessions nested within animal for real data.  A joint claim is
supported only when every registered component supports; missing or failed
evidence makes it inconclusive.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.model_comparison import paired_bootstrap  # noqa: E402


EXP23 = "exp23_closed_loop_local_controller"
EXP24 = "exp24_factorized_control_benchmark"
EXP25 = "exp25_compositional_tasks_real"
EXPERIMENTS = (EXP23, EXP24, EXP25)
TERMINAL_RUN_STATUSES = {"complete", "complete_with_failures"}
ROW_STATUSES = {"complete", "failed", "invalid"}
CONCLUSIONS = {"support", "oppose", "inconclusive"}

EXP23_TASKS = ("current", "delayed")
EXP23_CONDITIONS = (
    "frozen",
    "current_off_policy",
    "random_update",
    "exact_forward_sensitivity",
    "bptt_axis_only",
    "local_eprop",
)
EXP24_TASKS = ("routing_dominant", "dynamics_dominant")
EXP24_MODES = ("frozen", "routing", "gain", "low_rank", "rgl")
EXP25_PROTOCOLS = (
    "leave-one-block-out",
    "leave-one-composition-out",
    "unseen-stimulus-action-composition",
    "cross-session-transfer",
)
EXP25_FAMILIES = (
    "common",
    "input-gated",
    "state-gated",
    "fully-gated",
    "separate-task",
)
EXP25_IMPLEMENTED_PROTOCOLS = EXP25_PROTOCOLS[:3]


@dataclass(frozen=True)
class FormalBundle:
    """Formal configuration, planned coverage, and retained raw records."""

    configs: Mapping[str, Mapping[str, Any]]
    coverage: pd.DataFrame
    raw: pd.DataFrame


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object JSONL row")
            result.append(value)
    return result


def _formal_config_paths(config_root: Path) -> dict[str, Path]:
    return {
        EXP23: config_root / "exp23_closed_loop_local_controller.json",
        EXP24: config_root / "exp24_factorized_control_benchmark.json",
        EXP25: config_root / "exp25_compositional_tasks_real.json",
    }


def _load_formal_configs(config_root: str | Path) -> dict[str, dict[str, Any]]:
    paths = _formal_config_paths(Path(config_root))
    configs: dict[str, dict[str, Any]] = {}
    for experiment, path in paths.items():
        value = _read_json(path)
        if not isinstance(value, dict) or value.get("profile") != "formal":
            raise ValueError(f"{path} must be a formal JSON object")
        seeds = value.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(f"{path} has an invalid formal seed registration")
        configs[experiment] = value
    if configs[EXP23]["seeds"] != list(range(30)):
        raise ValueError("Exp23 formal support requires registered seeds 0..29")
    if configs[EXP24]["seeds"] != list(range(30)):
        raise ValueError("Exp24 formal support requires registered seeds 0..29")
    if configs[EXP25]["seeds"] != [0]:
        raise ValueError("Exp25 real-data formal registration must use seed 0")
    return configs


def _expected_plans(experiment: str) -> list[dict[str, object]]:
    if experiment == EXP23:
        return [
            {
                "condition": condition,
                "condition_method_label": (
                    "frozen_neutral_trajectory_block_local_eprop"
                    if condition == "current_off_policy"
                    else condition
                ),
                "condition_key_is_legacy_alias": (
                    condition == "current_off_policy"
                ),
                "task_variant": task,
                "controller_parameterization": "population_gain_axis",
            }
            for task in EXP23_TASKS
            for condition in EXP23_CONDITIONS
        ]
    if experiment == EXP24:
        return [
            {
                "task": task,
                "condition": mode,
                "actuator_mode": mode,
                "controller_source": "oracle_true_context_actuator_isolation",
                "control_dim": 2,
            }
            for task in EXP24_TASKS
            for mode in EXP24_MODES
        ]
    if experiment == EXP25:
        return [
            {
                "condition": f"{protocol}:{family}",
                "protocol": protocol,
                "model_family": family,
                "evaluation_level": "animal_session",
            }
            for protocol in EXP25_PROTOCOLS
            for family in EXP25_FAMILIES
        ]
    raise ValueError(f"unknown experiment: {experiment}")


def _condition_key(experiment: str, row: Mapping[str, Any]) -> tuple[str, ...]:
    if experiment == EXP23:
        return str(row.get("task_variant")), str(row.get("condition"))
    if experiment == EXP24:
        return str(row.get("task")), str(row.get("condition"))
    if experiment == EXP25:
        return str(row.get("protocol")), str(row.get("model_family"))
    raise ValueError(f"unknown experiment: {experiment}")


def _normalize_planned(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("planned_conditions.json must contain a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("planned condition must be a JSON object")
        row = dict(item)
        if row.pop("condition_index", None) != index:
            raise ValueError("planned condition indexes must be ordered")
        result.append(row)
    return result


def _attempt_time(path: Path, status: Mapping[str, Any]) -> str:
    value = status.get("started_at")
    return str(value) if value else path.name


def _registered_config_matches(
    observed: Mapping[str, Any],
    registered: Mapping[str, Any],
) -> bool:
    """Allow execution receipts while requiring every registered value."""

    return all(observed.get(key) == value for key, value in registered.items())


def _latest_formal_attempt(
    results_root: Path,
    experiment: str,
    seed: int,
    registered_config: Mapping[str, Any],
) -> Path | None:
    seed_root = results_root / "runs" / experiment / f"seed_{seed:04d}"
    candidates: list[tuple[str, Path]] = []
    for attempt in seed_root.glob("*"):
        config_path = attempt / "config.json"
        status_path = attempt / "status.json"
        if not config_path.is_file() or not status_path.is_file():
            continue
        config = _read_json(config_path)
        status = _read_json(status_path)
        if (
            isinstance(config, Mapping)
            and config.get("profile") == "formal"
            and config.get("experiment") == experiment
            and int(config.get("seed", -1)) == seed
            and _registered_config_matches(config, registered_config)
            and isinstance(status, Mapping)
        ):
            candidates.append((_attempt_time(attempt, status), attempt))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def _failure_text(rows: Sequence[Mapping[str, Any]]) -> str:
    messages: list[str] = []
    for row in rows:
        for field in ("failure_reason", "reason", "error"):
            value = row.get(field)
            if isinstance(value, str) and value and value not in messages:
                messages.append(value)
    return " | ".join(messages)


def _exp25_condition_status(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "missing"
    statuses = [str(row.get("status", "failed")) for row in rows]
    if any(status == "invalid" for status in statuses):
        return "invalid"
    if any(status == "failed" for status in statuses):
        return "failed"
    aggregates = [
        row for row in rows if row.get("record_type") == "protocol_aggregate"
    ]
    if len(aggregates) != 1:
        return "failed"
    aggregate = aggregates[0]
    if aggregate.get("status") != "complete":
        return "failed"
    planned = aggregate.get("outer_folds_planned")
    complete = aggregate.get("outer_folds_complete")
    failed = aggregate.get("outer_folds_failed")
    try:
        fold_receipt_valid = (
            int(planned) > 0 and int(complete) == int(planned) and int(failed) == 0
        )
    except (TypeError, ValueError):
        fold_receipt_valid = False
    return "complete" if fold_receipt_valid else "failed"


def _condition_status(
    experiment: str,
    rows: Sequence[Mapping[str, Any]],
    run_status: str,
) -> str:
    if run_status not in TERMINAL_RUN_STATUSES:
        return "failed"
    if experiment == EXP25:
        return _exp25_condition_status(rows)
    if len(rows) != 1:
        return "missing" if not rows else "failed"
    status = str(rows[0].get("status", "failed"))
    return status if status in ROW_STATUSES else "failed"


def collect_planned_rows(
    results_root: str | Path,
    *,
    config_root: str | Path = PROJECT_ROOT / "configs" / "formal",
) -> FormalBundle:
    """Collect latest formal attempts while materializing every planned cell."""

    root = Path(results_root)
    configs = _load_formal_configs(config_root)
    coverage_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        expected = _expected_plans(experiment)
        expected_keys = {
            _condition_key(experiment, row): row for row in expected
        }
        for seed in configs[experiment]["seeds"]:
            attempt = _latest_formal_attempt(
                root,
                experiment,
                int(seed),
                configs[experiment],
            )
            if attempt is None:
                for planned in expected:
                    coverage_rows.append(
                        {
                            "experiment": experiment,
                            "seed": int(seed),
                            **planned,
                            "profile": "formal",
                            "condition_status": "missing",
                            "n_metric_records": 0,
                            "failure_detail": "no formal attempt exists",
                            "attempt_path": None,
                            "run_status": "missing",
                        }
                    )
                continue
            status_payload = _read_json(attempt / "status.json")
            run_status = str(status_payload.get("status", "failed"))
            planned_path = attempt / "planned_conditions.json"
            try:
                observed = _normalize_planned(_read_json(planned_path))
            except Exception as error:
                observed = []
                plan_error = f"malformed planned_conditions.json: {error}"
            else:
                plan_error = ""
            plan_valid = observed == expected
            metrics = _read_jsonl(attempt / "metrics.jsonl")
            grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {
                key: [] for key in expected_keys
            }
            for row in metrics:
                key = _condition_key(experiment, row)
                enriched = {
                    **row,
                    "_formal_profile": True,
                    "_attempt_path": str(attempt.resolve()),
                    "_run_status": run_status,
                }
                raw_rows.append(enriched)
                if key in grouped:
                    grouped[key].append(enriched)
            for key, planned in expected_keys.items():
                rows = grouped[key]
                condition_status = _condition_status(
                    experiment, rows, run_status
                )
                if not plan_valid:
                    condition_status = "failed"
                details = _failure_text(rows)
                if plan_error:
                    details = " | ".join(value for value in (plan_error, details) if value)
                elif not plan_valid:
                    details = "planned condition grid differs from formal registration"
                coverage_rows.append(
                    {
                        "experiment": experiment,
                        "seed": int(seed),
                        **planned,
                        "profile": "formal",
                        "condition_status": condition_status,
                        "n_metric_records": len(rows),
                        "failure_detail": details,
                        "attempt_path": str(attempt.resolve()),
                        "run_status": run_status,
                    }
                )
    coverage = pd.DataFrame(coverage_rows)
    raw = pd.DataFrame(raw_rows)
    return FormalBundle(configs=configs, coverage=coverage, raw=raw)


def _strict_bool(value: object, expected: bool = True) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value) is expected
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return (value.lower() == "true") is expected
    return False


def _numeric(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _series(
    frame: pd.DataFrame, column: str, default: object = None
) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index, dtype=object)


def _bootstrap_zero(
    values: Sequence[float],
    unit_ids: Sequence[object],
    *,
    unit: str,
    statistic: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    result = paired_bootstrap(
        np.asarray(values, dtype=float),
        np.zeros(len(values), dtype=float),
        unit_ids=np.asarray(unit_ids, dtype=object),
        replicate_unit=unit,
        statistic=statistic,  # type: ignore[arg-type]
        n_resamples=n_bootstrap,
        seed=seed,
    )
    return result.estimate, result.ci_low, result.ci_high


def _bootstrap_ratio_of_means(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float | None, float | None, float | None, bool]:
    """Bootstrap the registered ratio of across-seed mean gains."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if (
        numerator.ndim != 1
        or numerator.shape != denominator.shape
        or numerator.size < 2
        or not np.isfinite(numerator).all()
        or not np.isfinite(denominator).all()
    ):
        return None, None, None, False
    denominator_mean = float(np.mean(denominator))
    if denominator_mean <= 0.0:
        return None, None, None, False
    estimate = float(np.mean(numerator) / denominator_mean)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        numerator.size,
        size=(n_bootstrap, numerator.size),
        endpoint=False,
    )
    numerator_means = np.mean(numerator[indices], axis=1)
    denominator_means = np.mean(denominator[indices], axis=1)
    if np.any(denominator_means <= 0.0):
        return estimate, None, None, False
    ratios = numerator_means / denominator_means
    low, high = np.quantile(ratios, [0.025, 0.975])
    return estimate, float(low), float(high), True


def _one_sided_seed_pvalue(
    values: Sequence[float],
    *,
    statistic: str,
    n_resamples: int,
    seed: int,
) -> float | None:
    """Return a deterministic one-sided seed-level p-value.

    The input is the seed-level effect after subtracting the registered null
    threshold. Mean effects use paired sign-flip randomization. Median effects
    use the exact one-sided sign test, which remains valid for tied magnitudes.
    """

    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 1
        or array.size < 2
        or not np.isfinite(array).all()
        or statistic not in {"mean", "median"}
    ):
        return None
    if statistic == "median":
        nonzero = array[array != 0.0]
        if nonzero.size < 2:
            return None
        positive = int(np.count_nonzero(nonzero > 0.0))
        return float(
            sum(
                math.comb(int(nonzero.size), count)
                for count in range(positive, int(nonzero.size) + 1)
            )
            / (2 ** int(nonzero.size))
        )
    reducer = np.mean
    observed = float(reducer(array))
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(n_resamples, array.size),
        replace=True,
    )
    randomized = reducer(signs * array[None, :], axis=1)
    return float(
        (1 + np.count_nonzero(randomized >= observed))
        / (n_resamples + 1)
    )


def _holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-adjust the finite p-values while retaining invalid entries."""

    adjusted: list[float | None] = [None] * len(p_values)
    finite = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None and np.isfinite(value)
    ]
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    family_size = len(p_values)
    for order, (index, value) in enumerate(ordered):
        running = max(running, (family_size - order) * value)
        adjusted[index] = min(1.0, running)
    return adjusted


def _classify(
    *,
    formal_ready: bool,
    ci_low: float | None,
    ci_high: float | None,
    threshold: float,
    direction: str = "greater",
    strict: bool = False,
) -> str:
    if (
        not formal_ready
        or ci_low is None
        or ci_high is None
        or not np.isfinite([ci_low, ci_high]).all()
    ):
        return "inconclusive"
    if direction == "greater":
        support = ci_low > threshold if strict else ci_low >= threshold
        oppose = ci_high <= threshold if strict else ci_high < threshold
    elif direction == "less":
        support = ci_high < threshold if strict else ci_high <= threshold
        oppose = ci_low >= threshold if strict else ci_low > threshold
    else:
        raise ValueError("direction must be greater or less")
    return "support" if support else ("oppose" if oppose else "inconclusive")


def _claim_row(
    *,
    claim_id: str,
    experiment: str,
    scope: str,
    metric: str,
    comparison: str,
    stats_unit: str,
    n_planned: int,
    n_complete: int,
    n_failed: int,
    n_invalid: int,
    estimate: float | None,
    ci_low: float | None,
    ci_high: float | None,
    threshold: float | None,
    conclusion: str,
    criterion: str,
    note: str,
    n_sessions: int | None = None,
    p_value: float | None = None,
    p_adjusted: float | None = None,
    multiplicity_method: str | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    if conclusion not in CONCLUSIONS:
        raise ValueError("unknown claim conclusion")
    return {
        "row_kind": "claim",
        "claim_id": claim_id,
        "experiment": experiment,
        "scope": scope,
        "unit_id": None,
        "condition": None,
        "status": None,
        "metric": metric,
        "comparison": comparison,
        "stats_unit": stats_unit,
        "n_planned": n_planned,
        "n_complete": n_complete,
        "n_failed": n_failed,
        "n_invalid": n_invalid,
        "n_sessions": n_sessions,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "threshold": threshold,
        "p_value": p_value,
        "p_adjusted": p_adjusted,
        "multiplicity_method": multiplicity_method,
        "alpha": alpha,
        "conclusion": conclusion,
        "criterion": criterion,
        "note": note,
    }


def _coverage_rows(bundle: FormalBundle) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in bundle.coverage.to_dict(orient="records"):
        experiment = str(row["experiment"])
        status = str(row["condition_status"])
        scope = row.get(
            "task_variant", row.get("task", row.get("protocol"))
        )
        result.append(
            {
                "row_kind": "condition_coverage",
                "claim_id": None,
                "experiment": experiment,
                "scope": scope,
                "unit_id": f"seed:{int(row['seed'])}",
                "condition": row["condition"],
                "status": status,
                "metric": "planned_condition_coverage",
                "comparison": None,
                "stats_unit": "seed",
                "n_planned": 1,
                "n_complete": int(status == "complete"),
                "n_failed": int(status in {"failed", "missing"}),
                "n_invalid": int(status == "invalid"),
                "n_sessions": None,
                "estimate": None,
                "ci_low": None,
                "ci_high": None,
                "threshold": None,
                "p_value": None,
                "p_adjusted": None,
                "multiplicity_method": None,
                "alpha": None,
                "conclusion": "inconclusive",
                "criterion": "coverage row; not a hypothesis test",
                "note": str(row.get("failure_detail", "") or ""),
            }
        )
    return result


def _formal_condition_ready(
    coverage: pd.DataFrame,
    experiment: str,
    *,
    selectors: Mapping[str, object] | None = None,
) -> bool:
    selected = coverage.loc[coverage["experiment"].eq(experiment)]
    for column, value in (selectors or {}).items():
        selected = selected.loc[selected[column].eq(value)]
    return bool(len(selected) and selected["condition_status"].eq("complete").all())


def _exp23_mechanism_audit_ready(
    selected: pd.DataFrame,
    planned_seeds: Sequence[int],
) -> bool:
    """Require explicit pairing, leakage, and frozen-recurrent receipts."""

    required_columns = {
        "seed",
        "condition",
        "random_tape_id",
        "split_id",
        "network_init_id",
        "gate_checkpoint_id",
        "readout_checkpoint_id",
        "readout_fit_data_id",
        "pairing_bundle_id",
        "paired_network_gate_readout_split_tape",
        "recurrent_frozen_hash",
        "recurrent_weights_hash_before_condition",
        "recurrent_weights_hash_after_condition",
        "recurrent_weights_snapshot_is_copy",
        "recurrent_copy_isolation_audit_passed",
        "recurrent_hash_audit_passed",
        "recurrent_weights_bitwise_frozen",
        "recurrent_weights_initial_id",
        "recurrent_weights_final_id",
        "recurrent_weights_audit",
        "recurrent_learning",
        "readout_fit_train_only",
        "readout_fit_scope",
        "gate_fit_train_only",
        "gate_fit_scope",
        "axis_fit_dev_only",
        "axis_fit_scope",
        "test_used_for_axis_fit",
        "axis_selection_accessed_test",
        "gate_test_accessed_true_context",
        "third_factor_accessed_true_context",
        "hidden_context_access_audit_passed",
        "local_learning",
        "used_autograd",
        "used_bptt",
        "local_rule_autograd_free",
        "local_rule_bptt_free",
    }
    if not required_columns <= set(selected):
        return False
    if set(pd.to_numeric(selected["seed"], errors="coerce")) != set(planned_seeds):
        return False
    pairing_ids = (
        "random_tape_id",
        "split_id",
        "network_init_id",
        "gate_checkpoint_id",
        "readout_checkpoint_id",
        "readout_fit_data_id",
        "pairing_bundle_id",
    )
    true_receipts = (
        "paired_network_gate_readout_split_tape",
        "recurrent_weights_snapshot_is_copy",
        "recurrent_copy_isolation_audit_passed",
        "recurrent_hash_audit_passed",
        "recurrent_weights_bitwise_frozen",
        "readout_fit_train_only",
        "gate_fit_train_only",
        "axis_fit_dev_only",
        "hidden_context_access_audit_passed",
    )
    false_receipts = (
        "recurrent_learning",
        "test_used_for_axis_fit",
        "axis_selection_accessed_test",
        "gate_test_accessed_true_context",
        "third_factor_accessed_true_context",
    )
    for seed in planned_seeds:
        rows = selected.loc[pd.to_numeric(selected["seed"]).eq(seed)]
        if (
            len(rows) != len(EXP23_CONDITIONS)
            or set(rows["condition"]) != set(EXP23_CONDITIONS)
        ):
            return False
        for field in pairing_ids:
            values = rows[field].dropna().astype(str)
            if len(values) != len(rows) or values.str.len().eq(0).any():
                return False
            if values.nunique() != 1:
                return False
        hashes = pd.concat(
            [
                rows["recurrent_frozen_hash"],
                rows["recurrent_weights_hash_before_condition"],
                rows["recurrent_weights_hash_after_condition"],
            ],
            ignore_index=True,
        ).dropna().astype(str)
        if len(hashes) != 3 * len(rows) or hashes.str.len().eq(0).any():
            return False
        if hashes.nunique() != 1:
            return False
        training_hashes = pd.concat(
            [
                rows["recurrent_weights_initial_id"],
                rows["recurrent_weights_final_id"],
            ],
            ignore_index=True,
        ).dropna().astype(str)
        if (
            len(training_hashes) != 2 * len(rows)
            or training_hashes.str.len().eq(0).any()
            or training_hashes.nunique() != 1
            or not rows["recurrent_weights_audit"]
            .eq("independent_copy_and_sha256")
            .all()
        ):
            return False
        if any(not rows[field].map(_strict_bool).all() for field in true_receipts):
            return False
        if any(
            not rows[field].map(lambda value: _strict_bool(value, False)).all()
            for field in false_receipts
        ):
            return False
        if not rows["readout_fit_scope"].eq("training_split_only").all():
            return False
        if not rows["gate_fit_scope"].eq("training_split_only").all():
            return False
        if not rows["axis_fit_scope"].eq("development_split_only").all():
            return False
        local = rows.loc[rows["condition"].eq("local_eprop")]
        if len(local) != 1:
            return False
        local_row = local.iloc[0]
        if not _strict_bool(local_row["local_learning"]):
            return False
        if not _strict_bool(local_row["used_autograd"], False):
            return False
        if not _strict_bool(local_row["used_bptt"], False):
            return False
        if not _strict_bool(local_row["local_rule_autograd_free"]):
            return False
        if not _strict_bool(local_row["local_rule_bptt_free"]):
            return False
    return True


def _exp23_rows(bundle: FormalBundle, n_bootstrap: int) -> list[dict[str, Any]]:
    config = bundle.configs[EXP23]
    thresholds = dict(config["registered_claim_thresholds"])
    planned_seeds = tuple(int(value) for value in config["seeds"])
    raw = bundle.raw.loc[
        _series(bundle.raw, "experiment").eq(EXP23)
        & _series(bundle.raw, "status").eq("complete")
    ].copy()
    for column in (
        "task_variant",
        "condition",
        "seed",
        "behavior_balanced_accuracy",
        "functional_budget_satisfied",
        "median_update_cosine_to_exact",
        "gate_moment_anchor_identifiable",
        "gate_mean_absolute_signed_belief_dev",
    ):
        if column not in raw:
            raw[column] = pd.Series(index=raw.index, dtype=object)
    claims: list[dict[str, Any]] = []
    joint_by_task: dict[str, str] = {}
    components_by_task: dict[str, list[dict[str, Any]]] = {}
    for task_index, task in enumerate(EXP23_TASKS):
        selected = raw.loc[raw["task_variant"].eq(task)]
        pivot = selected.pivot_table(
            index="seed",
            columns="condition",
            values="behavior_balanced_accuracy",
            aggfunc="first",
        )
        local_rows = selected.loc[selected["condition"].eq("local_eprop")].set_index(
            "seed"
        )
        eligible = sorted(
            set(planned_seeds)
            & set(pivot.index)
            & set(local_rows.index)
            & set(pivot.dropna().index)
        )
        required_columns = {
            "frozen",
            "random_update",
            "bptt_axis_only",
            "local_eprop",
        }
        if not required_columns <= set(pivot):
            eligible = []
        if eligible:
            pivot = pivot.loc[eligible]
            local_rows = local_rows.loc[eligible]
        coverage_ready = _formal_condition_ready(
            bundle.coverage, EXP23, selectors={"task_variant": task}
        )
        nonfrozen_rows = selected.loc[selected["condition"].ne("frozen")]
        budget_ready = bool(
            len(nonfrozen_rows)
            and set(nonfrozen_rows["condition"])
            == set(EXP23_CONDITIONS) - {"frozen"}
            and nonfrozen_rows["functional_budget_satisfied"]
            .map(_strict_bool)
            .all()
        )
        identifiable = local_rows["gate_moment_anchor_identifiable"].map(
            _strict_bool
        )
        belief_strength = pd.to_numeric(
            local_rows["gate_mean_absolute_signed_belief_dev"],
            errors="coerce",
        )
        gate_ready = bool(
            len(local_rows)
            and float(np.mean(identifiable))
            >= float(thresholds["minimum_gate_identifiable_fraction"])
            and belief_strength.notna().all()
            and float(np.min(belief_strength))
            >= float(thresholds["minimum_mean_absolute_signed_belief"])
        )
        mechanism_audit_ready = _exp23_mechanism_audit_ready(
            selected,
            planned_seeds,
        )
        formal_ready = (
            coverage_ready
            and budget_ready
            and gate_ready
            and mechanism_audit_ready
            and tuple(eligible) == planned_seeds
            and len(planned_seeds)
            >= int(thresholds["minimum_formal_seeds"])
        )
        n_failed = len(planned_seeds) - len(eligible)
        local = np.array([], dtype=float)
        frozen = np.array([], dtype=float)
        random = np.array([], dtype=float)
        bptt = np.array([], dtype=float)
        ratios = np.array([], dtype=float)
        ratio_numerator = np.array([], dtype=float)
        ratio_denominator = np.array([], dtype=float)
        cosine = np.array([], dtype=float)
        ratio_invalid = True
        if eligible:
            local = pivot["local_eprop"].to_numpy(float)
            frozen = pivot["frozen"].to_numpy(float)
            random = pivot["random_update"].to_numpy(float)
            bptt = pivot["bptt_axis_only"].to_numpy(float)
            bptt_gain = bptt - frozen
            local_gain = local - frozen
            ratio_numerator = local_gain
            ratio_denominator = bptt_gain
            ratios = np.full_like(local_gain, np.nan)
            nonzero = bptt_gain != 0.0
            ratios[nonzero] = local_gain[nonzero] / bptt_gain[nonzero]
            ratio_invalid = not bool(
                np.isfinite(local_gain).all()
                and np.isfinite(bptt_gain).all()
                and np.mean(bptt_gain) > 0.0
            )
            cosine = pd.to_numeric(
                local_rows["median_update_cosine_to_exact"], errors="coerce"
            ).to_numpy(float)
        values: list[
            tuple[str, str, np.ndarray, float, str, bool, str]
        ] = [
            (
                "gain_vs_frozen",
                "local_eprop - frozen held-out balanced accuracy",
                local - frozen,
                float(thresholds["minimum_local_gain_vs_frozen"]),
                "mean",
                False,
                "local gain over frozen is at least 0.03",
            ),
            (
                "gain_vs_random",
                "local_eprop - random_update held-out balanced accuracy",
                local - random,
                float(thresholds["minimum_local_gain_vs_random"]),
                "mean",
                False,
                "local gain over random update is at least 0.03",
            ),
            (
                "fraction_of_bptt_gain",
                "(local - frozen) / (BPTT-axis - frozen)",
                ratios,
                float(thresholds["minimum_fraction_of_bptt_axis_gain"]),
                "mean",
                ratio_invalid,
                "local retains at least 60% of the BPTT-axis gain",
            ),
            (
                "median_update_cosine",
                "local update cosine with exact forward sensitivity",
                cosine,
                float(thresholds["minimum_median_update_cosine"]),
                "median",
                not bool(cosine.size and np.isfinite(cosine).all()),
                "median local/exact update cosine is strictly positive",
            ),
        ]
        component_rows: list[dict[str, Any]] = []
        for component_index, (
            suffix,
            comparison,
            observations,
            threshold,
            statistic,
            component_invalid,
            criterion,
        ) in enumerate(values):
            finite = np.asarray(observations, dtype=float)
            valid = (
                np.isfinite(ratio_numerator)
                & np.isfinite(ratio_denominator)
                if suffix == "fraction_of_bptt_gain"
                else np.isfinite(finite)
            )
            estimate = ci_low = ci_high = None
            if suffix == "fraction_of_bptt_gain":
                estimate, ci_low, ci_high, ratio_bootstrap_valid = (
                    _bootstrap_ratio_of_means(
                        ratio_numerator,
                        ratio_denominator,
                        n_bootstrap=n_bootstrap,
                        seed=2300 + task_index * 10 + component_index,
                    )
                )
                component_invalid = (
                    component_invalid or not ratio_bootstrap_valid
                )
            elif np.count_nonzero(valid) >= 2:
                estimate, ci_low, ci_high = _bootstrap_zero(
                    finite[valid],
                    np.asarray(eligible, dtype=object)[valid],
                    unit="seed",
                    statistic=statistic,
                    n_bootstrap=n_bootstrap,
                    seed=2300 + task_index * 10 + component_index,
                )
            ready = formal_ready and not component_invalid and bool(valid.all())
            null_effects = (
                ratio_numerator - threshold * ratio_denominator
                if suffix == "fraction_of_bptt_gain"
                else finite - threshold
            )
            p_value = (
                _one_sided_seed_pvalue(
                    null_effects,
                    statistic=statistic,
                    n_resamples=n_bootstrap,
                    seed=2350 + task_index * 10 + component_index,
                )
                if not component_invalid and bool(valid.all())
                else None
            )
            conclusion = _classify(
                formal_ready=ready,
                ci_low=ci_low,
                ci_high=ci_high,
                threshold=threshold,
                strict=suffix == "median_update_cosine",
            )
            row = _claim_row(
                claim_id=f"exp23_{task}_{suffix}",
                experiment=EXP23,
                scope=task,
                metric="behavior_balanced_accuracy"
                if "cosine" not in suffix
                else "median_update_cosine_to_exact",
                comparison=comparison,
                stats_unit="seed",
                n_planned=len(planned_seeds),
                n_complete=int(np.count_nonzero(valid)),
                n_failed=n_failed,
                n_invalid=int(
                    coverage_ready
                    and (
                        component_invalid
                        or not budget_ready
                        or not gate_ready
                        or not mechanism_audit_ready
                    )
                ),
                estimate=estimate,
                ci_low=ci_low,
                ci_high=ci_high,
                threshold=threshold,
                conclusion=conclusion,
                criterion=(
                    f"{criterion}; one-sided seed-level p-value "
                    "must pass Holm familywise correction at alpha=0.05"
                ),
                note=(
                    "95% whole-seed bootstrap; mean effects use a paired sign-flip "
                    "test and the median cosine uses an exact sign test. Support "
                    "additionally requires the "
                    "complete 30-seed, 12-cell formal grid and a valid functional "
                    "state-displacement budget plus an identifiable non-collapsed "
                    "belief gate and the explicit paired/frozen/leakage/local-rule "
                    "mechanism audit."
                ),
                p_value=p_value,
                multiplicity_method=(
                    "Holm within the four registered components for this task"
                ),
                alpha=0.05,
            )
            component_rows.append(row)
        adjusted_p_values = _holm_adjust(
            [row["p_value"] for row in component_rows]
        )
        for row, adjusted_p_value in zip(
            component_rows,
            adjusted_p_values,
            strict=True,
        ):
            row["p_adjusted"] = adjusted_p_value
            if row["conclusion"] == "support" and (
                adjusted_p_value is None or adjusted_p_value > 0.05
            ):
                row["conclusion"] = "inconclusive"
            claims.append(row)
        component_conclusions = [row["conclusion"] for row in component_rows]
        joint = (
            "support"
            if component_conclusions
            and all(value == "support" for value in component_conclusions)
            else "oppose"
            if any(value == "oppose" for value in component_conclusions)
            else "inconclusive"
        )
        joint_by_task[task] = joint
        components_by_task[task] = component_rows
        claims.append(
            _claim_row(
                claim_id=f"exp23_{task}_joint_closed_loop_local_controller",
                experiment=EXP23,
                scope=task,
                metric="registered_AND_gate",
                comparison="all four Exp23 registered local-controller criteria",
                stats_unit="seed",
                n_planned=len(planned_seeds),
                n_complete=len(eligible),
                n_failed=n_failed,
                n_invalid=int(
                    coverage_ready
                    and (
                        not budget_ready
                        or not gate_ready
                        or not mechanism_audit_ready
                    )
                ),
                estimate=None,
                ci_low=None,
                ci_high=None,
                threshold=None,
                conclusion=joint,
                criterion=(
                    "all four Holm-corrected component conclusions must support "
                    "(intersection-union AND)"
                ),
                note=(
                    "No OR criterion is used. All non-frozen conditions must pass "
                    "the matched functional budget, and the learned belief gate "
                    "must pass its preregistered identifiability/strength gate. "
                    "Every seed must also pass the explicit pairing, recurrent "
                    "hash/copy, split-leakage, and local no-autograd/BPTT audit. "
                    "The joint alternative is an intersection: requiring every "
                    "Holm-corrected component to support is a conservative IUT."
                ),
                multiplicity_method=(
                    "intersection-union AND over four Holm-corrected components"
                ),
                alpha=0.05,
            )
        )
    overall = (
        "support"
        if all(value == "support" for value in joint_by_task.values())
        else "oppose"
        if any(value == "oppose" for value in joint_by_task.values())
        else "inconclusive"
    )
    claims.append(
        _claim_row(
            claim_id="exp23_joint_both_tasks",
            experiment=EXP23,
            scope="current_and_delayed",
            metric="registered_AND_gate",
            comparison="current and delayed task joint claims",
            stats_unit="seed",
            n_planned=len(planned_seeds),
            n_complete=min(
                (
                    int(rows[0]["n_complete"])
                    for rows in components_by_task.values()
                    if rows
                ),
                default=0,
            ),
            n_failed=len(planned_seeds)
            - int(
                bundle.coverage.loc[
                    bundle.coverage["experiment"].eq(EXP23)
                ]
                .groupby("seed")["condition_status"]
                .apply(lambda values: values.eq("complete").all())
                .sum()
            ),
            n_invalid=max(
                (
                    int(rows[0]["n_invalid"])
                    for rows in components_by_task.values()
                    if rows
                ),
                default=0,
            ),
            estimate=None,
            ci_low=None,
            ci_high=None,
            threshold=None,
            conclusion=overall,
            criterion=(
                "both task-specific Holm-corrected IUT claims must support (AND)"
            ),
            note=(
                "If local-eprop fails to beat random update, the registered "
                "population-gain-axis local-learning route is opposed for that task. "
                "Combining the two task claims is itself an intersection-union test."
            ),
            multiplicity_method=(
                "intersection-union AND over two task-specific Holm families"
            ),
            alpha=0.05,
        )
    )
    return claims


def _exp24_budget_valid(row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    explicit = row.get(
        "joint_functional_budget_valid",
        row.get("functional_budget_valid"),
    )
    if explicit is not None:
        return bool(
            _strict_bool(explicit)
            and all(
                _strict_bool(row.get(f"functional_budget_{name}_valid"))
                for name in ("state", "rate", "gain", "event")
            )
            and row.get("functional_budget_fit_scope") == "training_blocks_only"
            and _strict_bool(row.get("parameter_norm_budget_used"), False)
            and int(row.get("control_dim", -1)) == 2
            and _strict_bool(row.get("shared_control_dim_across_actuators"))
        )
    mode = str(row.get("condition"))
    if mode == "frozen":
        return True
    tolerance = float(config["functional_budget"]["relative_tolerance"])
    relative_error = _numeric(row.get("functional_budget_relative_error"))
    return bool(
        _strict_bool(row.get("functional_budget_converged"))
        and relative_error is not None
        and abs(relative_error) <= tolerance
        and row.get("functional_budget_fit_scope") == "training_blocks_only"
        and _strict_bool(row.get("parameter_norm_budget_used"), False)
        and int(row.get("control_dim", -1)) == 2
        and _strict_bool(row.get("shared_control_dim_across_actuators"))
    )


def _exp24_rows(bundle: FormalBundle, n_bootstrap: int) -> list[dict[str, Any]]:
    config = bundle.configs[EXP24]
    planned_seeds = tuple(int(value) for value in config["seeds"])
    raw = bundle.raw.loc[
        _series(bundle.raw, "experiment").eq(EXP24)
        & _series(bundle.raw, "status").eq("complete")
    ].copy()
    for column in ("task", "condition", "seed", "test_balanced_accuracy"):
        if column not in raw:
            raw[column] = pd.Series(index=raw.index, dtype=object)
    comparisons = (
        (
            "routing_prefers_routing_to_low_rank",
            "routing_dominant",
            "routing",
            "low_rank",
        ),
        (
            "routing_prefers_gain_to_low_rank",
            "routing_dominant",
            "gain",
            "low_rank",
        ),
        (
            "dynamics_prefers_low_rank_to_routing",
            "dynamics_dominant",
            "low_rank",
            "routing",
        ),
        (
            "dynamics_prefers_rgl_to_routing",
            "dynamics_dominant",
            "rgl",
            "routing",
        ),
    )
    all_coverage_ready = _formal_condition_ready(bundle.coverage, EXP24)
    budget_rows = raw.loc[raw["condition"].ne("frozen")]
    all_budget_ready = bool(
        len(budget_rows)
        and all(
            _exp24_budget_valid(row, config)
            for row in budget_rows.to_dict(orient="records")
        )
    )
    component_rows: list[dict[str, Any]] = []
    for index, (claim_suffix, task, candidate, reference) in enumerate(comparisons):
        selected = raw.loc[raw["task"].eq(task)]
        pivot = selected.pivot_table(
            index="seed",
            columns="condition",
            values="test_balanced_accuracy",
            aggfunc="first",
        )
        eligible = (
            sorted(
                set(planned_seeds)
                & set(pivot.dropna(subset=[candidate, reference]).index)
            )
            if {candidate, reference} <= set(pivot)
            else []
        )
        differences = np.array([], dtype=float)
        estimate = ci_low = ci_high = None
        if len(eligible) >= 2:
            differences = (
                pivot.loc[eligible, candidate] - pivot.loc[eligible, reference]
            ).to_numpy(float)
            estimate, ci_low, ci_high = _bootstrap_zero(
                differences,
                eligible,
                unit="seed",
                statistic="mean",
                n_bootstrap=n_bootstrap,
                seed=2400 + index,
            )
        formal_ready = (
            all_coverage_ready
            and all_budget_ready
            and tuple(eligible) == planned_seeds
        )
        p_value = (
            _one_sided_seed_pvalue(
                differences,
                statistic="mean",
                n_resamples=n_bootstrap,
                seed=2450 + index,
            )
            if tuple(eligible) == planned_seeds
            else None
        )
        conclusion = _classify(
            formal_ready=formal_ready,
            ci_low=ci_low,
            ci_high=ci_high,
            threshold=0.0,
            strict=True,
        )
        row = _claim_row(
            claim_id=f"exp24_{claim_suffix}",
            experiment=EXP24,
            scope=task,
            metric="test_balanced_accuracy",
            comparison=f"{candidate} - {reference}",
            stats_unit="seed",
            n_planned=len(planned_seeds),
            n_complete=len(eligible),
            n_failed=len(planned_seeds) - len(eligible),
            n_invalid=int(all_coverage_ready and not all_budget_ready),
            estimate=estimate,
            ci_low=ci_low,
            ci_high=ci_high,
            threshold=0.0,
            conclusion=conclusion,
            criterion=(
                "registered task-specific actuator direction is strictly positive; "
                "one-sided seed-paired sign-flip p-value must pass Holm familywise "
                "correction at alpha=0.05"
            ),
            note=(
                "95% whole-seed bootstrap. Support requires all 30 seeds, the "
                "complete 10-cell grid, shared two-dimensional oracle controls, "
                "and a training-only joint state/rate/gain/event budget receipt."
            ),
            p_value=p_value,
            multiplicity_method=(
                "Holm across the four registered Exp24 actuator comparisons"
            ),
            alpha=0.05,
        )
        component_rows.append(row)
    adjusted_p_values = _holm_adjust(
        [row["p_value"] for row in component_rows]
    )
    for row, adjusted_p_value in zip(
        component_rows,
        adjusted_p_values,
        strict=True,
    ):
        row["p_adjusted"] = adjusted_p_value
        if row["conclusion"] == "support" and (
            adjusted_p_value is None or adjusted_p_value > 0.05
        ):
            row["conclusion"] = "inconclusive"
    conclusions = [row["conclusion"] for row in component_rows]
    joint = (
        "support"
        if all(value == "support" for value in conclusions)
        else "oppose"
        if any(value == "oppose" for value in conclusions)
        else "inconclusive"
    )
    return [
        *component_rows,
        _claim_row(
            claim_id="exp24_joint_task_dependent_actuator_specialization",
            experiment=EXP24,
            scope="routing_and_dynamics",
            metric="registered_AND_gate",
            comparison="all four task-specific actuator directions",
            stats_unit="seed",
            n_planned=len(planned_seeds),
            n_complete=min(
                (int(row["n_complete"]) for row in component_rows), default=0
            ),
            n_failed=max(
                (int(row["n_failed"]) for row in component_rows), default=0
            ),
            n_invalid=int(all_coverage_ready and not all_budget_ready),
            estimate=None,
            ci_low=None,
            ci_high=None,
            threshold=None,
            conclusion=joint,
            criterion=(
                "all four Holm-corrected direction claims must support "
                "(intersection-union AND)"
            ),
            note=(
                "This is an oracle actuator-isolation benchmark; it does not "
                "establish that a local rule learned the controller. The joint "
                "alternative is an intersection over the corrected components."
            ),
            multiplicity_method=(
                "intersection-union AND over four Holm-corrected components"
            ),
            alpha=0.05,
        ),
    ]


def _parse_sequence(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [dict(item) for item in decoded if isinstance(item, Mapping)]
    return []


def _exp25_session_frame(bundle: FormalBundle) -> pd.DataFrame:
    raw = bundle.raw.loc[
        _series(bundle.raw, "experiment").eq(EXP25)
        & _series(bundle.raw, "status").eq("complete")
        & _series(bundle.raw, "record_type").eq("outer_fold")
    ]
    records: list[dict[str, Any]] = []
    for row in raw.to_dict(orient="records"):
        parameter_count = _numeric(row.get("parameter_count"))
        for session in _parse_sequence(row.get("per_session")):
            observations = _numeric(session.get("n_observations"))
            log_likelihood = _numeric(session.get("log_likelihood"))
            null_likelihood = _numeric(session.get("null_log_likelihood"))
            if (
                observations is None
                or observations <= 0.0
                or log_likelihood is None
                or null_likelihood is None
                or parameter_count is None
            ):
                continue
            records.append(
                {
                    "protocol": row.get("protocol"),
                    "model_family": row.get("model_family"),
                    "fold_id": row.get("fold_id"),
                    "session_id": str(session.get("session_id")),
                    "animal_id": str(session.get("animal_id")),
                    "mean_log_likelihood": log_likelihood / observations,
                    "gain_vs_null_per_observation": (
                        log_likelihood - null_likelihood
                    )
                    / observations,
                    "parameter_count": parameter_count,
                }
            )
    return pd.DataFrame(records)


def _merge_families(
    frame: pd.DataFrame,
    families: Sequence[str],
    *,
    protocols: Sequence[str],
) -> pd.DataFrame:
    selected = frame.loc[
        frame["protocol"].isin(protocols)
        & frame["model_family"].isin(families)
    ]
    keys = ["protocol", "fold_id", "session_id", "animal_id"]
    value_columns = [
        "mean_log_likelihood",
        "gain_vs_null_per_observation",
        "parameter_count",
    ]
    pieces = []
    for family in families:
        piece = selected.loc[
            selected["model_family"].eq(family), [*keys, *value_columns]
        ].rename(columns={name: f"{family}:{name}" for name in value_columns})
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=keys, how="inner", validate="one_to_one")
    return merged


def _animal_effect(
    paired: pd.DataFrame,
    effect: Callable[[pd.DataFrame], pd.Series],
) -> tuple[np.ndarray, np.ndarray, int]:
    if paired.empty:
        return np.array([]), np.array([], dtype=object), 0
    session = paired.assign(_effect=effect(paired)).groupby(
        ["animal_id", "session_id"], as_index=False
    )["_effect"].mean()
    session = session.loc[np.isfinite(session["_effect"])]
    animal = session.groupby("animal_id", as_index=False)["_effect"].mean()
    return (
        animal["_effect"].to_numpy(float),
        animal["animal_id"].to_numpy(object),
        int(session["session_id"].nunique()),
    )


def _exp25_effect_claim(
    *,
    bundle: FormalBundle,
    frame: pd.DataFrame,
    claim_id: str,
    scope: str,
    families: Sequence[str],
    protocols: Sequence[str],
    effect: Callable[[pd.DataFrame], pd.Series],
    comparison: str,
    metric: str,
    n_bootstrap: int,
    seed: int,
    direction: str = "greater",
    strict: bool = True,
    prerequisite: Callable[[pd.DataFrame], pd.Series] | None = None,
    prerequisite_label: str = "",
) -> dict[str, Any]:
    config = bundle.configs[EXP25]
    paired = _merge_families(frame, families, protocols=protocols)
    values, animals, n_sessions = _animal_effect(paired, effect)
    minimum_animals = int(config["minimum_animals"])
    minimum_sessions = int(config["minimum_sessions"])
    relevant_ready = all(
        _formal_condition_ready(
            bundle.coverage,
            EXP25,
            selectors={"protocol": protocol, "model_family": family},
        )
        for protocol in protocols
        for family in families
    )
    count_ready = (
        len(set(animals.tolist())) >= minimum_animals
        and n_sessions >= minimum_sessions
    )
    prerequisite_ready = True
    prerequisite_note = ""
    if prerequisite is not None:
        prerequisite_values, prerequisite_animals, _ = _animal_effect(
            paired, prerequisite
        )
        prerequisite_estimate = prerequisite_ci_low = None
        if len(prerequisite_values) >= 2:
            (
                prerequisite_estimate,
                prerequisite_ci_low,
                _,
            ) = _bootstrap_zero(
                prerequisite_values,
                prerequisite_animals,
                unit="animal",
                statistic="mean",
                n_bootstrap=n_bootstrap,
                seed=seed + 100,
            )
        prerequisite_ready = bool(
            prerequisite_ci_low is not None and prerequisite_ci_low > 0.0
        )
        prerequisite_note = (
            f" Prerequisite `{prerequisite_label}` estimate/CI-low="
            f"{prerequisite_estimate!r}/{prerequisite_ci_low!r}; it must be "
            "strictly positive."
        )
    estimate = ci_low = ci_high = None
    if len(values) >= 2:
        estimate, ci_low, ci_high = _bootstrap_zero(
            values,
            animals,
            unit="animal",
            statistic="mean",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    conclusion = _classify(
        formal_ready=relevant_ready and count_ready and prerequisite_ready,
        ci_low=ci_low,
        ci_high=ci_high,
        threshold=0.0,
        direction=direction,
        strict=strict,
    )
    coverage = bundle.coverage.loc[
        bundle.coverage["experiment"].eq(EXP25)
        & bundle.coverage["protocol"].isin(protocols)
        & bundle.coverage["model_family"].isin(families)
    ]
    return _claim_row(
        claim_id=claim_id,
        experiment=EXP25,
        scope=scope,
        metric=metric,
        comparison=comparison,
        stats_unit="animal (sessions nested)",
        n_planned=minimum_animals,
        n_complete=len(set(animals.tolist())),
        n_failed=max(0, minimum_animals - len(set(animals.tolist()))),
        n_invalid=int(coverage["condition_status"].eq("invalid").sum())
        + int(relevant_ready and not prerequisite_ready),
        n_sessions=n_sessions,
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        threshold=0.0,
        conclusion=conclusion,
        criterion=(
            f"strictly {direction}; >= {minimum_animals} animals and "
            f">= {minimum_sessions} sessions; every relevant outer fold complete"
        ),
        note=(
            "Likelihood effects are normalized per held-out observation, averaged "
            "within session, then within animal before whole-animal bootstrap. "
            "Neuron and time bin are never independent repeats."
            + prerequisite_note
        ),
    )


def _exp25_rows(bundle: FormalBundle, n_bootstrap: int) -> list[dict[str, Any]]:
    frame = _exp25_session_frame(bundle)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "protocol",
                "model_family",
                "fold_id",
                "session_id",
                "animal_id",
                "mean_log_likelihood",
                "gain_vs_null_per_observation",
                "parameter_count",
            ]
        )
    mean_ll = "mean_log_likelihood"
    params = "parameter_count"
    common = f"common:{mean_ll}"
    fully = f"fully-gated:{mean_ll}"
    input_gated = f"input-gated:{mean_ll}"
    state_gated = f"state-gated:{mean_ll}"
    separate = f"separate-task:{mean_ll}"
    fully_params = f"fully-gated:{params}"
    separate_params = f"separate-task:{params}"
    claims = [
        _exp25_effect_claim(
            bundle=bundle,
            frame=frame,
            claim_id="exp25_fully_gated_vs_common",
            scope="implemented_outer_protocols",
            families=("common", "fully-gated"),
            protocols=EXP25_IMPLEMENTED_PROTOCOLS,
            effect=lambda value: value[fully] - value[common],
            comparison="fully-gated - common held-out mean log likelihood",
            metric="heldout_mean_log_likelihood",
            n_bootstrap=n_bootstrap,
            seed=2500,
        ),
        _exp25_effect_claim(
            bundle=bundle,
            frame=frame,
            claim_id="exp25_input_gain_exceeds_state_gain",
            scope="implemented_outer_protocols",
            families=("input-gated", "state-gated"),
            protocols=EXP25_IMPLEMENTED_PROTOCOLS,
            effect=lambda value: value[input_gated] - value[state_gated],
            comparison="input-gated - state-gated held-out mean log likelihood",
            metric="heldout_mean_log_likelihood",
            n_bootstrap=n_bootstrap,
            seed=2501,
        ),
        _exp25_effect_claim(
            bundle=bundle,
            frame=frame,
            claim_id="exp25_fully_retains_90pct_separate_gain",
            scope="implemented_outer_protocols",
            families=("common", "fully-gated", "separate-task"),
            protocols=EXP25_IMPLEMENTED_PROTOCOLS,
            effect=lambda value: (value[fully] - value[common])
            - 0.9 * (value[separate] - value[common]),
            comparison=(
                "(fully-common) - 0.9 * (separate-task-common) held-out gain"
            ),
            metric="heldout_mean_log_likelihood_retention_margin",
            n_bootstrap=n_bootstrap,
            seed=2502,
            prerequisite=lambda value: value[separate] - value[common],
            prerequisite_label="separate-task > common held-out gain",
        ),
        _exp25_effect_claim(
            bundle=bundle,
            frame=frame,
            claim_id="exp25_fully_uses_fewer_parameters",
            scope="implemented_outer_protocols",
            families=("fully-gated", "separate-task"),
            protocols=EXP25_IMPLEMENTED_PROTOCOLS,
            effect=lambda value: value[fully_params] - value[separate_params],
            comparison="fully-gated - separate-task parameter count",
            metric="parameter_count",
            n_bootstrap=n_bootstrap,
            seed=2503,
            direction="less",
        ),
        _exp25_effect_claim(
            bundle=bundle,
            frame=frame,
            claim_id="exp25_unseen_composition_shared_vs_separate",
            scope="unseen-stimulus-action-composition",
            families=("fully-gated", "separate-task"),
            protocols=("unseen-stimulus-action-composition",),
            effect=lambda value: value[fully] - value[separate],
            comparison=(
                "fully-gated - separate-task held-out mean log likelihood on "
                "unseen composition"
            ),
            metric="heldout_mean_log_likelihood",
            n_bootstrap=n_bootstrap,
            seed=2504,
        ),
        _exp25_effect_claim(
            bundle=bundle,
            frame=frame,
            claim_id="exp25_cross_session_fully_vs_common",
            scope="cross-session-transfer",
            families=("common", "fully-gated"),
            protocols=("cross-session-transfer",),
            effect=lambda value: value[fully] - value[common],
            comparison=(
                "fully-gated - common held-out mean log likelihood in "
                "cross-session transfer"
            ),
            metric="heldout_mean_log_likelihood",
            n_bootstrap=n_bootstrap,
            seed=2505,
        ),
    ]
    conclusions = [row["conclusion"] for row in claims]
    joint = (
        "support"
        if all(value == "support" for value in conclusions)
        else "oppose"
        if any(value == "oppose" for value in conclusions)
        else "inconclusive"
    )
    claims.append(
        _claim_row(
            claim_id="exp25_joint_reusable_shared_belief_dynamics",
            experiment=EXP25,
            scope="all_registered_real_data_criteria",
            metric="registered_AND_gate",
            comparison="all Exp25 success criteria including cross-session transfer",
            stats_unit="animal (sessions nested)",
            n_planned=int(bundle.configs[EXP25]["minimum_animals"]),
            n_complete=min(
                (int(row["n_complete"]) for row in claims), default=0
            ),
            n_failed=max((int(row["n_failed"]) for row in claims), default=0),
            n_invalid=max((int(row["n_invalid"]) for row in claims), default=0),
            n_sessions=min(
                (
                    int(row["n_sessions"])
                    for row in claims
                    if row["n_sessions"] is not None
                ),
                default=0,
            ),
            estimate=None,
            ci_low=None,
            ci_high=None,
            threshold=None,
            conclusion=joint,
            criterion="every Exp25 component must support (AND)",
            note=(
                "An invalid cross-session protocol can never support the joint "
                "claim; it leaves the result inconclusive unless another component "
                "conclusively opposes the hypothesis."
            ),
        )
    )
    return claims


def summarize_claims(
    bundle: FormalBundle,
    *,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Return condition coverage and formal support/oppose/inconclusive claims."""

    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    rows = [
        *_coverage_rows(bundle),
        *_exp23_rows(bundle, n_bootstrap),
        *_exp24_rows(bundle, n_bootstrap),
        *_exp25_rows(bundle, n_bootstrap),
    ]
    summary = pd.DataFrame(rows)
    order = {"condition_coverage": 0, "claim": 1}
    summary["_order"] = summary["row_kind"].map(order)
    return (
        summary.sort_values(
            [
                "_order",
                "experiment",
                "scope",
                "condition",
                "unit_id",
                "claim_id",
            ],
            kind="stable",
            na_position="last",
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if value is None:
            return ""
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, rule, *body))


def _plot_claims(claims: pd.DataFrame, path: Path) -> bool:
    plotted = claims.loc[
        claims["row_kind"].eq("claim")
        & claims["estimate"].map(lambda value: _numeric(value) is not None)
        & claims["threshold"].map(lambda value: _numeric(value) is not None)
    ].copy()
    if plotted.empty:
        return False
    plotted["margin"] = pd.to_numeric(plotted["estimate"]) - pd.to_numeric(
        plotted["threshold"]
    )
    plotted["margin_low"] = pd.to_numeric(plotted["ci_low"]) - pd.to_numeric(
        plotted["threshold"]
    )
    plotted["margin_high"] = pd.to_numeric(plotted["ci_high"]) - pd.to_numeric(
        plotted["threshold"]
    )
    plotted = plotted.loc[
        np.isfinite(
            plotted[["margin", "margin_low", "margin_high"]].to_numpy(float)
        ).all(axis=1)
    ]
    if plotted.empty:
        return False
    colors = {
        "support": "#0072B2",
        "oppose": "#D55E00",
        "inconclusive": "#7F7F7F",
    }
    labels = [
        f"{row.experiment.replace('exp', 'E')} · {row.claim_id.split('_', 1)[-1]}"
        for row in plotted.itertuples()
    ]
    height = max(4.0, 0.32 * len(plotted) + 1.5)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
        }
    ):
        figure, axis = plt.subplots(figsize=(9.0, height), constrained_layout=True)
        y = np.arange(len(plotted))
        for index, row in enumerate(plotted.itertuples()):
            axis.errorbar(
                float(row.margin),
                y[index],
                xerr=np.asarray(
                    [
                        [max(0.0, float(row.margin - row.margin_low))],
                        [max(0.0, float(row.margin_high - row.margin))],
                    ]
                ),
                fmt="o",
                color=colors[str(row.conclusion)],
                capsize=3,
                markersize=4,
            )
        axis.axvline(0.0, color="0.25", linestyle="--", linewidth=0.8)
        axis.set_yticks(y, labels)
        axis.set_ylim(len(plotted) - 0.5, -0.5)
        axis.set_xlabel("Effect minus registered threshold (95% unit bootstrap CI)")
        axis.set_title("Exp23-25 formal claim margins", loc="left")
        axis.spines[["top", "right"]].set_visible(False)
        figure.savefig(
            path,
            dpi=220,
            bbox_inches="tight",
            metadata={"Software": "summarize_exp23_exp25.py"},
        )
        plt.close(figure)
    return True


def write_summary_artifacts(
    bundle: FormalBundle,
    *,
    output_dir: str | Path,
    n_bootstrap: int = 5000,
) -> dict[str, Path]:
    """Write ``summary.csv``, ``report.md``, and one data-dependent figure."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_claims(bundle, n_bootstrap=n_bootstrap)
    paths = {
        "summary": output / "summary.csv",
        "report": output / "report.md",
        "figure": output / "exp23_exp25_summary.png",
    }
    summary.to_csv(paths["summary"], index=False, lineterminator="\n")
    claims = summary.loc[summary["row_kind"].eq("claim")]
    coverage = summary.loc[summary["row_kind"].eq("condition_coverage")]
    incomplete = coverage.loc[~coverage["status"].eq("complete")]
    incomplete_report = pd.DataFrame()
    if len(incomplete):
        incomplete_report = (
            incomplete.groupby(
                ["experiment", "scope", "condition", "status"],
                as_index=False,
                dropna=False,
            )
            .agg(
                n_units=("unit_id", "nunique"),
                unit_ids=(
                    "unit_id",
                    lambda values: ", ".join(sorted(set(values.astype(str)))),
                ),
                note=(
                    "note",
                    lambda values: " | ".join(
                        sorted(
                            {
                                str(value)
                                for value in values
                                if isinstance(value, str) and value
                            }
                        )
                    ),
                ),
            )
        )
    primary = claims.loc[
        claims["claim_id"].isin(
            [
                "exp23_joint_both_tasks",
                "exp24_joint_task_dependent_actuator_specialization",
                "exp25_joint_reusable_shared_belief_dynamics",
            ]
        )
    ]
    report = [
        "# Exp23-25 formal evidence summary",
        "",
        (
            "This report is fail-closed. It reads only attempts whose saved "
            "`config.json` declares `profile=formal`; smoke and pilot attempts "
            "are ignored even when their numerical metrics are favorable."
        ),
        "",
        (
            "Every registered condition is represented in `summary.csv`. Missing, "
            "failed, and invalid cells are retained and prevent formal support in "
            "the affected AND gate."
        ),
        "",
        "All formal joint claims use AND, never OR.",
        (
            "Exp23 and Exp24 component inference use Holm correction; Exp23 formal "
            "readiness requires frozen-recurrent hash/copy receipts."
        ),
        "",
        "## Core conclusions",
        "",
        _markdown_table(
            primary[
                [
                    "claim_id",
                    "stats_unit",
                    "n_planned",
                    "n_complete",
                    "n_failed",
                    "n_invalid",
                    "n_sessions",
                    "conclusion",
                    "criterion",
                ]
            ]
        ),
        "",
        "## Component claims",
        "",
        _markdown_table(
            claims.loc[~claims.index.isin(primary.index)][
                [
                    "claim_id",
                    "scope",
                    "comparison",
                    "estimate",
                    "ci_low",
                    "ci_high",
                    "threshold",
                    "p_value",
                    "p_adjusted",
                    "multiplicity_method",
                    "n_complete",
                    "n_sessions",
                    "conclusion",
                ]
            ]
        ),
        "",
        "## Retained failed, invalid, or missing conditions",
        "",
        (
            _markdown_table(incomplete_report)
            if len(incomplete_report)
            else "None."
        ),
        "",
        "## Interpretation boundary",
        "",
        textwrap.fill(
            "Exp23 and Exp24 use seed as the independent unit. Exp25 first "
            "normalizes likelihood within held-out session, then averages sessions "
            "within animal and bootstraps animals; neuron and time bin are never "
            "replicates. Exp24 is an oracle actuator-isolation benchmark and does "
            "not itself establish local controller learning. Exp25 scores exact "
            "one-step conditional Poisson likelihood rather than a full marginal "
            "PLDS likelihood or autonomous forecast. A currently invalid "
            "cross-session transfer condition cannot support the real-data joint "
            "claim. Exp23/24 mean-effect p-values use one-sided paired sign-flip "
            "tests, the median-cosine component uses an exact sign test, and all "
            "four use Holm correction within each task family; task "
            "and cross-task conclusions are conservative intersection-union AND "
            "gates, never OR. Exp23 formal readiness also requires explicit "
            "pairing IDs, frozen-recurrent hash/copy receipts, train/dev/test "
            "separation, no true-context access, and local-eprop no-autograd/BPTT "
            "receipts for every registered seed.",
            width=96,
        ),
        "",
    ]
    paths["report"].write_text("\n".join(report), encoding="utf-8")
    if not _plot_claims(claims, paths["figure"]):
        paths.pop("figure")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--config-root", default=str(PROJECT_ROOT / "configs" / "formal")
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args(argv)
    bundle = collect_planned_rows(
        args.results_root,
        config_root=args.config_root,
    )
    paths = write_summary_artifacts(
        bundle,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
