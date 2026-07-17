"""Audit, summarize, and plot the one-shot Exp29 selector test."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import load_json_config  # noqa: E402
from experiments.exp26_actuator_phase_diagram import (  # noqa: E402
    canonical_config_sha256,
    git_identity,
)
from experiments.exp29_confirmatory_actuator_selector import (  # noqa: E402
    CONFIRMATORY_ELIGIBLE,
    DECISION_MODES,
    DECISION_RECEIPT_SCHEMA,
    EXPECTED_EVALUATION_SEEDS,
    EXPECTED_META_SEEDS,
    EXPERIMENT,
    FIT_RECEIPT_SCHEMA,
    INFERENCE_SCOPE,
    INFERENCE_STATUS,
    PROFILE,
    PROTOCOL_VERSION,
    REQUIRED_RUN_LABEL,
    SELECTORS,
    SOURCE_RECEIPT_SCHEMA,
    _decision_fingerprint,
    _file_sha256,
    _fit_frozen_selectors,
    _load_sources,
    _payload_sha256,
    _planned_conditions,
    _semantic_records,
    _validate_config,
    _validate_implementation_binding,
    analysis_contract_sha256,
)
from src.analysis.actuator_phase_statistics import holm_adjust  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "formal" / "exp29_confirmatory_actuator_selector.json"
)
PLAN_KEYS = (
    "selector_fit_seed",
    "evaluation_seed",
    "source_seed",
    "generator_split",
    "generator_id",
    "alpha",
    "transition_rank",
    "input_rank",
    "delay",
    "noise",
    "selector",
)
PROBABILITY_COLUMNS = tuple(f"selection_probability_{mode}" for mode in DECISION_MODES)
CANDIDATE_UTILITY_COLUMNS = tuple(
    f"candidate_{mode}_utility" for mode in DECISION_MODES
)
ACTIVE_FEASIBILITY_COLUMNS = tuple(
    f"candidate_{mode}_feasible" for mode in DECISION_MODES[1:]
)
TRAIN_UTILITY_COLUMNS = tuple(
    f"train_mean_candidate_{mode}_utility" for mode in DECISION_MODES[1:]
)
REGISTERED_ANALYSIS_CONTRACT_SHA256 = (
    "c680985c2d23c0b230be185b7f28ceb8a41e0377429f6fbd085c24a87e379e69"
)


@dataclass(frozen=True)
class ConfirmatorySeedEndpoint:
    seed: int
    registered_generators: int
    oracle_utility: float
    gru_bptt_utility: float
    local_three_factor_utility: float
    fixed_best_utility: float
    frozen_utility: float
    routing_utility: float
    gain_utility: float
    low_rank_utility: float
    local_minus_fixed_best: float
    local_noninferiority_contrast: float
    routing_infeasible_rate: float
    gain_infeasible_rate: float
    low_rank_infeasible_rate: float
    local_fallback_rate: float
    local_matched_budget_eligible_rate: float
    local_update_l1: float
    local_update_l2: float
    gru_update_l1: float
    gru_update_l2: float


@dataclass(frozen=True)
class ConfirmatoryContrast:
    name: str
    mean: float
    lower_confidence: float
    upper_confidence: float
    p_value: float
    p_value_holm: float
    opposition_p_value: float
    opposition_p_value_holm: float


@dataclass(frozen=True)
class Exp29SelectorConclusion:
    conclusion: str
    reason: str
    evidence_valid: bool
    confirmatory_eligible: bool
    statistics_unit: str
    n_seeds: int
    unconditional_all_registered_cells: bool
    complete_primary_coverage: bool
    noninferiority_fraction: float
    primary_contrasts: tuple[ConfirmatoryContrast, ...]
    seed_endpoints: tuple[ConfirmatorySeedEndpoint, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Exp29Collection:
    raw: pd.DataFrame
    config: Mapping[str, Any]
    config_sha256: str
    attempt_path: str
    run_git_commit: str
    run_git_tree: str
    runtime_identity: Mapping[str, Any]
    run_label: str
    evidence_valid: bool
    invalid_reason: str | None
    source_receipt: Mapping[str, Any] | None
    fit_receipt: Mapping[str, Any] | None
    decision_receipt: Mapping[str, Any] | None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Exp29 artifact {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read Exp29 rows {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank Exp29 row at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid Exp29 row at line {line_number}") from error
        if not isinstance(value, Mapping):
            raise ValueError("Exp29 metric row must be an object")
        rows.append(dict(value))
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _strict_boolean(series: pd.Series, *, name: str) -> pd.Series:
    if not bool(series.map(lambda value: isinstance(value, (bool, np.bool_))).all()):
        raise TypeError(f"{name} must contain literal booleans")
    return series.astype(bool)


def _finite_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric vector")
    vector = np.asarray(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite non-empty vector")
    return vector


def _bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, samples: int, confidence: float
) -> tuple[float, float]:
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid Exp29 bootstrap settings")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = np.mean(values[indices], axis=1)
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _one_sided_sign_flip(values: np.ndarray, *, seed: int, samples: int) -> float:
    observed = float(np.mean(values))
    if values.size <= 20:
        signs = np.asarray(list(product((-1.0, 1.0), repeat=values.size)))
    else:
        if samples < 1:
            raise ValueError("permutation_samples must be positive")
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(samples, values.size))
    null = np.mean(signs * values[np.newaxis, :], axis=1)
    return float((1.0 + np.sum(null >= observed - 1e-15)) / (null.size + 1.0))


def _validate_self_hash(value: Any, *, name: str, schema: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != schema:
        raise ValueError(f"{name} schema mismatch")
    payload = dict(value)
    observed = payload.pop("receipt_sha256", None)
    if observed != _payload_sha256(payload):
        raise ValueError(f"{name} self-hash mismatch")
    return dict(value)


def validate_confirmatory_selector_records(records: pd.DataFrame) -> pd.DataFrame:
    """Validate all 30x44x4 unconditional confirmatory rows."""

    required = {
        "seed",
        "selector_fit_seed",
        "evaluation_seed",
        "outer_seed",
        "source_seed",
        "generator_id",
        "generator_split",
        "selector",
        "requested_mode",
        "mode_selected",
        "deployed_mode",
        "oracle_mode",
        "fixed_best_mode",
        "utility",
        "oracle_utility",
        "fixed_best_utility",
        "status",
        "plasticity_l1",
        "plasticity_l2",
        "training_source_seed_count",
        "evaluation_seed_excluded_from_training",
        "confirmatory_rows_used_for_fit",
        "selector_fit_count",
        "train_split",
        "train_endpoint",
        "test_endpoint",
        "strict_unseen_composition",
        "unconditional_cell_retained",
        "primary_endpoint_eligible",
        "primary_scope",
        "deployment_fallback_applied",
        "selected_active_actuator_feasible",
        "matched_budget_support_eligible",
        *PROBABILITY_COLUMNS,
        *CANDIDATE_UTILITY_COLUMNS,
        *ACTIVE_FEASIBILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Exp29 selector rows lack columns: {sorted(missing)}")
    values = records.copy()
    integer_columns = (
        "seed",
        "selector_fit_seed",
        "evaluation_seed",
        "outer_seed",
        "source_seed",
        "training_source_seed_count",
        "confirmatory_rows_used_for_fit",
        "selector_fit_count",
    )
    for column in integer_columns:
        numeric = pd.to_numeric(values[column], errors="raise")
        if not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"{column} must contain exact integers")
        values[column] = numeric.astype(np.int64)
    if set(values["seed"]) != {2801} or set(values["selector_fit_seed"]) != {2801}:
        raise ValueError("Exp29 rows are not bound to fit seed 2801")
    expected_seeds = set(EXPECTED_EVALUATION_SEEDS)
    if set(values["evaluation_seed"]) != expected_seeds:
        raise ValueError("Exp29 evaluation seed coverage mismatch")
    if not bool((values["outer_seed"] == values["evaluation_seed"]).all()) or not bool(
        (values["source_seed"] == values["evaluation_seed"]).all()
    ):
        raise ValueError("Exp29 source/outer seed differs from evaluation seed")
    if not bool(values["status"].eq("complete").all()):
        raise ValueError("unexpected Exp29 failure invalidates confirmatory inference")
    if not bool(values["generator_split"].eq("heldout").all()):
        raise ValueError("Exp29 selector may evaluate only heldout cells")
    if not bool(values["train_split"].eq("discovery").all()) or not bool(
        values["train_endpoint"].eq("validation_balanced_accuracy").all()
    ):
        raise ValueError("Exp29 selector fit split/endpoint mismatch")
    if not bool(
        values["test_endpoint"].eq("test_balanced_accuracy_with_frozen_fallback").all()
    ):
        raise ValueError("Exp29 test endpoint does not include frozen fallback")
    if set(values["training_source_seed_count"]) != {len(EXPECTED_META_SEEDS)}:
        raise ValueError("Exp29 must train on exactly 30 meta seeds")
    if set(values["confirmatory_rows_used_for_fit"]) != {0}:
        raise ValueError("Exp29 confirmatory rows entered selector fitting")
    if set(values["selector_fit_count"]) != {1}:
        raise ValueError("Exp29 selectors were not fitted exactly once")
    excluded = _strict_boolean(
        values["evaluation_seed_excluded_from_training"],
        name="evaluation_seed_excluded_from_training",
    )
    retained = _strict_boolean(
        values["unconditional_cell_retained"], name="unconditional_cell_retained"
    )
    primary = _strict_boolean(
        values["primary_endpoint_eligible"], name="primary_endpoint_eligible"
    )
    _strict_boolean(
        values["strict_unseen_composition"], name="strict_unseen_composition"
    )
    if not bool(excluded.all()) or not bool(retained.all()) or not bool(primary.all()):
        raise ValueError("Exp29 unconditional/no-leakage flags are invalid")
    if not bool(
        values["primary_scope"].eq("unconditional_all_registered_heldout_cells").all()
    ):
        raise ValueError("Exp29 primary scope is not unconditional")

    group_keys = ["evaluation_seed", "generator_id"]
    counts = values.groupby(group_keys, sort=False)["selector"].agg(["size", "nunique"])
    if not bool(
        (
            (counts["size"] == len(SELECTORS)) & (counts["nunique"] == len(SELECTORS))
        ).all()
    ):
        raise ValueError("each Exp29 cell must contain four selector rows")
    selector_sets = values.groupby(group_keys, sort=False)["selector"].agg(set)
    if not bool(selector_sets.map(lambda item: item == set(SELECTORS)).all()):
        raise ValueError("Exp29 selector registry differs from preregistration")
    heldout_counts = values.groupby("evaluation_seed")["generator_id"].nunique()
    if len(heldout_counts) != 30 or set(heldout_counts) != {44}:
        raise ValueError("Exp29 must retain all 44 heldout cells per seed")
    if len(values) != 30 * 44 * len(SELECTORS):
        raise ValueError("Exp29 unconditional Cartesian coverage is incomplete")

    numeric_columns = (
        "utility",
        "oracle_utility",
        "fixed_best_utility",
        "plasticity_l1",
        "plasticity_l2",
        *PROBABILITY_COLUMNS,
        *CANDIDATE_UTILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    )
    for column in numeric_columns:
        values[column] = pd.to_numeric(values[column], errors="raise")
    numeric = values[list(numeric_columns)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Exp29 utilities, probabilities, or costs are non-finite")
    utility_columns = (
        "utility",
        "oracle_utility",
        "fixed_best_utility",
        *CANDIDATE_UTILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    )
    utility_values = values[list(utility_columns)].to_numpy(dtype=np.float64)
    if np.any((utility_values < 0.0) | (utility_values > 1.0)):
        raise ValueError("Exp29 balanced accuracy must lie in [0, 1]")
    if np.any(values[["plasticity_l1", "plasticity_l2"]].to_numpy() < 0.0):
        raise ValueError("Exp29 update costs must be non-negative")
    probabilities = values[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Exp29 decision probabilities are invalid")
    selected_indices = np.argmax(probabilities, axis=1)
    modes = np.asarray(DECISION_MODES, dtype=object)
    if not np.array_equal(values["mode_selected"].to_numpy(), modes[selected_indices]):
        raise ValueError("Exp29 probability argmax differs from mode_selected")
    if not np.array_equal(values["requested_mode"], values["mode_selected"]):
        raise ValueError("Exp29 requested and selected modes differ")
    candidate = values[list(CANDIDATE_UTILITY_COLUMNS)].to_numpy(dtype=np.float64)
    selected_utility = candidate[np.arange(len(values)), selected_indices]
    if not np.array_equal(
        values["utility"].to_numpy(dtype=np.float64), selected_utility
    ):
        raise ValueError("Exp29 utility differs from requested deployment utility")

    feasible = np.column_stack(
        [
            np.ones(len(values), dtype=bool),
            *[
                _strict_boolean(values[column], name=column).to_numpy()
                for column in ACTIVE_FEASIBILITY_COLUMNS
            ],
        ]
    )
    fallback = _strict_boolean(
        values["deployment_fallback_applied"],
        name="deployment_fallback_applied",
    ).to_numpy()
    matched = _strict_boolean(
        values["matched_budget_support_eligible"],
        name="matched_budget_support_eligible",
    ).to_numpy()
    for row, index in enumerate(selected_indices):
        requested = DECISION_MODES[index]
        if index == 0:
            if (
                values.iloc[row]["deployed_mode"] != "frozen"
                or bool(fallback[row])
                or bool(matched[row])
                or not pd.isna(values.iloc[row]["selected_active_actuator_feasible"])
            ):
                raise ValueError("Exp29 direct frozen deployment semantics are invalid")
        else:
            selected_feasible = bool(feasible[row, index])
            observed_selected = values.iloc[row]["selected_active_actuator_feasible"]
            if (
                not isinstance(observed_selected, (bool, np.bool_))
                or bool(observed_selected) != selected_feasible
            ):
                raise ValueError("Exp29 selected feasibility flag is invalid")
            expected_deployed = requested if selected_feasible else "frozen"
            if (
                values.iloc[row]["deployed_mode"] != expected_deployed
                or bool(fallback[row]) == selected_feasible
                or bool(matched[row]) != selected_feasible
            ):
                raise ValueError("Exp29 active fallback semantics are invalid")
            if not selected_feasible and candidate[row, index] != candidate[row, 0]:
                raise ValueError("Exp29 infeasible active utility differs from frozen")

    paired = (
        "oracle_mode",
        "fixed_best_mode",
        *CANDIDATE_UTILITY_COLUMNS,
        *ACTIVE_FEASIBILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    )
    if bool(
        (
            values.groupby(group_keys, sort=False)[list(paired)].nunique(dropna=False)
            != 1
        ).any(axis=None)
    ):
        raise ValueError("Exp29 paired cell data differ across selectors")
    oracle_rows = values[values["selector"] == "oracle"]
    oracle_candidate = oracle_rows[list(CANDIDATE_UTILITY_COLUMNS)].to_numpy(
        dtype=np.float64
    )
    oracle_feasible = np.column_stack(
        [
            np.ones(len(oracle_rows), dtype=bool),
            *[
                oracle_rows[column].astype(bool).to_numpy()
                for column in ACTIVE_FEASIBILITY_COLUMNS
            ],
        ]
    )
    expected_oracle = modes[
        np.argmax(np.where(oracle_feasible, oracle_candidate, -np.inf), axis=1)
    ]
    if not np.array_equal(oracle_rows["mode_selected"].to_numpy(), expected_oracle):
        raise ValueError("Exp29 oracle violates feasible-plus-frozen choice set")
    fixed_rows = values[values["selector"] == "fixed_best"]
    if fixed_rows["fixed_best_mode"].nunique() != 1 or not bool(
        (fixed_rows["mode_selected"] == fixed_rows["fixed_best_mode"]).all()
    ):
        raise ValueError("Exp29 fixed-best policy is not globally frozen")
    train_means = fixed_rows.iloc[0][list(TRAIN_UTILITY_COLUMNS)].to_numpy(
        dtype=np.float64
    )
    if (
        fixed_rows["fixed_best_mode"].iloc[0]
        != DECISION_MODES[1:][int(np.argmax(train_means))]
    ):
        raise ValueError("Exp29 fixed-best mode is not the meta-train argmax")
    return values.sort_values(
        ["evaluation_seed", "generator_id", "selector"], kind="mergesort"
    ).reset_index(drop=True)


def _seed_endpoints(
    records: pd.DataFrame, *, noninferiority_fraction: float
) -> tuple[ConfirmatorySeedEndpoint, ...]:
    values = validate_confirmatory_selector_records(records)
    endpoints: list[ConfirmatorySeedEndpoint] = []
    for seed, seed_rows in values.groupby("evaluation_seed", sort=True):
        by_selector = {
            selector: seed_rows[seed_rows["selector"] == selector]
            for selector in SELECTORS
        }
        means = {
            selector: float(np.mean(frame["utility"]))
            for selector, frame in by_selector.items()
        }
        local = by_selector["local_three_factor"]
        gru = by_selector["gru_bptt"]
        cell_rows = by_selector["oracle"]
        fixed_gain = means["oracle"] - means["fixed_best"]
        endpoints.append(
            ConfirmatorySeedEndpoint(
                seed=int(seed),
                registered_generators=int(seed_rows["generator_id"].nunique()),
                oracle_utility=means["oracle"],
                gru_bptt_utility=means["gru_bptt"],
                local_three_factor_utility=means["local_three_factor"],
                fixed_best_utility=means["fixed_best"],
                frozen_utility=float(np.mean(cell_rows["candidate_frozen_utility"])),
                routing_utility=float(np.mean(cell_rows["candidate_routing_utility"])),
                gain_utility=float(np.mean(cell_rows["candidate_gain_utility"])),
                low_rank_utility=float(
                    np.mean(cell_rows["candidate_low_rank_utility"])
                ),
                local_minus_fixed_best=(
                    means["local_three_factor"] - means["fixed_best"]
                ),
                local_noninferiority_contrast=(
                    means["local_three_factor"]
                    - means["fixed_best"]
                    - noninferiority_fraction * fixed_gain
                ),
                routing_infeasible_rate=float(
                    np.mean(~cell_rows["candidate_routing_feasible"].astype(bool))
                ),
                gain_infeasible_rate=float(
                    np.mean(~cell_rows["candidate_gain_feasible"].astype(bool))
                ),
                low_rank_infeasible_rate=float(
                    np.mean(~cell_rows["candidate_low_rank_feasible"].astype(bool))
                ),
                local_fallback_rate=float(
                    np.mean(local["deployment_fallback_applied"].astype(bool))
                ),
                local_matched_budget_eligible_rate=float(
                    np.mean(local["matched_budget_support_eligible"].astype(bool))
                ),
                local_update_l1=float(np.mean(local["plasticity_l1"])),
                local_update_l2=float(np.mean(local["plasticity_l2"])),
                gru_update_l1=float(np.mean(gru["plasticity_l1"])),
                gru_update_l2=float(np.mean(gru["plasticity_l2"])),
            )
        )
    return tuple(endpoints)


def _invalid_conclusion(
    reason: str, *, noninferiority_fraction: float
) -> Exp29SelectorConclusion:
    return Exp29SelectorConclusion(
        conclusion="invalid",
        reason=reason,
        evidence_valid=False,
        confirmatory_eligible=False,
        statistics_unit="evaluation_seed",
        n_seeds=0,
        unconditional_all_registered_cells=True,
        complete_primary_coverage=False,
        noninferiority_fraction=noninferiority_fraction,
        primary_contrasts=(),
        seed_endpoints=(),
    )


def summarize_confirmatory_selector(
    records: pd.DataFrame,
    *,
    invalid_reason: str | None = None,
    noninferiority_fraction: float,
    bootstrap_samples: int,
    permutation_samples: int,
    confidence: float,
    random_seed: int,
    alpha: float,
) -> Exp29SelectorConclusion:
    if invalid_reason is not None:
        return _invalid_conclusion(
            invalid_reason, noninferiority_fraction=noninferiority_fraction
        )
    endpoints = _seed_endpoints(
        records, noninferiority_fraction=noninferiority_fraction
    )
    if {item.seed for item in endpoints} != set(EXPECTED_EVALUATION_SEEDS):
        return _invalid_conclusion(
            "confirmatory seed coverage is incomplete",
            noninferiority_fraction=noninferiority_fraction,
        )
    endpoint_values = {
        "local_noninferiority_contrast": _finite_vector(
            [item.local_noninferiority_contrast for item in endpoints],
            name="local_noninferiority_contrast",
        ),
        "local_minus_fixed_best": _finite_vector(
            [item.local_minus_fixed_best for item in endpoints],
            name="local_minus_fixed_best",
        ),
    }
    positive = np.asarray(
        [
            _one_sided_sign_flip(
                values, seed=random_seed + 100 + index, samples=permutation_samples
            )
            for index, values in enumerate(endpoint_values.values())
        ]
    )
    negative = np.asarray(
        [
            _one_sided_sign_flip(
                -values, seed=random_seed + 200 + index, samples=permutation_samples
            )
            for index, values in enumerate(endpoint_values.values())
        ]
    )
    positive_holm = holm_adjust(positive)
    negative_holm = holm_adjust(negative)
    contrasts: list[ConfirmatoryContrast] = []
    for index, (name, values) in enumerate(endpoint_values.items()):
        lower, upper = _bootstrap_mean_interval(
            values,
            seed=random_seed + index,
            samples=bootstrap_samples,
            confidence=confidence,
        )
        contrasts.append(
            ConfirmatoryContrast(
                name=name,
                mean=float(np.mean(values)),
                lower_confidence=lower,
                upper_confidence=upper,
                p_value=float(positive[index]),
                p_value_holm=float(positive_holm[index]),
                opposition_p_value=float(negative[index]),
                opposition_p_value_holm=float(negative_holm[index]),
            )
        )
    supported = all(
        item.lower_confidence > 0.0 and item.p_value_holm < alpha for item in contrasts
    )
    opposed = any(
        item.upper_confidence <= 0.0 and item.opposition_p_value_holm < alpha
        for item in contrasts
    )
    if supported:
        conclusion = "support"
        reason = "both unconditional seed-level confirmatory endpoints passed"
    elif opposed:
        conclusion = "oppose"
        reason = "at least one preregistered unconditional endpoint was contradicted"
    else:
        conclusion = "inconclusive"
        reason = "the two preregistered confirmatory thresholds were not both met"
    return Exp29SelectorConclusion(
        conclusion=conclusion,
        reason=reason,
        evidence_valid=True,
        confirmatory_eligible=CONFIRMATORY_ELIGIBLE,
        statistics_unit="evaluation_seed",
        n_seeds=len(endpoints),
        unconditional_all_registered_cells=True,
        complete_primary_coverage=True,
        noninferiority_fraction=noninferiority_fraction,
        primary_contrasts=tuple(contrasts),
        seed_endpoints=endpoints,
    )


def _matching_attempt(
    results_root: Path, *, selector_fit_seed: int, run_label: str
) -> Path:
    seed_root = results_root / "runs" / EXPERIMENT / f"seed_{selector_fit_seed:04d}"
    if not seed_root.is_dir():
        raise ValueError("Exp29 selector attempt is missing")
    attempts = [path for path in seed_root.iterdir() if path.is_dir()]
    if len(attempts) != 1:
        raise ValueError("Exp29 requires exactly one selector attempt")
    attempt = attempts[0]
    stored = _read_json(attempt / "config.json")
    if (
        not isinstance(stored, Mapping)
        or stored.get("seed") != selector_fit_seed
        or stored.get("run_label") != run_label
    ):
        raise ValueError("Exp29 one-shot attempt identity mismatch")
    return attempt


def collect_exp29(
    results_root: str | Path,
    *,
    config: Mapping[str, Any],
    run_label: str,
) -> Exp29Collection:
    _validate_config(config)
    if analysis_contract_sha256(config) != REGISTERED_ANALYSIS_CONTRACT_SHA256:
        raise ValueError("Exp29 summarizer analysis-contract SHA-256 mismatch")
    _validate_implementation_binding(config)
    if run_label != REQUIRED_RUN_LABEL:
        raise ValueError("Exp29 collector requires the registered run label")
    fit_seed = int(config["selector_fit_seed"])
    attempt = _matching_attempt(
        Path(results_root), selector_fit_seed=fit_seed, run_label=run_label
    )
    status = _read_json(attempt / "status.json")
    if not isinstance(status, Mapping):
        raise ValueError("Exp29 attempt status is malformed")
    stored_config = _read_json(attempt / "config.json")
    if not isinstance(stored_config, Mapping):
        raise ValueError("Exp29 stored config is malformed")
    for key, expected in config.items():
        if stored_config.get(key) != expected:
            raise ValueError(f"Exp29 stored config differs at {key}")
    config_sha = canonical_config_sha256(config)
    provenance = stored_config.get("evidence_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema_version") != "exp29_confirmatory_selector_evidence_v1"
        or provenance.get("canonical_config_sha256") != config_sha
        or provenance.get("inference_scope") != INFERENCE_SCOPE
        or provenance.get("inference_status") != INFERENCE_STATUS
        or provenance.get("confirmatory_eligible") is not True
        or provenance.get("one_shot_attempt") is not True
    ):
        raise ValueError("Exp29 stored evidence provenance mismatch")
    run_git = provenance.get("git")
    if not isinstance(run_git, Mapping) or run_git.get("dirty") is not False:
        raise ValueError("Exp29 evidence Git identity is dirty or missing")
    commit = run_git.get("commit")
    tree = run_git.get("tree")
    if not isinstance(commit, str) or not isinstance(tree, str):
        raise ValueError("Exp29 evidence Git commit/tree is missing")
    environment = _read_json(attempt / "environment.json")
    if not isinstance(environment, Mapping) or environment.get("git") != run_git:
        raise ValueError("Exp29 environment Git identity mismatch")
    plans = _read_json(attempt / "planned_conditions.json")
    metrics = _read_jsonl(attempt / "metrics.jsonl")
    if not isinstance(plans, list):
        raise ValueError("Exp29 planned conditions must be a JSON list")
    expected_plan = _planned_conditions(config)
    normalized_plan = [
        {key: row.get(key) for key in PLAN_KEYS}
        for row in plans
        if isinstance(row, Mapping)
    ]
    if normalized_plan != expected_plan:
        raise ValueError("Exp29 planned conditions differ from registration")
    if [row.get("condition_index") for row in plans] != list(range(len(plans))):
        raise ValueError("Exp29 condition indexes are not contiguous")
    if len(metrics) != len(expected_plan):
        raise ValueError("Exp29 failed to retain every planned condition")
    normalized_metrics = [{key: row.get(key) for key in PLAN_KEYS} for row in metrics]
    if normalized_metrics != expected_plan:
        raise ValueError("Exp29 metric ordering/coverage differs from plan")
    frame = pd.DataFrame(metrics)
    run_status = status.get("status")
    unexpected_rows = int(np.sum(~frame["status"].eq("complete")))
    if run_status != "complete" or unexpected_rows:
        reason = (
            f"unexpected Exp29 execution failure: run_status={run_status}, "
            f"noncomplete_rows={unexpected_rows}"
        )
        return Exp29Collection(
            raw=frame,
            config=dict(config),
            config_sha256=config_sha,
            attempt_path=str(attempt.resolve()),
            run_git_commit=commit,
            run_git_tree=tree,
            runtime_identity=dict(environment),
            run_label=run_label,
            evidence_valid=False,
            invalid_reason=reason,
            source_receipt=None,
            fit_receipt=None,
            decision_receipt=None,
        )

    source_path = attempt / "source_package_receipt.json"
    fit_path = attempt / "selector_fit_receipt.json"
    decision_path = attempt / "decision_receipt.json"
    source_receipt = _validate_self_hash(
        _read_json(source_path), name=source_path.name, schema=SOURCE_RECEIPT_SCHEMA
    )
    fit_receipt = _validate_self_hash(
        _read_json(fit_path), name=fit_path.name, schema=FIT_RECEIPT_SCHEMA
    )
    decision_receipt = _validate_self_hash(
        _read_json(decision_path),
        name=decision_path.name,
        schema=DECISION_RECEIPT_SCHEMA,
    )
    if (
        decision_receipt.get("selector_fit_receipt_sha256")
        != fit_receipt.get("receipt_sha256")
        or fit_receipt.get("selector_fit_seed") != fit_seed
        or fit_receipt.get("fit_count") != 1
        or fit_receipt.get("confirmatory_rows_used_for_fit") != 0
    ):
        raise ValueError("Exp29 fit/decision receipt linkage is invalid")
    decision_selectors = decision_receipt.get("selectors")
    if not isinstance(decision_selectors, Mapping) or set(decision_selectors) != set(
        SELECTORS
    ):
        raise ValueError("Exp29 decision receipt selector registry mismatch")
    for selector in SELECTORS:
        value = decision_selectors[selector]
        if not isinstance(value, Mapping):
            raise ValueError(f"Exp29 {selector} decision receipt is missing")
        payload = dict(value)
        digest = payload.pop("receipt_sha256", None)
        if digest != _payload_sha256(payload):
            raise ValueError(f"Exp29 {selector} decision self-hash mismatch")

    meta, folds, replay_source = _load_sources(config)
    replay_fit = _fit_frozen_selectors(meta, folds, config)
    if _payload_sha256(source_receipt) != _payload_sha256(replay_source):
        raise ValueError("Exp29 source receipt replay mismatch")
    if _payload_sha256(fit_receipt) != _payload_sha256(replay_fit.fit_receipt):
        raise ValueError("Exp29 frozen selector fit replay mismatch")
    if _payload_sha256(decision_receipt) != _payload_sha256(
        replay_fit.decision_receipt
    ):
        raise ValueError("Exp29 decision replay mismatch")
    replay_records = _semantic_records(
        meta,
        folds,
        replay_fit,
        config,
        run_label=run_label,
        config_sha256=config_sha,
        run_git=run_git,
        source_receipt=replay_source,
        source_receipt_file_sha256=_file_sha256(source_path),
        fit_receipt_file_sha256=_file_sha256(fit_path),
        decision_receipt_file_sha256=_file_sha256(decision_path),
    )
    expected_semantic = [
        {**dimensions, **row_metrics} for row_metrics, dimensions in replay_records
    ]
    observed_semantic = [
        {
            key: value
            for key, value in row.items()
            if key not in {"run_id", "experiment", "seed", "recorded_at"}
        }
        for row in metrics
    ]
    if [_payload_sha256(row) for row in observed_semantic] != [
        _payload_sha256(row) for row in expected_semantic
    ]:
        raise ValueError("Exp29 deterministic semantic-row replay mismatch")
    row_bindings = {
        "protocol": PROTOCOL_VERSION,
        "run_label": run_label,
        "selector_fit_seed": fit_seed,
        "exp29_config_sha256": config_sha,
        "run_git_commit": commit,
        "run_git_tree": tree,
        "run_git_dirty": False,
        "inference_scope": INFERENCE_SCOPE,
        "inference_status": INFERENCE_STATUS,
        "confirmatory_eligible": True,
        "profile": PROFILE,
    }
    for column, expected in row_bindings.items():
        if column not in frame or not bool(
            frame[column].map(lambda value: value == expected).all()
        ):
            raise ValueError(f"Exp29 row provenance mismatch: {column}")
    for selector in SELECTORS:
        receipt = decision_selectors[selector]
        rows = frame[frame["selector"] == selector]
        seeds = rows["evaluation_seed"].to_numpy(dtype=np.int64)
        identifiers = rows["generator_id"].astype(str).tolist()
        probabilities = rows[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if (
            receipt["evaluation_seeds"] != seeds.tolist()
            or receipt["generator_ids"] != identifiers
            or receipt["decision_modes"] != list(DECISION_MODES)
            or not np.array_equal(
                np.asarray(receipt["probabilities"], dtype=np.float64), probabilities
            )
            or receipt["decision_fingerprint"]
            != _decision_fingerprint(seeds, identifiers, probabilities)
        ):
            raise ValueError(f"Exp29 {selector} decisions are not receipt-bound")
    validate_confirmatory_selector_records(frame)
    current_git = git_identity()
    if current_git.get("dirty") is not False or (
        current_git.get("commit"),
        current_git.get("tree"),
    ) != (commit, tree):
        raise ValueError("Exp29 summary must run clean on the evidence commit/tree")
    return Exp29Collection(
        raw=frame,
        config=dict(config),
        config_sha256=config_sha,
        attempt_path=str(attempt.resolve()),
        run_git_commit=commit,
        run_git_tree=tree,
        runtime_identity=dict(environment),
        run_label=run_label,
        evidence_valid=True,
        invalid_reason=None,
        source_receipt=source_receipt,
        fit_receipt=fit_receipt,
        decision_receipt=decision_receipt,
    )


def plot_exp29_selector(
    conclusion: Exp29SelectorConclusion, output_base: str | Path
) -> tuple[Path, Path]:
    """Create the registered contrast and feasibility audit figure."""

    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    if not conclusion.evidence_valid:
        for axis in axes:
            axis.axis("off")
        axes[0].text(
            0.5,
            0.5,
            "INVALID CONFIRMATORY EVIDENCE",
            ha="center",
            va="center",
            fontsize=14,
            color="#b2182b",
            transform=axes[0].transAxes,
        )
        axes[1].text(
            0.5,
            0.5,
            conclusion.reason,
            ha="center",
            va="center",
            wrap=True,
            transform=axes[1].transAxes,
        )
    else:
        endpoints = conclusion.seed_endpoints
        contrast_names = (
            "local_noninferiority_contrast",
            "local_minus_fixed_best",
        )
        colors = ("#2166ac", "#b2182b")
        for index, (name, color) in enumerate(zip(contrast_names, colors, strict=True)):
            values = np.asarray([getattr(item, name) for item in endpoints])
            jitter = np.linspace(-0.09, 0.09, len(values))
            axes[0].scatter(
                np.full(len(values), index) + jitter,
                values,
                s=18,
                alpha=0.65,
                color=color,
            )
            summary = next(
                item for item in conclusion.primary_contrasts if item.name == name
            )
            axes[0].errorbar(
                index,
                summary.mean,
                yerr=[
                    [summary.mean - summary.lower_confidence],
                    [summary.upper_confidence - summary.mean],
                ],
                fmt="o",
                color="black",
                capsize=4,
                zorder=5,
            )
        axes[0].axhline(0.0, color="0.4", linewidth=1, linestyle="--")
        axes[0].set_xticks(range(2), ["NI vs oracle gain", "Local - fixed"])
        axes[0].set_ylabel("Seed-level contrast")
        axes[0].set_title("Unconditional confirmatory endpoints")

        feasibility = np.asarray(
            [
                [
                    item.routing_infeasible_rate,
                    item.gain_infeasible_rate,
                    item.low_rank_infeasible_rate,
                    item.local_fallback_rate,
                ]
                for item in endpoints
            ]
        )
        labels = ("routing", "gain", "low-rank", "local fallback")
        means = np.mean(feasibility, axis=0)
        axes[1].bar(range(4), means, color=("#4d9221", "#c51b7d", "#762a83", "#fdae61"))
        for index in range(4):
            jitter = np.linspace(-0.09, 0.09, feasibility.shape[0])
            axes[1].scatter(
                np.full(feasibility.shape[0], index) + jitter,
                feasibility[:, index],
                s=12,
                color="black",
                alpha=0.35,
            )
        axes[1].set_xticks(range(4), labels, rotation=20)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_ylabel("Fraction of registered cells")
        axes[1].set_title("Feasibility and frozen fallback")
    fig.suptitle(f"Exp29 selector: {conclusion.conclusion.upper()}")
    fig.tight_layout()
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _report(collection: Exp29Collection, conclusion: Exp29SelectorConclusion) -> str:
    if conclusion.primary_contrasts:
        rows = "\n".join(
            "| {name} | {mean:.6f} | [{lower:.6f}, {upper:.6f}] | {pos:.6g} | {neg:.6g} |".format(
                name=item.name,
                mean=item.mean,
                lower=item.lower_confidence,
                upper=item.upper_confidence,
                pos=item.p_value_holm,
                neg=item.opposition_p_value_holm,
            )
            for item in conclusion.primary_contrasts
        )
    else:
        rows = "| unavailable | — | — | — | — |"
    return f"""# Exp29 one-shot confirmatory actuator selector

## Conclusion

**{conclusion.conclusion.upper()}** — {conclusion.reason}.

Evidence validity: `{str(conclusion.evidence_valid).lower()}`. The statistical
unit is the evaluation seed (`n={conclusion.n_seeds}`). Every one of the 44
registered heldout cells per seed enters both primary endpoints, including
cells with infeasible active actuators. Infeasible selections receive their
same-cell frozen utility; they do not support a matched-budget mechanism claim.

| Endpoint | Seed mean | 95% bootstrap CI | Positive Holm p | Negative Holm p |
|---|---:|---:|---:|---:|
{rows}

## Audit boundary

- Meta fitting data: Exp26 seeds `0--29`, discovery/validation only.
- Confirmatory evaluation data: Exp29 seeds `60--89`, heldout/test only.
- Selector fit count: `1`; root fit seed: `2801`.
- Source rows used for fitting: `0`.
- Attempt path: `{collection.attempt_path}`.
- Git commit/tree: `{collection.run_git_commit}` / `{collection.run_git_tree}`.
- Config SHA-256: `{collection.config_sha256}`.
- Re-run or replacement attempts are forbidden.

The oracle is a test-aware feasible-plus-frozen ceiling. GRU-BPTT is an
isolated baseline; the local selector uses neither autograd nor BPTT. A support
classification concerns unconditional deployed selector utility over the
frozen dictionary. It does not convert infeasible fallback rows into
matched-budget evidence, establish hidden-context inference, or show online
scalar-reward learning.
"""


def write_exp29_summary(
    collection: Exp29Collection,
    output_dir: str | Path,
    *,
    make_figure: bool = True,
) -> Exp29SelectorConclusion:
    analysis = collection.config["analysis"]
    evaluation = collection.config["evaluation"]
    conclusion = summarize_confirmatory_selector(
        collection.raw,
        invalid_reason=collection.invalid_reason,
        noninferiority_fraction=float(
            evaluation["local_oracle_gain_fraction_threshold"]
        ),
        bootstrap_samples=int(analysis["bootstrap_samples"]),
        permutation_samples=int(analysis["permutation_samples"]),
        confidence=float(analysis["confidence"]),
        random_seed=int(analysis["statistics_seed"]),
        alpha=float(analysis["alpha"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "raw_metrics.csv.gz"
    collection.raw.to_csv(
        raw_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    endpoints = pd.DataFrame(asdict(item) for item in conclusion.seed_endpoints)
    endpoints.to_csv(output / "seed_endpoints.csv", index=False)
    pd.DataFrame(asdict(item) for item in conclusion.primary_contrasts).to_csv(
        output / "summary.csv", index=False
    )
    payload = {
        **conclusion.to_dict(),
        "profile": PROFILE,
        "protocol_version": PROTOCOL_VERSION,
        "run_label": collection.run_label,
        "inference_status": INFERENCE_STATUS,
        "inference_scope": INFERENCE_SCOPE,
        "canonical_config_sha256": collection.config_sha256,
        "run_git_commit": collection.run_git_commit,
        "run_git_tree": collection.run_git_tree,
        "raw_metrics_sha256": _file_sha256(raw_path),
        "attempt_path": collection.attempt_path,
        "source_receipt_sha256": (
            collection.source_receipt.get("receipt_sha256")
            if collection.source_receipt is not None
            else None
        ),
        "selector_fit_receipt_sha256": (
            collection.fit_receipt.get("receipt_sha256")
            if collection.fit_receipt is not None
            else None
        ),
        "decision_receipt_sha256": (
            collection.decision_receipt.get("receipt_sha256")
            if collection.decision_receipt is not None
            else None
        ),
    }
    _write_json(output / "conclusion.json", payload)
    _write_json(
        output / "provenance.json",
        {
            **payload,
            "runtime_identity": collection.runtime_identity,
            "source_receipt": collection.source_receipt,
            "fit_receipt": collection.fit_receipt,
            "decision_receipt": collection.decision_receipt,
        },
    )
    (output / "report.md").write_text(_report(collection, conclusion), encoding="utf-8")
    if make_figure:
        plot_exp29_selector(conclusion, output / "exp29_selector_evidence")
    return conclusion


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-label", default=REQUIRED_RUN_LABEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    config = load_json_config(args.config)
    collection = collect_exp29(
        args.results_root, config=config, run_label=args.run_label
    )
    conclusion = write_exp29_summary(
        collection, args.output_dir, make_figure=not args.no_figure
    )
    print(
        json.dumps(
            {
                "conclusion": conclusion.conclusion,
                "evidence_valid": conclusion.evidence_valid,
                "n_seeds": conclusion.n_seeds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
