"""Seed-level confirmatory metrics for the frozen-actuator selector.

Exp27 evaluates four selector policies on the same held-out generator cells,
but generator cells are not independent replicates.  This module therefore
validates the paired panel, reduces every endpoint to one value per outer
network seed, and only then bootstraps or permutes those seed-level values.
The oracle is a test-aware ceiling; it is never a trainable comparator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from src.analysis.actuator_phase_statistics import holm_adjust


FloatArray = NDArray[np.float64]
CANDIDATE_MODES = ("routing", "gain", "low_rank")
SELECTOR_MODES = ("oracle", "gru_bptt", "local_three_factor", "fixed_best")
CANDIDATE_UTILITY_COLUMNS = tuple(
    f"candidate_{mode}_utility" for mode in CANDIDATE_MODES
)
TRAIN_UTILITY_COLUMNS = tuple(
    f"train_mean_candidate_{mode}_utility" for mode in CANDIDATE_MODES
)


def _finite_vector(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric vector")
    vector = np.asarray(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _bootstrap_mean_interval(
    values: FloatArray,
    *,
    seed: int,
    samples: int,
    confidence: float,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = np.mean(values[indices], axis=1)
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _one_sided_sign_flip(
    values: FloatArray,
    *,
    seed: int,
    samples: int,
) -> float:
    """Test a positive paired seed-level mean against zero."""

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


def _strict_boolean(series: pd.Series, *, name: str) -> pd.Series:
    if not bool(series.map(lambda value: isinstance(value, (bool, np.bool_))).all()):
        raise TypeError(f"{name} must contain literal booleans")
    return series.astype(bool)


def validate_selector_records(
    records: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_primary_generators_per_seed: int | None = None,
) -> pd.DataFrame:
    """Return the validated strict-unseen held-out selector panel.

    Candidate utilities are repeated on all four policy rows so the audit can
    prove that every policy was scored on the exact same immutable Exp26 cell.
    Any missing policy, duplicate row, non-finite utility, or selected-utility
    mismatch raises instead of silently reducing the evidential panel.
    """

    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    expected = tuple(int(seed) for seed in expected_seeds)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_seeds must be non-empty and unique")
    if expected_primary_generators_per_seed is not None and (
        isinstance(expected_primary_generators_per_seed, bool)
        or expected_primary_generators_per_seed < 1
    ):
        raise ValueError("expected_primary_generators_per_seed must be positive")
    required = {
        "seed",
        "outer_seed",
        "source_seed",
        "generator_id",
        "generator_split",
        "strict_unseen_composition",
        "primary_endpoint_eligible",
        "composition_overlap_secondary",
        "selector",
        "mode_selected",
        "oracle_mode",
        "fixed_best_mode",
        "utility",
        "status",
        "plasticity_l1",
        "plasticity_l2",
        "training_source_seed_count",
        "outer_seed_excluded_from_training",
        "train_split",
        "train_endpoint",
        "test_endpoint",
        *CANDIDATE_UTILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"selector records missing columns: {sorted(missing)}")

    values = records.copy()
    for column in ("seed", "outer_seed", "source_seed"):
        numeric = pd.to_numeric(values[column], errors="raise")
        if not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"{column} must contain exact integers")
        values[column] = numeric.astype(np.int64)
    if not bool((values["seed"] == values["outer_seed"]).all()):
        raise ValueError("ExperimentRun seed and outer_seed differ")
    if not bool((values["source_seed"] == values["outer_seed"]).all()):
        raise ValueError("held-out source_seed must equal outer_seed")
    observed = set(int(seed) for seed in values["outer_seed"].unique())
    if observed != set(expected):
        raise ValueError(
            f"observed outer seeds {sorted(observed)} differ from {list(expected)}"
        )
    if not bool((values["generator_split"] == "heldout").all()):
        raise ValueError("selector output may contain only heldout generator rows")
    if not bool((values["train_split"] == "discovery").all()):
        raise ValueError("selector fitting must use only discovery generators")
    if not bool((values["train_endpoint"] == "validation_balanced_accuracy").all()):
        raise ValueError("selector fitting must use only validation utility")
    if not bool((values["test_endpoint"] == "test_balanced_accuracy").all()):
        raise ValueError("selector evaluation must use the held-out test utility")
    excluded = _strict_boolean(
        values["outer_seed_excluded_from_training"],
        name="outer_seed_excluded_from_training",
    )
    if not bool(excluded.all()):
        raise ValueError("outer seed entered selector training")
    training_seed_count = pd.to_numeric(
        values["training_source_seed_count"], errors="raise"
    )
    if not np.all(np.equal(training_seed_count, np.floor(training_seed_count))):
        raise ValueError("training_source_seed_count must contain exact integers")
    if not bool((training_seed_count == len(expected) - 1).all()):
        raise ValueError("training source seed count does not match outer LOSO")
    if not bool(values["status"].eq("complete").all()):
        raise ValueError("all selector rows must be complete")
    values["strict_unseen_composition"] = _strict_boolean(
        values["strict_unseen_composition"], name="strict_unseen_composition"
    )
    values["primary_endpoint_eligible"] = _strict_boolean(
        values["primary_endpoint_eligible"], name="primary_endpoint_eligible"
    )
    values["composition_overlap_secondary"] = _strict_boolean(
        values["composition_overlap_secondary"],
        name="composition_overlap_secondary",
    )
    if not bool(
        (
            values["strict_unseen_composition"] == values["primary_endpoint_eligible"]
        ).all()
    ):
        raise ValueError("strict-unseen and primary-eligibility flags differ")
    if not bool(
        (
            values["composition_overlap_secondary"]
            == ~values["strict_unseen_composition"]
        ).all()
    ):
        raise ValueError("composition-overlap audit is inconsistent")

    primary = values[values["strict_unseen_composition"]].copy()
    if primary.empty:
        raise ValueError("no strict-unseen held-out composition is present")
    group_keys = ["outer_seed", "generator_id"]
    selector_counts = primary.groupby(group_keys, sort=False)["selector"].agg(
        ["size", "nunique"]
    )
    if not bool(
        (
            (selector_counts["size"] == len(SELECTOR_MODES))
            & (selector_counts["nunique"] == len(SELECTOR_MODES))
        ).all()
    ):
        raise ValueError("every strict-unseen cell needs exactly four selector rows")
    selector_sets = primary.groupby(group_keys, sort=False)["selector"].agg(set)
    if not bool(selector_sets.map(lambda item: item == set(SELECTOR_MODES)).all()):
        raise ValueError("selector panel differs from the registered selector set")
    per_seed = primary.groupby("outer_seed", sort=True)["generator_id"].nunique()
    if per_seed.nunique() != 1:
        raise ValueError("strict-unseen generator count differs across outer seeds")
    if expected_primary_generators_per_seed is not None and not bool(
        (per_seed == expected_primary_generators_per_seed).all()
    ):
        raise ValueError("strict-unseen generator count differs from registration")

    numeric_columns = (
        "utility",
        "plasticity_l1",
        "plasticity_l2",
        *CANDIDATE_UTILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    )
    for column in numeric_columns:
        primary[column] = pd.to_numeric(primary[column], errors="raise")
    if not np.all(np.isfinite(primary[list(numeric_columns)].to_numpy())):
        raise ValueError("selector utilities and plasticity costs must be finite")
    if bool(
        (
            (
                primary[["utility", *CANDIDATE_UTILITY_COLUMNS, *TRAIN_UTILITY_COLUMNS]]
                < 0.0
            )
            | (
                primary[["utility", *CANDIDATE_UTILITY_COLUMNS, *TRAIN_UTILITY_COLUMNS]]
                > 1.0
            )
        ).any(axis=None)
    ):
        raise ValueError("balanced-accuracy utilities must lie in [0, 1]")
    if bool((primary[["plasticity_l1", "plasticity_l2"]] < 0.0).any(axis=None)):
        raise ValueError("plasticity costs must be non-negative")

    paired_columns = (
        "source_seed",
        "oracle_mode",
        "fixed_best_mode",
        *CANDIDATE_UTILITY_COLUMNS,
        *TRAIN_UTILITY_COLUMNS,
    )
    paired_counts = primary.groupby(group_keys, sort=False)[
        list(paired_columns)
    ].nunique(dropna=False)
    if bool((paired_counts != 1).any(axis=None)):
        raise ValueError("paired candidate data differ across selector rows")
    if not bool(primary["mode_selected"].isin(CANDIDATE_MODES).all()):
        raise ValueError("mode_selected is outside the frozen candidate dictionary")
    if not bool(primary["oracle_mode"].isin(CANDIDATE_MODES).all()):
        raise ValueError("oracle_mode is outside the frozen candidate dictionary")
    if not bool(primary["fixed_best_mode"].isin(CANDIDATE_MODES).all()):
        raise ValueError("fixed_best_mode is outside the frozen candidate dictionary")

    candidate_matrix = primary[list(CANDIDATE_UTILITY_COLUMNS)].to_numpy()
    selected_indices = (
        primary["mode_selected"]
        .map({mode: index for index, mode in enumerate(CANDIDATE_MODES)})
        .to_numpy(dtype=np.int64)
    )
    selected_utility = candidate_matrix[np.arange(primary.shape[0]), selected_indices]
    if not np.allclose(
        primary["utility"].to_numpy(), selected_utility, rtol=0.0, atol=1e-12
    ):
        raise ValueError("reported utility does not match the selected frozen actuator")
    oracle_rows = primary[primary["selector"] == "oracle"]
    if not np.allclose(
        oracle_rows["utility"].to_numpy(),
        oracle_rows[list(CANDIDATE_UTILITY_COLUMNS)].max(axis=1).to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("oracle rows are not the per-cell test-aware ceiling")
    if not bool((oracle_rows["mode_selected"] == oracle_rows["oracle_mode"]).all()):
        raise ValueError("oracle mode_selected differs from oracle_mode")
    registered_order = np.asarray(CANDIDATE_MODES)
    expected_oracle_mode = registered_order[
        np.argmax(oracle_rows[list(CANDIDATE_UTILITY_COLUMNS)].to_numpy(), axis=1)
    ]
    if not np.array_equal(oracle_rows["oracle_mode"].to_numpy(), expected_oracle_mode):
        raise ValueError("oracle_mode violates the registered argmax tie break")
    fixed_rows = primary[primary["selector"] == "fixed_best"]
    if not bool((fixed_rows["mode_selected"] == fixed_rows["fixed_best_mode"]).all()):
        raise ValueError("fixed-best selector does not use its train-only fixed mode")
    fixed_modes_per_seed = fixed_rows.groupby("outer_seed")["fixed_best_mode"].nunique()
    if not bool((fixed_modes_per_seed == 1).all()):
        raise ValueError("fixed-best mode must be constant within outer seed")
    fixed_training = fixed_rows.groupby("outer_seed", sort=True).first()
    expected_fixed_mode = registered_order[
        np.argmax(fixed_training[list(TRAIN_UTILITY_COLUMNS)].to_numpy(), axis=1)
    ]
    if not np.array_equal(
        fixed_training["fixed_best_mode"].to_numpy(), expected_fixed_mode
    ):
        raise ValueError("fixed_best_mode is not the train-only registered argmax")
    return primary.sort_values([*group_keys, "selector"], kind="mergesort").reset_index(
        drop=True
    )


@dataclass(frozen=True)
class SeedSelectorEndpoint:
    seed: int
    strict_unseen_generators: int
    fixed_best_mode: str
    routing_utility: float
    gain_utility: float
    low_rank_utility: float
    fixed_best_utility: float
    oracle_utility: float
    gru_bptt_utility: float
    local_three_factor_utility: float
    local_minus_fixed_best: float
    oracle_minus_fixed_best: float
    gru_minus_fixed_best: float
    local_noninferiority_contrast: float
    local_regret: float
    gru_regret: float
    local_selection_accuracy: float
    gru_selection_accuracy: float
    local_recovered_oracle_gain: float
    gru_recovered_oracle_gain: float
    positive_oracle_gain_cells: int
    local_cell_recovery_mean: float
    gru_cell_recovery_mean: float
    local_update_l1: float
    local_update_l2: float
    gru_update_l1: float
    gru_update_l2: float


def seed_selector_endpoints(
    records: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_primary_generators_per_seed: int | None = None,
    noninferiority_fraction: float = 0.8,
) -> tuple[SeedSelectorEndpoint, ...]:
    """Reduce the paired strict-unseen panel to independent seed endpoints."""

    if not np.isfinite(noninferiority_fraction) or not (
        0.0 < noninferiority_fraction <= 1.0
    ):
        raise ValueError("noninferiority_fraction must lie in (0, 1]")
    primary = validate_selector_records(
        records,
        expected_seeds=expected_seeds,
        expected_primary_generators_per_seed=expected_primary_generators_per_seed,
    )
    endpoints: list[SeedSelectorEndpoint] = []
    for seed, seed_frame in primary.groupby("outer_seed", sort=True):
        policy = seed_frame.pivot(
            index="generator_id", columns="selector", values="utility"
        )
        cell = seed_frame.groupby("generator_id", sort=True).first()
        fixed_gain = policy["oracle"] - policy["fixed_best"]
        applicable = fixed_gain > 1e-12
        local_gain = policy["local_three_factor"] - policy["fixed_best"]
        gru_gain = policy["gru_bptt"] - policy["fixed_best"]
        oracle_gain_mean = float(np.mean(fixed_gain))
        local_gain_mean = float(np.mean(local_gain))
        gru_gain_mean = float(np.mean(gru_gain))
        local_rows = seed_frame[seed_frame["selector"] == "local_three_factor"]
        gru_rows = seed_frame[seed_frame["selector"] == "gru_bptt"]
        endpoints.append(
            SeedSelectorEndpoint(
                seed=int(seed),
                strict_unseen_generators=int(policy.shape[0]),
                fixed_best_mode=str(cell["fixed_best_mode"].iloc[0]),
                routing_utility=float(np.mean(cell["candidate_routing_utility"])),
                gain_utility=float(np.mean(cell["candidate_gain_utility"])),
                low_rank_utility=float(np.mean(cell["candidate_low_rank_utility"])),
                fixed_best_utility=float(np.mean(policy["fixed_best"])),
                oracle_utility=float(np.mean(policy["oracle"])),
                gru_bptt_utility=float(np.mean(policy["gru_bptt"])),
                local_three_factor_utility=float(np.mean(policy["local_three_factor"])),
                local_minus_fixed_best=local_gain_mean,
                oracle_minus_fixed_best=oracle_gain_mean,
                gru_minus_fixed_best=gru_gain_mean,
                local_noninferiority_contrast=float(
                    local_gain_mean - noninferiority_fraction * oracle_gain_mean
                ),
                local_regret=float(
                    np.mean(policy["oracle"] - policy["local_three_factor"])
                ),
                gru_regret=float(np.mean(policy["oracle"] - policy["gru_bptt"])),
                local_selection_accuracy=float(
                    np.mean(
                        local_rows["mode_selected"].to_numpy()
                        == local_rows["oracle_mode"].to_numpy()
                    )
                ),
                gru_selection_accuracy=float(
                    np.mean(
                        gru_rows["mode_selected"].to_numpy()
                        == gru_rows["oracle_mode"].to_numpy()
                    )
                ),
                local_recovered_oracle_gain=float(
                    local_gain_mean / oracle_gain_mean
                    if oracle_gain_mean > 1e-12
                    else float("nan")
                ),
                gru_recovered_oracle_gain=float(
                    gru_gain_mean / oracle_gain_mean
                    if oracle_gain_mean > 1e-12
                    else float("nan")
                ),
                positive_oracle_gain_cells=int(np.sum(applicable)),
                local_cell_recovery_mean=float(
                    np.mean(
                        (local_gain[applicable] / fixed_gain[applicable]).to_numpy()
                    )
                    if bool(applicable.any())
                    else float("nan")
                ),
                gru_cell_recovery_mean=float(
                    np.mean((gru_gain[applicable] / fixed_gain[applicable]).to_numpy())
                    if bool(applicable.any())
                    else float("nan")
                ),
                local_update_l1=float(np.mean(local_rows["plasticity_l1"])),
                local_update_l2=float(np.mean(local_rows["plasticity_l2"])),
                gru_update_l1=float(np.mean(gru_rows["plasticity_l1"])),
                gru_update_l2=float(np.mean(gru_rows["plasticity_l2"])),
            )
        )
    return tuple(endpoints)


@dataclass(frozen=True)
class SelectorContrastSummary:
    name: str
    null_value: float
    mean: float
    lower_confidence: float
    upper_confidence: float
    p_value: float | None
    p_value_holm: float | None
    opposition_p_value: float | None
    opposition_p_value_holm: float | None
    confirmatory: bool


@dataclass(frozen=True)
class ActuatorSelectorConclusion:
    conclusion: str
    reason: str
    statistics_unit: str
    n_seeds: int
    strict_unseen_only: bool
    complete_primary_coverage: bool
    noninferiority_fraction: float
    primary_contrasts: tuple[SelectorContrastSummary, ...]
    descriptive_contrasts: tuple[SelectorContrastSummary, ...]
    seed_endpoints: tuple[SeedSelectorEndpoint, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _contrast_summary(
    name: str,
    values: FloatArray,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence: float,
    p_value: float | None,
    p_value_holm: float | None,
    opposition_p_value: float | None,
    opposition_p_value_holm: float | None,
    confirmatory: bool,
) -> SelectorContrastSummary:
    lower, upper = _bootstrap_mean_interval(
        values,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
        confidence=confidence,
    )
    return SelectorContrastSummary(
        name=name,
        null_value=0.0,
        mean=float(np.mean(values)),
        lower_confidence=lower,
        upper_confidence=upper,
        p_value=p_value,
        p_value_holm=p_value_holm,
        opposition_p_value=opposition_p_value,
        opposition_p_value_holm=opposition_p_value_holm,
        confirmatory=confirmatory,
    )


def summarize_actuator_selector(
    records: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_primary_generators_per_seed: int | None = None,
    noninferiority_fraction: float = 0.8,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 100_000,
    confidence: float = 0.95,
    random_seed: int = 2701,
    force_inconclusive: bool = False,
) -> ActuatorSelectorConclusion:
    """Apply the preregistered two-endpoint intersection-union decision."""

    endpoints = seed_selector_endpoints(
        records,
        expected_seeds=expected_seeds,
        expected_primary_generators_per_seed=expected_primary_generators_per_seed,
        noninferiority_fraction=noninferiority_fraction,
    )
    expected = tuple(int(seed) for seed in expected_seeds)
    complete = bool(
        len(endpoints) == len(expected)
        and {endpoint.seed for endpoint in endpoints} == set(expected)
        and all(
            endpoint.strict_unseen_generators == endpoints[0].strict_unseen_generators
            for endpoint in endpoints
        )
    )
    endpoint_map = {
        "local_noninferiority_contrast": _finite_vector(
            [item.local_noninferiority_contrast for item in endpoints],
            name="local_noninferiority_contrast",
        ),
        "local_minus_fixed_best": _finite_vector(
            [item.local_minus_fixed_best for item in endpoints],
            name="local_minus_fixed_best",
        ),
    }
    raw_p = np.asarray(
        [
            _one_sided_sign_flip(
                values,
                seed=random_seed + 100 + index,
                samples=permutation_samples,
            )
            for index, values in enumerate(endpoint_map.values())
        ],
        dtype=np.float64,
    )
    adjusted = holm_adjust(raw_p)
    opposition_raw_p = np.asarray(
        [
            _one_sided_sign_flip(
                -values,
                seed=random_seed + 200 + index,
                samples=permutation_samples,
            )
            for index, values in enumerate(endpoint_map.values())
        ],
        dtype=np.float64,
    )
    opposition_adjusted = holm_adjust(opposition_raw_p)
    primary = tuple(
        _contrast_summary(
            name,
            values,
            bootstrap_seed=random_seed + index,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            p_value=float(raw_p[index]),
            p_value_holm=float(adjusted[index]),
            opposition_p_value=float(opposition_raw_p[index]),
            opposition_p_value_holm=float(opposition_adjusted[index]),
            confirmatory=True,
        )
        for index, (name, values) in enumerate(endpoint_map.items())
    )
    descriptive_values = {
        "local_minus_routing": np.asarray(
            [
                item.local_three_factor_utility - item.routing_utility
                for item in endpoints
            ]
        ),
        "local_minus_gain": np.asarray(
            [item.local_three_factor_utility - item.gain_utility for item in endpoints]
        ),
        "local_minus_low_rank": np.asarray(
            [
                item.local_three_factor_utility - item.low_rank_utility
                for item in endpoints
            ]
        ),
        "local_minus_gru_bptt": np.asarray(
            [
                item.local_three_factor_utility - item.gru_bptt_utility
                for item in endpoints
            ]
        ),
    }
    descriptive = tuple(
        _contrast_summary(
            name,
            _finite_vector(values, name=name),
            bootstrap_seed=random_seed + 20 + index,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            p_value=None,
            p_value_holm=None,
            opposition_p_value=None,
            opposition_p_value_holm=None,
            confirmatory=False,
        )
        for index, (name, values) in enumerate(descriptive_values.items())
    )
    primary_pass = all(
        item.lower_confidence > 0.0
        and item.p_value_holm is not None
        and item.p_value_holm < 0.05
        for item in primary
    )
    contradicted = any(
        item.upper_confidence <= 0.0
        and item.opposition_p_value_holm is not None
        and item.opposition_p_value_holm < 0.05
        for item in primary
    )
    if force_inconclusive:
        conclusion = "inconclusive"
        reason = "development/smoke profile is forced inconclusive"
    elif complete and primary_pass:
        conclusion = "support"
        reason = "both strict-unseen seed-level confirmatory endpoints passed"
    elif complete and contradicted:
        conclusion = "oppose"
        reason = "at least one preregistered selector claim was contradicted"
    else:
        conclusion = "inconclusive"
        reason = (
            "primary coverage incomplete"
            if not complete
            else "the two preregistered selector thresholds were not both met"
        )
    return ActuatorSelectorConclusion(
        conclusion=conclusion,
        reason=reason,
        statistics_unit="outer_seed",
        n_seeds=len(endpoints),
        strict_unseen_only=True,
        complete_primary_coverage=complete,
        noninferiority_fraction=float(noninferiority_fraction),
        primary_contrasts=primary,
        descriptive_contrasts=descriptive,
        seed_endpoints=endpoints,
    )


__all__ = [
    "ActuatorSelectorConclusion",
    "CANDIDATE_MODES",
    "CANDIDATE_UTILITY_COLUMNS",
    "SELECTOR_MODES",
    "TRAIN_UTILITY_COLUMNS",
    "SeedSelectorEndpoint",
    "SelectorContrastSummary",
    "seed_selector_endpoints",
    "summarize_actuator_selector",
    "validate_selector_records",
]
