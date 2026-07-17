"""Seed-level confirmatory statistics for the actuator phase diagram.

Generator cells are prediction targets, not independent statistical units.
Every confirmatory endpoint is first reduced to one value per network seed;
confidence intervals and sign-flip tests then operate only on those seed-level
values.  Discovery generators tune the decision threshold, while held-out
generators and test blocks remain untouched until final evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


FloatArray = NDArray[np.float64]
PRIMARY_MODES = ("frozen", "routing", "gain", "low_rank")


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


def holm_adjust(p_values: ArrayLike) -> FloatArray:
    """Return Holm family-wise adjusted p-values in original order."""

    values = _finite_vector(p_values, name="p_values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    adjusted_sorted = np.maximum.accumulate(
        (values.size - np.arange(values.size)) * values[order]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    adjusted.setflags(write=False)
    return adjusted


def _spearman(first: FloatArray, second: FloatArray) -> float:
    if first.size != second.size or first.size < 3:
        return float("nan")
    first_rank = rankdata(first, method="average")
    second_rank = rankdata(second, method="average")
    if np.std(first_rank) == 0.0 or np.std(second_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _family_advantage(frame: pd.DataFrame, performance: str) -> pd.DataFrame:
    required = {
        "seed",
        "generator_id",
        "generator_split",
        "actuator_mode",
        "chi",
        "alpha",
        performance,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"records missing required columns: {sorted(missing)}")
    values = frame.copy()
    values[performance] = pd.to_numeric(values[performance], errors="raise")
    values["chi"] = pd.to_numeric(values["chi"], errors="raise")
    values["alpha"] = pd.to_numeric(values["alpha"], errors="raise")
    if not np.all(np.isfinite(values[[performance, "chi", "alpha"]].to_numpy())):
        raise ValueError("performance, chi, and alpha must be finite")
    index = ["seed", "generator_id", "generator_split"]
    pivot = values.pivot(index=index, columns="actuator_mode", values=performance)
    for mode in ("routing", "gain", "low_rank"):
        if mode not in pivot:
            raise ValueError(f"records do not contain actuator mode {mode!r}")
    chi_counts = values.groupby(index, sort=False)["chi"].nunique(dropna=False)
    if bool((chi_counts != 1).any()):
        raise ValueError("chi differs across paired actuator modes")
    chi = values.groupby(index, sort=False)["chi"].first()
    alpha_counts = values.groupby(index, sort=False)["alpha"].nunique(dropna=False)
    if bool((alpha_counts != 1).any()):
        raise ValueError("alpha differs across paired actuator modes")
    alpha = values.groupby(index, sort=False)["alpha"].first()
    result = pivot.reset_index()
    result["chi"] = chi.reindex(pivot.index).to_numpy()
    result["alpha"] = alpha.reindex(pivot.index).to_numpy()
    result["advantage"] = result["low_rank"] - result[["routing", "gain"]].max(
        axis=1
    )
    return result


def fit_chi_threshold(
    chi: ArrayLike,
    labels: ArrayLike,
) -> float:
    """Fit a deterministic balanced-accuracy threshold on discovery tasks."""

    values = _finite_vector(chi, name="chi")
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.shape != values.shape:
        raise ValueError("labels must be a vector matching chi")
    if raw_labels.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError("labels must be binary numeric values")
    resolved_labels = np.asarray(raw_labels, dtype=np.int64)
    if not np.all(np.isin(resolved_labels, (0, 1))):
        raise ValueError("labels must contain only 0 and 1")
    if np.unique(resolved_labels).size != 2:
        raise ValueError("both task families are required to fit a threshold")
    unique = np.unique(values)
    candidates = np.concatenate(
        (
            np.array([np.nextafter(unique[0], -np.inf)]),
            0.5 * (unique[:-1] + unique[1:]),
            np.array([np.nextafter(unique[-1], np.inf)]),
        )
    )
    scores = np.array(
        [
            balanced_accuracy_score(resolved_labels, values >= threshold)
            for threshold in candidates
        ]
    )
    best = np.flatnonzero(np.isclose(scores, np.max(scores), atol=1e-12))
    distances = np.abs(candidates[best] - 0.5)
    closest = best[np.flatnonzero(np.isclose(distances, np.min(distances)))]
    return float(np.min(candidates[closest]))


@dataclass(frozen=True)
class SeedPhaseEndpoint:
    seed: int
    discovery_threshold: float
    discovery_alpha_threshold: float
    heldout_generators: int
    heldout_ties: int
    spearman_rho: float
    classifier_balanced_accuracy: float
    classifier_auroc: float
    alpha_classifier_balanced_accuracy: float
    alpha_classifier_auroc: float
    chi_minus_alpha_auroc: float


def seed_phase_endpoints(
    records: pd.DataFrame,
    *,
    tie_margin: float = 0.01,
) -> list[SeedPhaseEndpoint]:
    """Compute LOSO-discovery/held-out-test endpoints for every seed."""

    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    if not np.isfinite(tie_margin) or tie_margin < 0.0:
        raise ValueError("tie_margin must be finite and non-negative")
    discovery = _family_advantage(records, "validation_balanced_accuracy")
    heldout = _family_advantage(records, "test_balanced_accuracy")
    seeds = sorted(int(value) for value in records["seed"].unique())
    endpoints: list[SeedPhaseEndpoint] = []
    for seed in seeds:
        training = discovery[
            (discovery["generator_split"] == "discovery")
            & (discovery["seed"] != seed)
        ].copy()
        training = training[np.abs(training["advantage"]) >= tie_margin]
        labels = (training["advantage"].to_numpy() > 0.0).astype(np.int64)
        threshold = fit_chi_threshold(training["chi"].to_numpy(), labels)
        alpha_threshold = fit_chi_threshold(training["alpha"].to_numpy(), labels)
        evaluation = heldout[
            (heldout["generator_split"] == "heldout")
            & (heldout["seed"] == seed)
        ].copy()
        chi = evaluation["chi"].to_numpy(dtype=np.float64)
        alpha = evaluation["alpha"].to_numpy(dtype=np.float64)
        advantage = evaluation["advantage"].to_numpy(dtype=np.float64)
        rho = _spearman(chi, advantage)
        tied = np.abs(advantage) < tie_margin
        retained = ~tied
        family = (advantage[retained] > 0.0).astype(np.int64)
        if np.unique(family).size != 2:
            classifier_ba = float("nan")
            auroc = float("nan")
            alpha_classifier_ba = float("nan")
            alpha_auroc = float("nan")
        else:
            classifier_ba = float(
                balanced_accuracy_score(family, chi[retained] >= threshold)
            )
            auroc = float(roc_auc_score(family, chi[retained]))
            alpha_classifier_ba = float(
                balanced_accuracy_score(
                    family, alpha[retained] >= alpha_threshold
                )
            )
            alpha_auroc = float(roc_auc_score(family, alpha[retained]))
        endpoints.append(
            SeedPhaseEndpoint(
                seed=seed,
                discovery_threshold=threshold,
                discovery_alpha_threshold=alpha_threshold,
                heldout_generators=int(evaluation.shape[0]),
                heldout_ties=int(np.sum(tied)),
                spearman_rho=rho,
                classifier_balanced_accuracy=classifier_ba,
                classifier_auroc=auroc,
                alpha_classifier_balanced_accuracy=alpha_classifier_ba,
                alpha_classifier_auroc=alpha_auroc,
                chi_minus_alpha_auroc=float(auroc - alpha_auroc),
            )
        )
    return endpoints


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
    centered_values: FloatArray,
    *,
    seed: int,
    samples: int,
) -> float:
    observed = float(np.mean(centered_values))
    n_values = centered_values.size
    if n_values <= 20:
        signs = np.asarray(list(product((-1.0, 1.0), repeat=n_values)))
    else:
        if samples < 1:
            raise ValueError("permutation_samples must be positive")
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(samples, n_values))
    null = np.mean(signs * centered_values[np.newaxis, :], axis=1)
    return float((1.0 + np.sum(null >= observed - 1e-15)) / (null.size + 1.0))


@dataclass(frozen=True)
class EndpointSummary:
    name: str
    null_value: float
    mean: float
    lower_confidence: float
    upper_confidence: float
    p_value: float
    p_value_holm: float


@dataclass(frozen=True)
class PhaseDiagramConclusion:
    conclusion: str
    statistics_unit: str
    n_seeds: int
    complete_primary_coverage: bool
    endpoint_summaries: tuple[EndpointSummary, ...]
    incremental_auc_summary: EndpointSummary
    gramian_predictor_beats_alpha: bool
    seed_endpoints: tuple[SeedPhaseEndpoint, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_phase_diagram(
    records: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    tie_margin: float = 0.01,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 100_000,
    confidence: float = 0.95,
    random_seed: int = 2601,
) -> PhaseDiagramConclusion:
    """Apply the preregistered three-endpoint intersection-union test."""

    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    expected = tuple(int(seed) for seed in expected_seeds)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_seeds must be non-empty and unique")
    required = {
        "seed",
        "generator_id",
        "generator_split",
        "actuator_mode",
        "status",
        "functional_budget_valid",
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records missing required columns: {sorted(missing)}")
    primary = records[records["actuator_mode"].isin(PRIMARY_MODES)]
    observed_seeds = set(int(value) for value in primary["seed"].unique())
    mode_counts = primary.groupby(
        ["seed", "generator_id", "generator_split"], sort=False
    )["actuator_mode"].nunique()
    complete = bool(
        observed_seeds == set(expected)
        and not mode_counts.empty
        and bool((mode_counts == len(PRIMARY_MODES)).all())
        and bool((primary["status"] == "complete").all())
        and bool(primary["functional_budget_valid"].astype(bool).all())
    )
    endpoints = seed_phase_endpoints(primary, tie_margin=tie_margin)
    endpoint_names = (
        "spearman_rho",
        "classifier_balanced_accuracy",
        "classifier_auroc",
    )
    nulls = (0.0, 0.5, 0.5)
    raw: list[tuple[str, float, float, float, float, float]] = []
    for index, (name, null) in enumerate(zip(endpoint_names, nulls, strict=True)):
        values = np.asarray([getattr(item, name) for item in endpoints])
        if values.size != len(expected) or not np.all(np.isfinite(values)):
            complete = False
            raw.append((name, null, float("nan"), float("nan"), float("nan"), 1.0))
            continue
        lower, upper = _bootstrap_mean_interval(
            values,
            seed=random_seed + index,
            samples=bootstrap_samples,
            confidence=confidence,
        )
        p_value = _one_sided_sign_flip(
            values - null,
            seed=random_seed + 100 + index,
            samples=permutation_samples,
        )
        raw.append((name, null, float(np.mean(values)), lower, upper, p_value))
    adjusted = holm_adjust([item[-1] for item in raw])
    summaries = tuple(
        EndpointSummary(
            name=item[0],
            null_value=item[1],
            mean=item[2],
            lower_confidence=item[3],
            upper_confidence=item[4],
            p_value=item[5],
            p_value_holm=float(adjusted[index]),
        )
        for index, item in enumerate(raw)
    )
    lookup = {item.name: item for item in summaries}
    incremental_values = np.asarray(
        [item.chi_minus_alpha_auroc for item in endpoints], dtype=np.float64
    )
    if incremental_values.size != len(expected) or not np.all(
        np.isfinite(incremental_values)
    ):
        complete = False
        incremental_lower = float("nan")
        incremental_upper = float("nan")
        incremental_p = 1.0
        incremental_mean = float("nan")
    else:
        incremental_lower, incremental_upper = _bootstrap_mean_interval(
            incremental_values,
            seed=random_seed + 20,
            samples=bootstrap_samples,
            confidence=confidence,
        )
        incremental_p = _one_sided_sign_flip(
            incremental_values,
            seed=random_seed + 120,
            samples=permutation_samples,
        )
        incremental_mean = float(np.mean(incremental_values))
    incremental_summary = EndpointSummary(
        name="chi_minus_alpha_auroc",
        null_value=0.0,
        mean=incremental_mean,
        lower_confidence=incremental_lower,
        upper_confidence=incremental_upper,
        p_value=incremental_p,
        p_value_holm=incremental_p,
    )
    gramian_beats_alpha = bool(
        complete and incremental_lower > 0.0 and incremental_p < 0.05
    )
    support = bool(
        complete
        and lookup["spearman_rho"].lower_confidence > 0.30
        and lookup["classifier_balanced_accuracy"].lower_confidence > 0.60
        and lookup["classifier_auroc"].lower_confidence > 0.65
        and all(item.p_value_holm < 0.05 for item in summaries)
        and gramian_beats_alpha
    )
    oppose = bool(
        complete
        and lookup["spearman_rho"].upper_confidence <= 0.0
        and lookup["classifier_balanced_accuracy"].upper_confidence <= 0.5
    )
    if support:
        conclusion = "support"
        reason = "all three held-out seed-level confirmatory endpoints passed"
    elif oppose:
        conclusion = "oppose"
        reason = "held-out correlation and classifier jointly contradicted prediction"
    else:
        conclusion = "inconclusive"
        reason = (
            "primary coverage incomplete"
            if not complete
            else "the preregistered intersection-union thresholds were not all met"
        )
    return PhaseDiagramConclusion(
        conclusion=conclusion,
        statistics_unit="seed",
        n_seeds=len(endpoints),
        complete_primary_coverage=complete,
        endpoint_summaries=summaries,
        incremental_auc_summary=incremental_summary,
        gramian_predictor_beats_alpha=gramian_beats_alpha,
        seed_endpoints=tuple(endpoints),
        reason=reason,
    )


__all__ = [
    "EndpointSummary",
    "PRIMARY_MODES",
    "PhaseDiagramConclusion",
    "SeedPhaseEndpoint",
    "fit_chi_threshold",
    "holm_adjust",
    "seed_phase_endpoints",
    "summarize_phase_diagram",
]
