"""Session/animal-level comparisons for exp14 conditional count dynamics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from src.models.hierarchical_count_dynamics import HierarchicalCountScore


@dataclass(frozen=True, slots=True)
class IntervalEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    bootstrap_p_two_sided: float
    null_value: float
    holm_adjusted_p: float
    n_sessions: int
    n_animals: int


@dataclass(frozen=True, slots=True)
class IBLNeuralComparison:
    shared_vs_common: IntervalEstimate
    full_vs_common: IntervalEstimate
    retention_margin: IntervalEstimate
    retained_full_gain_ratio: float
    retention_defined: bool
    shared_parameter_count: int
    full_parameter_count: int
    shared_has_fewer_parameters: bool
    complete_cohort: bool
    conclusion: str
    inference_unit: str = "animal_with_session_nested"


def _score_map(score: HierarchicalCountScore) -> Mapping[str, object]:
    result = {item.session_id: item for item in score.per_session}
    if len(result) != len(score.per_session):
        raise ValueError("score contains duplicate session IDs")
    return result


def _animal_nested_bootstrap(
    values: Mapping[str, Sequence[float]],
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    animals = tuple(sorted(values))
    if not animals or any(len(values[animal]) < 1 for animal in animals):
        raise ValueError("every animal must contribute at least one session")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        sampled_animals = rng.choice(animals, size=len(animals), replace=True)
        animal_means = []
        for animal in sampled_animals:
            session_values = np.asarray(values[str(animal)], dtype=float)
            sampled_sessions = rng.choice(
                session_values, size=len(session_values), replace=True
            )
            animal_means.append(float(np.mean(sampled_sessions)))
        draws[draw] = float(np.mean(animal_means))
    return draws


def _interval(
    values_by_animal: Mapping[str, Sequence[float]],
    *,
    n_sessions: int,
    n_bootstrap: int,
    seed: int,
    null_value: float = 0.0,
) -> IntervalEstimate:
    animal_means = [
        float(np.mean(values_by_animal[animal])) for animal in sorted(values_by_animal)
    ]
    estimate = float(np.mean(animal_means))
    draws = _animal_nested_bootstrap(
        values_by_animal, n_bootstrap=n_bootstrap, seed=seed
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    p_value = min(
        1.0,
        2.0
        * min(
            float(np.mean(draws <= null_value)),
            float(np.mean(draws >= null_value)),
        ),
    )
    return IntervalEstimate(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        bootstrap_p_two_sided=p_value,
        null_value=float(null_value),
        holm_adjusted_p=float("nan"),
        n_sessions=n_sessions,
        n_animals=len(values_by_animal),
    )


def compare_count_families(
    scores: Mapping[str, HierarchicalCountScore],
    *,
    planned_sessions: int,
    planned_animals: int,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> IBLNeuralComparison:
    """Compare common/shared/full without treating bins or units as replicates."""

    if set(scores) != {"common", "shared", "full"}:
        raise ValueError("scores must contain exactly common/shared/full")
    maps = {family: _score_map(score) for family, score in scores.items()}
    session_ids = set(maps["common"])
    if any(set(values) != session_ids for values in maps.values()):
        raise ValueError("model families must score the same sessions")
    if not session_ids:
        raise ValueError("at least one paired session is required")
    improvements: dict[str, list[float]] = {}
    full_improvements: dict[str, list[float]] = {}
    retention_margins: dict[str, list[float]] = {}
    for session_id in sorted(session_ids):
        common = maps["common"][session_id]
        shared = maps["shared"][session_id]
        full = maps["full"][session_id]
        animal_ids = {item.animal_id for item in (common, shared, full)}
        if len(animal_ids) != 1:
            raise ValueError("paired session animal IDs disagree")
        animal = next(iter(animal_ids))
        shared_gain = float(common.nll_per_count - shared.nll_per_count)
        full_gain = float(common.nll_per_count - full.nll_per_count)
        improvements.setdefault(animal, []).append(shared_gain)
        full_improvements.setdefault(animal, []).append(full_gain)
        retention_margins.setdefault(animal, []).append(shared_gain - 0.9 * full_gain)
    improvement_interval = _interval(
        improvements,
        n_sessions=len(session_ids),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    full_interval = _interval(
        full_improvements,
        n_sessions=len(session_ids),
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    retention_interval = _interval(
        retention_margins,
        n_sessions=len(session_ids),
        n_bootstrap=n_bootstrap,
        seed=seed + 2,
    )
    intervals = [improvement_interval, full_interval, retention_interval]
    finite = [
        (index, item.bootstrap_p_two_sided)
        for index, item in enumerate(intervals)
        if np.isfinite(item.bootstrap_p_two_sided)
    ]
    adjusted = [float("nan")] * len(intervals)
    running = 0.0
    for order, (index, p_value) in enumerate(sorted(finite, key=lambda item: item[1])):
        value = min(1.0, (len(finite) - order) * p_value)
        running = max(running, value)
        adjusted[index] = running
    improvement_interval = replace(improvement_interval, holm_adjusted_p=adjusted[0])
    full_interval = replace(full_interval, holm_adjusted_p=adjusted[1])
    retention_interval = replace(retention_interval, holm_adjusted_p=adjusted[2])
    retained_ratio = (
        improvement_interval.estimate / full_interval.estimate
        if full_interval.estimate > 0.0
        else float("nan")
    )
    retention_defined = bool(
        full_interval.estimate > 0.0 and full_interval.ci_low > 0.0
    )
    n_animals = len(improvements)
    complete = len(session_ids) >= planned_sessions and n_animals >= planned_animals
    fewer = scores["shared"].parameter_count < scores["full"].parameter_count
    if (
        complete
        and improvement_interval.ci_low > 0.0
        and improvement_interval.holm_adjusted_p < 0.05
        and full_interval.ci_low > 0.0
        and full_interval.holm_adjusted_p < 0.05
        and retention_interval.ci_low >= 0.0
        and retention_interval.holm_adjusted_p < 0.05
        and fewer
    ):
        conclusion = "support"
    elif complete and (
        (
            improvement_interval.ci_high < 0.0
            and improvement_interval.holm_adjusted_p < 0.05
        )
        or (
            retention_defined
            and retention_interval.ci_high < 0.0
            and retention_interval.holm_adjusted_p < 0.05
        )
    ):
        conclusion = "oppose"
    else:
        conclusion = "inconclusive"
    return IBLNeuralComparison(
        shared_vs_common=improvement_interval,
        full_vs_common=full_interval,
        retention_margin=retention_interval,
        retained_full_gain_ratio=retained_ratio,
        retention_defined=retention_defined,
        shared_parameter_count=scores["shared"].parameter_count,
        full_parameter_count=scores["full"].parameter_count,
        shared_has_fewer_parameters=fewer,
        complete_cohort=complete,
        conclusion=conclusion,
    )


__all__ = [
    "IBLNeuralComparison",
    "IntervalEstimate",
    "compare_count_families",
]
