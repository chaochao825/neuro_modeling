"""Animal-primary paired inference for real belief-gated count dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.models.hierarchical_count_dynamics import HierarchicalCountScore


@dataclass(frozen=True, slots=True)
class BeliefDynamicsContrast:
    """Paired comparator NLL minus intact-MD NLL at the animal level."""

    comparison: str
    estimate: float
    ci_low: float
    ci_high: float
    bootstrap_p_two_sided: float
    holm_adjusted_p: float
    n_sessions: int
    n_animals: int
    complete_cohort: bool
    conclusion: str
    effect_definition: str = "comparator_nll_per_count_minus_md_shared_nll_per_count"
    inference_unit: str = "animal_with_session_nested"


def _score_map(score: HierarchicalCountScore) -> dict[str, object]:
    result = {item.session_id: item for item in score.per_session}
    if not result or len(result) != len(score.per_session):
        raise ValueError("score must contain unique non-empty session IDs")
    return result


def _nested_draws(
    values: Mapping[str, Sequence[float]], *, n_bootstrap: int, seed: int
) -> np.ndarray:
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    animals = tuple(sorted(values))
    if not animals or any(not values[animal] for animal in animals):
        raise ValueError("every animal must contribute at least one session")
    rng = np.random.default_rng(seed)
    result = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        sampled_animals = rng.choice(animals, size=len(animals), replace=True)
        animal_values = []
        for animal in sampled_animals:
            sessions = np.asarray(values[str(animal)], dtype=float)
            animal_values.append(float(np.mean(rng.choice(sessions, len(sessions)))))
        result[draw] = float(np.mean(animal_values))
    return result


def _holm(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Holm inputs must be a finite vector")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def compare_belief_dynamics_conditions(
    scores: Mapping[str, HierarchicalCountScore],
    *,
    intact_condition: str,
    comparator_conditions: Sequence[str],
    planned_sessions: int,
    planned_animals: int,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[BeliefDynamicsContrast, ...]:
    """Compare paired conditions without treating units/bins as replicates."""

    comparators = tuple(str(value) for value in comparator_conditions)
    if not comparators or len(set(comparators)) != len(comparators):
        raise ValueError("comparator_conditions must be unique and non-empty")
    required = {str(intact_condition), *comparators}
    if set(scores) != required:
        raise ValueError("scores must contain exactly intact plus comparators")
    maps = {condition: _score_map(score) for condition, score in scores.items()}
    session_ids = set(maps[str(intact_condition)])
    if any(set(values) != session_ids for values in maps.values()):
        raise ValueError("all conditions must score the same sessions")

    estimates = []
    raw_p = []
    intervals = []
    animal_counts = []
    for offset, comparator in enumerate(comparators):
        by_animal: dict[str, list[float]] = {}
        for session_id in sorted(session_ids):
            intact = maps[str(intact_condition)][session_id]
            other = maps[comparator][session_id]
            if intact.animal_id != other.animal_id:
                raise ValueError("paired session animal IDs disagree")
            by_animal.setdefault(intact.animal_id, []).append(
                float(other.nll_per_count - intact.nll_per_count)
            )
        estimate = float(
            np.mean([np.mean(by_animal[animal]) for animal in sorted(by_animal)])
        )
        draws = _nested_draws(
            by_animal,
            n_bootstrap=n_bootstrap,
            seed=seed + 104729 * offset,
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        p_value = min(
            1.0,
            2.0 * min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0))),
        )
        estimates.append(estimate)
        raw_p.append(p_value)
        intervals.append((float(low), float(high)))
        animal_counts.append(len(by_animal))

    adjusted = _holm(raw_p)
    result = []
    for comparator, estimate, (low, high), p_value, p_holm, n_animals in zip(
        comparators,
        estimates,
        intervals,
        raw_p,
        adjusted,
        animal_counts,
        strict=True,
    ):
        complete = len(session_ids) >= planned_sessions and n_animals >= planned_animals
        conclusion = "inconclusive"
        if complete and low > 0.0 and p_holm < 0.05:
            conclusion = "support"
        elif complete and high < 0.0 and p_holm < 0.05:
            conclusion = "oppose"
        result.append(
            BeliefDynamicsContrast(
                comparison=f"{intact_condition}_vs_{comparator}",
                estimate=estimate,
                ci_low=low,
                ci_high=high,
                bootstrap_p_two_sided=p_value,
                holm_adjusted_p=float(p_holm),
                n_sessions=len(session_ids),
                n_animals=n_animals,
                complete_cohort=complete,
                conclusion=conclusion,
            )
        )
    return tuple(result)


__all__ = ["BeliefDynamicsContrast", "compare_belief_dynamics_conditions"]
