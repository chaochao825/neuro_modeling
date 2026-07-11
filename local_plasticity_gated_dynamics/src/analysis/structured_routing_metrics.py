"""Leakage-safe routing and task-level inference for frozen candidate tapes.

This module provides a secondary functional benchmark.  It does not interpret
structured-task routing as evidence about biological neural mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.structured_task_dataset import (
    StructuredCandidate,
    StructuredTask,
    StructuredTaskDataset,
)


ROUTING_CONDITIONS = (
    "no_router",
    "shared_router",
    "per_task_oracle_ceiling",
)


@dataclass(frozen=True)
class SharedCandidateRouter:
    """Train-only standardized, K-dimensional candidate selector."""

    scaler: StandardScaler
    projection: PCA
    classifier: LogisticRegression
    router_dim: int
    fit_task_ids: tuple[str, ...]
    fit_candidate_count: int

    def score(self, candidates: Sequence[StructuredCandidate]) -> np.ndarray:
        if not candidates:
            return np.empty(0, dtype=float)
        features = np.stack([candidate.features for candidate in candidates])
        if features.shape[1] != self.scaler.n_features_in_:
            raise ValueError("candidate feature dimension differs from fitted router")
        latent = self.projection.transform(self.scaler.transform(features))
        return np.asarray(self.classifier.predict_proba(latent)[:, 1], dtype=float)


@dataclass(frozen=True)
class RoutingEvaluation:
    condition: str
    task_metrics: pd.DataFrame
    summary: Mapping[str, object]


def fit_shared_candidate_router(
    dataset: StructuredTaskDataset,
    *,
    router_dim: int,
    seed: int,
    regularization_c: float = 1.0,
) -> SharedCandidateRouter:
    """Fit a shared router using candidate labels from train tasks only."""

    if isinstance(router_dim, (bool, np.bool_)) or not isinstance(
        router_dim, (int, np.integer)
    ):
        raise TypeError("router_dim must be an integer")
    if router_dim < 1:
        raise ValueError("router_dim must be positive")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(regularization_c) or regularization_c <= 0.0:
        raise ValueError("regularization_c must be positive and finite")

    train_candidates = [
        candidate for task in dataset.train_tasks for candidate in task.candidates
    ]
    if not train_candidates:
        raise ValueError("router training tasks contain no candidates")
    features = np.stack([candidate.features for candidate in train_candidates])
    labels = np.asarray(
        [candidate.exact_correct for candidate in train_candidates], dtype=int
    )
    if np.unique(labels).size != 2:
        raise ValueError(
            "router training candidates must contain both correctness classes"
        )
    max_dimension = min(features.shape)
    if router_dim > max_dimension:
        raise ValueError(
            f"router_dim={router_dim} exceeds train-only PCA limit {max_dimension}"
        )

    scaler = StandardScaler().fit(features)
    projection = PCA(n_components=router_dim, svd_solver="full").fit(
        scaler.transform(features)
    )
    classifier = LogisticRegression(
        C=float(regularization_c),
        class_weight="balanced",
        solver="liblinear",
        random_state=int(seed),
        max_iter=1000,
    ).fit(projection.transform(scaler.transform(features)), labels)
    return SharedCandidateRouter(
        scaler=scaler,
        projection=projection,
        classifier=classifier,
        router_dim=int(router_dim),
        fit_task_ids=tuple(task.task_id for task in dataset.train_tasks),
        fit_candidate_count=len(train_candidates),
    )


def _argmax_with_stable_tie_break(
    candidates: Sequence[StructuredCandidate], scores: np.ndarray
) -> StructuredCandidate:
    if len(candidates) != scores.size or not candidates:
        raise ValueError("scores must match a non-empty candidate set")
    if not np.isfinite(scores).all():
        raise ValueError("candidate selection scores must be finite")
    # Candidate IDs make tape order irrelevant; baseline score is a secondary
    # tie breaker for a shared-router probability tie.
    ordered = sorted(
        zip(candidates, scores.tolist(), strict=True),
        key=lambda item: (item[1], item[0].baseline_score, item[0].candidate_id),
        reverse=True,
    )
    return ordered[0][0]


def _select_candidate(
    task: StructuredTask,
    *,
    condition: str,
    router: SharedCandidateRouter | None,
) -> StructuredCandidate | None:
    if not task.candidates:
        return None
    if condition == "no_router":
        scores = np.asarray(
            [candidate.baseline_score for candidate in task.candidates], dtype=float
        )
        return _argmax_with_stable_tie_break(task.candidates, scores)
    if condition == "shared_router":
        if router is None:
            raise ValueError("shared_router condition requires a fitted router")
        return _argmax_with_stable_tie_break(
            task.candidates, router.score(task.candidates)
        )
    if condition == "per_task_oracle_ceiling":
        correct = [
            candidate for candidate in task.candidates if candidate.exact_correct
        ]
        if correct:
            return sorted(correct, key=lambda candidate: candidate.candidate_id)[0]
        # When generation did not cover the answer, the ceiling remains wrong;
        # choosing a deterministic baseline candidate keeps the task retained.
        scores = np.asarray(
            [candidate.baseline_score for candidate in task.candidates], dtype=float
        )
        return _argmax_with_stable_tie_break(task.candidates, scores)
    raise ValueError(f"unknown routing condition {condition!r}")


def bootstrap_task_mean(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval with complete task as the resampling unit."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite vector")
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(
        n_bootstrap, (int, np.integer)
    ):
        raise TypeError("n_bootstrap must be an integer")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(n_bootstrap), array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def evaluate_routing_condition(
    dataset: StructuredTaskDataset,
    *,
    condition: str,
    router_dim: int,
    bootstrap_seed: int,
    n_bootstrap: int,
    router: SharedCandidateRouter | None = None,
) -> RoutingEvaluation:
    """Score every test task, including uncovered and empty candidate sets."""

    if condition not in ROUTING_CONDITIONS:
        raise ValueError(f"condition must be one of {ROUTING_CONDITIONS}")
    if condition == "shared_router" and router is None:
        raise ValueError("shared_router condition requires a fitted router")
    if router_dim < 1:
        raise ValueError("router_dim must be positive")

    rows: list[dict[str, object]] = []
    for task in dataset.test_tasks:
        selected = _select_candidate(task, condition=condition, router=router)
        n_candidates = len(task.candidates)
        raw_compute = float(
            sum(candidate.compute_cost for candidate in task.candidates)
        )
        # Every condition is charged the same K-dimensional scoring allowance,
        # padding cheaper controls.  Candidate generation/action execution cost
        # is therefore paired exactly, while oracle answer access remains marked.
        matched_compute = raw_compute + float(router_dim * n_candidates)
        rows.append(
            {
                "task_id": task.task_id,
                "condition": condition,
                "task_status": (
                    "missing_candidate_set" if not task.candidates else "complete"
                ),
                "candidate_covered": bool(task.candidate_covered),
                "selected_candidate_id": (
                    selected.candidate_id if selected is not None else None
                ),
                "selected_exact_correct": int(
                    selected.exact_correct if selected is not None else False
                ),
                "n_candidates": n_candidates,
                "candidate_compute_raw": raw_compute,
                "matched_compute_budget": matched_compute,
                "candidate_fingerprint": task.candidate_fingerprint,
                "task_provenance_hash": task.provenance_hash,
                "selection_privileged": condition == "per_task_oracle_ceiling",
                "selection_accessed_test_exact_correct": condition
                == "per_task_oracle_ceiling",
                "fit_accessed_test_exact_correct": False,
                "statistics_unit": "task",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("dataset has no test tasks")
    exact = frame["selected_exact_correct"].to_numpy(dtype=float)
    coverage = frame["candidate_covered"].to_numpy(dtype=float)
    low, high = bootstrap_task_mean(
        exact,
        seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )
    covered_mask = coverage.astype(bool)
    covered_accuracy = (
        float(exact[covered_mask].mean()) if covered_mask.any() else float("nan")
    )
    summary: dict[str, object] = {
        "status": "complete",
        "condition": condition,
        "primary_metric": "exact_task_accuracy",
        "exact_task_accuracy": float(exact.mean()),
        "exact_task_accuracy_ci_low": low,
        "exact_task_accuracy_ci_high": high,
        "candidate_coverage": float(coverage.mean()),
        "covered_task_accuracy": covered_accuracy,
        "missing_candidate_fraction": float(
            (frame["task_status"] == "missing_candidate_set").mean()
        ),
        "n_test_tasks": int(len(frame)),
        "n_covered_tasks": int(covered_mask.sum()),
        "matched_compute_budget_total": float(frame["matched_compute_budget"].sum()),
        "candidate_compute_raw_total": float(frame["candidate_compute_raw"].sum()),
        "router_dim": int(router_dim),
        "tape_fingerprint": dataset.tape_fingerprint,
        "task_split_disjoint": not bool(
            {task.task_id for task in dataset.train_tasks}
            & {task.task_id for task in dataset.test_tasks}
        ),
        "all_preprocessing_fit_on_train_tasks": True,
        "fit_accessed_test_exact_correct": False,
        "selection_accessed_test_exact_correct": condition == "per_task_oracle_ceiling",
        "candidate_generation_frozen": bool(dataset.frozen_before_evaluation),
        "candidate_generator_commit": dataset.candidate_generator_commit,
        # Hash-shaped, train-example-scoped declarations are schema checks only.
        # This adapter does not reproduce the external feature extractor, so the
        # tape cannot yet establish that features exclude held-out correctness.
        "input_feature_provenance_schema_attested": True,
        "input_feature_provenance_recomputed": False,
        "input_feature_provenance_validated": False,
        "provenance_validation_level": "schema_attestation_only",
        "candidate_set_matched": True,
        "compute_budget_matched_by_padding": True,
        "measured_compute_matched": False,
        "efficiency_claim_eligible": False,
        "statistics_unit": "task",
        "neural_evidence_claim": False,
        "interpretation_scope": "secondary_functional_routing_only",
    }
    if router is not None and condition == "shared_router":
        summary.update(
            router_fit_task_ids=list(router.fit_task_ids),
            router_fit_candidate_count=router.fit_candidate_count,
            router_effective_dim=router.router_dim,
        )
    return RoutingEvaluation(condition=condition, task_metrics=frame, summary=summary)


def assert_matched_routing_contract(
    evaluations: Sequence[RoutingEvaluation],
) -> None:
    """Fail closed unless all selectors used identical candidate/action tapes."""

    if {evaluation.condition for evaluation in evaluations} != set(ROUTING_CONDITIONS):
        raise ValueError("matched contract requires all registered routing conditions")
    reference: pd.DataFrame | None = None
    columns = [
        "task_id",
        "candidate_fingerprint",
        "n_candidates",
        "candidate_compute_raw",
        "matched_compute_budget",
        "task_provenance_hash",
    ]
    for evaluation in evaluations:
        current = (
            evaluation.task_metrics[columns]
            .sort_values("task_id")
            .reset_index(drop=True)
        )
        if reference is None:
            reference = current
            continue
        if not current.equals(reference):
            raise ValueError(
                "candidate set or charged compute budget differs across conditions"
            )


def paired_task_accuracy_difference(
    candidate: RoutingEvaluation,
    reference: RoutingEvaluation,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, float]:
    """Paired task-bootstrap difference without treating candidates as replicates."""

    left = candidate.task_metrics[["task_id", "selected_exact_correct"]].rename(
        columns={"selected_exact_correct": "candidate"}
    )
    right = reference.task_metrics[["task_id", "selected_exact_correct"]].rename(
        columns={"selected_exact_correct": "reference"}
    )
    paired = left.merge(right, on="task_id", how="outer", validate="one_to_one")
    if paired.isna().any().any():
        raise ValueError("paired routing evaluations must contain identical test tasks")
    differences = paired["candidate"].to_numpy(dtype=float) - paired[
        "reference"
    ].to_numpy(dtype=float)
    low, high = bootstrap_task_mean(differences, seed=seed, n_bootstrap=n_bootstrap)
    return {
        "accuracy_difference": float(differences.mean()),
        "accuracy_difference_ci_low": low,
        "accuracy_difference_ci_high": high,
    }


def evaluate_structured_routing(
    dataset: StructuredTaskDataset,
    *,
    router_dim: int,
    seed: int,
    n_bootstrap: int,
    regularization_c: float = 1.0,
) -> tuple[RoutingEvaluation, ...]:
    """Fit the train-only router and evaluate the three matched selectors."""

    router = fit_shared_candidate_router(
        dataset,
        router_dim=router_dim,
        seed=seed,
        regularization_c=regularization_c,
    )
    evaluations = tuple(
        evaluate_routing_condition(
            dataset,
            condition=condition,
            router_dim=router_dim,
            bootstrap_seed=seed,
            n_bootstrap=n_bootstrap,
            router=router if condition == "shared_router" else None,
        )
        for condition in ROUTING_CONDITIONS
    )
    assert_matched_routing_contract(evaluations)
    baseline = next(item for item in evaluations if item.condition == "no_router")
    updated: list[RoutingEvaluation] = []
    for evaluation in evaluations:
        summary = dict(evaluation.summary)
        summary.update(
            paired_task_accuracy_difference(
                evaluation,
                baseline,
                seed=seed,
                n_bootstrap=n_bootstrap,
            )
        )
        updated.append(
            RoutingEvaluation(
                condition=evaluation.condition,
                task_metrics=evaluation.task_metrics,
                summary=summary,
            )
        )
    return tuple(updated)
