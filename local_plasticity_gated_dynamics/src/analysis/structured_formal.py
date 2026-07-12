"""Fail-closed formal aggregation for the ARC exp13 task panel."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.analysis.structured_benchmark import STRUCTURED_CONDITIONS


FORMAL_COMPARISONS = (
    ("hierarchical_vs_flat", "hierarchical_local", "flat_local", "superiority"),
    ("trace_vs_flat", "trace_local", "flat_local", "superiority"),
    (
        "hierarchical_vs_support_heuristic",
        "hierarchical_local",
        "support_heuristic",
        "superiority",
    ),
    (
        "hierarchical_vs_gru_bptt",
        "hierarchical_local",
        "gru_bptt",
        "superiority",
    ),
    (
        "hierarchical_retains_90pct_gru",
        "hierarchical_local",
        "gru_bptt",
        "noninferiority_90pct",
    ),
    (
        "trace_vs_hierarchical",
        "trace_local",
        "hierarchical_local",
        "superiority",
    ),
)


def _holm_adjust(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("p-values must be a finite vector")
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _component_ids(frame: pd.DataFrame) -> pd.Series:
    """Join source and augmentation dependencies before task inference."""

    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    observed: dict[tuple[str, str], int] = {}
    for index, row in enumerate(frame.itertuples(index=False)):
        for kind, value in (
            ("source", str(row.source_group)),
            ("augmentation", str(row.augmentation_group)),
        ):
            key = kind, value
            previous = observed.setdefault(key, index)
            union(index, previous)
    labels = [f"component_{find(index):06d}" for index in range(len(frame))]
    return pd.Series(labels, index=frame.index, dtype="object")


def _bootstrap_mean(
    values: np.ndarray, *, seed: int, n_bootstrap: int
) -> tuple[float, float]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
        raise ValueError("bootstrap requires at least two finite component values")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    rng = np.random.default_rng(int(seed))
    distribution = np.empty(int(n_bootstrap), dtype=float)
    chunk_size = 4096
    for start in range(0, int(n_bootstrap), chunk_size):
        stop = min(start + chunk_size, int(n_bootstrap))
        indices = rng.integers(0, vector.size, size=(stop - start, vector.size))
        distribution[start:stop] = vector[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def _validate_raw_panel(
    raw: pd.DataFrame, *, expected_seeds: Sequence[int]
) -> pd.DataFrame:
    required = {
        "seed",
        "condition",
        "task_id",
        "source_group",
        "augmentation_group",
        "exact",
        "candidate_covered",
        "candidate_fingerprint",
        "parameter_count",
        "trainable_parameter_count",
        "used_bptt",
        "control_dim",
        "control_operator_rank",
        "run_id",
        "run_git_commit",
        "run_git_dirty",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"exp13 formal raw panel lacks columns: {sorted(missing)}")
    frame = raw.copy()
    expected = tuple(int(value) for value in expected_seeds)
    if len(expected) < 2 or len(set(expected)) != len(expected):
        raise ValueError("expected_seeds must contain unique seeds")
    if set(frame["seed"].astype(int)) != set(expected):
        raise ValueError("exp13 formal panel is missing or adds seeds")
    if set(frame["condition"].astype(str)) != set(STRUCTURED_CONDITIONS):
        raise ValueError("exp13 formal condition family is incomplete")
    if frame["run_git_dirty"].astype(bool).any():
        raise ValueError("exp13 formal inference requires clean run worktrees")
    if frame["run_git_commit"].astype(str).nunique() != 1:
        raise ValueError("exp13 formal seeds must share one code commit")
    keys = ["seed", "condition", "task_id"]
    if frame.duplicated(keys).any():
        raise ValueError("exp13 formal panel has duplicate seed/condition/task rows")
    expected_rows = len(expected) * len(STRUCTURED_CONDITIONS)
    task_counts = frame.groupby("task_id", sort=False).size()
    if not task_counts.eq(expected_rows).all():
        raise ValueError("every task must be retained for every seed and condition")
    condition_counts = frame.groupby(["seed", "task_id"])["condition"].nunique()
    if not condition_counts.eq(len(STRUCTURED_CONDITIONS)).all():
        raise ValueError("condition panel is not paired within seed/task")
    fingerprints = frame.groupby("task_id")["candidate_fingerprint"].nunique(
        dropna=False
    )
    if not fingerprints.eq(1).all():
        raise ValueError("candidate panels differ across condition or seed")
    for column in ("exact", "candidate_covered"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not numeric.isin([0, 1]).all():
            raise ValueError(f"{column} must be boolean/0/1")
        frame[column] = numeric.astype(float)
    return frame


def _component_condition_panel(raw: pd.DataFrame) -> pd.DataFrame:
    task_level = (
        raw.groupby(
            [
                "task_id",
                "source_group",
                "augmentation_group",
                "condition",
            ],
            as_index=False,
            sort=False,
        )[["exact", "candidate_covered"]]
        .mean()
    )
    identities = task_level[
        ["task_id", "source_group", "augmentation_group"]
    ].drop_duplicates("task_id")
    identities["component_id"] = _component_ids(identities).to_numpy()
    task_level = task_level.merge(
        identities[["task_id", "component_id"]],
        on="task_id",
        validate="many_to_one",
    )
    return (
        task_level.groupby(["component_id", "condition"], as_index=False)[
            ["exact", "candidate_covered"]
        ]
        .mean()
    )


def summarize_structured_formal(
    raw: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    seed: int = 20260712,
    n_bootstrap: int = 100_000,
    minimum_candidate_coverage: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return absolute condition results and paired component comparisons."""

    if not 0.0 <= minimum_candidate_coverage <= 1.0:
        raise ValueError("minimum_candidate_coverage must lie in [0, 1]")
    frame = _validate_raw_panel(raw, expected_seeds=expected_seeds)
    component = _component_condition_panel(frame)
    n_components = int(component["component_id"].nunique())
    n_tasks = int(frame["task_id"].nunique())
    condition_rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(STRUCTURED_CONDITIONS):
        selected = component.loc[component["condition"] == condition]
        exact = selected["exact"].to_numpy(dtype=float)
        low, high = _bootstrap_mean(
            exact, seed=seed + condition_index, n_bootstrap=n_bootstrap
        )
        metadata = frame.loc[frame["condition"] == condition]
        for column in (
            "parameter_count",
            "trainable_parameter_count",
            "used_bptt",
            "control_dim",
            "control_operator_rank",
        ):
            if metadata[column].nunique(dropna=False) != 1:
                raise ValueError(f"{condition} has varying {column} across formal runs")
        condition_rows.append(
            {
                "condition": condition,
                "exact_accuracy": float(exact.mean()),
                "exact_accuracy_ci_low": low,
                "exact_accuracy_ci_high": high,
                "candidate_coverage": float(selected["candidate_covered"].mean()),
                "minimum_candidate_coverage": float(minimum_candidate_coverage),
                "coverage_gate_passed": bool(
                    selected["candidate_covered"].mean()
                    >= minimum_candidate_coverage
                ),
                "n_tasks": n_tasks,
                "n_dependency_components": n_components,
                "n_seeds": len(tuple(expected_seeds)),
                "statistics_unit": "source_augmentation_dependency_component",
                "seed_nested_within_task": True,
                "parameter_count": int(metadata["parameter_count"].iloc[0]),
                "trainable_parameter_count": int(
                    metadata["trainable_parameter_count"].iloc[0]
                ),
                "used_bptt": bool(metadata["used_bptt"].iloc[0]),
                "control_dim": int(metadata["control_dim"].iloc[0]),
                "control_operator_rank": int(
                    metadata["control_operator_rank"].iloc[0]
                ),
            }
        )
    condition_summary = pd.DataFrame(condition_rows)

    panel = component.pivot(
        index="component_id", columns="condition", values="exact"
    ).loc[:, list(STRUCTURED_CONDITIONS)]
    if panel.isna().any().any():
        raise ValueError("formal component panel is incomplete")
    comparison_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for index, (name, candidate, reference, mode) in enumerate(FORMAL_COMPARISONS):
        candidate_values = panel[candidate].to_numpy(dtype=float)
        reference_values = panel[reference].to_numpy(dtype=float)
        if mode == "noninferiority_90pct":
            differences = candidate_values - 0.9 * reference_values
            estimand = "candidate_accuracy_minus_0.9_times_reference_accuracy"
        else:
            differences = candidate_values - reference_values
            estimand = "candidate_accuracy_minus_reference_accuracy"
        low, high = _bootstrap_mean(
            differences,
            seed=seed + 100 + index,
            n_bootstrap=n_bootstrap,
        )
        nonzero = np.count_nonzero(differences)
        p_value = (
            1.0
            if nonzero == 0
            else float(
                wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        )
        raw_p_values.append(p_value)
        comparison_rows.append(
            {
                "comparison": name,
                "candidate": candidate,
                "reference": reference,
                "comparison_mode": mode,
                "estimand": estimand,
                "estimate": float(differences.mean()),
                "ci_low": low,
                "ci_high": high,
                "wilcoxon_p": p_value,
                "n_dependency_components": len(differences),
                "n_nonzero_components": int(nonzero),
                "multiple_comparison_family": "exp13_arc_six_registered_comparisons",
                "candidate_coverage": float(
                    component["candidate_covered"].mean()
                ),
                "minimum_candidate_coverage": float(minimum_candidate_coverage),
            }
        )
    adjusted = _holm_adjust(raw_p_values)
    for row, holm_p in zip(comparison_rows, adjusted, strict=True):
        row["wilcoxon_p_holm"] = float(holm_p)
        coverage_passed = bool(
            float(row["candidate_coverage"]) >= minimum_candidate_coverage
        )
        row["coverage_gate_passed"] = coverage_passed
        row["core_claim_eligible"] = coverage_passed
        if coverage_passed and holm_p < 0.05 and float(row["ci_low"]) > 0.0:
            conclusion = "support"
        elif coverage_passed and holm_p < 0.05 and float(row["ci_high"]) < 0.0:
            conclusion = "oppose"
        else:
            conclusion = "inconclusive"
        row["conclusion"] = conclusion
    comparisons = pd.DataFrame(comparison_rows)
    return condition_summary, comparisons
