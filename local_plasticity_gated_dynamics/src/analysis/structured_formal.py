"""Fail-closed formal aggregation for an exp13 structured-task panel."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

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

PUBLISHED_EXP13_RAW_SHA256 = (
    "0865d15177be359194b454ca8c94620df34e3ddf449b096b5e796847e0533b4d"
)
PUBLISHED_EXP13_RUN_MANIFEST_SHA256 = (
    "7e46d4e0b62106e20b047f89ea25903619806f738683fc1be38d6ae4a87e8ead"
)
PUBLISHED_EXP13_SOURCE_MANIFEST_SHA256 = (
    "76e2360f6673093730676345fd3db8bf289be3f58179c002980a4e91ae0d9cda"
)
PUBLISHED_EXP13_SOURCE_REVISION = "399030444e0ab0cc8b4e199870fb20b863846f34"
PUBLISHED_EXP13_DATASET_NAME = "ARC-AGI-1"
PUBLISHED_EXP13_TEST_SPLIT_ROLE = "ood"


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
    task_level = raw.groupby(
        [
            "task_id",
            "source_group",
            "augmentation_group",
            "condition",
        ],
        as_index=False,
        sort=False,
    )[["exact", "candidate_covered"]].mean()
    identities = task_level[
        ["task_id", "source_group", "augmentation_group"]
    ].drop_duplicates("task_id")
    identities["component_id"] = _component_ids(identities).to_numpy()
    task_level = task_level.merge(
        identities[["task_id", "component_id"]],
        on="task_id",
        validate="many_to_one",
    )
    return task_level.groupby(["component_id", "condition"], as_index=False)[
        ["exact", "candidate_covered"]
    ].mean()


def summarize_structured_formal(
    raw: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    seed: int = 20260712,
    n_bootstrap: int = 100_000,
    minimum_candidate_coverage: float = 0.9,
    task_family: str = "arc",
    test_split_role: str = "ood",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return absolute condition results and paired component comparisons."""

    if not 0.0 <= minimum_candidate_coverage <= 1.0:
        raise ValueError("minimum_candidate_coverage must lie in [0, 1]")
    task_family = str(task_family).strip().lower()
    if task_family not in {"arc", "maze", "sudoku"}:
        raise ValueError("task_family must be arc, maze, or sudoku")
    test_split_role = str(test_split_role).strip().lower()
    if test_split_role not in {"ood", "non_ood"}:
        raise ValueError("test_split_role must be ood or non_ood")
    registered_ood_split = test_split_role == "ood"
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
                    selected["candidate_covered"].mean() >= minimum_candidate_coverage
                ),
                "n_tasks": n_tasks,
                "n_dependency_components": n_components,
                "n_seeds": len(tuple(expected_seeds)),
                "statistics_unit": "source_augmentation_dependency_component",
                "seed_nested_within_task": True,
                "test_split_role": test_split_role,
                "registered_ood_split": registered_ood_split,
                "parameter_count": int(metadata["parameter_count"].iloc[0]),
                "trainable_parameter_count": int(
                    metadata["trainable_parameter_count"].iloc[0]
                ),
                "used_bptt": bool(metadata["used_bptt"].iloc[0]),
                "control_dim": int(metadata["control_dim"].iloc[0]),
                "control_operator_rank": int(metadata["control_operator_rank"].iloc[0]),
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
                "multiple_comparison_family": (
                    f"exp13_{task_family}_six_registered_comparisons"
                ),
                "candidate_coverage": float(component["candidate_covered"].mean()),
                "minimum_candidate_coverage": float(minimum_candidate_coverage),
                "test_split_role": test_split_role,
                "registered_ood_split": registered_ood_split,
            }
        )
    adjusted = _holm_adjust(raw_p_values)
    for row, holm_p in zip(comparison_rows, adjusted, strict=True):
        row["wilcoxon_p_holm"] = float(holm_p)
        coverage_passed = bool(
            float(row["candidate_coverage"]) >= minimum_candidate_coverage
        )
        row["coverage_gate_passed"] = coverage_passed
        core_claim_eligible = coverage_passed and registered_ood_split
        row["core_claim_eligible"] = core_claim_eligible
        if core_claim_eligible and holm_p < 0.05 and float(row["ci_low"]) > 0.0:
            conclusion = "support"
        elif core_claim_eligible and holm_p < 0.05 and float(row["ci_high"]) < 0.0:
            conclusion = "oppose"
        else:
            conclusion = "inconclusive"
        row["conclusion"] = conclusion
    comparisons = pd.DataFrame(comparison_rows)
    condition_summary["bootstrap_seed"] = int(seed)
    condition_summary["n_bootstrap"] = int(n_bootstrap)
    comparisons["bootstrap_seed"] = int(seed)
    comparisons["n_bootstrap"] = int(n_bootstrap)
    return condition_summary, comparisons


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_text(frame: pd.DataFrame, column: str, *, source: str) -> str:
    if column not in frame:
        raise ValueError(f"{source} lacks binding column {column}")
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"{source} binding {column} is not unique")
    return str(values[0])


def _assert_every_row_matches(
    frame: pd.DataFrame, column: str, expected: str, *, source: str
) -> None:
    if column not in frame:
        raise ValueError(f"{source} lacks binding column {column}")
    if (
        not frame[column].notna().all()
        or not frame[column].astype(str).eq(expected).all()
    ):
        raise ValueError(f"{source} {column} differs from the run manifest")


def _assert_recomputed_table(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    key: str,
    source: str,
) -> None:
    if observed[key].astype(str).duplicated().any():
        raise ValueError(f"{source} has duplicate {key} rows")
    indexed = observed.set_index(observed[key].astype(str), drop=False)
    expected_keys = expected[key].astype(str).tolist()
    if set(indexed.index) != set(expected_keys):
        raise ValueError(f"{source} registered row family differs from recomputation")
    indexed = indexed.loc[expected_keys].reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    missing = sorted(set(expected.columns) - set(indexed.columns))
    if missing:
        raise ValueError(f"{source} lacks recomputed columns: {missing}")
    for column in expected.columns:
        left = indexed[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(right) and not pd.api.types.is_bool_dtype(
            right
        ):
            if not np.allclose(
                pd.to_numeric(left, errors="raise").to_numpy(dtype=float),
                pd.to_numeric(right, errors="raise").to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{source} column {column} differs from raw recomputation"
                )
        elif left.astype(str).tolist() != right.astype(str).tolist():
            raise ValueError(f"{source} column {column} differs from raw recomputation")


def load_validated_structured_snapshot(
    results_root: str | Path,
    *,
    prefix: str = "exp13_arc_formal",
    minimum_candidate_coverage: float = 0.9,
    require_published_root: bool = False,
    task_family: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and independently recompute every exp13 derived summary table."""

    root = Path(results_root)
    paths = {
        "conditions": root / f"{prefix}_conditions.csv",
        "comparisons": root / f"{prefix}_comparisons.csv",
        "raw": root / f"{prefix}_raw.csv.gz",
        "run_manifest": root / f"{prefix}_run_manifest.csv",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"exp13 formal snapshot is incomplete: {missing}")
    conditions = pd.read_csv(paths["conditions"])
    comparisons = pd.read_csv(paths["comparisons"])
    if task_family is None:
        task_family = (
            _one_text(comparisons, "task_family", source="comparisons")
            if "task_family" in comparisons
            else "arc"
        )
    task_family = str(task_family).strip().lower()
    if require_published_root and task_family != "arc":
        raise ValueError("the published exp13 trusted root is ARC-specific")
    raw_sha = _file_sha256(paths["raw"])
    run_sha = _file_sha256(paths["run_manifest"])
    for frame, source in ((conditions, "conditions"), (comparisons, "comparisons")):
        if _one_text(frame, "scoped_raw_sha256", source=source) != raw_sha:
            raise ValueError(f"{source} is not bound to the actual exp13 raw file")
        if _one_text(frame, "run_manifest_sha256", source=source) != run_sha:
            raise ValueError(f"{source} is not bound to the actual exp13 run manifest")
    if require_published_root and (
        raw_sha != PUBLISHED_EXP13_RAW_SHA256
        or run_sha != PUBLISHED_EXP13_RUN_MANIFEST_SHA256
    ):
        raise ValueError("exp13 raw/run files differ from the published trusted root")
    if require_published_root:
        # The immutable published ARC tables predate explicit dataset/split-role
        # columns.  Enrich only the in-memory view after the raw/run hashes have
        # authenticated the legacy snapshot; never migrate trusted files.
        legacy_bindings = {
            "dataset_name": PUBLISHED_EXP13_DATASET_NAME,
            "registered_ood_split": True,
            "task_family": "arc",
            "test_split_role": PUBLISHED_EXP13_TEST_SPLIT_ROLE,
        }
        for frame in (conditions, comparisons):
            for column, value in legacy_bindings.items():
                if column not in frame:
                    frame[column] = value
    raw = pd.read_csv(paths["raw"], low_memory=False)
    run_manifest = pd.read_csv(paths["run_manifest"])

    expected_seeds = sorted(raw["seed"].astype(int).unique().tolist())
    if (
        len(expected_seeds) < 2
        or set(run_manifest["seed"].astype(int)) != set(expected_seeds)
        or run_manifest["seed"].astype(int).duplicated().any()
    ):
        raise ValueError("exp13 raw/run manifest seed panels disagree")
    if set(run_manifest["status"].astype(str)) != {"complete"}:
        raise ValueError("exp13 run manifest contains incomplete runs")
    if run_manifest["git_dirty"].astype(bool).any():
        raise ValueError("exp13 run manifest contains dirty runs")
    commits = run_manifest["git_commit"].astype(str).unique()
    if len(commits) != 1:
        raise ValueError("exp13 run manifest contains multiple commits")
    for frame, source in ((conditions, "conditions"), (comparisons, "comparisons")):
        if _one_text(frame, "run_git_commit", source=source) != commits[0]:
            raise ValueError(f"{source} commit differs from the run manifest")
    if require_published_root:
        binding_values = {
            "source_manifest_sha256": PUBLISHED_EXP13_SOURCE_MANIFEST_SHA256,
            "source_revision": PUBLISHED_EXP13_SOURCE_REVISION,
            "dataset_name": PUBLISHED_EXP13_DATASET_NAME,
            "test_split_role": PUBLISHED_EXP13_TEST_SPLIT_ROLE,
        }
    else:
        binding_values = {
            column: _one_text(run_manifest, column, source="run_manifest")
            for column in (
                "source_manifest_sha256",
                "source_revision",
                "dataset_name",
                "formal_config_sha256",
                "test_split_role",
            )
        }
    for column, expected in binding_values.items():
        for frame, source in (
            (conditions, "conditions"),
            (comparisons, "comparisons"),
        ):
            _assert_every_row_matches(frame, column, expected, source=source)
    test_split_role = binding_values["test_split_role"]
    bootstrap_seed = int(_one_text(conditions, "bootstrap_seed", source="conditions"))
    n_bootstrap = int(_one_text(conditions, "n_bootstrap", source="conditions"))
    if (
        int(_one_text(comparisons, "bootstrap_seed", source="comparisons"))
        != bootstrap_seed
    ):
        raise ValueError("exp13 bootstrap seed differs across derived tables")
    if int(_one_text(comparisons, "n_bootstrap", source="comparisons")) != n_bootstrap:
        raise ValueError("exp13 bootstrap count differs across derived tables")
    recomputed_conditions, recomputed_comparisons = summarize_structured_formal(
        raw,
        expected_seeds=expected_seeds,
        seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
        minimum_candidate_coverage=minimum_candidate_coverage,
        task_family=task_family,
        test_split_role=test_split_role,
    )
    _assert_recomputed_table(
        conditions,
        recomputed_conditions,
        key="condition",
        source="exp13 conditions",
    )
    _assert_recomputed_table(
        comparisons,
        recomputed_comparisons,
        key="comparison",
        source="exp13 comparisons",
    )
    condition_order = list(STRUCTURED_CONDITIONS)
    comparison_order = [item[0] for item in FORMAL_COMPARISONS]
    conditions = conditions.set_index("condition").loc[condition_order].reset_index()
    comparisons = (
        comparisons.set_index("comparison").loc[comparison_order].reset_index()
    )
    return conditions, comparisons, raw, run_manifest


__all__ = [
    "FORMAL_COMPARISONS",
    "PUBLISHED_EXP13_DATASET_NAME",
    "PUBLISHED_EXP13_RAW_SHA256",
    "PUBLISHED_EXP13_RUN_MANIFEST_SHA256",
    "PUBLISHED_EXP13_SOURCE_MANIFEST_SHA256",
    "PUBLISHED_EXP13_SOURCE_REVISION",
    "PUBLISHED_EXP13_TEST_SPLIT_ROLE",
    "load_validated_structured_snapshot",
    "summarize_structured_formal",
]
