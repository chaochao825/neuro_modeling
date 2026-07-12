from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from figures.exp13_structured_reasoning_plot import plot_exp13
from figures.plot_style import save_figure
from src.analysis.structured_benchmark import STRUCTURED_CONDITIONS
from src.analysis.structured_formal import summarize_structured_formal
from scripts.build_report import append_exp13_structured_claims


def _panel(n_seeds: int = 3) -> pd.DataFrame:
    rows = []
    exact_by_condition = {
        "support_heuristic": 0.35,
        "flat_local": 0.25,
        "hierarchical_local": 0.75,
        "trace_local": 0.70,
        "gru_bptt": 0.65,
        "candidate_oracle": 0.90,
    }
    for seed in range(n_seeds):
        for task in range(40):
            for condition in STRUCTURED_CONDITIONS:
                threshold = exact_by_condition[condition]
                exact = float(((task * 17 + seed * 3) % 100) / 100 < threshold)
                is_local = condition in {
                    "flat_local",
                    "hierarchical_local",
                    "trace_local",
                }
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "task_id": f"task-{task}",
                        "source_group": f"task-{task}",
                        "augmentation_group": f"task-{task}",
                        "exact": exact,
                        "candidate_covered": 1.0,
                        "candidate_fingerprint": f"fingerprint-{task}",
                        "parameter_count": 100 if is_local else 120,
                        "trainable_parameter_count": 10 if is_local else 120,
                        "used_bptt": condition == "gru_bptt",
                        "control_dim": (
                            4
                            if condition in {"hierarchical_local", "trace_local"}
                            else 0
                        ),
                        "control_operator_rank": (
                            4
                            if condition in {"hierarchical_local", "trace_local"}
                            else 0
                        ),
                        "run_id": f"run-{seed}",
                        "run_git_commit": "a" * 40,
                        "run_git_dirty": False,
                    }
                )
    return pd.DataFrame(rows)


def test_formal_summary_uses_tasks_with_nested_seed_averages() -> None:
    conditions, comparisons = summarize_structured_formal(
        _panel(), expected_seeds=(0, 1, 2), seed=7, n_bootstrap=500
    )
    assert set(conditions["condition"]) == set(STRUCTURED_CONDITIONS)
    hierarchical = conditions.set_index("condition").loc["hierarchical_local"]
    flat = conditions.set_index("condition").loc["flat_local"]
    assert hierarchical["exact_accuracy"] > flat["exact_accuracy"]
    assert hierarchical["control_dim"] == 4
    assert hierarchical["used_bptt"] == np.bool_(False)
    comparison = comparisons.set_index("comparison").loc["hierarchical_vs_flat"]
    assert comparison["estimate"] > 0.0
    assert comparison["conclusion"] == "support"
    assert set(comparisons["conclusion"]) <= {
        "support",
        "oppose",
        "inconclusive",
    }


def test_formal_summary_rejects_candidate_or_seed_mismatch() -> None:
    raw = _panel()
    raw.loc[
        (raw["seed"] == 2)
        & (raw["task_id"] == "task-0")
        & (raw["condition"] == "flat_local"),
        "candidate_fingerprint",
    ] = "changed"
    with pytest.raises(ValueError, match="candidate panels differ"):
        summarize_structured_formal(raw, expected_seeds=(0, 1, 2), n_bootstrap=100)

    incomplete = _panel().loc[lambda frame: frame["seed"] != 2]
    with pytest.raises(ValueError, match="missing or adds seeds"):
        summarize_structured_formal(
            incomplete, expected_seeds=(0, 1, 2), n_bootstrap=100
        )


def test_exp13_formal_figure_is_bound_to_summary_and_raw(tmp_path) -> None:
    raw = _panel()
    conditions, comparisons = summarize_structured_formal(
        raw, expected_seeds=(0, 1, 2), seed=9, n_bootstrap=100
    )
    prefix = "exp13_test"
    raw_path = tmp_path / f"{prefix}_raw.csv.gz"
    manifest_path = tmp_path / f"{prefix}_run_manifest.csv"
    raw.to_csv(
        raw_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(
        {
            "seed": range(3),
            "status": "complete",
            "git_commit": "a" * 40,
            "git_dirty": False,
        }
    ).to_csv(manifest_path, index=False)
    bindings = {
        "source_revision": "revision",
        "scoped_raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "run_git_commit": "a" * 40,
        "run_git_dirty": False,
    }
    for key, value in bindings.items():
        conditions[key] = value
        comparisons[key] = value
    conditions.to_csv(tmp_path / f"{prefix}_conditions.csv", index=False)
    comparisons.to_csv(tmp_path / f"{prefix}_comparisons.csv", index=False)
    figure = plot_exp13(tmp_path, prefix)
    assert len(figure.axes) == 4
    save_figure(figure, "exp13_test", tmp_path)
    assert (tmp_path / "exp13_test.pdf").stat().st_size > 0
    assert (tmp_path / "exp13_test.png").stat().st_size > 0
    plt.close(figure)


def test_exp13_global_claims_are_bound_to_clean_snapshot(tmp_path) -> None:
    raw = _panel(30)
    conditions, comparisons = summarize_structured_formal(
        raw, expected_seeds=range(30), seed=9, n_bootstrap=100
    )
    raw_path = tmp_path / "exp13_arc_formal_raw.csv.gz"
    manifest_path = tmp_path / "exp13_arc_formal_run_manifest.csv"
    raw.to_csv(
        raw_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(
        {
            "seed": range(30),
            "run_id": [f"run-{seed}" for seed in range(30)],
            "status": "complete",
            "git_commit": "c" * 40,
            "git_dirty": False,
        }
    ).to_csv(manifest_path, index=False)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    bindings = {
        "source_revision": "revision",
        "scoped_raw_sha256": sha(raw_path),
        "run_manifest_sha256": sha(manifest_path),
        "run_git_commit": "c" * 40,
        "run_git_dirty": False,
    }
    for key, value in bindings.items():
        conditions[key] = value
        comparisons[key] = value
    conditions.to_csv(tmp_path / "exp13_arc_formal_conditions.csv", index=False)
    comparisons.to_csv(tmp_path / "exp13_arc_formal_comparisons.csv", index=False)
    summary = append_exp13_structured_claims(
        pd.DataFrame(), tmp_path, require_published_root=False
    )
    assert len(summary) == 6
    assert set(summary["conclusion"]) <= {"support", "oppose", "inconclusive"}
    assert all("no neural/biological claim" in note for note in summary["note"])
    t5 = summary.set_index("claim_id").loc["T5_arc_hierarchical_90pct_gru"]
    assert "minus 0.9 times" in t5["comparison"]
    assert "non-inferiority margin" in t5["criterion"]

    tampered = comparisons.copy()
    tampered.loc[0, "estimate"] = 999.0
    tampered.to_csv(tmp_path / "exp13_arc_formal_comparisons.csv", index=False)
    with pytest.raises(ValueError, match="differs from raw recomputation"):
        append_exp13_structured_claims(
            pd.DataFrame(), tmp_path, require_published_root=False
        )
    comparisons.to_csv(tmp_path / "exp13_arc_formal_comparisons.csv", index=False)
    tampered_conditions = conditions.copy()
    tampered_conditions.loc[0, "exact_accuracy"] = 999.0
    tampered_conditions.to_csv(
        tmp_path / "exp13_arc_formal_conditions.csv", index=False
    )
    with pytest.raises(ValueError, match="differs from raw recomputation"):
        append_exp13_structured_claims(
            pd.DataFrame(), tmp_path, require_published_root=False
        )
    conditions.to_csv(tmp_path / "exp13_arc_formal_conditions.csv", index=False)

    raw_path.write_bytes(raw_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="not bound"):
        append_exp13_structured_claims(
            pd.DataFrame(), tmp_path, require_published_root=False
        )
