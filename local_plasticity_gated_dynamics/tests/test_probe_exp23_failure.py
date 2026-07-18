from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.probe_exp23_failure import CONDITIONS, TASKS, load_archive, summarize


def _synthetic_raw() -> pd.DataFrame:
    rows = []
    offsets = {
        "frozen": 0.0,
        "local_eprop": -0.005,
        "exact_forward_sensitivity": 0.002,
        "bptt_axis_only": 0.01,
        "random_update": -0.002,
        "current_off_policy": -0.004,
    }
    for seed in range(30):
        for task in TASKS:
            for condition in CONDITIONS:
                gain = offsets[condition] + seed * 1e-5
                natural_l2 = 0.01 if condition != "frozen" else 0.0
                matched_l2 = natural_l2 * (10.0 if condition == "local_eprop" else 1.0)
                rows.append(
                    {
                        "seed": seed,
                        "task_variant": task,
                        "condition": condition,
                        "behavior_balanced_accuracy": 0.8 + gain,
                        "natural_behavior_balanced_accuracy": 0.8 + gain / 2,
                        "test_task_loss": 0.5 - gain,
                        "control_axis_l2": matched_l2,
                        "natural_control_axis_l2": natural_l2,
                        "median_update_cosine_to_exact": (
                            0.5 if condition == "local_eprop" else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_exp23_probe_keeps_negative_result_and_reports_scale_sensitivity() -> None:
    conditions, seeds, summary = summarize(_synthetic_raw())
    assert len(conditions) == 12
    assert len(seeds) == 60
    assert summary["existing_claim_remains"].startswith("inconclusive")
    assert summary["formal_eligibility_reverified"] is False
    assert summary["general_local_learning_conclusion"] == "inconclusive"
    assert "matched_vs_natural" in summary["probe_classification"]
    assert (
        summary["task_summary"]["delayed"]["local_eprop_median_axis_rescale_ratio"]
        == 10.0
    )


def test_exp23_probe_rejects_unbound_archive(tmp_path: Path) -> None:
    path = tmp_path / "unbound.tar.gz"
    path.write_bytes(b"not the frozen archive")
    with pytest.raises(RuntimeError, match="SHA-256"):
        load_archive(path)
