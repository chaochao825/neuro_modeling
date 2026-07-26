from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from experiments.common import load_json_config
from figures.exp38_stream51_soft_memory_plot import (
    COMPARISON_LABELS,
    EXTERNAL_CONDITIONS,
    make_external_figure,
    make_qualification_figure,
    save_all,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return deepcopy(
        load_json_config(
            PROJECT_ROOT / "configs/prospective/exp38_stream51_soft_memory.json"
        )
    )


def _receipt() -> dict[str, object]:
    seed_results = []
    for seed in _config()["seeds"]:
        qualification = {
            "stable_accumulation_gain": 0.03,
            "oracle_adaptation_headroom": 0.04,
            "cumulative_post_switch_harm": 0.05,
            "reachability_auc": 0.71,
            "reachability_recall": 0.45,
            "reachability_false_alarms_per_1000": 12.0,
            "reachability_median_delay": 5.0,
            "stable_accumulation_gate": True,
            "oracle_headroom_gate": True,
            "cumulative_harm_gate": True,
            "reachability_gate": True,
            "passed": True,
        }
        seed_results.append(
            {"seed": seed, "passed": True, "qualification": qualification}
        )
    return {
        "receipt_type": "exp38_qualification_gate",
        "external_stage_authorized": True,
        "seed_results": seed_results,
    }


def test_qualification_figure_binds_all_four_gates(tmp_path: Path) -> None:
    figure = make_qualification_figure(_receipt(), _config())
    assert len(figure.axes) == 4
    assert len(figure.axes[3].get_yticklabels()) == len(_config()["seeds"])
    assert len(figure.axes[3].get_xticklabels()) == 4
    outputs = save_all(figure, tmp_path / "qualification")
    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".png"}
    assert all(path.stat().st_size > 0 for path in outputs)
    plt.close(figure)


def test_incomplete_qualification_receipt_fails_closed() -> None:
    receipt = _receipt()
    del receipt["seed_results"][0]["qualification"]["reachability_auc"]
    with pytest.raises(ValueError, match="misses registered"):
        make_qualification_figure(receipt, _config())


def test_external_figure_requires_exact_registered_cells() -> None:
    conditions = pd.DataFrame(
        [
            {
                "panel": panel,
                "condition": condition,
                "video_equal_accuracy": 0.7,
            }
            for panel in ("natural", "hidden_switch")
            for condition in EXTERNAL_CONDITIONS
        ]
    )
    comparisons = pd.DataFrame(
        [
            {
                "comparison": comparison,
                "mean_difference": 0.03,
                "ci_low": 0.01,
                "ci_high": 0.05,
                "holm_adjusted_pvalue": 0.01,
            }
            for comparison in COMPARISON_LABELS
        ]
    )
    figure = make_external_figure(conditions, comparisons, _config())
    assert len(figure.axes) == 4
    plt.close(figure)
    with pytest.raises(ValueError, match="registered cells"):
        make_external_figure(conditions.iloc[:-1], comparisons, _config())
