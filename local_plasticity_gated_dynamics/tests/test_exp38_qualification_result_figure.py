from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.common import load_json_config
from figures.exp38_stream51_qualification_result_plot import make_result_figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_result_figure_removes_overlapping_numeric_seed_annotations() -> None:
    config = deepcopy(
        load_json_config(
            PROJECT_ROOT / "configs/prospective/exp38_stream51_soft_memory.json"
        )
    )
    seed_results = []
    for seed in config["seeds"]:
        qualification = {
            "stable_accumulation_gain": 0.03,
            "oracle_adaptation_headroom": 0.04,
            "cumulative_post_switch_harm": 0.7,
            "reachability_auc": 0.75,
            "reachability_recall": 0.2,
            "reachability_false_alarms_per_1000": 15.0,
            "reachability_median_delay": 0.0,
            "stable_accumulation_gate": True,
            "oracle_headroom_gate": True,
            "cumulative_harm_gate": True,
            "reachability_gate": False,
            "passed": False,
        }
        seed_results.append(
            {"seed": seed, "passed": False, "qualification": qualification}
        )
    receipt = {
        "receipt_type": "exp38_qualification_gate",
        "external_stage_authorized": False,
        "seed_results": seed_results,
    }
    figure = make_result_figure(receipt, config)
    for ax in (figure.axes[1], figure.axes[2]):
        assert not any(text.get_text().isdigit() for text in ax.texts)
    assert "orange" in figure.axes[3].get_xlabel()
    plt.close(figure)
