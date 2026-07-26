from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from experiments.exp36_change_aware_prefix import CONDITIONS, PANELS
from figures.exp36_change_aware_prefix_plot import make_figure
from scripts.summarize_exp36 import PRIMARY_COMPARISONS


def test_exp36_figure_is_bound_to_all_registered_comparisons() -> None:
    collectors = pd.DataFrame(
        [
            {
                "collector_id": collector,
                "panel": panel,
                "condition": condition,
                "accuracy": 0.6 + 0.01 * index,
                "false_alarms_per_1000": 1.0,
                "median_detection_delay": 4.0,
            }
            for collector in ("P1", "P2", "P3")
            for panel in PANELS
            for index, condition in enumerate(CONDITIONS)
        ]
    )
    comparisons = pd.DataFrame(
        [
            {
                "comparison": item["comparison"],
                "mean_difference": 0.02,
                "ci_low": 0.01,
                "ci_high": 0.03,
            }
            for item in PRIMARY_COMPARISONS
        ]
    )
    figure = make_figure(collectors, comparisons)
    assert len(figure.axes) == 4
    assert len(figure.axes[0].get_xticklabels()) == 5
    assert len(figure.axes[1].get_yticklabels()) == len(PRIMARY_COMPARISONS)
    plt.close(figure)
