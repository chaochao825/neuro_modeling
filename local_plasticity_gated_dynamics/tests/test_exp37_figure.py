from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from experiments.exp37_core50_change_aware_prefix import CONDITIONS, PANELS
from figures.exp37_core50_change_aware_prefix_plot import make_figure
from scripts.summarize_exp37 import PRIMARY_COMPARISONS


def test_exp37_figure_is_bound_to_registered_session_panel() -> None:
    sessions = pd.DataFrame(
        [
            {
                "session_id": session,
                "panel": panel,
                "condition": condition,
                "accuracy": 0.6 + 0.01 * index,
                "false_alarms_per_1000": 1.0,
                "median_detection_delay": 4.0,
            }
            for session in ("s3", "s4", "s5")
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
    figure = make_figure(sessions, comparisons)
    assert len(figure.axes) == 4
    assert len(figure.axes[0].get_xticklabels()) == len(
        ("current_frame", "cumulative", "fixed_forgetting", "sliding_window", "bocpd_change_reset", "oracle_change_reset")
    )
    assert len(figure.axes[1].get_yticklabels()) == len(PRIMARY_COMPARISONS)
    plt.close(figure)
