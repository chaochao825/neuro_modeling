from __future__ import annotations

import numpy as np
import pandas as pd

from figures.exp33_orbit_streaming_fewshot_plot import make_figure
from scripts.summarize_exp33 import summarize_panel


def _raw_panel() -> pd.DataFrame:
    rows = []
    conditions = (
        "prototype",
        "gain",
        "delta",
        "temporal",
        "train_fixed_best",
        "reward_only_local",
        "credit_shuffled_local",
        "oracle_per_frame",
    )
    values = {
        "prototype": 0.60,
        "gain": 0.62,
        "delta": 0.61,
        "temporal": 0.64,
        "train_fixed_best": 0.64,
        "reward_only_local": 0.68,
        "credit_shuffled_local": 0.63,
        "oracle_per_frame": 0.78,
    }
    for seed in (1, 2):
        for user_number, user_id in enumerate(("u0", "u1", "u2")):
            for condition in conditions:
                rows.append(
                    {
                        "seed": seed,
                        "user_id": user_id,
                        "task_index": 0,
                        "video_id": f"v{user_number}",
                        "condition": condition,
                        "n_frames": 20,
                        "frame_accuracy": values[condition] + 0.01 * user_number,
                        "action_0_fraction": 0.25,
                        "action_1_fraction": 0.25,
                        "action_2_fraction": 0.25,
                        "action_3_fraction": 0.25,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_averages_seeds_within_user() -> None:
    raw = _raw_panel()
    diagnostics = pd.DataFrame(
        {
            "seed": [1, 1, 1, 2, 2, 2],
            "user_id": ["u0", "u1", "u2"] * 2,
            "oracle_gain": [0.1] * 6,
            "action_disagreement": [0.2] * 6,
        }
    )
    config = {
        "profile": "smoke",
        "analysis": {
            "minimum_oracle_headroom": 0.01,
            "minimum_action_disagreement": 0.05,
            "bootstrap_samples": 1000,
            "statistics_seed": 3,
        },
    }
    comparisons, payload = summarize_panel(raw, diagnostics, config=config)
    assert payload["summary"]["n_users"] == 3
    assert payload["summary"]["n_seeds"] == 2
    assert payload["summary"]["scale_decision"] == "scale-authorized"
    assert np.isclose(comparisons.iloc[0]["mean_difference"], 0.04)


def test_exp33_plot_is_data_bound() -> None:
    raw = _raw_panel()
    user_panel = (
        raw.groupby(["user_id", "condition"], as_index=False)["frame_accuracy"]
        .mean()
        .rename(columns={"frame_accuracy": "user_video_mean_accuracy"})
    )
    headroom = pd.DataFrame(
        {
            "user_id": ["u0", "u1", "u2"],
            "oracle_gain": [0.1, 0.08, 0.12],
            "action_disagreement": [0.2, 0.3, 0.25],
        }
    )
    figure = make_figure(user_panel, headroom, raw)
    assert len(figure.axes) == 4
