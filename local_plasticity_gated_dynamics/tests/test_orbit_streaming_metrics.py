from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.orbit_streaming_metrics import (
    actuator_headroom,
    holm_adjust,
    paired_user_inference,
    reduce_to_user_accuracy,
    task_video_accuracy_rows,
)


def test_task_video_rows_preserve_public_scoring_atom() -> None:
    rows = task_video_accuracy_rows(
        user_id="u0",
        task_index=0,
        condition="local",
        labels=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        video_ids=["a", "a", "b", "b"],
        selected_actions=[0, 1, 1, 1],
    )
    assert rows["frame_accuracy"].tolist() == [0.5, 1.0]
    user = reduce_to_user_accuracy(rows)
    assert user.loc[0, "user_video_mean_accuracy"] == 0.75


def test_headroom_separates_fixed_and_per_frame_oracle() -> None:
    headroom = actuator_headroom(
        [0, 1, 0, 1],
        np.asarray([[0, 1], [0, 1], [1, 0], [1, 0]]),
    )
    assert np.allclose(headroom.per_action_accuracy, [0.5, 0.5])
    assert headroom.best_fixed_accuracy == 0.5
    assert headroom.oracle_accuracy == 1.0
    assert headroom.oracle_gain == 0.5
    assert headroom.action_disagreement == 1.0


def test_inference_uses_paired_users_and_holm_is_monotone() -> None:
    rows = pd.DataFrame(
        {
            "user_id": ["u0", "u0", "u1", "u1", "u2", "u2"],
            "condition": ["method", "base"] * 3,
            "user_video_mean_accuracy": [0.8, 0.6, 0.7, 0.6, 0.9, 0.7],
        }
    )
    result = paired_user_inference(
        rows,
        method="method",
        comparator="base",
        bootstrap_samples=1000,
        seed=5,
    )
    assert result.n_users == 3
    assert np.isclose(result.mean_difference, 1.0 / 6.0)
    assert result.positive_users == 3
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
