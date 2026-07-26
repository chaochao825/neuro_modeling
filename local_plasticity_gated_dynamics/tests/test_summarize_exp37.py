from __future__ import annotations

import pandas as pd
import pytest

from experiments.exp37_core50_change_aware_prefix import CONDITIONS, PANELS
from scripts.summarize_exp37 import summarize_panel


def _config() -> dict[str, object]:
    return {
        "profile": "prospective_external",
        "evidence_provenance": "prospective_core50_session_holdout",
        "seeds": [1, 2, 3, 4, 5],
        "external_sessions": [f"s{i}" for i in range(3, 12)],
        "n_external_tasks_per_session": 2,
        "analysis": {
            "bootstrap_samples": 1000,
            "statistics_seed": 7,
            "alpha": 0.05,
            "hidden_switch_mcid": 0.03,
            "natural_noninferiority_margin": 0.02,
            "max_natural_point_loss": 0.01,
            "max_median_delay": 8.0,
            "max_false_alarms_per_1000": 5.0,
        },
    }


def _supporting_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3, 4, 5):
        for session in (f"s{i}" for i in range(3, 12)):
            for task in range(2):
                for panel in PANELS:
                    for condition in CONDITIONS:
                        accuracy = 0.75
                        post = 0.75
                        if panel == "hidden_switch":
                            values = {
                                "cumulative": (0.64, 0.55),
                                "fixed_forgetting": (0.69, 0.64),
                                "sliding_window": (0.68, 0.63),
                                "bocpd_change_reset": (0.75, 0.72),
                                "bocpd_score_no_reset": (0.65, 0.56),
                                "matched_shifted_reset": (0.66, 0.59),
                                "bocpd_posterior": (0.73, 0.70),
                                "oracle_change_reset": (0.80, 0.79),
                            }
                            accuracy, post = values.get(condition, (accuracy, post))
                        elif condition == "cumulative":
                            accuracy = 0.80
                        elif condition == "bocpd_change_reset":
                            accuracy = 0.795
                        switches = 5 if panel == "hidden_switch" else 0
                        detector = condition in {"bocpd_change_reset", "bocpd_score_no_reset"}
                        alarms = 5 if panel == "hidden_switch" and detector else 0
                        rows.append(
                            {
                                "seed": seed,
                                "session_id": session,
                                "task_index": task,
                                "panel": panel,
                                "condition": condition,
                                "accuracy": accuracy,
                                "post_switch_accuracy": post if panel == "hidden_switch" else float("nan"),
                                "detection_precision": 1.0 if alarms else float("nan"),
                                "detection_recall": 1.0 if alarms else float("nan"),
                                "false_alarms_per_1000": 0.0,
                                "median_detection_delay": 4.0 if alarms else float("nan"),
                                "n_frames": 192 if panel == "hidden_switch" else 512,
                                "n_switches": switches,
                                "n_alarms": alarms,
                                "n_matched_switches": alarms,
                                "n_resets": alarms,
                                "mean_state_l1": 1.0,
                                "object_ids": "o1|o2|o3|o4",
                                "source_cells": f"{session}/o1",
                                "status": "complete",
                            }
                        )
    return pd.DataFrame(rows)


def test_exp37_summary_uses_session_and_passes_joint_gate() -> None:
    sessions, comparisons, summary = summarize_panel(_supporting_panel(), config=_config())
    assert len(sessions) == 9 * len(PANELS) * len(CONDITIONS)
    assert summary["n_sessions"] == 9
    assert summary["statistical_unit"] == "session"
    assert summary["conclusion"] == "support"
    assert not summary["stop_rule_triggered"]
    assert all(summary["support_gates"].values())
    assert len(comparisons) == 4
    assert comparisons["n_users"].eq(9).all()


def test_exp37_stop_rule_when_stationary_window_matches() -> None:
    raw = _supporting_panel()
    mask = (raw["panel"] == "hidden_switch") & (raw["condition"] == "sliding_window")
    raw.loc[mask, "accuracy"] = 0.76
    _, _, summary = summarize_panel(raw, config=_config())
    assert summary["stop_rule_triggered"]
    assert summary["conclusion"] != "support"


def test_exp37_summary_fails_closed_on_missing_cell() -> None:
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        summarize_panel(_supporting_panel().iloc[:-1], config=_config())
