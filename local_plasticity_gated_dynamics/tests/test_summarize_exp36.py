from __future__ import annotations

import pandas as pd
import pytest

from experiments.exp36_change_aware_prefix import CONDITIONS, PANELS
from scripts.summarize_exp36 import summarize_panel


def _config() -> dict[str, object]:
    return {
        "profile": "prospective_external",
        "evidence_provenance": "prospective_external_confirmation",
        "seeds": [1, 2],
        "external_collectors": [f"P{index}" for index in range(1, 13)],
        "n_external_tasks_per_collector": 2,
        "analysis": {
            "bootstrap_samples": 1000,
            "statistics_seed": 7,
            "alpha": 0.05,
            "hidden_switch_mcid": 0.03,
            "natural_noninferiority_margin": 0.02,
            "max_median_delay": 8.0,
            "max_false_alarms_per_1000": 5.0,
        },
    }


def _supporting_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in (1, 2):
        for collector in (f"P{index}" for index in range(1, 13)):
            for task in range(2):
                for panel in PANELS:
                    for condition in CONDITIONS:
                        accuracy = 0.70
                        post = 0.70
                        if panel == "hidden_switch":
                            if condition == "cumulative":
                                accuracy, post = 0.62, 0.55
                            elif condition == "fixed_forgetting":
                                accuracy, post = 0.66, 0.61
                            elif condition == "jsd_change_reset":
                                accuracy, post = 0.70, 0.67
                            elif condition == "jsd_score_no_reset":
                                accuracy, post = 0.63, 0.56
                            elif condition == "matched_shifted_reset":
                                accuracy, post = 0.64, 0.58
                        elif condition == "cumulative":
                            accuracy = 0.80
                        elif condition == "jsd_change_reset":
                            accuracy = 0.795
                        switches = 3 if panel == "hidden_switch" else 0
                        is_detector = condition in {
                            "jsd_change_reset",
                            "jsd_score_no_reset",
                        }
                        alarms = 3 if panel == "hidden_switch" and is_detector else 0
                        rows.append(
                            {
                                "seed": seed,
                                "unit_id": collector,
                                "task_index": task,
                                "panel": panel,
                                "condition": condition,
                                "accuracy": accuracy,
                                "post_switch_accuracy": (
                                    post if panel == "hidden_switch" else float("nan")
                                ),
                                "detection_precision": 1.0 if alarms else float("nan"),
                                "detection_recall": 1.0 if alarms else float("nan"),
                                "false_alarms_per_1000": 0.0,
                                "median_detection_delay": 4.0 if alarms else float("nan"),
                                "n_frames": 128,
                                "n_switches": switches,
                                "n_alarms": alarms,
                                "n_matched_switches": alarms,
                                "n_resets": alarms,
                                "mean_state_l1": 1.0,
                                "source_video_ids": "v0|v1|v2|v3",
                                "status": "complete",
                            }
                        )
    return pd.DataFrame(rows)


def test_exp36_summary_uses_collector_as_unit_and_passes_joint_gate() -> None:
    collectors, comparisons, summary = summarize_panel(
        _supporting_panel(), config=_config()
    )
    assert len(collectors) == 12 * len(PANELS) * len(CONDITIONS)
    assert summary["n_seeds"] == 2
    assert summary["n_collectors"] == 12
    assert summary["statistical_unit"] == "collector"
    assert summary["conclusion"] == "support"
    assert not summary["stop_rule_triggered"]
    assert all(summary["support_gates"].values())
    assert len(comparisons) == 4
    assert comparisons["n_users"].eq(12).all()


def test_exp36_stop_rule_fires_when_fixed_forgetting_matches_detector() -> None:
    raw = _supporting_panel()
    mask = (raw["panel"] == "hidden_switch") & (
        raw["condition"] == "fixed_forgetting"
    )
    raw.loc[mask, "accuracy"] = 0.71
    _, _, summary = summarize_panel(raw, config=_config())
    assert summary["stop_rule_triggered"]
    assert summary["conclusion"] != "support"


def test_exp36_missing_detector_delays_fail_closed() -> None:
    raw = _supporting_panel()
    mask = (raw["panel"] == "hidden_switch") & (
        raw["condition"] == "jsd_change_reset"
    )
    raw.loc[mask, ["n_alarms", "n_matched_switches"]] = 0
    raw.loc[mask, "median_detection_delay"] = float("nan")
    _, _, summary = summarize_panel(raw, config=_config())
    assert summary["cohort_median_detection_delay"] is None
    assert not summary["support_gates"]["detection_delay"]
    assert summary["conclusion"] != "support"


def test_exp36_summary_fails_closed_on_missing_registered_cell() -> None:
    raw = _supporting_panel().iloc[:-1].copy()
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        summarize_panel(raw, config=_config())
