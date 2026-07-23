from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.exp34_orbit_causal_consensus import formal_config_fingerprint
from figures.exp34_orbit_causal_consensus_plot import make_figure
from scripts.summarize_exp34 import build_authorization_receipt, summarize_panel


CONDITIONS = (
    "prototype",
    "gain",
    "delta",
    "temporal",
    "selection_fixed_best",
    "causal_consensus",
    "memoryless_consensus",
    "delayed_consensus",
    "oracle_per_frame",
)


def _raw_panel() -> pd.DataFrame:
    values = {
        "prototype": 0.60,
        "gain": 0.66,
        "delta": 0.58,
        "temporal": 0.64,
        "selection_fixed_best": 0.64,
        "causal_consensus": 0.74,
        "memoryless_consensus": 0.65,
        "delayed_consensus": 0.70,
        "oracle_per_frame": 0.82,
    }
    rows = []
    for seed in (1, 2):
        for user_number, user_id in enumerate(("u0", "u1", "u2")):
            for condition in CONDITIONS:
                rows.append(
                    {
                        "seed": seed,
                        "user_id": user_id,
                        "task_index": 0,
                        "video_id": f"v{user_number}",
                        "condition": condition,
                        "frame_accuracy": values[condition] + 0.005 * user_number,
                        "mean_event_l1": 4.0 if "consensus" in condition else 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _config() -> dict[str, object]:
    return {
        "profile": "smoke",
        "analysis": {
            "minimum_accuracy_gain": 0.005,
            "minimum_retained_oracle_headroom": 0.2,
            "bootstrap_samples": 1000,
            "statistics_seed": 7,
        },
    }


def test_exp34_summary_uses_user_not_seed_as_independent_unit() -> None:
    raw = _raw_panel()
    diagnostics = pd.DataFrame(
        {
            "seed": [1, 1, 1, 2, 2, 2],
            "user_id": ["u0", "u1", "u2"] * 2,
            "oracle_gain": [0.12] * 6,
            "action_disagreement": [0.30] * 6,
        }
    )
    comparisons, payload = summarize_panel(raw, diagnostics, config=_config())
    summary = payload["summary"]
    assert summary["n_users"] == 3
    assert summary["n_seeds"] == 2
    assert summary["scale_decision"] == "scale-authorized"
    assert np.isclose(comparisons.iloc[0]["mean_difference"], 0.10)
    assert comparisons.iloc[0]["positive_users"] == 3
    assert np.isclose(
        summary["official_task_video_mean_accuracy"]["causal_consensus"], 0.745
    )


def test_exp34_authorization_receipt_is_self_hashed(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.csv"
    summary_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("seed\n1\n", encoding="utf-8")
    formal = {
        "profile": "formal",
        "protocol_version": "exp34_orbit_causal_consensus_v2_support_annotation_safe",
        "seeds": [1],
        "analysis": {
            "minimum_accuracy_gain": 0.005,
            "minimum_retained_oracle_headroom": 0.2,
        },
    }
    summary = {
        "profile": "smoke",
        "scale_decision": "scale-authorized",
        "n_users": 3,
        "n_seeds": 2,
        "retained_oracle_headroom_fraction": 0.8,
        "comparisons": [],
    }
    receipt = build_authorization_receipt(
        summary,
        formal_config=formal,
        development_summary_path=summary_path,
        run_manifest_path=manifest_path,
    )
    assert receipt["authorized"] is True
    assert len(receipt["receipt_sha256"]) == 64
    moved = dict(formal)
    formal["config_path"] = "/server/a/formal.json"
    moved["config_path"] = "/server/b/formal.json"
    assert formal_config_fingerprint(formal) == formal_config_fingerprint(moved)


def test_exp34_plot_is_bound_to_accuracy_cost_and_interventions() -> None:
    raw = _raw_panel()
    user_panel = (
        raw.groupby(["user_id", "condition"], as_index=False)["frame_accuracy"]
        .mean()
        .rename(columns={"frame_accuracy": "user_video_mean_accuracy"})
    )
    headroom = pd.DataFrame(
        {
            "user_id": ["u0", "u1", "u2"],
            "oracle_gain": [0.12, 0.10, 0.11],
            "action_disagreement": [0.3, 0.4, 0.35],
        }
    )
    figure = make_figure(user_panel, headroom, raw)
    assert len(figure.axes) == 4
