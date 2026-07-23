from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp34_orbit_causal_consensus import formal_config_fingerprint
from figures.exp34_orbit_causal_consensus_plot import make_figure
from scripts.summarize_exp34 import (
    build_authorization_receipt,
    load_panel,
    summarize_panel,
)


CONDITIONS = (
    "prototype",
    "gain",
    "delta",
    "temporal",
    "selection_fixed_best",
    "instantaneous_majority",
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
        "instantaneous_majority": 0.67,
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
                        "mean_event_l1": (
                            4.0
                            if "consensus" in condition
                            or condition == "instantaneous_majority"
                            else 1.0
                        ),
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
    selection_features = tmp_path / "selection-features"
    evaluation_features = tmp_path / "evaluation-features"
    for root, split in (
        (selection_features, "validation"),
        (evaluation_features, "test"),
    ):
        root.mkdir()
        (root / "feature_manifest.csv").write_text("split\n" + split + "\n")
        (root / f"provenance_{split}.json").write_text("{}\n")
    formal = {
        "profile": "formal",
        "protocol_version": "exp34_orbit_causal_consensus_v3_official_video_exclusion",
        "seeds": [1],
        "selection_split": "validation",
        "eval_split": "test",
        "selection_feature_root": str(selection_features),
        "eval_feature_root": str(evaluation_features),
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


def test_exp34_loader_blocks_any_failed_condition_or_missing_coverage(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(
        '{"seed": 1, "status": "complete", "condition_failures": 0, '
        '"condition_invalid": 0}\n',
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        '{"profile": "smoke", "run_label": "smoke"}\n', encoding="utf-8"
    )
    (run / "config.json").write_text(
        '{"profile": "smoke", "protocol_version": '
        '"exp34_orbit_causal_consensus_v3_official_video_exclusion"}\n',
        encoding="utf-8",
    )
    rows = []
    for condition in CONDITIONS:
        rows.append(
            {
                "user_id": "u0",
                "task_index": 0,
                "video_id": "v0",
                "condition": condition,
                "frame_accuracy": 0.5,
                "status": "complete",
                "episode_fingerprint": "episode",
                "trace_fingerprint": "trace",
            }
        )
    pd.DataFrame(rows).to_csv(run / "raw_video_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "user_id": "u0",
                "task_index": 0,
                "oracle_gain": 0.1,
                "action_disagreement": 0.2,
                "excluded_query_video_ids": "[]",
            }
        ]
    ).to_csv(run / "actuator_headroom.csv", index=False)
    config = {
        "profile": "smoke",
        "seeds": [1],
        "eval_user_ids": ["u0"],
        "eval_split": "validation",
        "official_splits_path": "unused.json",
        "n_eval_tasks_per_user": 1,
    }
    _, _, _, coverage = load_panel([run], config=config)
    assert coverage["complete"] is True

    (run / "status.json").write_text(
        '{"seed": 1, "status": "complete_with_failures", '
        '"condition_failures": 1, "condition_invalid": 0}\n',
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(RuntimeError, "formal inference is blocked"):
        load_panel([run], config=config)

    (run / "status.json").write_text(
        '{"seed": 1, "status": "complete", "condition_failures": 0, '
        '"condition_invalid": 0}\n',
        encoding="utf-8",
    )
    config["n_eval_tasks_per_user"] = 2
    with np.testing.assert_raises_regex(RuntimeError, "coverage is incomplete"):
        load_panel([run], config=config)
