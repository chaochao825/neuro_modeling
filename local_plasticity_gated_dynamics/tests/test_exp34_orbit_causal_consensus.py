from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp34_orbit_causal_consensus import (
    EVALUATION_CONDITIONS,
    run_seed,
)
from src.data.orbit_streaming import FEATURE_MANIFEST_COLUMNS


def _config(tmp_path: Path) -> dict[str, object]:
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["train-only"],
                "validation": ["select-user", "eval-user"],
                "test": ["test-only"],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "features"
    root.mkdir()
    rows = []
    for user_number, user in enumerate(("select-user", "eval-user")):
        for label, object_name in enumerate(("cup", "keys")):
            for video_type in ("clean", "clutter"):
                video_id = f"{user}-{object_name}-{video_type}"
                relative = Path("validation") / f"{video_id}.npz"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                rng = np.random.default_rng(10 * user_number + label)
                center = np.zeros(6)
                center[label] = 2.0
                embeddings = center + rng.normal(0.0, 0.15, size=(8, 6))
                np.savez_compressed(
                    path,
                    embeddings=embeddings,
                    frame_indices=np.arange(8),
                    object_present=np.ones(8, dtype=np.bool_),
                )
                rows.append(
                    {
                        "split": "validation",
                        "user_id": user,
                        "object_name": object_name,
                        "video_type": video_type,
                        "video_id": video_id,
                        "feature_path": relative.as_posix(),
                        "n_frames": 8,
                        "feature_dim": 6,
                        "source_fingerprint": video_id,
                    }
                )
    pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS).to_csv(
        root / "feature_manifest.csv", index=False
    )
    return {
        "profile": "smoke",
        "protocol_version": "exp34_orbit_causal_consensus_v1",
        "training_algorithm": "causal_label_free_count_belief",
        "used_query_labels": False,
        "used_future_frames": False,
        "used_autograd": False,
        "used_bptt": False,
        "feature_root": str(root),
        "official_splits_path": str(split_path),
        "selection_split": "validation",
        "eval_split": "validation",
        "selection_user_ids": ["select-user"],
        "eval_user_ids": ["eval-user"],
        "require_complete_selection_split": True,
        "require_complete_eval_split": True,
        "n_selection_tasks_per_user": 1,
        "n_eval_tasks_per_user": 1,
        "sampling": {
            "support_stride": 2,
            "max_support_frames_per_video": 4,
            "query_frames_per_video": 4,
            "min_query_frames_per_video": 2,
            "max_frames_per_video": 8,
            "support_video_limit": None,
        },
        "actuators": {},
        "gate": {
            "retention": 1.0,
            "prior_count": 0.0,
            "delay_frames": 0,
            "reset_each_frame": False,
            "tie_break_order": [3, 1, 0, 2],
        },
        "analysis": {
            "delay_intervention_frames": 2,
            "minimum_accuracy_gain": 0.0,
            "bootstrap_samples": 1000,
        },
    }


def test_exp34_retains_all_causal_interventions(tmp_path: Path) -> None:
    path = run_seed(_config(tmp_path), seed=8, results_root=tmp_path / "results")
    raw = pd.read_csv(path / "raw_video_metrics.csv")
    assert set(raw["condition"]) == set(EVALUATION_CONDITIONS)
    consensus = raw.loc[raw["condition"] == "causal_consensus"]
    assert set(consensus["compute_scope"]) == {"full_actuator_bank"}
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert summary["conclusion"] == "inconclusive"
    assert (path / "selection_audit.json").is_file()
