from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp33_orbit_streaming_fewshot import (
    EVALUATION_CONDITIONS,
    run_seed,
)
from src.data.orbit_streaming import FEATURE_MANIFEST_COLUMNS


def _synthetic_config(tmp_path: Path) -> dict[str, object]:
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["train-only"],
                "validation": ["fit-user", "eval-user"],
                "test": ["test-only"],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "features"
    root.mkdir()
    rows = []
    for user_index, user in enumerate(("fit-user", "eval-user")):
        for label, object_name in enumerate(("cup", "keys")):
            for video_type in ("clean", "clutter"):
                video_id = f"{user}-{object_name}-{video_type}"
                relative = Path("validation") / f"{video_id}.npz"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                rng = np.random.default_rng(100 * user_index + 10 * label)
                center = np.zeros(6)
                center[label] = 2.0
                embeddings = center + rng.normal(0.0, 0.1, size=(8, 6))
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
        "protocol_version": "exp33_orbit_causal_cluve_v1",
        "training_algorithm": "executed_reward_only_contextual_bandit",
        "used_autograd": False,
        "used_bptt": False,
        "feature_root": str(root),
        "official_splits_path": str(split_path),
        "fit_split": "validation",
        "eval_split": "validation",
        "fit_user_ids": ["fit-user"],
        "eval_user_ids": ["eval-user"],
        "require_complete_fit_split": True,
        "require_complete_eval_split": True,
        "n_fit_tasks_per_user": 1,
        "n_eval_tasks_per_user": 1,
        "training_frame_stride": 1,
        "sampling": {
            "support_stride": 2,
            "max_support_frames_per_video": 4,
            "query_frames_per_video": 4,
            "min_query_frames_per_video": 2,
            "max_frames_per_video": 8,
            "support_video_limit": None,
        },
        "actuators": {},
        "controller": {"epsilon": 0.2, "belief_retention": 0.0},
        "analysis": {"minimum_accuracy_gain": 0.0},
    }


def test_exp33_preserves_all_conditions_and_receipts(tmp_path: Path) -> None:
    run = run_seed(
        _synthetic_config(tmp_path), seed=4, results_root=tmp_path / "results"
    )
    raw = pd.read_csv(run / "raw_video_metrics.csv")
    assert set(raw["condition"]) == set(EVALUATION_CONDITIONS)
    assert set(raw["status"]) == {"complete"}
    receipt = json.loads((run / "controller_receipts.json").read_text(encoding="utf-8"))
    assert receipt["main"]["used_autograd"] is False
    assert receipt["main"]["used_bptt"] is False
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["conclusion"] == "inconclusive"
    assert (run / "planned_conditions.json").is_file()
