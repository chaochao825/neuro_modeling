from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.merge_orbit_feature_shards import merge_feature_shards
from src.data.orbit_streaming import FEATURE_MANIFEST_COLUMNS, OrbitFeatureStore


def _shard(root: Path, *, user: str, split: str = "validation") -> Path:
    root.mkdir()
    relative = Path(split) / user / "cup" / "clean" / f"{user}-video.npz"
    path = root / relative
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        embeddings=np.ones((3, 4), dtype=np.float32),
        frame_indices=np.arange(3),
        object_present=np.ones(3, dtype=np.bool_),
    )
    pd.DataFrame(
        [
            {
                "split": split,
                "user_id": user,
                "object_name": "cup",
                "video_type": "clean",
                "video_id": f"{user}-video",
                "feature_path": relative.as_posix(),
                "n_frames": 3,
                "feature_dim": 4,
                "source_fingerprint": user,
            }
        ],
        columns=FEATURE_MANIFEST_COLUMNS,
    ).to_csv(root / "feature_manifest.csv", index=False)
    (root / f"failures_{split}.csv").write_text("\n", encoding="utf-8")
    return root


def test_merge_shards_builds_complete_validated_store(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps({"train": ["train"], "validation": ["u0", "u1"], "test": ["test"]}),
        encoding="utf-8",
    )
    output = merge_feature_shards(
        (_shard(tmp_path / "s0", user="u0"), _shard(tmp_path / "s1", user="u1")),
        output_root=tmp_path / "merged",
        split="validation",
        official_splits_path=split_path,
    )
    store = OrbitFeatureStore(
        output,
        split="validation",
        official_splits_path=split_path,
        require_complete_split=True,
    )
    assert store.users == ("u0", "u1")
    assert len(store.frame) == 2


def test_merge_shards_rejects_duplicate_video(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps({"train": ["train"], "validation": ["u0"], "test": ["test"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate videos"):
        merge_feature_shards(
            (_shard(tmp_path / "s0", user="u0"), _shard(tmp_path / "s1", user="u0")),
            output_root=tmp_path / "merged",
            split="validation",
            official_splits_path=split_path,
        )
