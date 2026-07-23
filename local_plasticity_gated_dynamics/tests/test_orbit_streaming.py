from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.orbit_streaming import (
    FEATURE_MANIFEST_COLUMNS,
    OrbitEpisodeSamplingConfig,
    OrbitFeatureStore,
    load_official_orbit_splits,
    validate_user_disjoint_stores,
)


def _make_store(tmp_path: Path) -> tuple[Path, Path]:
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["train_user"],
                "validation": ["validation_user"],
                "test": ["test_user"],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "features"
    root.mkdir()
    rows: list[dict[str, object]] = []
    for split, user in (
        ("train", "train_user"),
        ("validation", "validation_user"),
        ("test", "test_user"),
    ):
        for class_index, object_name in enumerate(("cup", "keys")):
            for video_type in ("clean", "clutter"):
                video_id = f"{user}-{object_name}-{video_type}"
                relative = Path(split) / f"{video_id}.npz"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                basis = np.zeros(4, dtype=np.float32)
                basis[class_index] = 1.0
                embeddings = np.stack(
                    [basis + 0.01 * index for index in range(8)], axis=0
                )
                np.savez_compressed(
                    path,
                    embeddings=embeddings,
                    frame_indices=np.arange(8, dtype=np.int64),
                    object_present=np.ones(8, dtype=np.bool_),
                )
                rows.append(
                    {
                        "split": split,
                        "user_id": user,
                        "object_name": object_name,
                        "video_type": video_type,
                        "video_id": video_id,
                        "feature_path": relative.as_posix(),
                        "n_frames": 8,
                        "feature_dim": 4,
                        "source_fingerprint": video_id,
                    }
                )
    pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS).to_csv(
        root / "feature_manifest.csv", index=False
    )
    return root, split_path


def test_official_split_loader_rejects_user_leakage(tmp_path: Path) -> None:
    root, split_path = _make_store(tmp_path)
    splits = load_official_orbit_splits(split_path)
    assert tuple(map(len, splits.values())) == (1, 1, 1)

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["test"] = payload["train"]
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        load_official_orbit_splits(split_path)

    # The feature cache remains intact after the fail-closed check.
    assert (root / "feature_manifest.csv").is_file()


def test_episode_is_reproducible_chronological_and_label_free(tmp_path: Path) -> None:
    root, split_path = _make_store(tmp_path)
    store = OrbitFeatureStore(
        root,
        split="validation",
        official_splits_path=split_path,
        require_complete_split=True,
    )
    config = OrbitEpisodeSamplingConfig(
        support_stride=2,
        max_support_frames_per_video=3,
        query_frames_per_video=5,
        min_query_frames_per_video=2,
        max_frames_per_video=8,
    )
    first = store.sample_episode("validation_user", seed=7, task_index=3, config=config)
    second = store.sample_episode(
        "validation_user", seed=7, task_index=3, config=config
    )
    assert first.fingerprint == second.fingerprint
    assert np.array_equal(first.query_frame_indices, second.query_frame_indices)
    assert first.n_classes == 2
    observation = first.query_observation
    assert not hasattr(observation, "labels")
    assert not hasattr(observation, "query_labels")
    for video_id in np.unique(observation.video_ids):
        indices = observation.frame_indices[observation.video_ids == video_id]
        assert np.all(np.diff(indices) >= 0)
    assert not observation.embeddings.flags.writeable


def test_store_pair_validation_uses_user_as_independent_unit(tmp_path: Path) -> None:
    root, split_path = _make_store(tmp_path)
    train = OrbitFeatureStore(root, split="train", official_splits_path=split_path)
    validation = OrbitFeatureStore(
        root, split="validation", official_splits_path=split_path
    )
    validate_user_disjoint_stores((train, validation))
    with pytest.raises(ValueError, match="share users"):
        validate_user_disjoint_stores((train, train))
