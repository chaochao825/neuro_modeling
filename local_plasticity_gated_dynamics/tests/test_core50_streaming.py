from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.core50_streaming import (
    Core50FeatureStore,
    cosine_probability_evidence,
    fit_object_prototypes,
    prepare_core50_task,
)


def _store(tmp_path: Path) -> Core50FeatureStore:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(4)
    for session in ("s1", "s2"):
        for object_index in range(1, 6):
            object_id = f"o{object_index}"
            path = Path(session) / f"{object_id}.npz"
            (tmp_path / session).mkdir(exist_ok=True)
            features = rng.normal(size=(160, 8))
            features[:, object_index % 8] += 4.0
            np.savez_compressed(tmp_path / path, features=features)
            rows.append(
                {
                    "session_id": session,
                    "object_id": object_id,
                    "feature_path": path.as_posix(),
                    "n_frames": len(features),
                    "feature_dim": features.shape[1],
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "feature_manifest.csv", index=False)
    return Core50FeatureStore(
        tmp_path,
        expected_sessions=("s1", "s2"),
        expected_objects=tuple(f"o{i}" for i in range(1, 6)),
    )


def test_feature_store_fails_closed_on_missing_cell(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.load("s1", "o1").shape == (160, 8)
    manifest = pd.read_csv(tmp_path / "feature_manifest.csv").iloc[:-1]
    manifest.to_csv(tmp_path / "feature_manifest.csv", index=False)
    with pytest.raises(ValueError, match="incomplete"):
        Core50FeatureStore(
            tmp_path,
            expected_sessions=("s1", "s2"),
            expected_objects=tuple(f"o{i}" for i in range(1, 6)),
        )


def test_probability_evidence_is_normalized() -> None:
    features = np.eye(3)
    result = cosine_probability_evidence(features, np.eye(3), temperature=5.0)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(result, axis=1), np.arange(3))


def test_task_is_deterministic_and_hides_boundaries(tmp_path: Path) -> None:
    store = _store(tmp_path)
    prototypes = fit_object_prototypes(
        store,
        support_session="s1",
        object_ids=store.objects,
        stride=2,
        max_frames_per_object=40,
    )
    config = {
        "n_classes": 4,
        "segments_per_stream": 6,
        "segment_frames": 16,
        "natural_frames": 64,
    }
    first = prepare_core50_task(
        store,
        prototypes=prototypes,
        session_id="s2",
        seed=7,
        task_index=3,
        temperature=5.0,
        stream_config=config,
    )
    second = prepare_core50_task(
        store,
        prototypes=prototypes,
        session_id="s2",
        seed=7,
        task_index=3,
        temperature=5.0,
        stream_config=config,
    )
    assert np.array_equal(first[1].evidence, second[1].evidence)
    assert len(np.unique(first[0].stream_ids)) == 4
    assert len(np.unique(first[1].stream_ids)) == 1
    assert first[1].switch_flags.sum() == 5
    boundaries = np.flatnonzero(first[1].switch_flags)
    assert np.all(first[1].labels[boundaries] != first[1].labels[boundaries - 1])
