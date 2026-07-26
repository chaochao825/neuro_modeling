from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.exp36_change_aware_prefix import (
    CONDITIONS,
    PROTOCOL_VERSION,
    build_hidden_switch_stream,
    run_development_seed,
    run_seed,
    validate_config,
    validate_preregistration,
)
from src.data.orbit_streaming import (
    FEATURE_MANIFEST_COLUMNS,
    OrbitEpisodeSamplingConfig,
    OrbitFeatureStore,
)
from src.models.streaming_fewshot_actuators import PersonalizedStreamingActuators


def _write_store(
    root: Path,
    *,
    split: str,
    user: str,
    seed: int,
) -> None:
    root.mkdir()
    rows: list[dict[str, object]] = []
    for label, object_name in enumerate(("cup", "keys", "book", "bottle")):
        for video_type in ("clean", "clutter"):
            video_id = f"{user}-{object_name}-{video_type}"
            relative = Path(split) / f"{video_id}.npz"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed + 10 * label + (video_type == "clutter"))
            center = np.zeros(8, dtype=np.float64)
            center[label] = 3.0
            embeddings = center + rng.normal(0.0, 0.1, size=(16, 8))
            np.savez_compressed(
                path,
                embeddings=embeddings,
                frame_indices=np.arange(16, dtype=np.int64),
                object_present=np.ones(16, dtype=np.bool_),
            )
            rows.append(
                {
                    "split": split,
                    "user_id": user,
                    "object_name": object_name,
                    "video_type": video_type,
                    "video_id": video_id,
                    "feature_path": relative.as_posix(),
                    "n_frames": 16,
                    "feature_dim": 8,
                    "source_fingerprint": video_id,
                }
            )
    pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS).to_csv(
        root / "feature_manifest.csv", index=False
    )


def _config(tmp_path: Path) -> dict[str, object]:
    development_root = tmp_path / "development"
    external_root = tmp_path / "external"
    _write_store(development_root, split="validation", user="dev", seed=1)
    _write_store(external_root, split="external", user="ext", seed=2)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {"train": ["train"], "validation": ["dev"], "test": ["test"]}
        ),
        encoding="utf-8",
    )
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps({"split": "external", "collectors": ["ext"]}),
        encoding="utf-8",
    )
    return {
        "profile": "smoke",
        "protocol_version": PROTOCOL_VERSION,
        "evidence_provenance": "development_scale_gate",
        "claim_upgrade_allowed": False,
        "seeds": [3],
        "used_development_labels_for_selection": True,
        "used_external_labels_for_fit": False,
        "used_future_frames": False,
        "used_autograd": False,
        "used_bptt": False,
        "development_feature_root": str(development_root),
        "development_split": "validation",
        "development_splits_path": str(splits),
        "external_feature_root": str(external_root),
        "external_split": "external",
        "external_cohort_path": str(cohort),
        "external_collectors": ["ext"],
        "require_complete_development_split": True,
        "require_complete_external_cohort": True,
        "cache_features_in_memory": True,
        "n_development_tasks_per_user": 1,
        "n_external_tasks_per_collector": 1,
        "retention_grid": [0.0, 0.9, 1.0],
        "window_grid": [2, 4, 8],
        "detector_grid": {
            "fast_retention": [0.0],
            "jsd_threshold": [0.01, 0.2],
            "patience": [1],
            "min_run_frames": [2],
        },
        "selection_constraints": {
            "max_natural_accuracy_loss": 1.0,
            "max_false_alarms_per_1000": 1000.0,
        },
        "stream": {
            "segments_per_stream": 4,
            "segment_frames": 4,
            "post_switch_window": 3,
            "detection_tolerance": 4,
        },
        "sampling": {
            "support_stride": 2,
            "max_support_frames_per_video": 6,
            "query_frames_per_video": 8,
            "min_query_frames_per_video": 4,
            "max_frames_per_video": 16,
            "support_video_limit": None,
        },
        "actuators": {},
        "analysis": {"bootstrap_samples": 1000},
    }


def test_exp36_runs_complete_paired_panels(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = run_seed(config, seed=3, results_root=tmp_path / "results")
    raw = pd.read_csv(path / "external_task_metrics.csv")
    assert set(raw["condition"]) == set(CONDITIONS)
    assert set(raw["panel"]) == {"natural", "hidden_switch"}
    assert len(raw) == 2 * len(CONDITIONS)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["complete"] is True
    assert summary["statistical_unit"] == "collector"
    selected = json.loads(
        (path / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    assert selected["used_external_labels"] is False
    assert (path / "development_selection_audit.csv").is_file()


def test_exp36_development_gate_never_opens_external_store(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["profile"] = "development"
    config["external_feature_root"] = str(tmp_path / "deliberately-missing")
    path = run_development_seed(
        config, seed=3, results_root=tmp_path / "development-results"
    )
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert summary["claim_upgrade_allowed"] is False
    assert summary["selected_hyperparameters"]["used_external_data"] is False


def test_hidden_stream_hides_source_boundaries_from_actuators(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = OrbitFeatureStore(
        config["external_feature_root"],
        split="external",
        external_cohort_path=config["external_cohort_path"],
        require_complete_split=True,
    )
    episode = store.sample_episode(
        "ext",
        seed=2,
        task_index=0,
        config=OrbitEpisodeSamplingConfig(**config["sampling"]),
    )
    fitted = PersonalizedStreamingActuators.fit(
        episode.support, n_classes=episode.n_classes
    )
    stream = build_hidden_switch_stream(
        episode,
        fitted,
        seed=4,
        segments_per_stream=4,
        segment_frames=4,
    )
    assert np.unique(stream.stream_ids).size == 1
    assert np.flatnonzero(stream.switch_flags).tolist() == [4, 8, 12]
    assert len(stream.source_video_ids) == 4
    assert len(set(stream.labels[index] for index in (0, 4, 8, 12))) == 4


def test_exp36_rejects_external_label_fitting(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["used_external_labels_for_fit"] = True
    with pytest.raises(ValueError, match="used_external_labels_for_fit"):
        validate_config(config)


def test_exp36_preregistration_hashes_match_frozen_files() -> None:
    project = Path(__file__).resolve().parents[1]
    receipt = validate_preregistration(
        project / "configs/prospective/exp36_change_aware_prefix.json"
    )
    assert receipt["protocol_version"] == PROTOCOL_VERSION
    assert receipt["external_outcomes_inspected"] is False
