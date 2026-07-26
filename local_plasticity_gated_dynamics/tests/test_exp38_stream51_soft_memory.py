from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.common import load_json_config
from experiments.exp38_stream51_soft_memory import (
    _make_streams,
    _oracle_reset_prediction,
    controller_candidates,
    evaluate_qualification,
    fit_control_standardizer,
    partition_development_videos,
    qualification_parameters_for_seed,
    select_controller,
    select_stationary_baselines,
    select_temperature,
    validate_config,
    validate_qualification_receipt,
)
from scripts.build_stream51_cohort import build_cohort
from src.data.stream51_streaming import Stream51FeatureStore, fit_stream51_vmf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_store(tmp_path: Path) -> Stream51FeatureStore:
    ordering = tmp_path / "train.txt"
    lines: list[str] = []
    for class_id in range(2):
        for video_id in range(9):
            for frame_id in range(8):
                lines.append(
                    f"train/{class_id + 1}-class{class_id}/"
                    f"{video_id:03d}_{video_id:03d}_{frame_id:03d}.jpg {class_id}"
                )
    ordering.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cohort_path = tmp_path / "cohort.json"
    cohort = build_cohort(
        ordering,
        cohort_path,
        split_salt="exp38-test",
        source_repo_commit="c" * 40,
    )
    root = tmp_path / "features"
    root.mkdir()
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(38)
    for video in cohort["videos"]:
        class_id = int(video["class_id"])
        center = np.zeros(6)
        center[class_id] = 1.0
        features = center + rng.normal(0.0, 0.35, size=(8, 6))
        relative = Path(str(video["split"])) / f"{video['video_key']}.npz"
        (root / relative).parent.mkdir(exist_ok=True)
        np.savez_compressed(root / relative, features=features)
        rows.append(
            {
                "video_key": video["video_key"],
                "class_id": class_id,
                "split": video["split"],
                "feature_path": relative.as_posix(),
                "n_frames": 8,
                "feature_dim": 6,
            }
        )
    pd.DataFrame(rows).to_csv(root / "feature_manifest.csv", index=False)
    return Stream51FeatureStore(
        root,
        cohort_path=cohort_path,
        required_splits=("support", "development"),
        cache_in_memory=True,
    )


def _small_config() -> dict[str, object]:
    config = load_json_config(
        PROJECT_ROOT / "configs/prospective/exp38_stream51_soft_memory.json"
    )
    config = deepcopy(config)
    config["temperature_grid"] = [1.0, 5.0]
    config["retention_grid"] = [0.0, 0.8, 1.0]
    config["window_grid"] = [2, 4]
    config["controller_grid"] = {
        "retention_floor": [0.0],
        "retention_ceiling": [0.95],
        "gain": [1.0],
        "threshold": [-0.5, 0.0],
        "feature_templates": [[1.0, 0.0, 1.0]],
        "fast_retention": 0.5,
        "slow_retention": 0.98,
        "evidence_weight_floor": 1.0,
    }
    config["task"] = {
        "natural_max_frames": 8,
        "segment_frames": 4,
        "videos_per_stream": 2,
        "post_switch_window": 2,
        "detection_tolerance": 2,
        "refractory_frames": 2,
        "min_run_frames": 2,
    }
    config["selection_constraints"] = {
        "max_natural_accuracy_loss": 1.0,
        "min_switch_recall": 0.0,
        "max_false_alarms_per_1000": 1000.0,
    }
    return config


def test_frozen_config_and_candidate_grid_are_valid() -> None:
    config = load_json_config(
        PROJECT_ROOT / "configs/prospective/exp38_stream51_soft_memory.json"
    )
    validate_config(config)
    assert len(controller_candidates(config)) == 960
    assert all(candidate.control_dimension == 3 for candidate in controller_candidates(config))


def test_development_fit_and_qualification_are_disjoint_by_video(tmp_path: Path) -> None:
    store = _tiny_store(tmp_path)
    fit, qualification = partition_development_videos(
        store, salt="fixed", fit_fraction=0.5
    )
    assert set(fit).isdisjoint(qualification)
    assert set(fit) | set(qualification) == set(store.video_keys("development"))
    assert {store.class_id(key) for key in fit} == {0, 1}
    assert {store.class_id(key) for key in qualification} == {0, 1}


def test_minimal_development_pipeline_retains_all_diagnostics(tmp_path: Path) -> None:
    store = _tiny_store(tmp_path)
    config = _small_config()
    fit_keys, qualification_keys = partition_development_videos(
        store, salt="fixed", fit_fraction=0.5
    )
    model = fit_stream51_vmf(store)
    temperature, temperature_audit = select_temperature(
        store, model, video_keys=fit_keys, grid=config["temperature_grid"]
    )
    fit_natural, fit_hidden = _make_streams(
        store,
        model,
        split="development",
        video_keys=fit_keys,
        temperature=temperature,
        seed=38,
        task=config["task"],
    )
    retention, window, stationary_audit = select_stationary_baselines(
        fit_natural, fit_hidden, config=config
    )
    standardizer = fit_control_standardizer(
        (*fit_natural, *fit_hidden), config=config
    )
    controller, eligible, controller_audit = select_controller(
        fit_natural,
        fit_hidden,
        standardizer=standardizer,
        reference_natural_accuracy=0.0,
        config=config,
    )
    qualification_natural, qualification_hidden = _make_streams(
        store,
        model,
        split="development",
        video_keys=qualification_keys,
        temperature=temperature,
        seed=38,
        task=config["task"],
    )
    summary, video_rows = evaluate_qualification(
        qualification_natural,
        qualification_hidden,
        retention=retention,
        window_frames=window,
        controller=controller,
        controller_fit_eligible=eligible,
        standardizer=standardizer,
        config=config,
    )
    assert temperature_audit["selected"].sum() == 1
    assert stationary_audit["selected"].sum() == 2
    assert controller_audit["selected"].sum() == 1
    assert set(video_rows["panel"]) == {"natural", "hidden_switch"}
    assert isinstance(summary["passed"], bool)
    assert "stable_accumulation_gate" in summary
    assert "reachability_gate" in summary


def test_external_stage_refuses_failed_qualification_receipt(tmp_path: Path) -> None:
    config = load_json_config(
        PROJECT_ROOT / "configs/prospective/exp38_stream51_soft_memory.json"
    )
    receipt = tmp_path / "qualification.json"
    receipt.write_text(
        json.dumps(
            {
                "protocol_version": "exp38_stream51_soft_memory_v1",
                "all_registered_seeds_passed": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="locked"):
        validate_qualification_receipt(receipt, config=config)


def test_external_parameters_are_loaded_from_frozen_seed_result() -> None:
    selected = {
        "temperature": 5.0,
        "retention": 0.85,
        "window_frames": 8,
        "controller": {
            "retention_floor": 0.25,
            "retention_ceiling": 0.97,
            "bias": -1.0,
            "surprise_weight": 1.0,
            "entropy_weight": 0.0,
            "disagreement_weight": 1.0,
            "fast_retention": 0.5,
            "slow_retention": 0.98,
            "evidence_weight_floor": 1.0,
            "epsilon": 1e-12,
        },
        "controller_standardizer": {
            "mean": [0.1, 0.2, 0.3],
            "scale": [1.0, 2.0, 3.0],
        },
        "controller_fit_eligible": True,
    }
    receipt = {
        "seed_results": [
            {
                "seed": 13800,
                "passed": True,
                "selected_hyperparameters": selected,
            }
        ]
    }
    loaded = qualification_parameters_for_seed(receipt, seed=13800)
    assert loaded == selected
    with pytest.raises(ValueError, match="one seed"):
        qualification_parameters_for_seed(receipt, seed=13801)


def test_oracle_is_cumulative_with_exact_switch_reset(tmp_path: Path) -> None:
    store = _tiny_store(tmp_path)
    config = _small_config()
    model = fit_stream51_vmf(store)
    keys = store.video_keys("development")[:4]
    _, hidden = _make_streams(
        store,
        model,
        split="development",
        video_keys=keys,
        temperature=1.0,
        seed=38,
        task=config["task"],
    )
    stream = hidden[0]
    prediction = _oracle_reset_prediction(stream)
    switch_index = int(np.flatnonzero(stream.switch_flags)[0])
    assert prediction[switch_index] == np.argmax(stream.evidence[switch_index])
