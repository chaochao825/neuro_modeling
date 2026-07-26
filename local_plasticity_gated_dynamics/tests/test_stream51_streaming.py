from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_stream51_cohort import build_cohort
from src.data.stream51_streaming import (
    Stream51FeatureStore,
    assign_stream51_video_splits,
    fit_stream51_vmf,
    make_stream51_hidden_streams,
    make_stream51_natural_streams,
    parse_stream51_ordering_line,
    read_stream51_ordering,
)


def _ordering_lines(*, classes: int = 2, videos: int = 9, frames: int = 4) -> list[str]:
    lines: list[str] = []
    for class_id in range(classes):
        for video_id in range(videos):
            for frame_id in range(frames):
                lines.append(
                    f"train/{class_id + 1}-class{class_id}/"
                    f"{video_id:03d}_{video_id:03d}_{frame_id:03d}.jpg {class_id}"
                )
    return lines


def _cohort_and_cache(tmp_path: Path) -> tuple[Path, Path]:
    ordering = tmp_path / "train.txt"
    ordering.write_text("\n".join(_ordering_lines()) + "\n", encoding="utf-8")
    cohort_path = tmp_path / "cohort.json"
    cohort = build_cohort(
        ordering,
        cohort_path,
        split_salt="unit-test",
        source_repo_commit="a" * 40,
    )
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    rows: list[dict[str, object]] = []
    for video in cohort["videos"]:
        video_key = str(video["video_key"])
        class_id = int(video["class_id"])
        base = np.zeros(4)
        base[class_id] = 1.0
        features = np.stack([base + 0.01 * index for index in range(4)])
        relative = Path(str(video["split"])) / f"{video_key}.npz"
        (feature_root / relative).parent.mkdir(exist_ok=True)
        np.savez_compressed(feature_root / relative, features=features)
        rows.append(
            {
                "video_key": video_key,
                "class_id": class_id,
                "split": video["split"],
                "feature_path": relative.as_posix(),
                "n_frames": 4,
                "feature_dim": 4,
            }
        )
    pd.DataFrame(rows).to_csv(feature_root / "feature_manifest.csv", index=False)
    return cohort_path, feature_root


def test_parse_official_ordering_schema() -> None:
    record = parse_stream51_ordering_line(
        "train/24-whale/043_036_000.jpg 23"
    )
    assert record.class_id == 23
    assert record.class_name == "whale"
    assert record.clip_id == 43
    assert record.video_id == 36
    assert record.frame_id == 0
    assert record.video_key == "c23_clip043_video036"

    spaced = parse_stream51_ordering_line(
        "train/37-irish terrier/019_026_000.jpg 36"
    )
    assert spaced.class_name == "irish terrier"


@pytest.mark.parametrize(
    "line",
    [
        "train/24-whale/043_036_000.jpg",
        "../24-whale/043_036_000.jpg 23",
        "train/24-whale/043_036_000.jpg 22",
        "train/24-whale/not_a_frame.jpg 23",
    ],
)
def test_bad_ordering_rows_fail_closed(line: str) -> None:
    with pytest.raises(ValueError):
        parse_stream51_ordering_line(line)


def test_outcome_blind_split_keeps_whole_videos_and_each_class(tmp_path: Path) -> None:
    ordering = tmp_path / "train.txt"
    ordering.write_text("\n".join(_ordering_lines()) + "\n", encoding="utf-8")
    records = read_stream51_ordering(ordering)
    first = assign_stream51_video_splits(records, salt="fixed")
    second = assign_stream51_video_splits(records, salt="fixed")
    assert first == second
    assert first["split_counts"] == {"development": 6, "external": 6, "support": 6}
    videos = pd.DataFrame(first["videos"])
    assert not videos["video_key"].duplicated().any()
    per_class = videos.groupby(["class_id", "split"]).size().unstack()
    assert (per_class == 3).all().all()


def test_cohort_receipt_records_source_hash_without_outcome_access(tmp_path: Path) -> None:
    ordering = tmp_path / "train.txt"
    ordering.write_text("\n".join(_ordering_lines()) + "\n", encoding="utf-8")
    output = tmp_path / "cohort.json"
    payload = build_cohort(
        ordering,
        output,
        split_salt="fixed",
        source_repo_commit="b" * 40,
    )
    reloaded = json.loads(output.read_text(encoding="utf-8"))
    assert reloaded == payload
    assert payload["outcome_fields_inspected"] is False
    assert len(str(payload["source_ordering_sha256"])) == 64


def test_feature_store_and_streams_are_split_safe(tmp_path: Path) -> None:
    cohort_path, feature_root = _cohort_and_cache(tmp_path)
    store = Stream51FeatureStore(
        feature_root,
        cohort_path=cohort_path,
        required_splits=("support", "development"),
    )
    assert store.n_classes == 2
    assert len(store.video_keys("support")) == 6
    model = fit_stream51_vmf(store)
    natural = make_stream51_natural_streams(
        store,
        model,
        split="development",
        temperature=1.0,
        max_frames=3,
    )
    hidden = make_stream51_hidden_streams(
        store,
        model,
        split="development",
        seed=7,
        temperature=1.0,
        segment_frames=3,
        videos_per_stream=3,
        video_keys=store.video_keys("development")[:6],
    )
    assert len(natural) == 6
    assert len(hidden) == 2
    assert all(stream.panel == "natural" for stream in natural)
    assert all(stream.panel == "hidden_switch" for stream in hidden)
    for stream in hidden:
        switch_indices = np.flatnonzero(stream.switch_flags)
        assert switch_indices.size == 2
        assert np.all(stream.labels[switch_indices] != stream.labels[switch_indices - 1])
    all_sources = np.concatenate([stream.source_video_ids for stream in hidden])
    assert len(set(all_sources.tolist())) == 6
    with pytest.raises(KeyError):
        store.video_keys("external")


def test_feature_store_rejects_incomplete_required_split(tmp_path: Path) -> None:
    cohort_path, feature_root = _cohort_and_cache(tmp_path)
    manifest_path = feature_root / "feature_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    victim = manifest[manifest["split"] == "development"].index[0]
    manifest = manifest.drop(victim)
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="does not cover"):
        Stream51FeatureStore(
            feature_root,
            cohort_path=cohort_path,
            required_splits=("support", "development"),
        )
