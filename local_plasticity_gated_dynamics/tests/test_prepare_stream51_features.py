from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from scripts.prepare_stream51_features import (
    _embed_video,
    crop_registered_bbox,
    merge_failure_history,
    select_registered_frames,
    validate_bbox_schema,
)


def _metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for index, split in enumerate(("support", "development", "external")):
        video_key = f"c00_clip00{index}_video00{index}"
        cohort_rows.append(
            {
                "video_key": video_key,
                "class_id": 0,
                "split": split,
                "n_available_frames": 6,
            }
        )
        for frame_id in range(6):
            metadata_rows.append(
                {
                    "video_key": video_key,
                    "class_id": 0,
                    "frame_id": frame_id,
                    "source_path": f"train/1-class/{index:03d}_{index:03d}_{frame_id:03d}.jpg",
                }
            )
    return pd.DataFrame(metadata_rows), pd.DataFrame(cohort_rows)


def test_registered_frame_selection_is_uniform_and_stage_locked() -> None:
    metadata, cohort = _metadata()
    selected = select_registered_frames(
        metadata,
        cohort,
        splits=("support", "development"),
        frames_per_split={"support": 3, "development": 4, "external": 5},
    )
    assert set(selected) == {
        "c00_clip000_video000",
        "c00_clip001_video001",
    }
    assert selected["c00_clip000_video000"]["frame_id"].tolist() == [0, 2, 5]
    assert len(selected["c00_clip001_video001"]) == 4
    assert not any("video002" in key for key in selected)


def test_registered_bbox_crop_matches_official_coordinate_order() -> None:
    image = Image.new("RGB", (100, 80), color="white")
    row = {
        "bbox_xmax": 80,
        "bbox_xmin": 20,
        "bbox_ymax": 60,
        "bbox_ymin": 10,
    }
    cropped = crop_registered_bbox(image, row, ratio=1.0)
    assert cropped.size == (60, 50)
    padded = crop_registered_bbox(image, row, ratio=1.1)
    assert padded.size[0] > cropped.size[0]
    assert padded.size[1] > cropped.size[1]


def test_official_tracker_overflow_is_clipped_not_rejected() -> None:
    shape, bounds = validate_bbox_schema([1920, 816], [1921, -4, 820, 0])
    assert shape.tolist() == [1920, 816]
    assert bounds.tolist() == [1921, -4, 820, 0]
    image = Image.new("RGB", (1920, 816), color="white")
    cropped = crop_registered_bbox(
        image,
        {
            "bbox_xmax": 1921,
            "bbox_xmin": -4,
            "bbox_ymax": 820,
            "bbox_ymin": 0,
        },
        ratio=1.0,
    )
    assert cropped.size == (1920, 816)


def test_bbox_empty_after_clipping_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty after"):
        validate_bbox_schema([100, 80], [120, 101, 50, 10])


def test_empty_bbox_and_bad_stage_fail_closed() -> None:
    image = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError):
        crop_registered_bbox(
            image,
            {
                "bbox_xmax": 5,
                "bbox_xmin": 5,
                "bbox_ymax": 4,
                "bbox_ymin": 4,
            },
            ratio=1.0,
        )
    metadata, cohort = _metadata()
    with pytest.raises(ValueError):
        select_registered_frames(
            metadata,
            cohort,
            splits=("forbidden",),
            frames_per_split={"support": 3, "development": 3, "external": 3},
        )


def test_feature_failure_history_is_append_only() -> None:
    columns = (
        "attempt_id",
        "attempted_at",
        "video_key",
        "class_id",
        "split",
        "error_type",
        "error",
    )
    prior = pd.DataFrame(
        [["a1", "t1", "video1", 0, "support", "OSError", "old"]],
        columns=columns,
    )
    merged = merge_failure_history(
        prior,
        [
            {
                "attempt_id": "a2",
                "attempted_at": "t2",
                "video_key": "video2",
                "class_id": 1,
                "split": "development",
                "error_type": "ValueError",
                "error": "new",
            }
        ],
    )
    assert merged["attempt_id"].tolist() == ["a1", "a2"]
    assert merged["error"].tolist() == ["old", "new"]


def test_parallel_decode_preserves_registered_frame_order(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    rows = []
    with zipfile.ZipFile(archive_path, "w") as archive:
        for frame_id, intensity in enumerate((20, 80, 160, 240)):
            buffer = BytesIO()
            Image.new("RGB", (8, 6), color=(intensity, 0, 0)).save(
                buffer, format="PNG"
            )
            member = f"frame_{frame_id}.png"
            archive.writestr(member, buffer.getvalue())
            rows.append(
                {
                    "archive_member": member,
                    "frame_id": frame_id,
                    "bbox_xmax": 8,
                    "bbox_xmin": 0,
                    "bbox_ymax": 6,
                    "bbox_ymin": 0,
                }
            )
    frame = pd.DataFrame(rows)

    def transform(image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(
            2, 0, 1
        )

    model = torch.nn.Sequential(torch.nn.Flatten())
    with zipfile.ZipFile(archive_path) as archive:
        serial = _embed_video(
            frame,
            archive=archive,
            model=model,
            transform=transform,
            device="cpu",
            batch_size=4,
            decode_workers=1,
            bbox_ratio=1.0,
        )
    with zipfile.ZipFile(archive_path) as archive:
        parallel = _embed_video(
            frame,
            archive=archive,
            model=model,
            transform=transform,
            device="cpu",
            batch_size=4,
            decode_workers=4,
            bbox_ratio=1.0,
        )
    np.testing.assert_array_equal(parallel, serial)
    assert serial[:, 0].tolist() == [20.0, 80.0, 160.0, 240.0]
