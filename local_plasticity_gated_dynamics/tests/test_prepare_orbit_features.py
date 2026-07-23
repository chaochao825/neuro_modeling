from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from scripts.prepare_orbit_features import (
    _protocol_object_present_mask,
    discover_orbit_videos,
    parse_frame_index,
    parse_user_ids,
)


def test_frame_index_parser_is_strict() -> None:
    assert parse_frame_index("video-name-00042.jpg") == 42
    assert parse_frame_index(Path("nested/video-3.JPEG")) == 3
    with pytest.raises(ValueError, match="cannot parse"):
        parse_frame_index("frame.jpg")


def test_user_shard_parser_is_explicit_and_unique() -> None:
    assert parse_user_ids("u2,u0") == ("u2", "u0")
    assert parse_user_ids(None) is None
    with pytest.raises(ValueError, match="unique"):
        parse_user_ids("u0,u0")


def test_video_discovery_enforces_user_boundary(tmp_path: Path) -> None:
    video = tmp_path / "validation" / "u0" / "keys" / "clean" / "video0"
    video.mkdir(parents=True)
    assert discover_orbit_videos(
        tmp_path, split="validation", allowed_users=["u0"]
    ) == [("u0", "keys", "clean", video)]
    with pytest.raises(ValueError, match="outside official split"):
        discover_orbit_videos(
            tmp_path, split="validation", allowed_users=["different-user"]
        )


def test_clean_support_never_reads_extra_frame_annotations(tmp_path: Path) -> None:
    frames = [tmp_path / "video-00001.jpg", tmp_path / "video-00002.jpg"]
    mask = _protocol_object_present_mask(
        frames,
        video_type="clean",
        annotations_root=tmp_path / "missing-annotations",
        split="validation",
        video_id="clean-video",
    )
    assert np.array_equal(mask, np.ones(2, dtype=np.bool_))
    with pytest.raises(FileNotFoundError, match="annotations not found"):
        _protocol_object_present_mask(
            frames,
            video_type="clutter",
            annotations_root=tmp_path / "missing-annotations",
            split="validation",
            video_id="clutter-video",
        )
