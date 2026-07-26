from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from scripts.prepare_orbit_features import (
    _annotation_mask,
    _protocol_object_present_mask,
    discover_orbit_videos,
    file_md5,
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


def test_archive_md5_is_streamed_and_validated(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"frozen external bytes")
    assert file_md5(archive, chunk_bytes=3) == "197b668f953e11506fa1fbecb6f6bcad"
    with pytest.raises(ValueError, match="positive"):
        file_md5(archive, chunk_bytes=0)


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


def test_external_video_discovery_uses_dataset_root(tmp_path: Path) -> None:
    video = tmp_path / "Dataset" / "P1" / "keys" / "clutter" / "video0"
    video.mkdir(parents=True)
    assert discover_orbit_videos(
        tmp_path, split="external", allowed_users=["P1"]
    ) == [("P1", "keys", "clutter", video)]


def test_external_annotations_have_no_split_directory(tmp_path: Path) -> None:
    frames = [tmp_path / "video-00001.jpeg", tmp_path / "video-00002.jpeg"]
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "video.json").write_text(
        '{"video-00001.jpeg":{"object_not_present_issue":false},'
        '"video-00002.jpeg":{"object_not_present_issue":true}}',
        encoding="utf-8",
    )
    mask = _annotation_mask(
        frames,
        annotations_root=tmp_path / "annotations",
        split="external",
        video_id="video",
    )
    assert np.array_equal(mask, np.asarray([True, False]))


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
