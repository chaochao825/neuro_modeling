from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_orbit_features import discover_orbit_videos, parse_frame_index


def test_frame_index_parser_is_strict() -> None:
    assert parse_frame_index("video-name-00042.jpg") == 42
    assert parse_frame_index(Path("nested/video-3.JPEG")) == 3
    with pytest.raises(ValueError, match="cannot parse"):
        parse_frame_index("frame.jpg")


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
