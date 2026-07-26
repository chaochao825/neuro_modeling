from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.prepare_core50_features import _sha256, discover_core50_cells, locate_dataset_root


def _raw_tree(root: Path) -> Path:
    dataset = root / "nested" / "core50_128x128"
    for session in ("s1", "s2"):
        for object_id in ("o1", "o2"):
            cell = dataset / session / object_id
            cell.mkdir(parents=True)
            for frame in (2, 10, 1):
                (cell / f"C_{frame}.png").write_bytes(b"image")
    return dataset


def test_sha256_is_streamed(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"registered bytes")
    assert _sha256(path, chunk_bytes=2) == hashlib.sha256(b"registered bytes").hexdigest()
    with pytest.raises(ValueError, match="positive"):
        _sha256(path, chunk_bytes=0)


def test_discovery_requires_exact_schema_and_natural_order(tmp_path: Path) -> None:
    dataset = _raw_tree(tmp_path)
    assert locate_dataset_root(tmp_path, expected_sessions=("s1", "s2")) == dataset
    root, cells = discover_core50_cells(
        tmp_path,
        expected_sessions=("s1", "s2"),
        expected_objects=("o1", "o2"),
    )
    assert root == dataset
    assert len(cells) == 4
    assert [path.name for path in cells[0][2]] == ["C_1.png", "C_2.png", "C_10.png"]


def test_discovery_rejects_missing_object(tmp_path: Path) -> None:
    dataset = _raw_tree(tmp_path)
    missing = dataset / "s2" / "o2"
    staged = tmp_path / "trash" / "missing-o2"
    staged.parent.mkdir()
    missing.rename(staged)
    with pytest.raises(ValueError, match="schema mismatch"):
        discover_core50_cells(
            tmp_path,
            expected_sessions=("s1", "s2"),
            expected_objects=("o1", "o2"),
        )
