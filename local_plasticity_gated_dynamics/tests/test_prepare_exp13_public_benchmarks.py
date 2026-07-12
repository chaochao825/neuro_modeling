from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from experiments.exp13_structured_reasoning import _validated_preparation_manifest
from scripts import prepare_exp13_public_benchmarks as prep
from src.data.maze_tasks import load_maze_tasks
from src.data.sudoku_tasks import load_sudoku_tasks


_SOLUTION = np.asarray(
    [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ],
    dtype=np.int8,
)


def _dat(puzzle: np.ndarray) -> bytes:
    rows = [" ".join(str(int(value)) for value in row) for row in puzzle]
    return ("fixture phone\n64x64 PNG\n" + "\n".join(rows) + "\n").encode()


def _png_for_path(path: str) -> bytes:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    floor = np.asarray([72, 140, 60], dtype=np.uint8)
    wall = np.asarray([70, 70, 75], dtype=np.uint8)
    grid = np.ones((5, 5), dtype=np.int8)
    point = (0, 0)
    grid[point] = 0
    for character in path:
        delta = prep._DIRECTIONS[character]
        point = point[0] + delta[0], point[1] + delta[1]
        grid[point] = 0
    edges = np.rint(np.linspace(2, 62, 6)).astype(int)
    for row in range(5):
        for column in range(5):
            color = floor if grid[row, column] == 0 else wall
            image[
                edges[row] : edges[row + 1],
                edges[column] : edges[column + 1],
            ] = color
    # Non-base pixels inside the documented skin palette localize the player.
    image[5:8, 5:8] = np.asarray([250, 205, 160], dtype=np.uint8)
    buffer = io.BytesIO()
    mpimg.imsave(buffer, image, format="png")
    return buffer.getvalue()


def _write_cache(root: Path, files: dict[str, bytes]) -> None:
    for relative_path, payload in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _no_network(url: str) -> bytes:
    raise AssertionError(f"test unexpectedly attempted network access: {url}")


def test_sudoku_fixture_build_is_unique_official_split_and_hash_bound(
    tmp_path: Path, monkeypatch
) -> None:
    duplicated_puzzle = _SOLUTION.copy()
    duplicated_puzzle[0, 0] = 0
    train_only_puzzle = _SOLUTION.copy()
    train_only_puzzle[0, 1] = 0
    files = {
        "README.rst": b"fixture CC-BY-4.0\n",
        "datasets/v2_train.desc": b"images/image1.jpg\nimages/image2.jpg\n",
        "datasets/v2_test.desc": b"images/image3.jpg\n",
        "images/image1.dat": _dat(duplicated_puzzle),
        "images/image2.dat": _dat(train_only_puzzle),
        "images/image3.dat": _dat(duplicated_puzzle),
    }
    source_hashes = {
        name: prep.sha256_bytes(payload)
        for name, payload in files.items()
        if name in prep.SUDOKU_SOURCES
    }
    monkeypatch.setattr(prep, "SUDOKU_SOURCES", source_hashes)
    monkeypatch.setattr(prep, "SUDOKU_EXPECTED_SPLIT_COUNTS", {"train": 2, "test": 1})
    cache = tmp_path / "cache"
    _write_cache(cache, files)

    dataset_path, manifest_path = prep.prepare_sudoku(
        tmp_path / "out", cache_root=cache, fetcher=_no_network
    )
    dataset = load_sudoku_tasks(dataset_path, split=None)
    assert len(dataset.for_split("train")) == 1
    assert len(dataset.for_split("test")) == 1
    assert all(
        task.metadata["source_version"] == prep.SUDOKU_REVISION
        for task in dataset.tasks
    )
    assert {task.content_group for task in dataset.for_split("train")}.isdisjoint(
        task.content_group for task in dataset.for_split("test")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_with_exclusions"
    assert manifest["summary"] == {
        "n_excluded": 1,
        "n_excluded_duplicate_content": 1,
        "n_excluded_invalid": 0,
        "n_expected": 3,
        "n_included": 2,
        "n_unique_puzzles": 2,
    }
    assert manifest["split"]["n_cross_official_split_groups"] == 1
    duplicate_receipts = [
        row
        for row in manifest["records"]
        if row.get("exclusion_type") == "duplicate_puzzle_content"
    ]
    assert len(duplicate_receipts) == 1
    assert duplicate_receipts[0]["official_source_split"] == "train"
    assert duplicate_receipts[0]["assigned_split"] == "test"
    assert manifest["output"]["sha256"] == prep.sha256_bytes(dataset_path.read_bytes())
    data_config = {
        "dataset_name": "wichtounet/sudoku_dataset-v2",
        "license": "CC-BY-4.0",
        "manifest_path": str(manifest_path),
        "manifest_sha256": prep.sha256_bytes(manifest_path.read_bytes()),
        "revision": prep.SUDOKU_REVISION,
        "test_split_role": "non_ood",
    }
    _, validated, _ = _validated_preparation_manifest(
        data_config, family="sudoku", dataset_path=dataset_path
    )
    assert validated["summary"]["n_included"] == 2
    manifest_bytes = manifest_path.read_bytes()
    tampered_manifest = json.loads(manifest_bytes)
    tampered_manifest["builder"]["script_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(tampered_manifest, sort_keys=True), encoding="utf-8"
    )
    tampered_config = {
        **data_config,
        "manifest_sha256": prep.sha256_bytes(manifest_path.read_bytes()),
    }
    with pytest.raises(ValueError, match="current public-data builder"):
        _validated_preparation_manifest(
            tampered_config, family="sudoku", dataset_path=dataset_path
        )
    manifest_path.write_bytes(manifest_bytes)
    dataset_path.write_bytes(dataset_path.read_bytes() + b"tampered\n")
    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        _validated_preparation_manifest(
            data_config, family="sudoku", dataset_path=dataset_path
        )


def test_sudoku_parser_and_solver_retain_ambiguous_status() -> None:
    puzzle = _SOLUTION.copy()
    puzzle[8, 8] = 0
    parsed, metadata = prep.parse_sudoku_dat(_dat(puzzle))
    solution, count = prep.solve_unique_sudoku(parsed)
    assert metadata["capture_device"] == "fixture phone"
    assert count == 1
    assert np.array_equal(solution, _SOLUTION)
    _, ambiguous_count = prep.solve_unique_sudoku(np.zeros((9, 9), dtype=np.int8))
    assert ambiguous_count == 2


def test_tiny_png_maze_build_validates_all_paths_and_retains_unreachable(
    tmp_path: Path, monkeypatch
) -> None:
    annotations = {
        "gen_maze_001": {"reachable": True, "accepted_shortest_paths": ["RRRR"]},
        "gen_maze_002": {"reachable": True, "accepted_shortest_paths": ["DDDD"]},
        "gen_maze_003": {"reachable": False, "accepted_shortest_paths": []},
    }
    files = {
        "README.md": b"---\nlicense: mit\n---\n",
        "maze_annotations.json": json.dumps(annotations, sort_keys=True).encode(),
        "gen_maze_001.png": _png_for_path("RRRR"),
        "gen_maze_002.png": _png_for_path("DDDD"),
        "gen_maze_003.png": _png_for_path("RRRR"),
    }
    source_hashes = {
        name: prep.sha256_bytes(payload)
        for name, payload in files.items()
        if name in prep.MAZE_SOURCES
    }
    monkeypatch.setattr(prep, "MAZE_SOURCES", source_hashes)
    monkeypatch.setattr(prep, "MAZE_EXPECTED_IMAGES", 3)
    monkeypatch.setattr(prep, "MAZE_EXPECTED_REACHABLE", 2)
    monkeypatch.setattr(prep, "MAZE_IMAGE_SHAPE", (64, 64, 3))
    monkeypatch.setattr(prep, "MAZE_BOUNDS", (2, 62))
    monkeypatch.setattr(prep, "MAZE_CANDIDATE_SIZES", (5,))
    cache = tmp_path / "cache"
    _write_cache(cache, files)

    dataset_path, manifest_path = prep.prepare_maze(
        tmp_path / "out", cache_root=cache, fetcher=_no_network
    )
    dataset = load_maze_tasks(dataset_path, split=None)
    assert len(dataset.for_split("train")) == 1
    assert len(dataset.for_split("test")) == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_with_exclusions"
    assert manifest["summary"] == {
        "n_expected_images": 3,
        "n_included_reachable": 2,
        "n_parse_or_download_failures": 0,
        "n_upstream_unreachable": 1,
    }
    included = [row for row in manifest["records"] if row["status"] == "included"]
    assert all(row["validated_accepted_path_count"] == 1 for row in included)
    assert manifest["split"]["official_source_role"] == "evaluation_only"
    assert manifest["split"]["test_split_role"] == "non_ood"


def test_existing_nonidentical_artifact_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_bytes(b"old\n")
    try:
        prep._write_new_or_identical(path, b"new\n")
    except FileExistsError:
        pass
    else:
        raise AssertionError("non-identical existing artifact should fail closed")
    assert path.read_bytes() == b"old\n"
