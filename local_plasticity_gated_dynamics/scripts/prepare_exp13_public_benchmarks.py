"""Prepare pinned public Sudoku and MazeBench tasks for exp13.

The builder is deliberately fail-closed.  Trusted small source files are
checked against published SHA-256 roots, every downloaded sample is bound to
the pinned upstream revision, exclusions remain in the manifest, and existing
non-identical outputs are never overwritten.  The generated tasks are a
hybrid proposal-selection benchmark; they are neither an HRM/CTM reproduction
nor evidence about neural dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import matplotlib.image as mpimg
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.maze_tasks import shortest_path_distance
from src.data.structured_protocol import public_projection_sha256


SCHEMA_VERSION = "exp13_public_benchmark_manifest_v1"
BUILDER_VERSION = "2026-07-12-v2"

SUDOKU_REPOSITORY = "https://github.com/wichtounet/sudoku_dataset"
SUDOKU_REVISION = "0db6de036b80b4e8e4574abe6e15026331bd5c2c"
SUDOKU_RAW_ROOT = (
    f"https://raw.githubusercontent.com/wichtounet/sudoku_dataset/{SUDOKU_REVISION}"
)
SUDOKU_SOURCES = {
    "README.rst": "7f21464cab12e2cfdff99e52527932c97bb0b8e4ca21d793bb7fb6fde6fe8dd4",
    "datasets/v2_train.desc": "815ddbbb6fa0ecf00dc826e59a672ce62515b0e9e5fafd323b0b0e4889ef0210",
    "datasets/v2_test.desc": "195c8771d5aa02b1126b9a268a6989ce56390881a30d6adbe63948c06e09aa92",
}
SUDOKU_EXPECTED_SPLIT_COUNTS = {"train": 160, "test": 40}
_SUDOKU_IMAGE = re.compile(r"images/image[0-9]+\.jpg")

MAZE_REPOSITORY = "https://huggingface.co/datasets/albertoRodriguez97/MazeBench"
MAZE_REVISION = "a71a2d1e0931c79f74cd91c5accd13f164d34c73"
MAZE_RESOLVE_ROOT = f"{MAZE_REPOSITORY}/resolve/{MAZE_REVISION}"
MAZE_SOURCES = {
    "README.md": "60b158c5f8f0817ee8b17eba282396b22de8e12c197e4f7c4b9e1140171c56d0",
    "maze_annotations.json": "07eff8a015c1d4d2ff4b2804c21325f3b8aad850d362196ff496a36df2977ec5",
}
MAZE_EXPECTED_IMAGES = 110
MAZE_EXPECTED_REACHABLE = 79
MAZE_IMAGE_SHAPE = (1024, 1024, 3)
MAZE_BOUNDS = (62, 962)
MAZE_CANDIDATE_SIZES = tuple(range(5, 21))

Fetcher = Callable[[str], bytes]


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n"
    ).encode("utf-8")


def _write_new_or_identical(path: Path, payload: bytes) -> None:
    """Write a deterministic artifact without destroying a prior artifact."""

    if path.is_file():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to overwrite non-identical artifact {path}; preserve it "
                "and choose a new output root"
            )
        return
    if path.exists():
        raise FileExistsError(f"artifact path is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _network_fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "exp13-auditable-builder/1"})
    with urlopen(request, timeout=120) as response:  # noqa: S310 - pinned HTTPS URLs
        return response.read()


def _source_bytes(
    *,
    cache_root: Path,
    relative_path: str,
    url: str,
    fetcher: Fetcher,
    expected_sha256: str | None = None,
) -> bytes:
    path = cache_root / Path(relative_path)
    payload = path.read_bytes() if path.is_file() else fetcher(url)
    digest = sha256_bytes(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"source SHA-256 mismatch for {relative_path}: {digest} != "
            f"{expected_sha256}"
        )
    if not path.is_file():
        _write_new_or_identical(path, payload)
    return payload


def _builder_receipt() -> Mapping[str, str]:
    script = Path(__file__).resolve()
    return {
        "version": BUILDER_VERSION,
        "script": script.name,
        "script_sha256": sha256_bytes(script.read_bytes()),
    }


def parse_sudoku_dat(payload: bytes) -> tuple[np.ndarray, Mapping[str, str]]:
    """Parse two metadata rows followed by exactly nine puzzle rows."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Sudoku .dat file is not UTF-8") from error
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 11:
        raise ValueError("Sudoku .dat must contain 2 metadata and 9 grid rows")
    rows: list[list[int]] = []
    for row_index, line in enumerate(lines[2:]):
        tokens = line.split()
        if len(tokens) != 9 or any(not token.isdigit() for token in tokens):
            raise ValueError(f"Sudoku row {row_index} must contain nine integers")
        values = [int(token) for token in tokens]
        if any(value < 0 or value > 9 for value in values):
            raise ValueError(f"Sudoku row {row_index} entries must be in [0, 9]")
        rows.append(values)
    return np.asarray(rows, dtype=np.int8), {
        "capture_device": lines[0],
        "image_format": lines[1],
    }


def solve_unique_sudoku(puzzle: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Return the deterministic first solution and a count capped at two."""

    board = np.asarray(puzzle, dtype=np.int8).copy()
    if board.shape != (9, 9) or np.any((board < 0) | (board > 9)):
        raise ValueError("Sudoku puzzle must be a 9x9 board with entries in [0, 9]")
    rows = [set() for _ in range(9)]
    columns = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for row in range(9):
        for column in range(9):
            value = int(board[row, column])
            if value == 0:
                continue
            box = 3 * (row // 3) + column // 3
            if value in rows[row] or value in columns[column] or value in boxes[box]:
                return None, 0
            rows[row].add(value)
            columns[column].add(value)
            boxes[box].add(value)

    first_solution: np.ndarray | None = None
    n_solutions = 0

    def visit() -> None:
        nonlocal first_solution, n_solutions
        if n_solutions >= 2:
            return
        chosen: tuple[int, int] | None = None
        candidates: tuple[int, ...] = ()
        for row in range(9):
            for column in range(9):
                if board[row, column] != 0:
                    continue
                box = 3 * (row // 3) + column // 3
                available = tuple(
                    value
                    for value in range(1, 10)
                    if value not in rows[row]
                    and value not in columns[column]
                    and value not in boxes[box]
                )
                if not available:
                    return
                if chosen is None or len(available) < len(candidates):
                    chosen = row, column
                    candidates = available
                    if len(candidates) == 1:
                        break
            if len(candidates) == 1:
                break
        if chosen is None:
            n_solutions += 1
            if first_solution is None:
                first_solution = board.copy()
            return
        row, column = chosen
        box = 3 * (row // 3) + column // 3
        for value in candidates:
            board[row, column] = value
            rows[row].add(value)
            columns[column].add(value)
            boxes[box].add(value)
            visit()
            boxes[box].remove(value)
            columns[column].remove(value)
            rows[row].remove(value)
            board[row, column] = 0
            if n_solutions >= 2:
                return

    visit()
    return first_solution, n_solutions


def _listed_sudoku_paths(payload: bytes, *, expected_count: int) -> tuple[str, ...]:
    try:
        lines = tuple(line.strip() for line in payload.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        raise ValueError("Sudoku split description is not UTF-8") from error
    if len(lines) != expected_count or any(not line for line in lines):
        raise ValueError(
            f"Sudoku description expected {expected_count} non-empty entries"
        )
    if any(_SUDOKU_IMAGE.fullmatch(line) is None for line in lines):
        raise ValueError("Sudoku description contains a non-image*.jpg entry")
    dat_paths = tuple(
        str(Path(line).with_suffix(".dat")).replace("\\", "/") for line in lines
    )
    if len(dat_paths) != len(set(dat_paths)):
        raise ValueError("Sudoku description contains duplicate samples")
    return dat_paths


def prepare_sudoku(
    output_root: str | Path,
    *,
    cache_root: str | Path | None = None,
    fetcher: Fetcher = _network_fetch,
) -> tuple[Path, Path]:
    """Build a leakage-safe, content-deduplicated view of the official split."""

    root = Path(output_root)
    cache = (
        Path(cache_root)
        if cache_root is not None
        else root / "source_cache" / f"sudoku_dataset-{SUDOKU_REVISION}"
    )
    trusted: dict[str, Mapping[str, Any]] = {}
    source_payloads: dict[str, bytes] = {}
    for relative_path, expected_digest in SUDOKU_SOURCES.items():
        url = f"{SUDOKU_RAW_ROOT}/{relative_path}"
        payload = _source_bytes(
            cache_root=cache,
            relative_path=relative_path,
            url=url,
            fetcher=fetcher,
            expected_sha256=expected_digest,
        )
        source_payloads[relative_path] = payload
        trusted[relative_path] = {
            "url": url,
            "sha256": expected_digest,
            "n_bytes": len(payload),
        }

    descriptions = {
        "train": _listed_sudoku_paths(
            source_payloads["datasets/v2_train.desc"],
            expected_count=SUDOKU_EXPECTED_SPLIT_COUNTS["train"],
        ),
        "test": _listed_sudoku_paths(
            source_payloads["datasets/v2_test.desc"],
            expected_count=SUDOKU_EXPECTED_SPLIT_COUNTS["test"],
        ),
    }
    overlap = set(descriptions["train"]) & set(descriptions["test"])
    if overlap:
        raise ValueError(f"official Sudoku train/test descriptions overlap: {overlap}")

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    records: list[Mapping[str, Any]] = []
    receipts: list[Mapping[str, Any]] = []
    for split in ("train", "test"):
        for relative_path in descriptions[split]:
            url = f"{SUDOKU_RAW_ROOT}/{relative_path}"
            receipt: dict[str, Any] = {
                "source_path": relative_path,
                "official_source_split": split,
                "url": url,
            }
            try:
                payload = _source_bytes(
                    cache_root=cache,
                    relative_path=relative_path,
                    url=url,
                    fetcher=fetcher,
                )
                digest = sha256_bytes(payload)
                puzzle, metadata = parse_sudoku_dat(payload)
                solution, n_solutions = solve_unique_sudoku(puzzle)
                receipt.update(
                    {
                        "source_sha256": digest,
                        "n_bytes": len(payload),
                        "n_solutions_capped_at_two": n_solutions,
                    }
                )
                if solution is None or n_solutions != 1:
                    raise ValueError(
                        f"puzzle has {n_solutions} solutions (count capped at two)"
                    )
                puzzle_text = "".join(str(int(value)) for value in puzzle.ravel())
                puzzle_sha256 = public_projection_sha256(
                    {"family": "sudoku", "puzzle": puzzle}
                )
                content_group = f"sudoku-puzzle-sha256:{puzzle_sha256}"
                record = {
                    "augmentation_group": content_group,
                    "capture_device": metadata["capture_device"],
                    "image_format": metadata["image_format"],
                    "puzzle": puzzle_text,
                    "puzzle_sha256": puzzle_sha256,
                    "solution": "".join(str(int(value)) for value in solution.ravel()),
                    "source_group": content_group,
                    "source_license": "CC-BY-4.0",
                    "source_record_path": relative_path,
                    "source_record_sha256": digest,
                    "source_repository": SUDOKU_REPOSITORY,
                    "source_version": SUDOKU_REVISION,
                }
                receipt.update(
                    {
                        "content_group": content_group,
                        "puzzle_sha256": puzzle_sha256,
                        "status": "validated_pending_content_dedup",
                    }
                )
                candidates.append((record, receipt))
            except Exception as error:  # retain every malformed/non-unique sample
                receipt.update(
                    {
                        "status": "excluded",
                        "failure_type": type(error).__name__,
                        "failure_reason": str(error),
                    }
                )
            receipts.append(receipt)

    by_content: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for record, receipt in candidates:
        by_content.setdefault(str(receipt["content_group"]), []).append(
            (record, receipt)
        )
    cross_split_groups = 0
    for content_group in sorted(by_content):
        members = by_content[content_group]
        official_splits = sorted(
            {str(receipt["official_source_split"]) for _, receipt in members}
        )
        assigned_split = "test" if "test" in official_splits else "train"
        if len(official_splits) > 1:
            cross_split_groups += 1
        eligible = [
            item
            for item in members
            if item[1]["official_source_split"] == assigned_split
        ]
        representative_record, representative_receipt = min(
            eligible, key=lambda item: str(item[1]["source_path"])
        )
        puzzle_sha256 = str(representative_receipt["puzzle_sha256"])
        task_id = f"wichtounet-sudoku-v2:{puzzle_sha256}"
        source_records = sorted(
            (
                {
                    "official_source_split": str(receipt["official_source_split"]),
                    "source_path": str(receipt["source_path"]),
                    "source_sha256": str(receipt["source_sha256"]),
                }
                for _, receipt in members
            ),
            key=lambda item: (item["official_source_split"], item["source_path"]),
        )
        record = {
            **representative_record,
            "official_source_splits": official_splits,
            "source_records": source_records,
            "split": assigned_split,
            "split_provenance": "content_dedup_test_precedence_v1",
            "task_id": task_id,
        }
        records.append(record)
        derived_digest = sha256_bytes(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for _, receipt in members:
            receipt.update(
                {
                    "assigned_split": assigned_split,
                    "duplicate_group_size": len(members),
                    "official_source_splits": official_splits,
                    "retained_task_id": task_id,
                }
            )
            if receipt is representative_receipt:
                receipt.update(
                    {
                        "status": "included",
                        "task_id": task_id,
                        "derived_record_sha256": derived_digest,
                    }
                )
            else:
                receipt.update(
                    {
                        "status": "excluded",
                        "exclusion_type": "duplicate_puzzle_content",
                        "exclusion_reason": (
                            "exact model-visible puzzle duplicate; one deterministic "
                            "representative retained after test-precedence grouping"
                        ),
                    }
                )

    records.sort(key=lambda record: (record["split"], record["task_id"]))
    output_path = root / "sudoku_wichtounet_v2.jsonl"
    output_payload = _jsonl(records)
    _write_new_or_identical(output_path, output_payload)
    split_counts = Counter(str(record["split"]) for record in records)
    manifest = {
        "builder": _builder_receipt(),
        "dataset": {
            "license": "CC-BY-4.0",
            "license_evidence": "README.rst License section",
            "license_status": "verified",
            "name": "wichtounet/sudoku_dataset-v2",
            "official_source_split": True,
            "official_split": False,
            "repository": SUDOKU_REPOSITORY,
            "revision": SUDOKU_REVISION,
            "role": "functional_hybrid_proposal_selection_only",
        },
        "family": "sudoku",
        "output": {
            "n_records": len(records),
            "path": output_path.name,
            "sha256": sha256_bytes(output_payload),
            "split_counts": dict(sorted(split_counts.items())),
        },
        "records": receipts,
        "schema_version": SCHEMA_VERSION,
        "source_roots": trusted,
        "split": {
            "algorithm": "content_dedup_test_precedence_v1",
            "content_identity": (
                "sha256(canonical public projection of family and 9x9 puzzle)"
            ),
            "n_cross_official_split_groups": cross_split_groups,
            "n_unique_puzzles": len(by_content),
            "official_raw_split_counts": dict(SUDOKU_EXPECTED_SPLIT_COUNTS),
            "policy": (
                "deduplicate exact puzzles; if a puzzle occurs in official test, "
                "keep one test representative and exclude all train duplicates"
            ),
            "test_split_role": "non_ood",
        },
        "status": (
            "complete"
            if all(receipt["status"] == "included" for receipt in receipts)
            else "complete_with_exclusions"
        ),
        "summary": {
            "n_excluded": sum(receipt["status"] == "excluded" for receipt in receipts),
            "n_excluded_duplicate_content": sum(
                receipt.get("exclusion_type") == "duplicate_puzzle_content"
                for receipt in receipts
            ),
            "n_excluded_invalid": sum(
                receipt["status"] == "excluded"
                and receipt.get("exclusion_type") != "duplicate_puzzle_content"
                for receipt in receipts
            ),
            "n_expected": sum(SUDOKU_EXPECTED_SPLIT_COUNTS.values()),
            "n_included": len(records),
            "n_unique_puzzles": len(by_content),
        },
    }
    manifest_path = root / "sudoku_wichtounet_v2.manifest.json"
    _write_new_or_identical(manifest_path, _canonical_json(manifest))
    return output_path, manifest_path


def decode_png(payload: bytes) -> np.ndarray:
    """Decode a PNG as exact uint8 RGB using matplotlib's installed backend."""

    image = np.asarray(mpimg.imread(io.BytesIO(payload), format="png"))
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("MazeBench image must decode as RGB/RGBA")
    image = image[..., :3]
    if image.dtype.kind == "f":
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.asarray(image, dtype=np.uint8)
    return image


def _line_scores(
    image: np.ndarray,
    *,
    bounds: tuple[int, int],
    candidate_sizes: Sequence[int],
) -> Mapping[int, float]:
    lower, upper = bounds
    interior = image[lower:upper, lower:upper].astype(np.int32)
    if interior.shape[0] != interior.shape[1] or interior.shape[0] < 5:
        raise ValueError("maze crop must be a square with at least five pixels")
    horizontal = np.abs(np.diff(interior, axis=1)).sum(axis=(0, 2))
    vertical = np.abs(np.diff(interior, axis=0)).sum(axis=(1, 2))
    span = upper - lower
    scores: dict[int, float] = {}
    for raw_size in candidate_sizes:
        size = int(raw_size)
        if size < 2:
            raise ValueError("candidate grid sizes must be at least two")
        positions = np.rint(np.linspace(0, span, size + 1)[1:-1]).astype(int) - 1
        values = []
        for position in positions:
            left = max(0, int(position) - 2)
            right = min(len(horizontal), int(position) + 3)
            values.append(
                int(horizontal[left:right].max(initial=0))
                + int(vertical[left:right].max(initial=0))
            )
        scores[size] = float(np.mean(values)) if values else 0.0
    return scores


def _tile_features(
    image: np.ndarray, *, size: int, bounds: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lower, upper = bounds
    edges = np.rint(np.linspace(lower, upper, size + 1)).astype(int)
    base = np.empty((size, size, 3), dtype=np.uint8)
    skin = np.zeros((size, size), dtype=np.int64)
    gold = np.zeros((size, size), dtype=np.int64)
    confidence: list[float] = []
    for row in range(size):
        for column in range(size):
            y0, y1 = int(edges[row]), int(edges[row + 1])
            x0, x1 = int(edges[column]), int(edges[column + 1])
            margin = max(1, int(min(y1 - y0, x1 - x0) * 0.04))
            pixels = image[y0 + margin : y1 - margin, x0 + margin : x1 - margin]
            if pixels.size == 0:
                raise ValueError("candidate grid creates an empty tile")
            step = max(1, min(pixels.shape[:2]) // 20)
            sampled = pixels[::step, ::step]
            colors, counts = np.unique(
                sampled.reshape(-1, 3), axis=0, return_counts=True
            )
            winner = int(np.argmax(counts))
            base[row, column] = colors[winner]
            confidence.append(int(counts[winner]) / len(sampled.reshape(-1, 3)))
            skin_palette = (
                (pixels[..., 0] >= 190)
                & (pixels[..., 1] >= 170)
                & (pixels[..., 1] <= 230)
                & (pixels[..., 2] >= 120)
                & (pixels[..., 2] <= 200)
            )
            # Beige floor tiles themselves fall inside the broad skin range.
            # Excluding the dominant base retains the sprite evidence instead
            # of incorrectly treating every open tile as the player.
            not_base = np.any(pixels != base[row, column], axis=2)
            skin[row, column] = int(np.count_nonzero(skin_palette & not_base))
            signed = pixels.astype(np.int16)
            gold_palette = (
                (signed[..., 0] >= 180)
                & (signed[..., 1] >= 140)
                & (signed[..., 0] - signed[..., 2] >= 100)
                & (signed[..., 1] - signed[..., 2] >= 80)
            )
            gold[row, column] = int(np.count_nonzero(gold_palette))
    return base, skin, gold, float(np.mean(confidence))


_DIRECTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _path_delta(path: str) -> tuple[int, int]:
    if not path or any(character not in _DIRECTIONS for character in path):
        raise ValueError("accepted maze paths must be non-empty U/D/L/R strings")
    return tuple(
        sum(_DIRECTIONS[character][axis] for character in path) for axis in (0, 1)
    )


def _walk(
    grid: np.ndarray, start: tuple[int, int], path: str
) -> tuple[int, int] | None:
    point = start
    for character in path:
        delta = _DIRECTIONS[character]
        point = point[0] + delta[0], point[1] + delta[1]
        if (
            point[0] < 0
            or point[0] >= grid.shape[0]
            or point[1] < 0
            or point[1] >= grid.shape[1]
            or int(grid[point]) != 0
        ):
            return None
    return point


def infer_maze_grid(
    image: np.ndarray,
    accepted_shortest_paths: Sequence[str],
    *,
    bounds: tuple[int, int] = MAZE_BOUNDS,
    candidate_sizes: Sequence[int] = MAZE_CANDIDATE_SIZES,
) -> Mapping[str, Any]:
    """Infer a discrete grid and validate *all* published shortest paths."""

    paths = tuple(str(path) for path in accepted_shortest_paths)
    if not paths:
        raise ValueError("reachable maze needs at least one accepted shortest path")
    lengths = {len(path) for path in paths}
    deltas = {_path_delta(path) for path in paths}
    if len(lengths) != 1 or len(deltas) != 1:
        raise ValueError("accepted paths disagree on length or endpoint displacement")
    expected_distance = lengths.pop()
    displacement = deltas.pop()
    scores = _line_scores(image, bounds=bounds, candidate_sizes=candidate_sizes)
    valid: list[Mapping[str, Any]] = []
    for size in sorted(scores, key=lambda value: (-scores[value], value)):
        base, skin, gold, confidence = _tile_features(image, size=size, bounds=bounds)
        endpoint_candidates: list[
            tuple[tuple[int, int], tuple[int, int], str, int]
        ] = []
        goal_order = sorted(
            np.ndindex(gold.shape), key=lambda point: (-int(gold[point]), point)
        )
        for goal in goal_order:
            if int(gold[goal]) <= 0:
                break
            start = goal[0] - displacement[0], goal[1] - displacement[1]
            if 0 <= start[0] < size and 0 <= start[1] < size:
                endpoint_candidates.append(
                    (
                        start,
                        goal,
                        "gold_goal_reverse_path_displacement",
                        int(gold[goal]),
                    )
                )
        start_order = sorted(
            np.ndindex(skin.shape), key=lambda point: (-int(skin[point]), point)
        )
        for start in start_order:
            if int(skin[start]) <= 0:
                break
            goal = start[0] + displacement[0], start[1] + displacement[1]
            if 0 <= goal[0] < size and 0 <= goal[1] < size:
                candidate = (
                    start,
                    goal,
                    "skin_start_forward_path_displacement",
                    int(skin[start]),
                )
                if candidate[:2] not in {item[:2] for item in endpoint_candidates}:
                    endpoint_candidates.append(candidate)
        for start, goal, endpoint_locator, locator_score in endpoint_candidates:
            start_color = tuple(int(value) for value in base[start])
            goal_color = tuple(int(value) for value in base[goal])
            palettes = ((start_color,),)
            if goal_color != start_color:
                palettes += ((start_color, goal_color), (goal_color,))
            for palette in palettes:
                traversable = np.zeros((size, size), dtype=bool)
                for color in palette:
                    traversable |= np.all(base == np.asarray(color), axis=2)
                grid = (~traversable).astype(np.int8)
                # Endpoint sprites can dominate their tile; annotations are
                # used only to identify/validate the endpoints, never exposed
                # to a solver at evaluation time.
                grid[start] = 0
                grid[goal] = 0
                endpoints = tuple(_walk(grid, start, path) for path in paths)
                if any(endpoint != goal for endpoint in endpoints):
                    continue
                distance = shortest_path_distance(grid, start, goal, wall_value=1)
                if distance != expected_distance:
                    continue
                valid.append(
                    {
                        "base_confidence": confidence,
                        "goal": goal,
                        "grid": grid,
                        "endpoint_locator": endpoint_locator,
                        "line_score": scores[size],
                        "locator_score": locator_score,
                        "open_palette": palette,
                        "size": size,
                        "skin_score": int(skin[start]),
                        "start": start,
                    }
                )
                break
            if valid and valid[-1]["size"] == size:
                break
    if not valid:
        raise ValueError("no candidate grid validates every accepted shortest path")
    chosen = max(
        valid,
        key=lambda item: (
            float(item["line_score"]),
            str(item["endpoint_locator"]).startswith("gold_goal"),
            int(item["locator_score"]),
            int(item["skin_score"]),
            -int(item["size"]),
        ),
    )
    return {
        **chosen,
        "all_valid_sizes": tuple(sorted({int(item["size"]) for item in valid})),
        "shortest_distance": expected_distance,
    }


def _maze_splits(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], Mapping[str, Any]]:
    if len(records) < 2:
        raise ValueError("MazeBench preparation needs at least two reachable records")
    counts = Counter(int(record["grid_size"]) for record in records)
    sizes = sorted(counts, reverse=True)
    target = max(1, int(math.ceil(0.2 * len(records))))
    test_sizes: list[int] = []
    n_test = 0
    for size in sizes[:-1]:
        test_sizes.append(size)
        n_test += counts[size]
        if n_test >= target:
            break
    if test_sizes and n_test < len(records):
        assignments = {
            str(record["task_id"]): (
                "test" if int(record["grid_size"]) in test_sizes else "train"
            )
            for record in records
        }
        return assignments, {
            "algorithm": "largest_grid_sizes_ood_until_20_percent_v1",
            "official_source_role": "evaluation_only",
            "test_split_role": "ood",
            "test_grid_sizes": sorted(test_sizes),
            "train_grid_sizes": sorted(set(counts) - set(test_sizes)),
        }

    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{MAZE_REVISION}:{record['task_id']}".encode("utf-8")
        ).hexdigest(),
    )
    n_test = max(1, int(math.ceil(0.2 * len(ordered))))
    test_ids = {str(record["task_id"]) for record in ordered[:n_test]}
    assignments = {
        str(record["task_id"]): (
            "test" if str(record["task_id"]) in test_ids else "train"
        )
        for record in records
    }
    return assignments, {
        "algorithm": "pinned_revision_sha256_20_percent_v1",
        "official_source_role": "evaluation_only",
        "test_split_role": "non_ood",
        "test_grid_sizes": sorted(
            {
                int(record["grid_size"])
                for record in records
                if record["task_id"] in test_ids
            }
        ),
        "train_grid_sizes": sorted(
            {
                int(record["grid_size"])
                for record in records
                if record["task_id"] not in test_ids
            }
        ),
    }


def prepare_maze(
    output_root: str | Path,
    *,
    cache_root: str | Path | None = None,
    fetcher: Fetcher = _network_fetch,
) -> tuple[Path, Path]:
    """Convert the pinned MazeBench evaluation images into discrete tasks."""

    root = Path(output_root)
    cache = (
        Path(cache_root)
        if cache_root is not None
        else root / "source_cache" / f"MazeBench-{MAZE_REVISION}"
    )
    trusted: dict[str, Mapping[str, Any]] = {}
    source_payloads: dict[str, bytes] = {}
    for relative_path, expected_digest in MAZE_SOURCES.items():
        url = f"{MAZE_RESOLVE_ROOT}/{relative_path}?download=true"
        payload = _source_bytes(
            cache_root=cache,
            relative_path=relative_path,
            url=url,
            fetcher=fetcher,
            expected_sha256=expected_digest,
        )
        source_payloads[relative_path] = payload
        trusted[relative_path] = {
            "url": url,
            "sha256": expected_digest,
            "n_bytes": len(payload),
        }
    try:
        annotations = json.loads(source_payloads["maze_annotations.json"])
    except json.JSONDecodeError as error:
        raise ValueError("MazeBench annotations are invalid JSON") from error
    expected_keys = {
        f"gen_maze_{index:03d}" for index in range(1, MAZE_EXPECTED_IMAGES + 1)
    }
    if not isinstance(annotations, Mapping) or set(annotations) != expected_keys:
        raise ValueError("MazeBench annotations must contain exactly gen_maze_001..110")
    reachable_count = sum(
        bool(annotations[key].get("reachable")) for key in sorted(annotations)
    )
    if reachable_count != MAZE_EXPECTED_REACHABLE:
        raise ValueError(
            f"MazeBench expected {MAZE_EXPECTED_REACHABLE} reachable mazes, got "
            f"{reachable_count}"
        )

    provisional: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for key in sorted(annotations):
        relative_path = f"{key}.png"
        url = f"{MAZE_RESOLVE_ROOT}/{relative_path}?download=true"
        annotation = annotations[key]
        receipt: dict[str, Any] = {
            "annotation_reachable": bool(annotation.get("reachable")),
            "source_path": relative_path,
            "task_id": f"mazebench:{key}",
            "url": url,
        }
        try:
            payload = _source_bytes(
                cache_root=cache,
                relative_path=relative_path,
                url=url,
                fetcher=fetcher,
            )
            receipt.update(
                {"source_sha256": sha256_bytes(payload), "n_bytes": len(payload)}
            )
            image = decode_png(payload)
            if image.shape != MAZE_IMAGE_SHAPE:
                raise ValueError(
                    f"official MazeBench image must be {MAZE_IMAGE_SHAPE}, got "
                    f"{image.shape}"
                )
            paths = annotation.get("accepted_shortest_paths")
            if not isinstance(paths, list):
                raise ValueError("accepted_shortest_paths must be a list")
            if not bool(annotation.get("reachable")):
                if paths:
                    raise ValueError(
                        "unreachable annotation unexpectedly contains paths"
                    )
                receipt.update(
                    {
                        "status": "excluded",
                        "failure_type": "UpstreamUnreachable",
                        "failure_reason": "upstream annotation marks maze unreachable",
                    }
                )
                receipts.append(receipt)
                continue
            inferred = infer_maze_grid(
                image,
                paths,
                bounds=MAZE_BOUNDS,
                candidate_sizes=MAZE_CANDIDATE_SIZES,
            )
            task_id = f"mazebench:{key}"
            provisional.append(
                {
                    "accepted_path_count": len(paths),
                    "augmentation_group": task_id,
                    "goal": list(inferred["goal"]),
                    "grid_size": int(inferred["size"]),
                    "maze": np.asarray(inferred["grid"], dtype=int).tolist(),
                    "parse_pipeline": "periodic_lines_dominant_rgb_sprite_v1",
                    "shortest_distance": int(inferred["shortest_distance"]),
                    "source_group": task_id,
                    "source_license": "MIT",
                    "source_record_path": relative_path,
                    "source_record_sha256": receipt["source_sha256"],
                    "source_repository": MAZE_REPOSITORY,
                    "source_version": MAZE_REVISION,
                    "start": list(inferred["start"]),
                    "task_id": task_id,
                    "wall_value": 1,
                }
            )
            receipt.update(
                {
                    "all_valid_sizes": list(inferred["all_valid_sizes"]),
                    "base_confidence": inferred["base_confidence"],
                    "endpoint_locator": inferred["endpoint_locator"],
                    "grid_size": inferred["size"],
                    "line_score": inferred["line_score"],
                    "shortest_distance": inferred["shortest_distance"],
                    "skin_score": inferred["skin_score"],
                    "status": "included",
                    "validated_accepted_path_count": len(paths),
                }
            )
        except Exception as error:  # retain image/annotation/parse failures
            receipt.update(
                {
                    "status": "excluded",
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
        receipts.append(receipt)

    if len(receipts) != MAZE_EXPECTED_IMAGES:
        raise AssertionError("MazeBench receipt count changed unexpectedly")
    assignments, split_receipt = _maze_splits(provisional)
    records = []
    for record in provisional:
        split = assignments[str(record["task_id"])]
        records.append(
            {
                **record,
                "split": split,
                "split_provenance": split_receipt["algorithm"],
            }
        )
    records.sort(key=lambda record: (record["split"], record["task_id"]))
    output_path = root / "maze_mazebench_a71a2d1.jsonl"
    output_payload = _jsonl(records)
    _write_new_or_identical(output_path, output_payload)
    split_counts = Counter(str(record["split"]) for record in records)
    manifest = {
        "builder": _builder_receipt(),
        "dataset": {
            "license": "MIT",
            "license_evidence": "Hugging Face dataset card front matter",
            "license_status": "verified",
            "name": "albertoRodriguez97/MazeBench",
            "official_split": False,
            "repository": MAZE_REPOSITORY,
            "revision": MAZE_REVISION,
            "role": "functional_hybrid_proposal_selection_only",
            "upstream_role": "evaluation_only",
        },
        "family": "maze",
        "output": {
            "n_records": len(records),
            "path": output_path.name,
            "sha256": sha256_bytes(output_payload),
            "split_counts": dict(sorted(split_counts.items())),
        },
        "records": receipts,
        "schema_version": SCHEMA_VERSION,
        "source_roots": trusted,
        "split": split_receipt,
        "status": (
            "complete"
            if all(receipt["status"] == "included" for receipt in receipts)
            else "complete_with_exclusions"
        ),
        "summary": {
            "n_expected_images": MAZE_EXPECTED_IMAGES,
            "n_included_reachable": len(records),
            "n_parse_or_download_failures": sum(
                receipt["status"] == "excluded"
                and receipt.get("failure_type") != "UpstreamUnreachable"
                for receipt in receipts
            ),
            "n_upstream_unreachable": sum(
                receipt.get("failure_type") == "UpstreamUnreachable"
                for receipt in receipts
            ),
        },
    }
    manifest_path = root / "maze_mazebench_a71a2d1.manifest.json"
    _write_new_or_identical(manifest_path, _canonical_json(manifest))
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("sudoku", "maze", "all"), default="all")
    parser.add_argument("--output-root", default="data/structured")
    parser.add_argument("--sudoku-cache-root")
    parser.add_argument("--maze-cache-root")
    args = parser.parse_args()
    outputs: list[tuple[Path, Path]] = []
    if args.family in {"sudoku", "all"}:
        outputs.append(
            prepare_sudoku(
                args.output_root,
                cache_root=args.sudoku_cache_root,
            )
        )
    if args.family in {"maze", "all"}:
        outputs.append(prepare_maze(args.output_root, cache_root=args.maze_cache_root))
    for dataset_path, manifest_path in outputs:
        print(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "dataset_sha256": sha256_bytes(dataset_path.read_bytes()),
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
