"""Hash-locked task-generator manifests for Exp26.

The expensive E/I tier uses a balanced subset of a fully enumerated analytic
grid.  Selection depends only on a preregistered seed and parameter grid, never
on actuator performance.  Discovery and held-out generator tuples are
disjoint before any network seed is simulated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np


def _numeric_sequence(
    value: object,
    *,
    name: str,
    integer: bool,
) -> tuple[int, ...] | tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a non-empty sequence")
    if len(value) == 0:
        raise ValueError(f"{name} must be non-empty")
    resolved: list[int | float] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must contain real numeric values")
        number = float(item)
        if not np.isfinite(number):
            raise ValueError(f"{name} must contain finite values")
        if integer:
            if not number.is_integer():
                raise ValueError(f"{name} must contain integers")
            resolved.append(int(number))
        else:
            resolved.append(number)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(resolved)  # type: ignore[return-value]


@dataclass(frozen=True)
class GeneratorCell:
    generator_id: str
    generator_split: str
    alpha: float
    transition_rank: int
    input_rank: int
    delay: int
    noise_std: float
    rotation_seed: int

    def task_tuple(self) -> tuple[float, int, int, int, float]:
        return (
            self.alpha,
            self.transition_rank,
            self.input_rank,
            self.delay,
            self.noise_std,
        )


def _cell_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _rotation_seed(selection_seed: int, payload: Mapping[str, object]) -> int:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(
        f"exp26-rotation:{selection_seed}:{encoded}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def analytic_grid(grid: Mapping[str, object]) -> list[dict[str, int | float]]:
    """Enumerate the complete theory grid in a stable lexical order."""

    if not isinstance(grid, Mapping):
        raise TypeError("grid must be a mapping")
    alpha = _numeric_sequence(grid.get("alpha"), name="alpha", integer=False)
    transition_rank = _numeric_sequence(
        grid.get("transition_rank"), name="transition_rank", integer=True
    )
    input_rank = _numeric_sequence(
        grid.get("input_rank"), name="input_rank", integer=True
    )
    delay = _numeric_sequence(grid.get("delay"), name="delay", integer=True)
    noise = _numeric_sequence(
        grid.get("noise_std"), name="noise_std", integer=False
    )
    if any(not 0.0 <= value <= 1.0 for value in alpha):
        raise ValueError("alpha values must lie in [0, 1]")
    if any(value < 1 for value in (*transition_rank, *input_rank)):
        raise ValueError("rank values must be positive")
    if any(value < 0 for value in delay):
        raise ValueError("delay values must be non-negative")
    if any(value < 0.0 for value in noise):
        raise ValueError("noise_std values must be non-negative")
    return [
        {
            "alpha": float(a),
            "transition_rank": int(rank_a),
            "input_rank": int(rank_b),
            "delay": int(lag),
            "noise_std": float(noise_value),
        }
        for a, rank_a, rank_b, lag, noise_value in product(
            alpha, transition_rank, input_rank, delay, noise
        )
    ]


def _hamming(first: Mapping[str, object], second: Mapping[str, object]) -> int:
    keys = ("transition_rank", "input_rank", "delay", "noise_std")
    return sum(first[key] != second[key] for key in keys)


def _balanced_pick(
    candidates: list[dict[str, int | float]],
    *,
    count: int,
    existing: list[dict[str, int | float]],
    global_counts: dict[tuple[str, int | float], int],
    rng: np.random.Generator,
) -> list[dict[str, int | float]]:
    keys = ("transition_rank", "input_rank", "delay", "noise_std")
    available = list(candidates)
    selected: list[dict[str, int | float]] = []
    for _ in range(count):
        jitter = rng.random(len(available))
        scores: list[tuple[float, float, float]] = []
        for index, candidate in enumerate(available):
            references = [*existing, *selected]
            min_distance = (
                min(_hamming(candidate, reference) for reference in references)
                if references
                else len(keys)
            )
            imbalance = sum(
                global_counts.get((key, candidate[key]), 0) for key in keys
            )
            scores.append((float(min_distance), float(-imbalance), float(jitter[index])))
        best_index = max(range(len(available)), key=lambda index: scores[index])
        chosen = available.pop(best_index)
        selected.append(chosen)
        for key in keys:
            token = (key, chosen[key])
            global_counts[token] = global_counts.get(token, 0) + 1
    return selected


def select_generator_manifest(
    grid: Mapping[str, object],
    *,
    per_alpha_per_split: int,
    selection_seed: int,
) -> tuple[GeneratorCell, ...]:
    """Select a balanced, performance-independent discovery/held-out manifest."""

    if isinstance(per_alpha_per_split, (bool, np.bool_)) or not isinstance(
        per_alpha_per_split, (int, np.integer)
    ):
        raise TypeError("per_alpha_per_split must be an integer")
    count = int(per_alpha_per_split)
    if count < 1:
        raise ValueError("per_alpha_per_split must be positive")
    if isinstance(selection_seed, (bool, np.bool_)) or not isinstance(
        selection_seed, (int, np.integer)
    ):
        raise TypeError("selection_seed must be an integer")
    if int(selection_seed) < 0:
        raise ValueError("selection_seed must be non-negative")
    full = analytic_grid(grid)
    alpha_values = sorted({float(item["alpha"]) for item in full})
    rng = np.random.default_rng(int(selection_seed))
    selected_by_split: dict[str, list[dict[str, int | float]]] = {
        "discovery": [],
        "heldout": [],
    }
    counts_by_split: dict[str, dict[tuple[str, int | float], int]] = {
        "discovery": {},
        "heldout": {},
    }
    for alpha in alpha_values:
        candidates = [item for item in full if float(item["alpha"]) == alpha]
        if len(candidates) < 2 * count:
            raise ValueError(
                "each alpha requires at least twice per_alpha_per_split candidates"
            )
        discovery = _balanced_pick(
            candidates,
            count=count,
            existing=[],
            global_counts=counts_by_split["discovery"],
            rng=rng,
        )
        discovery_tokens = {
            tuple(item[key] for key in sorted(item)) for item in discovery
        }
        remaining = [
            item
            for item in candidates
            if tuple(item[key] for key in sorted(item)) not in discovery_tokens
        ]
        heldout = _balanced_pick(
            remaining,
            count=count,
            existing=discovery,
            global_counts=counts_by_split["heldout"],
            rng=rng,
        )
        selected_by_split["discovery"].extend(discovery)
        selected_by_split["heldout"].extend(heldout)
    cells: list[GeneratorCell] = []
    for split in ("discovery", "heldout"):
        for item in selected_by_split[split]:
            payload = {"generator_split": split, **item}
            rotation_seed = _rotation_seed(int(selection_seed), payload)
            identifier = _cell_id({**payload, "rotation_seed": rotation_seed})
            cells.append(
                GeneratorCell(
                    generator_id=identifier,
                    generator_split=split,
                    alpha=float(item["alpha"]),
                    transition_rank=int(item["transition_rank"]),
                    input_rank=int(item["input_rank"]),
                    delay=int(item["delay"]),
                    noise_std=float(item["noise_std"]),
                    rotation_seed=rotation_seed,
                )
            )
    return tuple(cells)


def manifest_hash(cells: Sequence[GeneratorCell]) -> str:
    """Return a stable full-manifest SHA-256 receipt."""

    if not cells:
        raise ValueError("cells must be non-empty")
    payload = [asdict(cell) for cell in cells]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "GeneratorCell",
    "analytic_grid",
    "manifest_hash",
    "select_generator_manifest",
]
