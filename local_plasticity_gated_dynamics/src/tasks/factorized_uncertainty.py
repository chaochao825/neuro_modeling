"""Orthogonal jump, drift, and observation-noise streams for Exp39."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
from typing import Iterable, Sequence

import numpy as np

from src.utils.reproducibility import make_rng


FACTOR_NAMES = ("h", "q", "r")


@dataclass(frozen=True)
class UncertaintyLevels:
    """Low/high values for the three independently manipulated factors."""

    hazard: tuple[float, float] = (0.0025, 0.06)
    process_variance: tuple[float, float] = (0.0025, 0.04)
    observation_variance: tuple[float, float] = (0.01, 0.16)

    def __post_init__(self) -> None:
        for name, values in (
            ("hazard", self.hazard),
            ("process_variance", self.process_variance),
            ("observation_variance", self.observation_variance),
        ):
            if len(values) != 2 or not 0.0 < values[0] < values[1]:
                raise ValueError(f"{name} must contain increasing positive levels")
        if self.hazard[1] >= 0.5:
            raise ValueError("hazard levels must be below 0.5")

    def values(self, cell: str) -> tuple[float, float, float]:
        bits = parse_cell(cell)
        return (
            self.hazard[bits[0]],
            self.process_variance[bits[1]],
            self.observation_variance[bits[2]],
        )


@dataclass(frozen=True)
class FactorialStreamConfig:
    """Sequence-level design with no random split of time points."""

    block_length: int = 96
    blocks_per_sequence: int = 16
    n_sequences: int = 4
    jump_variance: float = 4.0

    def __post_init__(self) -> None:
        for name in ("block_length", "blocks_per_sequence", "n_sequences"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.jump_variance) or self.jump_variance <= 0.0:
            raise ValueError("jump_variance must be positive")


@dataclass(frozen=True)
class UncertaintyTape:
    """One immutable observation tape shared by every compared filter."""

    observations: np.ndarray
    latent: np.ndarray
    hazard: np.ndarray
    process_variance: np.ndarray
    observation_variance: np.ndarray
    jump_flags: np.ndarray
    sequence_ids: np.ndarray
    block_ids: np.ndarray
    cells: np.ndarray
    split: str
    digest: str

    def __post_init__(self) -> None:
        arrays = (
            self.observations,
            self.latent,
            self.hazard,
            self.process_variance,
            self.observation_variance,
            self.jump_flags,
            self.sequence_ids,
            self.block_ids,
            self.cells,
        )
        lengths = {len(np.asarray(array)) for array in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("tape arrays must have one shared non-zero length")
        if not np.all(np.isfinite(self.observations)) or not np.all(
            np.isfinite(self.latent)
        ):
            raise ValueError("observations and latent state must be finite")


def parse_cell(cell: str) -> tuple[int, int, int]:
    """Parse a registered three-bit h/Q/R condition."""

    if not isinstance(cell, str) or len(cell) != 3 or set(cell) - {"0", "1"}:
        raise ValueError("cell must be a three-character binary string")
    return tuple(int(value) for value in cell)  # type: ignore[return-value]


def all_factorial_cells() -> tuple[str, ...]:
    return tuple("".join(map(str, bits)) for bits in product((0, 1), repeat=3))


def single_factor_training_cells() -> tuple[str, ...]:
    """Baseline plus one-at-a-time factor elevations."""

    return ("000", "100", "010", "001")


def heldout_composition_cells() -> tuple[str, ...]:
    """Pairwise and triple elevations absent from fitting."""

    return ("110", "101", "011", "111")


def _validate_cells(cells: Iterable[str], *, blocks: int) -> tuple[str, ...]:
    result = tuple(cells)
    if not result or len(set(result)) != len(result):
        raise ValueError("cells must be non-empty and unique")
    for cell in result:
        parse_cell(cell)
    if blocks % len(result):
        raise ValueError("blocks_per_sequence must be divisible by cell count")
    return result


def _balanced_order(
    rng: np.random.Generator, cells: Sequence[str], repeats: int
) -> list[str]:
    order: list[str] = []
    previous: str | None = None
    for _ in range(repeats):
        candidates = list(cells)
        rng.shuffle(candidates)
        if previous is not None and len(candidates) > 1 and candidates[0] == previous:
            swap = next(index for index, value in enumerate(candidates) if value != previous)
            candidates[0], candidates[swap] = candidates[swap], candidates[0]
        order.extend(candidates)
        previous = candidates[-1]
    return order


def _digest(*arrays: np.ndarray, labels: tuple[object, ...]) -> str:
    hasher = hashlib.sha256()
    for label in labels:
        encoded = repr(label).encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.dtype).encode("ascii"))
        hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def generate_uncertainty_tape(
    *,
    seed: int,
    split: str,
    cells: Sequence[str],
    levels: UncertaintyLevels = UncertaintyLevels(),
    config: FactorialStreamConfig = FactorialStreamConfig(),
) -> UncertaintyTape:
    """Generate balanced sequences from parameter-independent random variates."""

    if not isinstance(split, str) or not split:
        raise ValueError("split must be non-empty")
    selected = _validate_cells(cells, blocks=config.blocks_per_sequence)
    rng = make_rng(seed, "exp39-factorial-tape-v1", split)
    total = config.n_sequences * config.blocks_per_sequence * config.block_length
    observations = np.empty(total, dtype=np.float64)
    latent = np.empty(total, dtype=np.float64)
    hazards = np.empty(total, dtype=np.float64)
    process = np.empty(total, dtype=np.float64)
    observation = np.empty(total, dtype=np.float64)
    jumps = np.zeros(total, dtype=bool)
    sequence_ids = np.empty(total, dtype=np.int64)
    block_ids = np.empty(total, dtype=np.int64)
    cell_array = np.empty(total, dtype="U3")
    cursor = 0
    global_block = 0

    for sequence in range(config.n_sequences):
        order = _balanced_order(
            rng, selected, config.blocks_per_sequence // len(selected)
        )
        state = float(rng.normal(0.0, np.sqrt(config.jump_variance)))
        for cell in order:
            h_value, q_value, r_value = levels.values(cell)
            for _ in range(config.block_length):
                jump = bool(rng.random() < h_value)
                if jump:
                    state = float(
                        rng.normal(0.0, np.sqrt(config.jump_variance))
                    )
                else:
                    state += float(rng.normal(0.0, np.sqrt(q_value)))
                observed = state + float(rng.normal(0.0, np.sqrt(r_value)))
                observations[cursor] = observed
                latent[cursor] = state
                hazards[cursor] = h_value
                process[cursor] = q_value
                observation[cursor] = r_value
                jumps[cursor] = jump
                sequence_ids[cursor] = sequence
                block_ids[cursor] = global_block
                cell_array[cursor] = cell
                cursor += 1
            global_block += 1

    digest = _digest(
        observations,
        latent,
        hazards,
        process,
        observation,
        jumps,
        sequence_ids,
        block_ids,
        cell_array,
        labels=(seed, split, selected, config),
    )
    return UncertaintyTape(
        observations=observations,
        latent=latent,
        hazard=hazards,
        process_variance=process,
        observation_variance=observation,
        jump_flags=jumps,
        sequence_ids=sequence_ids,
        block_ids=block_ids,
        cells=cell_array,
        split=split,
        digest=digest,
    )
