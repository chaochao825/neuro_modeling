"""Fresh variable-duration uncertainty streams for the Exp43 exchange audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np

from src.tasks.factorized_uncertainty import (
    UncertaintyLevels,
    UncertaintyTape,
    parse_cell,
)
from src.utils.reproducibility import make_rng


@dataclass(frozen=True)
class FastSlowStreamConfig:
    """Balanced cell-by-duration blocks without random time-point splitting."""

    block_lengths: tuple[int, ...] = (64, 96, 128)
    blocks_per_sequence: int = 12
    n_sequences: int = 4
    jump_variance: float = 4.0

    def __post_init__(self) -> None:
        lengths = tuple(self.block_lengths)
        if not lengths or len(set(lengths)) != len(lengths):
            raise ValueError("block_lengths must be non-empty and unique")
        if any(
            isinstance(value, (bool, np.bool_))
            or int(value) != value
            or value <= 0
            for value in lengths
        ):
            raise ValueError("block_lengths must contain positive integers")
        for name in ("blocks_per_sequence", "n_sequences"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.jump_variance) or self.jump_variance <= 0.0:
            raise ValueError("jump_variance must be positive")


def _validated_cells(cells: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(cells)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("cells must be non-empty and unique")
    for cell in selected:
        parse_cell(cell)
    return selected


def _balanced_cell_duration_order(
    rng: np.random.Generator,
    *,
    cells: tuple[str, ...],
    block_lengths: tuple[int, ...],
    blocks_per_sequence: int,
) -> list[tuple[str, int]]:
    base = [(cell, length) for cell in cells for length in block_lengths]
    if blocks_per_sequence % len(base):
        raise ValueError(
            "blocks_per_sequence must be divisible by cells x block_lengths"
        )
    order: list[tuple[str, int]] = []
    previous_cell: str | None = None
    for _ in range(blocks_per_sequence // len(base)):
        candidates = list(base)
        rng.shuffle(candidates)
        if previous_cell is not None and candidates[0][0] == previous_cell:
            swap = next(
                (
                    index
                    for index, (cell, _) in enumerate(candidates)
                    if cell != previous_cell
                ),
                None,
            )
            if swap is not None:
                candidates[0], candidates[swap] = candidates[swap], candidates[0]
        order.extend(candidates)
        previous_cell = candidates[-1][0]
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


def generate_fast_slow_tape(
    *,
    seed: int,
    split: str,
    cells: Sequence[str],
    levels: UncertaintyLevels = UncertaintyLevels(),
    config: FastSlowStreamConfig = FastSlowStreamConfig(),
) -> UncertaintyTape:
    """Generate an immutable tape from an Exp43-specific RNG namespace."""

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be non-empty")
    selected = _validated_cells(cells)
    rng = make_rng(int(seed), "exp43-fast-slow-tape-v1", split)
    orders = [
        _balanced_cell_duration_order(
            rng,
            cells=selected,
            block_lengths=config.block_lengths,
            blocks_per_sequence=config.blocks_per_sequence,
        )
        for _ in range(config.n_sequences)
    ]
    total = sum(length for order in orders for _, length in order)
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

    for sequence, order in enumerate(orders):
        state = float(rng.normal(0.0, np.sqrt(config.jump_variance)))
        for cell, block_length in order:
            h_value, q_value, r_value = levels.values(cell)
            for _ in range(block_length):
                jump = bool(rng.random() < h_value)
                if jump:
                    state = float(rng.normal(0.0, np.sqrt(config.jump_variance)))
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
        labels=(int(seed), split, selected, config, "exp43-fast-slow-tape-v1"),
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


__all__ = ["FastSlowStreamConfig", "generate_fast_slow_tape"]
