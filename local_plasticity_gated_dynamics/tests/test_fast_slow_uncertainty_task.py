from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from src.tasks.factorized_uncertainty import UncertaintyLevels
from src.tasks.fast_slow_uncertainty import (
    FastSlowStreamConfig,
    generate_fast_slow_tape,
)


CELLS = ("000", "100", "010", "001")


def test_fast_slow_tape_is_deterministic_split_separated_and_balanced() -> None:
    config = FastSlowStreamConfig(
        block_lengths=(8, 12), blocks_per_sequence=8, n_sequences=2
    )
    first = generate_fast_slow_tape(
        seed=43, split="fit", cells=CELLS, config=config
    )
    replay = generate_fast_slow_tape(
        seed=43, split="fit", cells=CELLS, config=config
    )
    test = generate_fast_slow_tape(
        seed=43, split="test", cells=CELLS, config=config
    )

    assert first.digest == replay.digest
    assert np.array_equal(first.observations, replay.observations)
    assert first.digest != test.digest
    assert not np.array_equal(first.observations, test.observations)

    for sequence in range(config.n_sequences):
        sequence_mask = first.sequence_ids == sequence
        counts: Counter[tuple[str, int]] = Counter()
        for block in np.unique(first.block_ids[sequence_mask]):
            mask = first.block_ids == block
            counts[(str(first.cells[mask][0]), int(np.sum(mask)))] += 1
        assert counts == Counter(
            (cell, length) for cell in CELLS for length in config.block_lengths
        )


def test_fast_slow_tape_has_aligned_truth_and_no_block_reset_requirement() -> None:
    tape = generate_fast_slow_tape(
        seed=44,
        split="fit",
        cells=CELLS,
        config=FastSlowStreamConfig(
            block_lengths=(6,), blocks_per_sequence=4, n_sequences=2
        ),
    )
    arrays = (
        tape.observations,
        tape.latent,
        tape.hazard,
        tape.process_variance,
        tape.observation_variance,
        tape.jump_flags,
        tape.sequence_ids,
        tape.block_ids,
        tape.cells,
    )
    assert {len(value) for value in arrays} == {48}
    assert np.all(np.isfinite(tape.observations))
    assert np.all(np.diff(tape.sequence_ids) >= 0)
    assert set(tape.cells) == set(CELLS)


def test_fast_slow_stream_config_rejects_unbalanced_or_invalid_design() -> None:
    with pytest.raises(ValueError, match="unique"):
        FastSlowStreamConfig(block_lengths=(8, 8))
    with pytest.raises(ValueError, match="positive"):
        FastSlowStreamConfig(block_lengths=(0,))
    with pytest.raises(ValueError, match="divisible"):
        generate_fast_slow_tape(
            seed=1,
            split="fit",
            cells=CELLS,
            config=FastSlowStreamConfig(
                block_lengths=(8, 12), blocks_per_sequence=4, n_sequences=1
            ),
        )


def test_fast_slow_tape_accepts_explicit_levels_without_using_test_truth() -> None:
    levels = UncertaintyLevels(
        hazard=(0.01, 0.02),
        process_variance=(0.03, 0.04),
        observation_variance=(0.05, 0.06),
    )
    tape = generate_fast_slow_tape(
        seed=2,
        split="fit",
        cells=CELLS,
        levels=levels,
        config=FastSlowStreamConfig(
            block_lengths=(4,), blocks_per_sequence=4, n_sequences=1
        ),
    )
    assert set(np.unique(tape.hazard)) == {0.01, 0.02}
    assert set(np.unique(tape.process_variance)) == {0.03, 0.04}
    assert set(np.unique(tape.observation_variance)) == {0.05, 0.06}
