from __future__ import annotations

import numpy as np
import pytest

from src.tasks.factorized_uncertainty import (
    FactorialStreamConfig,
    UncertaintyLevels,
    all_factorial_cells,
    generate_uncertainty_tape,
    heldout_composition_cells,
    parse_cell,
    single_factor_training_cells,
)


def test_factorial_cell_partition_is_complete_and_disjoint() -> None:
    training = set(single_factor_training_cells())
    heldout = set(heldout_composition_cells())
    assert not training & heldout
    assert training | heldout == set(all_factorial_cells())
    assert parse_cell("101") == (1, 0, 1)
    with pytest.raises(ValueError, match="three-character"):
        parse_cell("10")


def test_uncertainty_tape_is_deterministic_balanced_and_sequence_grouped() -> None:
    config = FactorialStreamConfig(
        block_length=12, blocks_per_sequence=8, n_sequences=3
    )
    first = generate_uncertainty_tape(
        seed=39,
        split="unit",
        cells=all_factorial_cells(),
        config=config,
    )
    second = generate_uncertainty_tape(
        seed=39,
        split="unit",
        cells=all_factorial_cells(),
        config=config,
    )
    assert first.digest == second.digest
    np.testing.assert_array_equal(first.observations, second.observations)
    assert np.all(first.sequence_ids[1:] >= first.sequence_ids[:-1])
    counts = dict(zip(*np.unique(first.cells, return_counts=True), strict=True))
    assert set(counts.values()) == {config.block_length * config.n_sequences}
    assert len(first.observations) == 12 * 8 * 3


def test_levels_are_applied_orthogonally() -> None:
    levels = UncertaintyLevels(
        hazard=(0.01, 0.1),
        process_variance=(0.02, 0.2),
        observation_variance=(0.03, 0.3),
    )
    tape = generate_uncertainty_tape(
        seed=1,
        split="orthogonal",
        cells=all_factorial_cells(),
        levels=levels,
        config=FactorialStreamConfig(
            block_length=4, blocks_per_sequence=8, n_sequences=1
        ),
    )
    for cell in all_factorial_cells():
        mask = tape.cells == cell
        expected = levels.values(cell)
        assert np.unique(tape.hazard[mask]).tolist() == [expected[0]]
        assert np.unique(tape.process_variance[mask]).tolist() == [expected[1]]
        assert np.unique(tape.observation_variance[mask]).tolist() == [expected[2]]


def test_generator_rejects_unbalanced_or_invalid_design() -> None:
    with pytest.raises(ValueError, match="divisible"):
        generate_uncertainty_tape(
            seed=0,
            split="bad",
            cells=single_factor_training_cells(),
            config=FactorialStreamConfig(blocks_per_sequence=5),
        )
    with pytest.raises(ValueError, match="increasing"):
        UncertaintyLevels(hazard=(0.1, 0.01))
