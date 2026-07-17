"""Contracts for the preregistered Exp26 generator manifest."""

from __future__ import annotations

from collections import Counter

import pytest

from src.analysis.actuator_manifest import (
    analytic_grid,
    manifest_hash,
    select_generator_manifest,
)


FORMAL_GRID = {
    "alpha": [round(index / 10.0, 1) for index in range(11)],
    "transition_rank": [1, 2, 4, 8],
    "input_rank": [1, 2, 4],
    "delay": [0, 4, 12, 24],
    "noise_std": [0.1, 0.3, 0.6, 1.0],
}


def test_full_grid_and_manifest_are_balanced_disjoint_and_hash_locked() -> None:
    assert len(analytic_grid(FORMAL_GRID)) == 2112
    first = select_generator_manifest(
        FORMAL_GRID, per_alpha_per_split=4, selection_seed=2601
    )
    second = select_generator_manifest(
        FORMAL_GRID, per_alpha_per_split=4, selection_seed=2601
    )
    assert first == second
    assert len(first) == 88
    assert len({item.generator_id for item in first}) == 88
    assert len(manifest_hash(first)) == 64
    for split in ("discovery", "heldout"):
        cells = [item for item in first if item.generator_split == split]
        assert len(cells) == 44
        assert Counter(item.alpha for item in cells) == {
            round(index / 10.0, 1): 4 for index in range(11)
        }
        assert max(Counter(item.transition_rank for item in cells).values()) - min(
            Counter(item.transition_rank for item in cells).values()
        ) <= 2
    discovery = {item.task_tuple() for item in first if item.generator_split == "discovery"}
    heldout = {item.task_tuple() for item in first if item.generator_split == "heldout"}
    assert discovery.isdisjoint(heldout)


def test_manifest_selection_fails_closed() -> None:
    with pytest.raises(ValueError, match="twice"):
        select_generator_manifest(
            {
                "alpha": [0.0],
                "transition_rank": [1],
                "input_rank": [1],
                "delay": [0],
                "noise_std": [0.1],
            },
            per_alpha_per_split=1,
            selection_seed=2601,
        )
    with pytest.raises(ValueError, match="alpha"):
        analytic_grid({**FORMAL_GRID, "alpha": [1.2]})
