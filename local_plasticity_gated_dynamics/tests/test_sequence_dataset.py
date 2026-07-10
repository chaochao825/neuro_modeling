from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from src.data.sequence_dataset import (
    SequenceDataError,
    bin_spikes,
    block_split_trials,
    load_sequence_session,
    smooth_spike_counts,
    unseen_combination_split,
)


def _write_session(path: Path) -> None:
    pd.DataFrame(
        {
            "trial": range(6),
            "block": [0, 0, 1, 1, 2, 2],
            "item": ["a", "a", "b", "b", "a", "b"],
            "rule": ["f", "b", "f", "b", "f", "b"],
        }
    ).to_csv(path / "trials.csv", index=False)
    pd.DataFrame({"unit": [0, 1], "channel": [3, 4]}).to_csv(
        path / "units.csv", index=False
    )
    spikes = np.empty((6, 2), dtype=object)
    for trial in range(6):
        spikes[trial, 0] = np.array([0.01, 0.021, 0.079, 0.1])
        spikes[trial, 1] = np.array([0.03 + trial * 0.001])
    savemat(path / "spikes.mat", {"spikes": spikes})


def test_load_and_bin_sequence_session(tmp_path: Path) -> None:
    _write_session(tmp_path)
    session = load_sequence_session(tmp_path)
    assert session.spike_times.shape == (6, 2)
    counts, times = bin_spikes(session, window_s=(0.0, 0.1), bin_size_ms=20)
    assert counts.shape == (6, 5, 2)
    assert times.shape == (5,)
    assert counts[0, :, 0].sum() == 3
    assert not session.spike_times.flags.writeable
    assert not session.spike_times[0, 0].flags.writeable


def test_smoothing_never_leaks_across_trials_or_future() -> None:
    counts = np.zeros((2, 8, 1))
    counts[0, -1, 0] = 1.0
    causal = smooth_spike_counts(counts, bin_size_ms=20, sigma_ms=40, mode="causal")
    assert np.all(causal[1] == 0.0)
    assert np.all(causal[0, :-1] == 0.0)
    symmetric = smooth_spike_counts(
        counts, bin_size_ms=20, sigma_ms=40, mode="symmetric"
    )
    assert symmetric[0, -2, 0] > 0.0
    assert np.all(symmetric[1] == 0.0)


def test_block_split_and_unseen_combinations() -> None:
    trials = pd.DataFrame(
        {
            "block": np.repeat(np.arange(6), 4),
            "item": np.tile(["a", "a", "b", "b"], 6),
            "rule": np.tile(["f", "b", "f", "b"], 6),
        }
    )
    train, test, blocks = block_split_trials(
        trials, block_column="block", test_fraction=0.3, seed=2
    )
    assert set(blocks[train]).isdisjoint(set(blocks[test]))
    combo_train, combo_test, held = unseen_combination_split(
        trials, factor_columns=["item", "rule"], seed=1, holdout_fraction=0.25
    )
    assert combo_test.size > 0 and combo_train.size > 0 and held
    for column in ("item", "rule"):
        assert set(trials.iloc[combo_test][column]) <= set(trials.iloc[combo_train][column])
        assert set(trials[column]) == set(trials.iloc[combo_train][column])


def test_unseen_split_preserves_all_factor_levels_for_many_deterministic_orders() -> None:
    combinations = pd.DataFrame(
        [(a, b) for a in "abc" for b in "xyz"], columns=["item", "rule"]
    )
    trials = pd.concat([combinations, combinations], ignore_index=True)
    for seed in range(20):
        train, test, _ = unseen_combination_split(
            trials,
            factor_columns=["item", "rule"],
            seed=seed,
            holdout_fraction=0.4,
        )
        assert test.size and train.size
        for column in ("item", "rule"):
            assert set(trials.iloc[train][column]) == set(trials[column])


def test_mat_loader_ignores_dense_numeric_distractors(tmp_path: Path) -> None:
    pd.DataFrame({"trial": range(2)}).to_csv(tmp_path / "trials.csv", index=False)
    pd.DataFrame({"unit": range(2)}).to_csv(tmp_path / "units.csv", index=False)
    cells = np.empty((2, 2), dtype=object)
    cells[0, 0], cells[0, 1] = np.array([0.1]), np.array([0.2, 0.25])
    cells[1, 0], cells[1, 1] = np.array([0.3, 0.35, 0.37]), np.array([0.4])
    savemat(tmp_path / "spikes.mat", {"a_dense": np.ones((2, 2)), "cells": cells})
    session = load_sequence_session(tmp_path)
    assert session.spike_key == "cells"
    np.testing.assert_array_equal(session.spike_times[1, 1], [0.4])


def test_split_rejects_missing_blocks_and_factors() -> None:
    with pytest.raises(SequenceDataError, match="block labels"):
        block_split_trials(
            pd.DataFrame({"block": [0, np.nan, 1]}),
            block_column="block",
            seed=0,
        )
    with pytest.raises(SequenceDataError, match="factor columns"):
        unseen_combination_split(
            pd.DataFrame({"a": ["x", None], "b": [0, 1]}),
            factor_columns=["a", "b"],
            seed=0,
        )


def test_loader_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sequence_session(tmp_path)
