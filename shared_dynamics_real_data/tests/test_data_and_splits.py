from __future__ import annotations

import numpy as np
from scipy.io import savemat
from scipy.sparse import csc_matrix

from shared_dynamics_real_data.data import load_activity_mat
from shared_dynamics_real_data.splits import build_transitions, purged_contiguous_folds


def test_mat_reader_transposes_unit_time_matrix(tmp_path) -> None:
    path = tmp_path / "tiny.mat"
    source = np.arange(15, dtype=np.uint8).reshape(3, 5)
    savemat(path, {"X": source})

    loaded = load_activity_mat(path)

    assert loaded.shape == (5, 3)
    np.testing.assert_array_equal(loaded, source.T)
    assert loaded.dtype == np.float32
    assert not loaded.flags.writeable


def test_mat_reader_accepts_sparse_activity(tmp_path) -> None:
    path = tmp_path / "tiny_sparse.mat"
    source = np.eye(4, 6, dtype=np.uint8)
    savemat(path, {"X": csc_matrix(source)})

    loaded = load_activity_mat(path)

    np.testing.assert_array_equal(loaded, source.T)


def test_purged_blocks_are_disjoint_and_transitions_never_cross_gaps() -> None:
    recordings = {
        "a": np.column_stack([np.arange(30), np.ones(30)]),
        "b": np.column_stack([np.arange(30), -np.ones(30)]),
    }
    folds = purged_contiguous_folds(recordings, n_splits=3, purge=2)
    middle = folds[1]

    for context in recordings:
        train_idx = np.concatenate(
            [s.indices for s in middle.train if s.context == context]
        )
        test_idx = np.concatenate([s.indices for s in middle.test if s.context == context])
        assert not np.intersect1d(train_idx, test_idx).size
        assert np.min(np.abs(train_idx[:, None] - test_idx[None, :])) > 2

    current, following, labels = build_transitions(middle.train)
    # Feature zero is the absolute timestamp, so every valid transition advances by one.
    np.testing.assert_array_equal(following[:, 0] - current[:, 0], 1.0)
    assert set(labels) == {"a", "b"}
    # A naive concatenation would bridge 7 -> 22 in each context; it must not exist.
    assert not np.any((current[:, 0] == 7) & (following[:, 0] == 22))


def test_fold_construction_is_deterministic() -> None:
    recordings = {"ctx": np.arange(80, dtype=float).reshape(40, 2)}
    first = purged_contiguous_folds(recordings, n_splits=4, purge=1)
    second = purged_contiguous_folds(recordings, n_splits=4, purge=1)
    for left, right in zip(first, second):
        assert left.fold == right.fold
        for a, b in zip(left.train + left.test, right.train + right.test):
            np.testing.assert_array_equal(a.indices, b.indices)
            np.testing.assert_array_equal(a.values, b.values)
