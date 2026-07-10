"""Contiguous, purged block folds with explicit transition provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TimeSegment:
    """A contiguous slice from one context recording."""

    context: str
    values: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        indices = np.asarray(self.indices, dtype=np.int64)
        if not isinstance(self.context, str) or not self.context:
            raise ValueError("context must be a non-empty string")
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("values must have shape [time, feature]")
        if indices.shape != (values.shape[0],):
            raise ValueError("indices must contain one entry per time point")
        if np.any(indices < 0) or (indices.size > 1 and np.any(np.diff(indices) != 1)):
            raise ValueError("indices must be non-negative and strictly contiguous")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise TypeError("values must be real numeric")
        if not np.isfinite(values).all():
            raise ValueError("values contain non-finite entries")
        # Keep slice views of large recordings instead of copying a full matrix
        # once per CV fold. Integer inputs are promoted once when preprocessing.
        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float32)
        indices = indices.copy()
        values.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True)
class BlockFold:
    """Train/test segments for one held-out contiguous block per context."""

    fold: int
    train: tuple[TimeSegment, ...]
    test: tuple[TimeSegment, ...]
    purge: int

    def __post_init__(self) -> None:
        if self.fold < 0 or self.purge < 0:
            raise ValueError("fold and purge must be non-negative")
        if not self.train or not self.test:
            raise ValueError("a fold needs non-empty train and test segments")
        contexts = {segment.context for segment in self.test}
        if contexts != {segment.context for segment in self.train}:
            raise ValueError("every context must occur in both train and test")
        for context in contexts:
            train_idx = np.concatenate(
                [segment.indices for segment in self.train if segment.context == context]
            )
            test_idx = np.concatenate(
                [segment.indices for segment in self.test if segment.context == context]
            )
            if np.intersect1d(train_idx, test_idx).size:
                raise ValueError("train/test time points overlap")
            test_start, test_stop = int(test_idx.min()), int(test_idx.max())
            if self.purge and np.any(
                (train_idx >= test_start - self.purge)
                & (train_idx <= test_stop + self.purge)
            ):
                raise ValueError("train points violate the requested purge gap")


def _validate_recordings(recordings: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(recordings, Mapping) or not recordings:
        raise ValueError("recordings must be a non-empty context-to-array mapping")
    checked: dict[str, np.ndarray] = {}
    n_features: int | None = None
    for context, raw in recordings.items():
        if not isinstance(context, str) or not context:
            raise ValueError("recording keys must be non-empty strings")
        values = np.asarray(raw)
        if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 1:
            raise ValueError("each recording must have shape [time>=4, feature]")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise TypeError("recordings must be real numeric")
        if not np.isfinite(values).all():
            raise ValueError("recordings contain non-finite entries")
        if n_features is None:
            n_features = int(values.shape[1])
        elif values.shape[1] != n_features:
            raise ValueError("all contexts must contain the same aligned features")
        checked[context] = values
    return checked


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def purged_contiguous_folds(
    recordings: Mapping[str, np.ndarray],
    *,
    n_splits: int,
    purge: int = 1,
) -> list[BlockFold]:
    """Hold out contiguous blocks, removing adjacent train points.

    Each context is partitioned independently into ``n_splits`` chronological
    blocks.  Fold ``k`` holds out block ``k`` in every context.  Remaining time
    points are converted to separate contiguous segments, so downstream code
    cannot accidentally create a transition across a held-out or purged gap.
    """

    checked = _validate_recordings(recordings)
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise TypeError("n_splits must be an integer")
    if isinstance(purge, bool) or not isinstance(purge, (int, np.integer)):
        raise TypeError("purge must be an integer")
    n_splits, purge = int(n_splits), int(purge)
    if n_splits < 2 or purge < 0:
        raise ValueError("n_splits must be >=2 and purge must be non-negative")

    boundaries: dict[str, np.ndarray] = {}
    for context, values in checked.items():
        if values.shape[0] < n_splits * 2 + 2 * purge:
            raise ValueError(
                f"context {context!r} is too short for {n_splits} folds and purge={purge}"
            )
        boundaries[context] = np.linspace(
            0, values.shape[0], n_splits + 1, dtype=int
        )

    folds: list[BlockFold] = []
    for fold_index in range(n_splits):
        train_segments: list[TimeSegment] = []
        test_segments: list[TimeSegment] = []
        for context in sorted(checked):
            values = checked[context]
            start = int(boundaries[context][fold_index])
            stop = int(boundaries[context][fold_index + 1])
            if stop - start < 2:
                raise ValueError("every held-out block must contain at least two points")
            test_segments.append(
                TimeSegment(context, values[start:stop], np.arange(start, stop))
            )
            train_mask = np.ones(values.shape[0], dtype=bool)
            train_mask[max(0, start - purge) : min(values.shape[0], stop + purge)] = False
            for run_start, run_stop in _runs(train_mask):
                # Singletons cannot define a transition and are intentionally omitted.
                if run_stop - run_start >= 2:
                    train_segments.append(
                        TimeSegment(
                            context,
                            values[run_start:run_stop],
                            np.arange(run_start, run_stop),
                        )
                    )
            if not any(segment.context == context for segment in train_segments):
                raise ValueError(f"purging leaves no train transition for {context!r}")
        folds.append(
            BlockFold(
                fold=fold_index,
                train=tuple(train_segments),
                test=tuple(test_segments),
                purge=purge,
            )
        )
    return folds


def build_transitions(
    segments: Sequence[TimeSegment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate within-segment transitions and their context labels."""

    current: list[np.ndarray] = []
    following: list[np.ndarray] = []
    labels: list[str] = []
    for segment in segments:
        if not isinstance(segment, TimeSegment):
            raise TypeError("segments must contain TimeSegment objects")
        if segment.values.shape[0] < 2:
            continue
        if np.any(np.diff(segment.indices) != 1):
            raise RuntimeError("non-contiguous segment would create invalid transitions")
        current.append(segment.values[:-1])
        following.append(segment.values[1:])
        labels.extend([segment.context] * (segment.values.shape[0] - 1))
    if not current:
        raise ValueError("segments contain no transitions")
    return np.vstack(current), np.vstack(following), np.asarray(labels, dtype=object)
