"""Group-aware splits and train-only preprocessing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def _validated_seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    return int(seed)


def _group_key(value: object) -> tuple[str, str, object]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError("group labels cannot be missing")
    if isinstance(value, tuple):
        return "builtins", "tuple", tuple(_group_key(item) for item in value)
    if isinstance(value, (float, complex)) and not np.isfinite(value):
        raise ValueError("group labels must be finite")
    try:
        if not bool(value == value):
            raise ValueError("group labels cannot be missing")
    except (TypeError, ValueError):
        raise ValueError("group labels must be scalar and non-missing") from None
    try:
        hash(value)
    except TypeError as error:
        raise ValueError("group labels must be hashable scalars") from error
    value_type = type(value)
    return value_type.__module__, value_type.__qualname__, value


def _factorize_groups(groups: Iterable[object]) -> np.ndarray:
    values = list(groups)
    if len(values) < 2:
        raise ValueError("groups must be a one-dimensional sequence with at least two samples")
    lookup: dict[tuple[str, str, object], int] = {}
    codes = np.empty(len(values), dtype=int)
    for index, value in enumerate(values):
        key = _group_key(value)
        if key not in lookup:
            lookup[key] = len(lookup)
        codes[index] = lookup[key]
    return codes


def grouped_train_test_split(
    groups: Iterable[object], *, test_fraction: float = 0.2, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole groups, preserving all samples from a trial/block together."""

    codes = _factorize_groups(groups)
    unique = np.unique(codes)
    if unique.size < 2:
        raise ValueError("at least two distinct groups are required")
    if not np.isfinite(test_fraction) or not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    seed = _validated_seed(seed)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_test = min(unique.size - 1, max(1, int(np.ceil(test_fraction * unique.size))))
    test_groups = shuffled[:n_test]
    test_mask = np.isin(codes, test_groups)
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def grouped_kfold(groups: Iterable[object], n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize GroupKFold indices and assert disjoint group membership."""

    if isinstance(n_splits, (bool, np.bool_)) or not isinstance(
        n_splits, (int, np.integer)
    ) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    group_array = _factorize_groups(groups)
    if np.unique(group_array).size < n_splits:
        raise ValueError("n_splits exceeds the number of distinct groups")
    dummy = np.zeros((group_array.size, 1), dtype=float)
    folds = list(GroupKFold(n_splits=n_splits).split(dummy, groups=group_array))
    for train, test in folds:
        if np.intersect1d(group_array[train], group_array[test]).size:
            raise RuntimeError("group leakage detected")
    return folds


@dataclass
class TrainOnlyTransformer:
    """Standardization and optional PCA whose fit provenance is explicit."""

    n_components: int | float | None = None
    with_mean: bool = True
    with_std: bool = True

    def __post_init__(self) -> None:
        if self.n_components is not None:
            if isinstance(self.n_components, (bool, np.bool_)):
                raise TypeError("n_components must be an integer, float, or None")
            if isinstance(self.n_components, (int, np.integer)):
                if self.n_components < 1:
                    raise ValueError("integer n_components must be positive")
            elif isinstance(self.n_components, (float, np.floating)):
                if not np.isfinite(self.n_components) or not 0.0 < self.n_components < 1.0:
                    raise ValueError("float n_components must lie in (0, 1)")
            else:
                raise TypeError("n_components must be an integer, float, or None")
        if not isinstance(self.with_mean, (bool, np.bool_)) or not isinstance(
            self.with_std, (bool, np.bool_)
        ):
            raise TypeError("with_mean and with_std must be boolean")
        self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.pca = (
            PCA(n_components=self.n_components, svd_solver="full")
            if self.n_components is not None
            else None
        )
        self._fit_sample_ids: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, *, sample_ids: Iterable[object] | None = None) -> "TrainOnlyTransformer":
        x_train = np.asarray(x_train, dtype=float)
        if x_train.ndim != 2 or x_train.shape[0] < 2:
            raise ValueError("x_train must be a 2-D array with at least two rows")
        if x_train.shape[1] < 1 or not np.isfinite(x_train).all():
            raise ValueError("x_train must contain finite features")
        ids: np.ndarray | None = None
        if sample_ids is not None:
            if isinstance(sample_ids, (str, bytes)):
                raise TypeError("sample_ids must be a sequence of sample identifiers")
            items = list(sample_ids)
            ids = np.empty(len(items), dtype=object)
            ids[:] = items
            if ids.shape != (x_train.shape[0],):
                raise ValueError("sample_ids length does not match x_train")
        candidate_scaler = StandardScaler(
            with_mean=self.with_mean, with_std=self.with_std
        )
        candidate_pca = (
            PCA(n_components=self.n_components, svd_solver="full")
            if self.n_components is not None
            else None
        )
        scaled = candidate_scaler.fit_transform(x_train)
        if candidate_pca is not None:
            candidate_pca.fit(scaled)
        # Commit only after every train-fitted stage succeeds, leaving a prior
        # valid fit untouched if a refit fails.
        self.scaler = candidate_scaler
        self.pca = candidate_pca
        self._fit_sample_ids = None if ids is None else ids.copy()
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.scaler, "n_features_in_"):
            raise RuntimeError("transformer must be fit on training data first")
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.scaler.n_features_in_:
            raise ValueError("x must be a 2-D array with the fitted feature count")
        if not np.isfinite(values).all():
            raise ValueError("x must contain only finite values")
        scaled = self.scaler.transform(values)
        return self.pca.transform(scaled) if self.pca is not None else scaled

    def fit_transform(
        self, x_train: np.ndarray, *, sample_ids: Iterable[object] | None = None
    ) -> np.ndarray:
        self.fit(x_train, sample_ids=sample_ids)
        return self.transform(x_train)

    @property
    def fit_sample_ids(self) -> np.ndarray | None:
        return None if self._fit_sample_ids is None else self._fit_sample_ids.copy()
