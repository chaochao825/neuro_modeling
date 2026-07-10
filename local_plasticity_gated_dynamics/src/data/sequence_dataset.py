"""Loader and leakage-safe preprocessing for public sequence-memory sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

from src.utils.splits import grouped_train_test_split


class SequenceDataError(RuntimeError):
    """Raised when a public sequence session violates the expected contract."""


@dataclass(frozen=True)
class SequenceSession:
    path: Path
    trials: pd.DataFrame
    units: pd.DataFrame
    spike_times: np.ndarray
    spike_key: str

    def __post_init__(self) -> None:
        trials = self.trials.copy(deep=True)
        units = self.units.copy(deep=True)
        spikes = np.asarray(self.spike_times, dtype=object)
        if spikes.dtype != object or spikes.ndim != 2:
            raise SequenceDataError("spike_times must be a trial-by-unit object array")
        if spikes.shape != (len(trials), len(units)):
            raise SequenceDataError(
                "spike matrix shape does not match trials.csv rows and units.csv rows"
            )
        copied = np.empty(spikes.shape, dtype=object)
        for index in np.ndindex(spikes.shape):
            cell = np.asarray(spikes[index], dtype=float).reshape(-1)
            if not np.isfinite(cell).all():
                raise SequenceDataError("spike times must contain only finite values")
            cell = np.sort(cell).copy()
            cell.setflags(write=False)
            copied[index] = cell
        copied.setflags(write=False)
        if not isinstance(self.spike_key, str) or not self.spike_key:
            raise SequenceDataError("spike_key must be a non-empty string")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "spike_times", copied)

    @property
    def session_id(self) -> str:
        return self.path.name


def discover_sequence_sessions(root: str | Path) -> list[Path]:
    """Return sorted session directories containing all three required files."""

    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(base)
    if not base.is_dir():
        raise NotADirectoryError(base)
    required = {"trials.csv", "units.csv", "spikes.mat"}
    sessions = [
        path
        for path in base.rglob("*")
        if path.is_dir() and required.issubset({child.name for child in path.iterdir()})
    ]
    if required.issubset({child.name for child in base.iterdir()}):
        sessions.append(base)
    return sorted(set(sessions))


def _coerce_spike_cell_matrix(value: object) -> np.ndarray | None:
    original = np.asarray(value)
    # A dense numeric matrix is not a MATLAB trial-by-unit cell array.  Without
    # this guard, an unrelated numeric variable can be silently selected.
    if original.dtype != object:
        return None
    array = np.asarray(value, dtype=object)
    if array.ndim == 1:
        # A single trial or unit may be squeezed by MATLAB readers.
        array = array.reshape(1, -1)
    if array.ndim != 2:
        return None
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        try:
            cell = np.asarray(array[index], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(cell).all():
            raise SequenceDataError("spike cell contains non-finite times")
        result[index] = np.sort(cell)
    return result


def _load_v7_mat(path: Path) -> tuple[np.ndarray, str]:
    payload = loadmat(path, simplify_cells=True)
    preferred = ("spikes", "spike_times", "spiketimes", "trial_spikes")
    keys = [*preferred, *(key for key in payload if not key.startswith("__"))]
    seen: set[str] = set()
    for key in keys:
        if key in seen or key not in payload:
            continue
        seen.add(key)
        matrix = _coerce_spike_cell_matrix(payload[key])
        if matrix is not None:
            return matrix, key
    raise SequenceDataError("no trial-by-unit spike cell array found in spikes.mat")


def _hdf5_cell(dataset: h5py.Dataset, handle: h5py.File) -> np.ndarray | None:
    if h5py.check_dtype(ref=dataset.dtype) is None:
        return None
    references = np.asarray(dataset)
    if references.ndim != 2:
        return None
    result = np.empty(references.shape, dtype=object)
    for index in np.ndindex(references.shape):
        reference = references[index]
        if not reference:
            result[index] = np.empty(0, dtype=float)
        else:
            cell = np.asarray(handle[reference], dtype=float).reshape(-1)
            if not np.isfinite(cell).all():
                raise SequenceDataError("HDF5 spike cell contains non-finite times")
            result[index] = np.sort(cell)
    # MATLAB v7.3 stores matrix dimensions in column-major order.
    return result.T


def _load_v73_mat(path: Path) -> tuple[np.ndarray, str]:
    with h5py.File(path, "r") as handle:
        preferred = ("spikes", "spike_times", "spiketimes", "trial_spikes")
        keys = [*preferred, *handle.keys()]
        seen: set[str] = set()
        for key in keys:
            if key in seen or key not in handle:
                continue
            seen.add(key)
            item = handle[key]
            if isinstance(item, h5py.Dataset):
                matrix = _hdf5_cell(item, handle)
                if matrix is not None:
                    return matrix, key
    raise SequenceDataError("no HDF5 trial-by-unit spike cell array found")


def load_sequence_session(path: str | Path) -> SequenceSession:
    """Load `trials.csv`, `units.csv`, and either MATLAB spike format."""

    session_path = Path(path)
    required = [session_path / name for name in ("trials.csv", "units.csv", "spikes.mat")]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError("missing required session files: " + ", ".join(missing))
    trials = pd.read_csv(required[0])
    units = pd.read_csv(required[1])
    if trials.empty or units.empty:
        raise SequenceDataError("trials.csv and units.csv must both be non-empty")
    try:
        spike_times, key = _load_v7_mat(required[2])
    except (NotImplementedError, ValueError, OSError, SequenceDataError) as v7_error:
        try:
            spike_times, key = _load_v73_mat(required[2])
        except (OSError, SequenceDataError) as v73_error:
            raise SequenceDataError(
                f"could not load spikes.mat as v7 ({v7_error}) or v7.3 ({v73_error})"
            ) from v73_error

    if spike_times.shape == (len(units), len(trials)):
        spike_times = spike_times.T
    return SequenceSession(session_path, trials, units, spike_times, key)


def bin_spikes(
    session: SequenceSession,
    *,
    window_s: tuple[float, float],
    bin_size_ms: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin each trial/unit independently; returned time is left-bin time in s."""

    start, stop = map(float, window_s)
    if not np.isfinite([start, stop]).all() or stop <= start:
        raise ValueError("window_s must be finite and increasing")
    if isinstance(bin_size_ms, (bool, np.bool_)) or not isinstance(
        bin_size_ms, (int, np.integer)
    ) or bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be a positive integer")
    width = bin_size_ms / 1000.0
    n_bins_float = (stop - start) / width
    n_bins = int(round(n_bins_float))
    if n_bins < 1 or not np.isclose(n_bins_float, n_bins, atol=1e-9):
        raise ValueError("window duration must be an integer number of bins")
    edges = start + np.arange(n_bins + 1) * width
    counts = np.zeros((len(session.trials), n_bins, len(session.units)), dtype=float)
    for trial in range(len(session.trials)):
        for unit in range(len(session.units)):
            spikes = np.asarray(session.spike_times[trial, unit], dtype=float)
            # np.histogram includes the rightmost edge in its final bin; filter
            # explicitly so every analysis window is consistently [start, stop).
            spikes = spikes[(spikes >= start) & (spikes < stop)]
            counts[trial, :, unit], _ = np.histogram(
                spikes, bins=edges
            )
    return counts, edges[:-1]


def smooth_spike_counts(
    counts: np.ndarray,
    *,
    bin_size_ms: int,
    sigma_ms: float = 50.0,
    mode: Literal["causal", "symmetric"] = "causal",
) -> np.ndarray:
    """Smooth within trials only; causal mode is mandatory for prediction scores."""

    values = np.asarray(counts, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("counts must be finite [trial, time, unit] data")
    if isinstance(bin_size_ms, (bool, np.bool_)) or not isinstance(
        bin_size_ms, (int, np.integer)
    ) or bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be a positive integer")
    if not np.isfinite(sigma_ms) or sigma_ms <= 0:
        raise ValueError("sigma_ms must be positive and finite")
    sigma_bins = sigma_ms / bin_size_ms
    if mode == "symmetric":
        return gaussian_filter1d(values, sigma=sigma_bins, axis=1, mode="nearest")
    if mode != "causal":
        raise ValueError("mode must be causal or symmetric")
    length = max(2, int(np.ceil(5.0 * sigma_bins)))
    lag = np.arange(length, dtype=float)
    kernel = np.exp(-0.5 * (lag / sigma_bins) ** 2)
    kernel /= kernel.sum()
    # lfilter resets at each trial because time is axis 1 and trials are separate rows.
    return lfilter(kernel, [1.0], values, axis=1)


def infer_column(frame: pd.DataFrame, aliases: Sequence[str], *, purpose: str) -> str:
    if not all(isinstance(column, str) for column in frame.columns):
        raise SequenceDataError("all table column names must be strings")
    if (
        isinstance(aliases, (str, bytes))
        or not aliases
        or not all(isinstance(alias, str) for alias in aliases)
    ):
        raise ValueError("aliases must be a non-empty string sequence")
    lower = {column.lower(): column for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise SequenceDataError(
        f"cannot infer {purpose}; expected one of {list(aliases)}, got {list(frame.columns)}"
    )


def block_split_trials(
    trials: pd.DataFrame,
    *,
    block_column: str | None = None,
    contiguous_block_size: int | None = None,
    test_fraction: float = 0.2,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split complete blocks, deriving contiguous blocks only when explicitly asked."""

    if block_column is not None:
        if block_column not in trials:
            raise SequenceDataError(f"missing block column {block_column!r}")
        blocks = trials[block_column].to_numpy(copy=True)
        if pd.isna(blocks).any():
            raise SequenceDataError("block labels must not be missing")
    elif contiguous_block_size is not None:
        if isinstance(contiguous_block_size, (bool, np.bool_)) or not isinstance(
            contiguous_block_size, (int, np.integer)
        ) or contiguous_block_size < 1:
            raise ValueError("contiguous_block_size must be a positive integer")
        blocks = np.arange(len(trials), dtype=int) // contiguous_block_size
    else:
        raise ValueError("provide block_column or contiguous_block_size")
    train, test = grouped_train_test_split(blocks, test_fraction=test_fraction, seed=seed)
    return train, test, blocks


def unseen_combination_split(
    trials: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    seed: int,
    holdout_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[tuple[object, ...]]]:
    """Hold out complete factor combinations while retaining each factor in train."""

    if (
        isinstance(factor_columns, (str, bytes))
        or not factor_columns
        or any(column not in trials for column in factor_columns)
    ):
        raise SequenceDataError("all factor_columns must be present in trials")
    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError("factor_columns must not contain duplicates")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    factor_frame = trials[list(factor_columns)]
    if factor_frame.isna().any().any():
        raise SequenceDataError("factor columns must not contain missing values")
    tuples = [tuple(row) for row in factor_frame.itertuples(index=False, name=None)]
    try:
        unique = sorted(set(tuples), key=repr)
    except TypeError as error:
        raise SequenceDataError("factor values must be hashable scalars") from error
    if len(unique) < 2:
        raise SequenceDataError("at least two unique factor combinations are required")
    all_levels = [
        {combo[factor] for combo in unique} for factor in range(len(factor_columns))
    ]
    rng = np.random.default_rng(seed)
    target_count = max(1, int(np.ceil(holdout_fraction * len(unique))))
    chosen: list[tuple[object, ...]] | None = None
    for _ in range(256):
        candidate_chosen: list[tuple[object, ...]] = []
        for index in rng.permutation(len(unique)):
            candidate = unique[int(index)]
            proposed = {*candidate_chosen, candidate}
            training_combinations = [combo for combo in unique if combo not in proposed]
            if not training_combinations:
                continue
            valid = all(
                {combo[factor] for combo in training_combinations} == all_levels[factor]
                for factor in range(len(factor_columns))
            )
            if valid:
                candidate_chosen.append(candidate)
            if len(candidate_chosen) >= target_count:
                chosen = candidate_chosen
                break
        if chosen is not None:
            break
    if chosen is None:
        raise SequenceDataError(
            "requested holdout_fraction is infeasible while retaining all factor levels"
        )
    chosen_set = set(chosen)
    test_mask = np.array([combo in chosen_set for combo in tuples], dtype=bool)
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask), chosen
