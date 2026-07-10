"""Optional ONE adapter and trial-level preprocessing for IBL sessions.

ONE/ibllib imports are deliberately lazy and confined to this module. Unit
tests use the in-memory source protocol and never access the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from statsmodels.tools.tools import add_constant


class IBLDependencyError(ImportError):
    """Raised when the optional IBL dependency is requested but unavailable."""


class IBLDataError(RuntimeError):
    """Raised when an IBL session lacks required fields or datasets."""


@dataclass(frozen=True)
class ProbeSpikes:
    collection: str
    times: np.ndarray
    clusters: np.ndarray
    unit_ids: np.ndarray
    regions: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.collection, str) or not self.collection:
            raise IBLDataError("collection must be a non-empty string")
        times = np.asarray(self.times, dtype=float)
        raw_clusters = np.asarray(self.clusters)
        raw_unit_ids = np.asarray(self.unit_ids)
        regions = np.asarray(self.regions).astype(str)
        if times.ndim != 1 or raw_clusters.shape != times.shape:
            raise IBLDataError("spike times and cluster assignments must be matching vectors")
        if not np.isfinite(times).all():
            raise IBLDataError("spike times must be finite")
        if np.issubdtype(raw_clusters.dtype, np.bool_) or not np.issubdtype(
            raw_clusters.dtype, np.integer
        ):
            raise IBLDataError("spike cluster assignments must be integers")
        if raw_unit_ids.ndim != 1 or regions.shape != raw_unit_ids.shape:
            raise IBLDataError("unit_ids and regions must be matching vectors")
        if np.issubdtype(raw_unit_ids.dtype, np.bool_) or not np.issubdtype(
            raw_unit_ids.dtype, np.integer
        ):
            raise IBLDataError("unit_ids must be integers")
        unit_ids = raw_unit_ids.astype(int, copy=True)
        if np.any(unit_ids < 0) or np.unique(unit_ids).size != unit_ids.size:
            raise IBLDataError("unit_ids must be unique within a probe")
        order = np.argsort(times, kind="stable")
        copies = {
            "times": np.array(times[order], dtype=float, copy=True),
            "clusters": np.array(raw_clusters[order], dtype=int, copy=True),
            "unit_ids": unit_ids,
            "regions": np.array(regions, dtype=str, copy=True),
        }
        for name, values in copies.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)


class IBLSessionSource(Protocol):
    """Minimal source interface implemented by ONE and in-memory test fakes."""

    def search_sessions(self, *, limit: int) -> list[str]: ...

    def load_trials(self, eid: str) -> Mapping[str, Any]: ...

    def load_probe_spikes(self, eid: str) -> list[ProbeSpikes]: ...

    def load_wheel(self, eid: str) -> Mapping[str, Any] | None: ...

    def load_pose_summary(
        self,
        eid: str,
        events: np.ndarray,
        *,
        window_s: tuple[float, float] = (-0.5, 0.0),
    ) -> np.ndarray | None: ...

    def session_details(self, eid: str) -> Mapping[str, Any]: ...


def _field(mapping: Mapping[str, Any], *names: str, required: bool = True) -> np.ndarray | None:
    for name in names:
        if name in mapping:
            return np.asarray(mapping[name])
    if required:
        raise IBLDataError(f"missing trial field; expected one of {names}")
    return None


def contiguous_context_blocks(probability_left: Sequence[float]) -> np.ndarray:
    """Label contiguous probability blocks without treating equal distant blocks alike."""

    probability = np.asarray(probability_left, dtype=float)
    if probability.ndim != 1 or probability.size == 0 or not np.isfinite(probability).all():
        raise ValueError("probability_left must be a non-empty finite vector")
    blocks = np.zeros(probability.size, dtype=int)
    for index in range(1, probability.size):
        same = np.isclose(probability[index], probability[index - 1])
        blocks[index] = blocks[index - 1] + int(not same)
    return blocks


def _wheel_displacement(
    timestamps: np.ndarray,
    position: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    result = np.full(starts.shape, np.nan, dtype=float)
    if timestamps.ndim != 1 or position.shape != timestamps.shape:
        raise IBLDataError("wheel timestamps and position must be matching vectors")
    if not np.isfinite(timestamps).all() or not np.isfinite(position).all():
        raise IBLDataError("wheel timestamps and position must be finite")
    if np.any(np.diff(timestamps) < 0):
        raise IBLDataError("wheel timestamps must be sorted")
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            continue
        mask = (timestamps >= start) & (timestamps <= stop)
        samples = position[mask]
        if samples.size >= 2:
            result[index] = float(np.sum(np.abs(np.diff(samples))))
    return result


def build_trial_covariates(
    trials: Mapping[str, Any],
    *,
    wheel: Mapping[str, Any] | None = None,
    pose_summary: np.ndarray | None = None,
) -> pd.DataFrame:
    """Construct the preregistered nuisance table without silent imputation."""

    stim_on = np.asarray(_field(trials, "stimOn_times", "stim_on_times"), dtype=float)
    movement = np.asarray(
        _field(trials, "firstMovement_times", "first_movement_times"), dtype=float
    )
    n_trials = stim_on.size
    if stim_on.ndim != 1 or movement.shape != (n_trials,):
        raise IBLDataError("event fields must be matching trial vectors")
    contrast_left = np.asarray(
        _field(trials, "contrastLeft", "contrast_left"), dtype=float
    )
    contrast_right = np.asarray(
        _field(trials, "contrastRight", "contrast_right"), dtype=float
    )
    if contrast_left.shape != (n_trials,) or contrast_right.shape != (n_trials,):
        raise IBLDataError("trial contrast fields have inconsistent lengths")
    exactly_one_finite = np.logical_xor(
        np.isfinite(contrast_left), np.isfinite(contrast_right)
    )
    if not exactly_one_finite.all() or np.any(
        np.isinf(contrast_left) | np.isinf(contrast_right)
    ):
        raise IBLDataError(
            "each trial must have exactly one finite left/right stimulus contrast"
        )
    stimulus = np.nan_to_num(contrast_right, nan=0.0) - np.nan_to_num(
        contrast_left, nan=0.0
    )
    choice = np.asarray(_field(trials, "choice"), dtype=float)
    reward = _field(trials, "feedbackType", "rewardVolume", "reward", required=False)
    reward_array = (
        np.full(n_trials, np.nan) if reward is None else np.asarray(reward, dtype=float)
    )
    probability = np.asarray(
        _field(trials, "probabilityLeft", "probability_left"), dtype=float
    )
    response = _field(trials, "response_times", "responseTimes", required=False)
    response_array = movement if response is None else np.asarray(response, dtype=float)
    jointly_finite = np.isfinite(stim_on) & np.isfinite(movement)
    timing_valid = jointly_finite & (movement >= stim_on)
    # Early-movement and missing-event trials occur in public IBL sessions.
    # Mark their reaction time missing so the preregistered complete-case mask
    # excludes the individual trials instead of discarding the whole session.
    reaction_time = movement - stim_on
    reaction_time = np.asarray(reaction_time, dtype=float)
    reaction_time[~timing_valid] = np.nan
    required_vectors = {
        "contrastLeft": contrast_left,
        "contrastRight": contrast_right,
        "choice": choice,
        "reward": reward_array,
        "probabilityLeft": probability,
        "response_times": response_array,
    }
    for name, values in required_vectors.items():
        if values.shape != (n_trials,):
            raise IBLDataError(f"trial field {name} has inconsistent length")

    if wheel is None:
        wheel_movement = np.full(n_trials, np.nan)
    else:
        wheel_movement = _wheel_displacement(
            np.asarray(_field(wheel, "timestamps", "times"), dtype=float),
            np.asarray(_field(wheel, "position"), dtype=float),
            stim_on,
            response_array,
        )
    if pose_summary is None:
        pose = np.full(n_trials, np.nan)
    else:
        pose = np.asarray(pose_summary, dtype=float)
        if pose.shape != (n_trials,):
            raise IBLDataError("pose summary must have one value per trial")

    return pd.DataFrame(
        {
            "stimulus": stimulus,
            "choice": choice,
            "wheel": wheel_movement,
            "reward": reward_array,
            "reaction_time": reaction_time,
            "pose": pose,
            "probability_left": probability,
            "stim_on": stim_on,
            "first_movement": movement,
            "timing_valid": timing_valid,
            "block_id": contiguous_context_blocks(probability),
        }
    )


def concatenate_probe_spikes(probes: Sequence[ProbeSpikes]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate probes while remapping cluster labels to unique dense unit ids."""

    if not probes:
        raise IBLDataError("no probe spike sorting found")
    all_times: list[np.ndarray] = []
    all_clusters: list[np.ndarray] = []
    unit_labels: list[str] = []
    regions: list[str] = []
    offset = 0
    collections: set[str] = set()
    for probe in probes:
        if not isinstance(probe, ProbeSpikes):
            raise TypeError("probes must contain ProbeSpikes instances")
        if probe.collection in collections:
            raise IBLDataError(f"duplicate probe collection {probe.collection!r}")
        collections.add(probe.collection)
        cluster_to_local = {int(cluster): index for index, cluster in enumerate(probe.unit_ids)}
        keep = np.array([int(cluster) in cluster_to_local for cluster in probe.clusters], dtype=bool)
        mapped = np.array(
            [cluster_to_local[int(cluster)] + offset for cluster in probe.clusters[keep]],
            dtype=int,
        )
        all_times.append(np.asarray(probe.times[keep], dtype=float))
        all_clusters.append(mapped)
        unit_labels.extend(f"{probe.collection}:{cluster}" for cluster in probe.unit_ids)
        regions.extend(str(region) for region in probe.regions)
        offset += probe.unit_ids.size
    times = np.concatenate(all_times)
    clusters = np.concatenate(all_clusters)
    order = np.argsort(times, kind="stable")
    return times[order], clusters[order], np.asarray(unit_labels), np.asarray(regions)


def event_aligned_spike_counts(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    events: np.ndarray,
    *,
    n_units: int,
    window_s: tuple[float, float] = (-0.5, 0.0),
    bin_size_ms: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin pre-event activity independently for every trial."""

    times = np.asarray(spike_times, dtype=float)
    raw_clusters = np.asarray(spike_clusters)
    event_array = np.asarray(events, dtype=float)
    if times.ndim != 1 or raw_clusters.shape != times.shape:
        raise ValueError("spike_times and spike_clusters must be matching vectors")
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("spike_times must be finite and sorted")
    if np.issubdtype(raw_clusters.dtype, np.bool_) or not np.issubdtype(
        raw_clusters.dtype, np.integer
    ):
        raise ValueError("spike_clusters must contain integers")
    clusters = raw_clusters.astype(int, copy=False)
    if (
        isinstance(n_units, (bool, np.bool_))
        or not isinstance(n_units, (int, np.integer))
        or n_units < 1
        or np.any(clusters < 0)
        or np.any(clusters >= n_units)
    ):
        raise ValueError("cluster labels must lie in [0, n_units)")
    if event_array.ndim != 1:
        raise ValueError("events must be a one-dimensional trial vector")
    start, stop = map(float, window_s)
    if not np.isfinite([start, stop]).all():
        raise ValueError("window bounds must be finite")
    if isinstance(bin_size_ms, (bool, np.bool_)) or not isinstance(
        bin_size_ms, (int, np.integer)
    ) or bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be a positive integer")
    width = bin_size_ms / 1000.0
    n_bins_float = (stop - start) / width
    n_bins = int(round(n_bins_float))
    if start >= stop or n_bins < 1 or not np.isclose(n_bins_float, n_bins):
        raise ValueError("window must be increasing and divisible by bin size")
    relative_edges = start + np.arange(n_bins + 1) * width
    counts = np.zeros((event_array.size, n_bins, n_units), dtype=float)
    valid = np.isfinite(event_array)
    for trial in np.flatnonzero(valid):
        event = event_array[trial]
        left = np.searchsorted(times, event + start, side="left")
        right = np.searchsorted(times, event + stop, side="left")
        relative = times[left:right] - event
        bins = np.searchsorted(relative_edges, relative, side="right") - 1
        in_range = (bins >= 0) & (bins < n_bins)
        np.add.at(counts[trial], (bins[in_range], clusters[left:right][in_range]), 1.0)
    return counts, relative_edges[:-1], valid


def _camera_pose_motion_summary(
    camera: Mapping[str, Any],
    events: np.ndarray,
    *,
    window_s: tuple[float, float] = (-0.5, 0.0),
    likelihood_threshold: float = 0.9,
) -> np.ndarray:
    """Summarize DLC coordinate motion in a half-open pre-event window."""

    if "times" not in camera or "dlc" not in camera:
        raise IBLDataError("camera object must contain times and dlc")
    times = np.asarray(camera["times"], dtype=float)
    event_array = np.asarray(events, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise IBLDataError("camera times must be a finite strictly increasing vector")
    if event_array.ndim != 1:
        raise IBLDataError("pose events must be a one-dimensional trial vector")
    start, stop = map(float, window_s)
    if not np.isfinite([start, stop]).all() or stop <= start or stop > 0.0:
        raise ValueError("pose window must be finite, increasing, and pre-event")
    if not np.isfinite(likelihood_threshold) or not 0.0 <= likelihood_threshold <= 1.0:
        raise ValueError("likelihood_threshold must be in [0, 1]")

    dlc_value = camera["dlc"]
    if isinstance(dlc_value, pd.DataFrame):
        dlc = dlc_value.copy(deep=True)
    elif isinstance(dlc_value, Mapping):
        dlc = pd.DataFrame(dict(dlc_value))
    else:
        raw = np.asarray(dlc_value)
        if raw.dtype.names:
            dlc = pd.DataFrame.from_records(raw)
        else:
            raise IBLDataError("camera dlc must be a DataFrame, mapping, or structured array")
    if len(dlc) != times.size:
        raise IBLDataError("camera dlc rows must match camera times")
    coordinate_columns = [
        column
        for column in dlc.columns
        if isinstance(column, str) and (column.endswith("_x") or column.endswith("_y"))
    ]
    if not coordinate_columns:
        raise IBLDataError("camera dlc contains no coordinate columns")
    coordinates = dlc.loc[:, coordinate_columns].apply(pd.to_numeric, errors="coerce")
    for column_index, column in enumerate(coordinate_columns):
        feature = column[:-2]
        likelihood_column = f"{feature}_likelihood"
        if likelihood_column in dlc:
            likelihood = pd.to_numeric(dlc[likelihood_column], errors="coerce").to_numpy()
            low_quality = ~np.isfinite(likelihood) | (likelihood < likelihood_threshold)
            coordinates.iloc[low_quality, column_index] = np.nan
    values = coordinates.to_numpy(dtype=float)
    deltas = np.diff(values, axis=0) / np.diff(times)[:, None]
    valid_counts = np.sum(np.isfinite(deltas), axis=1)
    squared_sum = np.nansum(np.square(deltas), axis=1)
    motion = np.full(times.size - 1, np.nan, dtype=float)
    valid_motion = valid_counts > 0
    motion[valid_motion] = np.sqrt(squared_sum[valid_motion] / valid_counts[valid_motion])
    motion_times = times[1:]
    summary = np.full(event_array.shape, np.nan, dtype=float)
    for trial in np.flatnonzero(np.isfinite(event_array)):
        event = event_array[trial]
        selected = (motion_times >= event + start) & (motion_times < event + stop)
        if np.any(selected) and np.isfinite(motion[selected]).any():
            summary[trial] = float(np.nanmean(motion[selected]))
    return summary


@dataclass(frozen=True)
class IBLTrialData:
    eid: str
    animal_id: str
    covariates: pd.DataFrame
    view_covariates: Mapping[str, pd.DataFrame]
    activity: Mapping[str, np.ndarray]
    valid_masks: Mapping[str, np.ndarray]
    time_axes: Mapping[str, np.ndarray]
    unit_ids: np.ndarray
    regions: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.eid, str) or not self.eid:
            raise IBLDataError("eid must be a non-empty string")
        if not isinstance(self.animal_id, str) or not self.animal_id:
            raise IBLDataError("animal_id must be a non-empty string")
        covariates = self.covariates.copy(deep=True)
        unit_ids = np.asarray(self.unit_ids).astype(str)
        regions = np.asarray(self.regions).astype(str)
        if unit_ids.ndim != 1 or regions.shape != unit_ids.shape:
            raise IBLDataError("unit_ids and regions must be matching vectors")
        if np.any(unit_ids == "") or np.unique(unit_ids).size != unit_ids.size:
            raise IBLDataError("unit_ids must be unique non-empty strings")
        n_trials = len(covariates)
        activity: dict[str, np.ndarray] = {}
        valid_masks: dict[str, np.ndarray] = {}
        time_axes: dict[str, np.ndarray] = {}
        if set(self.activity) != set(self.valid_masks) or set(self.activity) != set(
            self.time_axes
        ):
            raise IBLDataError("activity, valid_masks, and time_axes must share keys")
        if set(self.view_covariates) != set(self.activity):
            raise IBLDataError("view_covariates must have one table per activity view")
        view_covariates: dict[str, pd.DataFrame] = {}
        for name, table in self.view_covariates.items():
            if not isinstance(table, pd.DataFrame) or len(table) != n_trials:
                raise IBLDataError(
                    f"view covariates {name!r} must be a per-trial DataFrame"
                )
            if tuple(table.columns) != tuple(covariates.columns):
                raise IBLDataError(
                    f"view covariates {name!r} must share the base covariate schema"
                )
            view_covariates[name] = table.copy(deep=True)
        for name in self.activity:
            values = np.array(self.activity[name], dtype=float, order="C", copy=True)
            mask_raw = np.asarray(self.valid_masks[name])
            axis = np.array(self.time_axes[name], dtype=float, order="C", copy=True)
            if values.ndim != 3 or values.shape[0] != n_trials or values.shape[2] != unit_ids.size:
                raise IBLDataError(f"activity view {name!r} has inconsistent shape")
            if not np.isfinite(values).all():
                raise IBLDataError(f"activity view {name!r} contains non-finite values")
            if mask_raw.shape != (n_trials,) or (
                mask_raw.dtype != bool and not np.isin(mask_raw, [0, 1]).all()
            ):
                raise IBLDataError(f"valid mask {name!r} must be binary per trial")
            if axis.shape != (values.shape[1],) or not np.isfinite(axis).all():
                raise IBLDataError(f"time axis {name!r} has inconsistent shape")
            mask = np.array(mask_raw, dtype=bool, copy=True)
            for array in (values, mask, axis):
                array.setflags(write=False)
            activity[name], valid_masks[name], time_axes[name] = values, mask, axis
        unit_copy = np.array(unit_ids, dtype=str, copy=True)
        region_copy = np.array(regions, dtype=str, copy=True)
        unit_copy.setflags(write=False)
        region_copy.setflags(write=False)
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "view_covariates", MappingProxyType(view_covariates))
        object.__setattr__(self, "activity", MappingProxyType(activity))
        object.__setattr__(self, "valid_masks", MappingProxyType(valid_masks))
        object.__setattr__(self, "time_axes", MappingProxyType(time_axes))
        object.__setattr__(self, "unit_ids", unit_copy)
        object.__setattr__(self, "regions", region_copy)


def load_ibl_trial_data(
    source: IBLSessionSource,
    eid: str,
    *,
    bin_size_ms: int = 20,
    pre_window_s: tuple[float, float] = (-0.5, 0.0),
) -> IBLTrialData:
    trials = source.load_trials(eid)
    stim_on = np.asarray(_field(trials, "stimOn_times", "stim_on_times"), dtype=float)
    movement = np.asarray(
        _field(trials, "firstMovement_times", "first_movement_times"), dtype=float
    )
    wheel = source.load_wheel(eid)
    event_views = (("stimulus_pre", stim_on), ("movement_pre", movement))
    view_covariates = {
        name: build_trial_covariates(
            trials,
            wheel=wheel,
            pose_summary=source.load_pose_summary(
                eid, events, window_s=pre_window_s
            ),
        )
        for name, events in event_views
    }
    # Retain the original stimulus-aligned table for compatibility; analyses
    # must select view_covariates[view] for view-specific nuisance regression.
    covariates = view_covariates["stimulus_pre"]
    times, clusters, unit_ids, regions = concatenate_probe_spikes(
        source.load_probe_spikes(eid)
    )
    activity: dict[str, np.ndarray] = {}
    valid_masks: dict[str, np.ndarray] = {}
    time_axes: dict[str, np.ndarray] = {}
    for name, events in event_views:
        counts, axis, valid = event_aligned_spike_counts(
            times,
            clusters,
            events,
            n_units=unit_ids.size,
            window_s=pre_window_s,
            bin_size_ms=bin_size_ms,
        )
        activity[name] = counts
        valid_masks[name] = valid
        time_axes[name] = axis
    details = source.session_details(eid)
    animal_value = details.get("subject", details.get("animal"))
    if animal_value is None or not str(animal_value).strip():
        raise IBLDataError(
            "session details must include subject/animal for animal-level statistics"
        )
    animal = str(animal_value)
    return IBLTrialData(
        eid=str(eid),
        animal_id=animal,
        covariates=covariates,
        view_covariates=view_covariates,
        activity=activity,
        valid_masks=valid_masks,
        time_axes=time_axes,
        unit_ids=unit_ids,
        regions=regions,
    )


class TrialNuisanceResidualizer:
    """Vectorized OLS residualizer with train-fitted design columns only."""

    def __init__(self, columns: Sequence[str]) -> None:
        if (
            isinstance(columns, (str, bytes))
            or not columns
            or not all(isinstance(column, str) and column for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError("at least one nuisance column is required")
        self.columns = tuple(columns)
        self._fitted = False

    def _design(self, covariates: pd.DataFrame, *, fitting: bool) -> np.ndarray:
        missing = [column for column in self.columns if column not in covariates]
        if missing:
            raise IBLDataError(f"missing nuisance columns: {missing}")
        selected = covariates.loc[:, self.columns].copy()
        if selected.isna().any().any():
            missing_columns = selected.columns[selected.isna().any()].tolist()
            raise IBLDataError(
                f"nuisance covariates contain missing values: {missing_columns}; "
                "load the required datasets or explicitly filter trials"
            )
        encoded = pd.get_dummies(selected, drop_first=False, dtype=float)
        if fitting:
            self.design_columns_ = tuple(encoded.columns)
        else:
            encoded = encoded.reindex(columns=self.design_columns_, fill_value=0.0)
        design = np.asarray(
            add_constant(encoded.to_numpy(dtype=float), prepend=True, has_constant="add"),
            dtype=float,
        )
        if not np.isfinite(design).all():
            raise IBLDataError("nuisance design contains non-finite values")
        return design

    def fit(
        self,
        covariates_train: pd.DataFrame,
        activity_train: np.ndarray,
        *,
        sample_ids: Sequence[object] | None = None,
    ) -> "TrialNuisanceResidualizer":
        self._fitted = False
        values = np.asarray(activity_train, dtype=float)
        if values.ndim < 2 or values.shape[0] != len(covariates_train):
            raise ValueError("activity must start with the trial dimension")
        if not np.isfinite(values).all():
            raise ValueError("activity contains non-finite values")
        if sample_ids is None:
            fit_ids = None
        else:
            if isinstance(sample_ids, (str, bytes)):
                raise TypeError("sample_ids must be a sequence of trial identifiers")
            items = list(sample_ids)
            fit_ids = np.empty(len(items), dtype=object)
            fit_ids[:] = items
            if fit_ids.shape != (values.shape[0],):
                raise ValueError("sample_ids must match training trials")
        design = self._design(covariates_train, fitting=True)
        flattened = values.reshape(values.shape[0], -1)
        coefficients = np.linalg.pinv(design) @ flattened
        coefficients.setflags(write=False)
        self.coefficients_ = coefficients
        self.feature_shape_ = values.shape[1:]
        if fit_ids is not None:
            fit_ids.setflags(write=False)
        self.fit_sample_ids_ = fit_ids
        self._fitted = True
        return self

    def transform(self, covariates: pd.DataFrame, activity: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("residualizer must be fit on training trials first")
        values = np.asarray(activity, dtype=float)
        if values.shape[0] != len(covariates) or values.shape[1:] != self.feature_shape_:
            raise ValueError("activity shape differs from fitted data")
        if not np.isfinite(values).all():
            raise ValueError("activity contains non-finite values")
        design = self._design(covariates, fitting=False)
        predicted = design @ self.coefficients_
        return (values.reshape(values.shape[0], -1) - predicted).reshape(values.shape)


class CachedIBLSessionSource:
    """Read a small, explicitly mapped set of IBL sessions from a ONE cache.

    The source never constructs a :class:`one.api.ONE` instance and therefore
    never contacts Alyx.  ``one.alf.io`` is imported lazily on the first data
    access.  Session paths must be relative to ``cache_dir`` and are resolved
    again on every access so that symlink replacement cannot escape the cache.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        session_paths: Mapping[str, str | Path],
    ) -> None:
        root = Path(cache_dir).expanduser()
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as error:
            raise IBLDataError(f"IBL cache directory does not exist: {root}") from error
        if not resolved_root.is_dir():
            raise IBLDataError(f"IBL cache path is not a directory: {resolved_root}")
        if not isinstance(session_paths, Mapping) or not 1 <= len(session_paths) <= 5:
            raise ValueError("session_paths must map 1-5 session ids to relative paths")

        checked: dict[str, Path] = {}
        for eid, raw_path in session_paths.items():
            if not isinstance(eid, str) or not eid.strip():
                raise ValueError("session ids must be non-empty strings")
            if not isinstance(raw_path, (str, Path)):
                raise TypeError("session paths must be strings or pathlib.Path instances")
            relative = Path(raw_path)
            if relative.is_absolute() or relative == Path("."):
                raise ValueError("session paths must be non-empty paths relative to cache_dir")
            candidate = resolved_root / relative
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(resolved_root)
            except (OSError, ValueError) as error:
                raise IBLDataError(
                    f"session {eid!r} is missing or escapes cache_dir: {relative}"
                ) from error
            if not resolved.is_dir():
                raise IBLDataError(f"session {eid!r} path is not a directory: {relative}")
            checked[eid] = relative
        self.cache_dir = resolved_root
        self.session_paths = MappingProxyType(checked)

    @staticmethod
    def _alf_io() -> Any:
        try:
            from one.alf import io as alf_io
        except ImportError as error:
            raise IBLDependencyError(
                "cached IBL access requires ONE-api: pip install '.[ibl]'"
            ) from error
        return alf_io

    def _session_path(self, eid: str) -> Path:
        if eid not in self.session_paths:
            raise IBLDataError(f"session {eid!r} is not present in the local cache mapping")
        relative = self.session_paths[eid]
        try:
            path = (self.cache_dir / relative).resolve(strict=True)
            path.relative_to(self.cache_dir)
        except (OSError, ValueError) as error:
            raise IBLDataError(
                f"session {eid!r} is missing or escapes cache_dir: {relative}"
            ) from error
        if not path.is_dir():
            raise IBLDataError(f"session {eid!r} path is not a directory: {relative}")
        return path

    @staticmethod
    def _resolve_within(path: Path, root: Path, kind: str) -> Path:
        resolved_root = root
        try:
            resolved_root = root.resolve(strict=True)
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise IBLDataError(f"{kind} is missing or escapes {resolved_root}") from error
        return resolved

    def _alf_path(self, eid: str) -> Path:
        session_path = self._session_path(eid)
        alf_path = self._resolve_within(
            session_path / "alf", session_path, f"session {eid!r} alf directory"
        )
        if not alf_path.is_dir():
            raise IBLDataError(f"session {eid!r} has no local alf directory")
        return alf_path

    @staticmethod
    def _is_revision(path: Path) -> bool:
        return len(path.name) > 2 and path.name.startswith("#") and path.name.endswith("#")

    @classmethod
    def _latest_table_directory(cls, alf_path: Path, filename: str) -> Path | None:
        candidates: list[tuple[Path, bool, str]] = []
        base_file = alf_path / filename
        if base_file.exists() or base_file.is_symlink():
            resolved_file = cls._resolve_within(base_file, alf_path, "ALF dataset")
            if resolved_file.is_file():
                candidates.append((alf_path, False, ""))
        for raw_directory in alf_path.iterdir():
            if not cls._is_revision(raw_directory):
                continue
            directory = cls._resolve_within(
                raw_directory, alf_path, "ALF revision directory"
            )
            if not directory.is_dir():
                continue
            dataset = directory / filename
            if not dataset.exists() and not dataset.is_symlink():
                continue
            resolved_dataset = cls._resolve_within(dataset, alf_path, "ALF dataset")
            if resolved_dataset.is_file():
                candidates.append((directory, True, raw_directory.name))
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate[1:])[0]

    @classmethod
    def _has_object(
        cls, directory: Path, object_name: str, containment_root: Path
    ) -> bool:
        found = False
        for path in directory.iterdir():
            matches = path.name.startswith(f"{object_name}.") or (
                f"_{object_name}." in path.name
            )
            if not matches:
                continue
            resolved = cls._resolve_within(path, containment_root, "ALF dataset")
            found = found or resolved.is_file()
        return found

    @classmethod
    def _load_object_directory(
        cls,
        directory: Path,
        object_name: str,
        *,
        containment_root: Path,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        directory = cls._resolve_within(
            directory, containment_root, "ALF object directory"
        )
        if not directory.is_dir() or not cls._has_object(
            directory, object_name, containment_root
        ):
            return {}
        kwargs = {} if namespace is None else {"namespace": namespace}
        try:
            loaded = cls._alf_io().load_object(directory, object_name, **kwargs)
        except IBLDependencyError:
            raise
        except Exception as error:
            raise IBLDataError(
                f"failed to load local ALF object {object_name!r} from {directory}"
            ) from error
        return dict(loaded)

    @classmethod
    def _load_base_and_revision(
        cls,
        base_directory: Path,
        selected_directory: Path,
        object_name: str,
        *,
        containment_root: Path,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        merged = cls._load_object_directory(
            base_directory,
            object_name,
            containment_root=containment_root,
            namespace=namespace,
        )
        if selected_directory != base_directory:
            merged.update(
                cls._load_object_directory(
                    selected_directory,
                    object_name,
                    containment_root=containment_root,
                    namespace=namespace,
                )
            )
        return merged

    def search_sessions(self, *, limit: int) -> list[str]:
        if isinstance(limit, (bool, np.bool_)) or not isinstance(
            limit, (int, np.integer)
        ) or not 1 <= limit <= 5:
            raise ValueError("initial IBL analysis is restricted to 1-5 sessions")
        return list(self.session_paths)[: int(limit)]

    def load_trials(self, eid: str) -> Mapping[str, Any]:
        alf_path = self._alf_path(eid)
        selected = self._latest_table_directory(alf_path, "_ibl_trials.table.pqt")
        if selected is None:
            raise IBLDataError(
                f"session {eid!r} has no complete _ibl_trials.table.pqt in its local cache"
            )
        trials = self._load_base_and_revision(
            alf_path,
            selected,
            "trials",
            containment_root=alf_path,
            namespace="ibl",
        )
        if not trials:
            raise IBLDataError(f"session {eid!r} local trials object is empty")
        return trials

    @classmethod
    def _logical_probe_directory(cls, directory: Path, alf_path: Path) -> Path:
        logical_raw = directory.parent if cls._is_revision(directory) else directory
        logical = cls._resolve_within(
            logical_raw, alf_path, "probe collection directory"
        )
        source = cls._resolve_within(directory, alf_path, "probe source directory")
        try:
            source.relative_to(logical)
        except ValueError as error:
            raise IBLDataError("probe source escaped its logical collection") from error
        return logical

    @staticmethod
    def _probe_name(logical_directory: Path, alf_path: Path) -> str | None:
        for part in logical_directory.relative_to(alf_path).parts:
            if part.startswith("probe"):
                return part
        return None

    @classmethod
    def _complete_spike_directories(cls, alf_path: Path) -> dict[Path, list[Path]]:
        by_collection: dict[Path, list[Path]] = {}
        for times_file in alf_path.rglob("spikes.times.npy"):
            directory = cls._resolve_within(
                times_file.parent, alf_path, "probe source directory"
            )
            resolved_times = cls._resolve_within(
                times_file, directory, "probe spikes.times dataset"
            )
            clusters_file = times_file.parent / "spikes.clusters.npy"
            if not clusters_file.exists() and not clusters_file.is_symlink():
                continue
            resolved_clusters = cls._resolve_within(
                clusters_file, directory, "probe spikes.clusters dataset"
            )
            if not resolved_times.is_file() or not resolved_clusters.is_file():
                continue
            logical = cls._logical_probe_directory(times_file.parent, alf_path)
            if cls._probe_name(logical, alf_path) is not None:
                by_collection.setdefault(logical, []).append(directory)
        return by_collection

    @classmethod
    def _selected_probe_directories(cls, alf_path: Path) -> list[tuple[Path, Path]]:
        candidates = cls._complete_spike_directories(alf_path)
        by_probe: dict[str, list[Path]] = {}
        for logical in candidates:
            probe_name = cls._probe_name(logical, alf_path)
            if probe_name is not None:
                by_probe.setdefault(probe_name, []).append(logical)
        selected: list[tuple[Path, Path]] = []
        for probe_name in sorted(by_probe):
            logical = max(
                by_probe[probe_name],
                key=lambda path: (
                    len(path.relative_to(alf_path).parts),
                    path.as_posix(),
                ),
            )
            source = max(
                candidates[logical],
                key=lambda path: (
                    path != logical,
                    path.name if path != logical else "",
                ),
            )
            selected.append((logical, source))
        return selected

    def load_probe_spikes(self, eid: str) -> list[ProbeSpikes]:
        alf_path = self._alf_path(eid)
        selected = self._selected_probe_directories(alf_path)
        if not selected:
            raise IBLDataError(
                f"session {eid!r} has no complete local probe spike sorting"
            )
        probes: list[ProbeSpikes] = []
        for logical, source in selected:
            spikes = self._load_base_and_revision(
                logical, source, "spikes", containment_root=alf_path
            )
            clusters_object = self._load_base_and_revision(
                logical, source, "clusters", containment_root=alf_path
            )
            if "times" not in spikes or "clusters" not in spikes:
                raise IBLDataError(
                    f"incomplete local spikes object in {source}: times/clusters required"
                )
            spike_clusters = np.asarray(spikes["clusters"])
            unit_ids_value = clusters_object.get("cluster_id")
            if unit_ids_value is None:
                channels = np.asarray(clusters_object.get("channels", []))
                unit_ids = (
                    np.arange(channels.size, dtype=int)
                    if channels.size
                    else np.unique(spike_clusters).astype(int, copy=False)
                )
            else:
                unit_ids = np.asarray(unit_ids_value, dtype=int)
            acronym = clusters_object.get(
                "acronyms", clusters_object.get("acronym")
            )
            regions = (
                np.asarray(acronym).astype(str)
                if acronym is not None and len(acronym) == unit_ids.size
                else np.repeat("unknown", unit_ids.size)
            )
            collection = "alf/" + logical.relative_to(alf_path).as_posix()
            probes.append(
                ProbeSpikes(
                    collection,
                    np.asarray(spikes["times"], dtype=float),
                    spike_clusters,
                    unit_ids,
                    regions,
                )
            )
        return probes

    @classmethod
    def _latest_object_revision(cls, alf_path: Path, object_name: str) -> Path:
        revisions: list[tuple[Path, str]] = []
        for raw_directory in alf_path.iterdir():
            if not cls._is_revision(raw_directory):
                continue
            directory = cls._resolve_within(
                raw_directory, alf_path, "ALF revision directory"
            )
            if directory.is_dir() and cls._has_object(
                directory, object_name, alf_path
            ):
                revisions.append((directory, raw_directory.name))
        return max(revisions, key=lambda item: item[1])[0] if revisions else alf_path

    def load_wheel(self, eid: str) -> Mapping[str, Any] | None:
        alf_path = self._alf_path(eid)
        selected = self._latest_object_revision(alf_path, "wheel")
        wheel = self._load_base_and_revision(
            alf_path,
            selected,
            "wheel",
            containment_root=alf_path,
            namespace="ibl",
        )
        if not wheel:
            return None
        if "timestamps" not in wheel or "position" not in wheel:
            raise IBLDataError(
                f"session {eid!r} local wheel object requires timestamps and position"
            )
        return wheel

    def _load_camera(self, eid: str, label: str) -> Mapping[str, Any] | None:
        alf_path = self._alf_path(eid)
        object_name = f"{label}Camera"
        filename = f"_ibl_{object_name}.dlc.pqt"
        selected = self._latest_table_directory(alf_path, filename)
        if selected is None:
            return None
        camera = self._load_base_and_revision(
            alf_path,
            selected,
            object_name,
            containment_root=alf_path,
            namespace="ibl",
        )
        if "times" not in camera or "dlc" not in camera:
            raise IBLDataError(
                f"session {eid!r} local {object_name} requires times and dlc"
            )
        return camera

    def load_pose_summary(
        self,
        eid: str,
        events: np.ndarray,
        *,
        window_s: tuple[float, float] = (-0.5, 0.0),
    ) -> np.ndarray | None:
        for label in ("left", "right", "body"):
            camera = self._load_camera(eid, label)
            if camera is not None:
                return _camera_pose_motion_summary(camera, events, window_s=window_s)
        return None

    def session_details(self, eid: str) -> Mapping[str, Any]:
        session_path = self._session_path(eid)
        parts = session_path.relative_to(self.cache_dir).parts
        for index, part in enumerate(parts[:-1]):
            if part.lower() == "subjects" and index + 1 < len(parts):
                subject = parts[index + 1]
                if subject:
                    return {"subject": subject}
        raise IBLDataError(
            f"session {eid!r} path does not contain a Subjects/<subject> component"
        )


class OneAPISource:
    """Thin, lazy ONE wrapper; all optional imports remain inside IBL code."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        base_url: str = "https://openalyx.internationalbrainlab.org",
        username: str | None = "intbrainlab",
        password: str | None = "international",
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if (username is None) != (password is None):
            raise ValueError("username and password must either both be supplied or both be None")
        if username is not None and (
            not isinstance(username, str)
            or not username
            or not isinstance(password, str)
            or not password
        ):
            raise ValueError("username and password must be non-empty strings")
        try:
            from one.api import ONE
        except ImportError as error:
            raise IBLDependencyError(
                "IBL access requires the optional dependency: pip install '.[ibl]'"
            ) from error
        constructor_kwargs: dict[str, Any] = {
            "base_url": base_url.strip(),
            "cache_dir": Path(cache_dir),
            "silent": True,
        }
        if username is not None:
            constructor_kwargs.update(username=username, password=password)
        # Credentials are forwarded directly to ONE and are deliberately not
        # retained as attributes of this wrapper or written to run artifacts.
        self.one = ONE(**constructor_kwargs)

    def search_sessions(self, *, limit: int) -> list[str]:
        """Return 1-5 sessions from the preregistered left-camera-DLC cohort."""

        if isinstance(limit, (bool, np.bool_)) or not isinstance(
            limit, (int, np.integer)
        ) or not 1 <= limit <= 5:
            raise ValueError("initial IBL analysis is restricted to 1-5 sessions")
        candidate_eids = self.one.search(
            datasets=[
                "_ibl_trials.table.pqt",
                "spikes.times.npy",
                "spikes.clusters.npy",
                "_ibl_leftCamera.dlc.pqt",
            ]
        )
        selected: list[str] = []
        for eid in candidate_eids:
            if self._session_has_pose_datasets(str(eid)):
                selected.append(str(eid))
                if len(selected) == limit:
                    break
        return selected

    def _session_has_pose_datasets(self, eid: str) -> bool:
        """Validate a complete camera pair within the left-DLC candidate cohort."""

        datasets = self.one.list_datasets(eid)
        if isinstance(datasets, pd.DataFrame):
            if "rel_path" in datasets:
                raw_paths = datasets["rel_path"].tolist()
            elif "name" in datasets:
                raw_paths = datasets["name"].tolist()
            else:
                raw_paths = list(datasets.index)
        elif isinstance(datasets, pd.Series):
            raw_paths = datasets.tolist()
        else:
            raw_paths = list(datasets)
        paths = [str(path).replace("\\", "/").lower() for path in raw_paths]
        for label in ("left", "right", "body"):
            stem = f"{label}camera"
            has_times = any(stem in path and "times" in path for path in paths)
            has_dlc = any(stem in path and "dlc" in path for path in paths)
            if has_times and has_dlc:
                return True
        return False

    def load_trials(self, eid: str) -> Mapping[str, Any]:
        return self.one.load_object(eid, "trials")

    def load_probe_spikes(self, eid: str) -> list[ProbeSpikes]:
        collections = self.one.list_collections(eid, filename="spikes.times.npy")
        probe_names = sorted(
            {
                part
                for collection in collections
                for part in str(collection).split("/")
                if part.startswith("probe")
            }
        )
        if not probe_names:
            return []
        try:
            from brainbox.io.one import SpikeSortingLoader
        except ImportError:
            return self._load_probe_spikes_direct(eid, collections)

        probes: list[ProbeSpikes] = []
        for probe_name in probe_names:
            loader = SpikeSortingLoader(eid=eid, pname=probe_name, one=self.one)
            spikes, clusters_object, channels = loader.load_spike_sorting()
            clusters_object = loader.merge_clusters(spikes, clusters_object, channels)
            unit_ids = clusters_object.get("cluster_id")
            if unit_ids is None:
                candidate_length = len(clusters_object.get("channels", []))
                unit_ids = (
                    np.arange(candidate_length, dtype=int)
                    if candidate_length
                    else np.unique(np.asarray(spikes["clusters"], dtype=int))
                )
            else:
                unit_ids = np.asarray(unit_ids, dtype=int)
            acronym = clusters_object.get(
                "acronyms", clusters_object.get("acronym")
            )
            regions = (
                np.asarray(acronym).astype(str)
                if acronym is not None and len(acronym) == unit_ids.size
                else np.repeat("unknown", unit_ids.size)
            )
            probes.append(
                ProbeSpikes(
                    str(loader.collection),
                    np.asarray(spikes["times"], dtype=float),
                    np.asarray(spikes["clusters"]),
                    unit_ids,
                    regions,
                )
            )
        return probes

    def _load_probe_spikes_direct(
        self, eid: str, collections: Sequence[str]
    ) -> list[ProbeSpikes]:
        """Safe fallback when ibllib's SpikeSortingLoader is unavailable.

        ONE may expose base and sorter-specific collections for the same probe.
        Select exactly one collection per probe, preferring an explicit sorter
        subcollection, to avoid duplicate spikes and pseudo-units.
        """

        by_probe: dict[str, list[str]] = {}
        for raw_collection in collections:
            collection = str(raw_collection)
            parts = collection.split("/")
            probe_names = [part for part in parts if part.startswith("probe")]
            if not probe_names:
                continue
            by_probe.setdefault(probe_names[0], []).append(collection)
        probes: list[ProbeSpikes] = []
        for probe_name in sorted(by_probe):
            candidates = by_probe[probe_name]
            collection = max(
                candidates,
                key=lambda value: (
                    "pykilosort" in value or "iblsorter" in value,
                    value.count("/"),
                    value,
                ),
            )
            spikes = self.one.load_object(
                eid, "spikes", collection=collection, attribute=["times", "clusters"]
            )
            clusters_object = self.one.load_object(eid, "clusters", collection=collection)
            unit_ids = clusters_object.get("cluster_id")
            if unit_ids is None:
                candidate_length = len(clusters_object.get("channels", []))
                unit_ids = (
                    np.arange(candidate_length, dtype=int)
                    if candidate_length
                    else np.unique(np.asarray(spikes["clusters"], dtype=int))
                )
            else:
                unit_ids = np.asarray(unit_ids, dtype=int)
            acronym = clusters_object.get(
                "acronyms", clusters_object.get("acronym")
            )
            regions = (
                np.asarray(acronym).astype(str)
                if acronym is not None and len(acronym) == unit_ids.size
                else np.repeat("unknown", unit_ids.size)
            )
            probes.append(
                ProbeSpikes(
                    str(collection),
                    np.asarray(spikes["times"], dtype=float),
                    np.asarray(spikes["clusters"]),
                    unit_ids,
                    regions,
                )
            )
        return probes

    def load_wheel(self, eid: str) -> Mapping[str, Any] | None:
        try:
            from one.alf.exceptions import ALFObjectNotFound
        except ImportError:
            missing_errors: tuple[type[BaseException], ...] = ()
        else:
            missing_errors = (ALFObjectNotFound,)
        try:
            return self.one.load_object(eid, "wheel")
        except missing_errors:
            return None

    def load_pose_summary(
        self,
        eid: str,
        events: np.ndarray,
        *,
        window_s: tuple[float, float] = (-0.5, 0.0),
    ) -> np.ndarray | None:
        try:
            from one.alf.exceptions import ALFObjectNotFound
        except ImportError:
            missing_errors: tuple[type[BaseException], ...] = ()
        else:
            missing_errors = (ALFObjectNotFound,)
        for label in ("left", "right", "body"):
            try:
                camera = self.one.load_object(
                    eid,
                    f"{label}Camera",
                    collection="alf",
                    attribute=["times", "dlc"],
                )
            except missing_errors:
                continue
            return _camera_pose_motion_summary(camera, events, window_s=window_s)
        return None

    def session_details(self, eid: str) -> Mapping[str, Any]:
        return self.one.get_details(eid)
