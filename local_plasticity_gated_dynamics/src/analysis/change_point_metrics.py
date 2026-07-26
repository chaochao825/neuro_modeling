"""Sequence- and collector-safe metrics for causal change-aware accumulators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class ChangePointMetrics:
    accuracy: float
    post_switch_accuracy: float
    detection_precision: float
    detection_recall: float
    false_alarms_per_1000: float
    median_detection_delay: float
    n_frames: int
    n_switches: int
    n_alarms: int
    n_matched_switches: int


def _boolean_vector(value: ArrayLike, *, name: str, n_frames: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.bool_)
    if result.shape != (n_frames,):
        raise ValueError(f"{name} must align with frames")
    return result


def change_point_metrics(
    predictions_value: ArrayLike,
    labels_value: ArrayLike,
    *,
    alarm_flags: ArrayLike,
    switch_flags: ArrayLike,
    post_switch_window: int,
    detection_tolerance: int,
) -> ChangePointMetrics:
    """Match each causal alarm to at most one true switch and summarize utility."""

    predictions = np.asarray(predictions_value)
    labels = np.asarray(labels_value)
    if predictions.dtype.kind not in {"i", "u"} or predictions.ndim != 1:
        raise ValueError("predictions must be an integer vector")
    if labels.dtype.kind not in {"i", "u"} or labels.shape != predictions.shape:
        raise ValueError("labels must be an aligned integer vector")
    if predictions.size == 0 or np.any(predictions < 0) or np.any(labels < 0):
        raise ValueError("prediction and label vectors must be non-empty and non-negative")
    n_frames = int(predictions.size)
    alarms = _boolean_vector(alarm_flags, name="alarm_flags", n_frames=n_frames)
    switches = _boolean_vector(switch_flags, name="switch_flags", n_frames=n_frames)
    for name, value in (
        ("post_switch_window", post_switch_window),
        ("detection_tolerance", detection_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    post_switch_window = int(post_switch_window)
    detection_tolerance = int(detection_tolerance)

    switch_indices = np.flatnonzero(switches)
    alarm_indices = np.flatnonzero(alarms)
    used_alarm = np.zeros(alarm_indices.size, dtype=np.bool_)
    delays: list[int] = []
    for switch in switch_indices:
        eligible = np.flatnonzero(
            (~used_alarm)
            & (alarm_indices >= switch)
            & (alarm_indices <= switch + detection_tolerance)
        )
        if eligible.size:
            match = int(eligible[0])
            used_alarm[match] = True
            delays.append(int(alarm_indices[match] - switch))

    post_mask = np.zeros(n_frames, dtype=np.bool_)
    for switch in switch_indices:
        post_mask[switch : min(n_frames, switch + post_switch_window)] = True
    matched = len(delays)
    false_alarms = int(alarm_indices.size - matched)
    accuracy = float(np.mean(predictions == labels))
    post_accuracy = (
        float(np.mean(predictions[post_mask] == labels[post_mask]))
        if np.any(post_mask)
        else float("nan")
    )
    precision = (
        float(matched / alarm_indices.size) if alarm_indices.size else float("nan")
    )
    recall = (
        float(matched / switch_indices.size) if switch_indices.size else float("nan")
    )
    delay = float(np.median(delays)) if delays else float("nan")
    return ChangePointMetrics(
        accuracy=accuracy,
        post_switch_accuracy=post_accuracy,
        detection_precision=precision,
        detection_recall=recall,
        false_alarms_per_1000=1000.0 * false_alarms / n_frames,
        median_detection_delay=delay,
        n_frames=n_frames,
        n_switches=int(switch_indices.size),
        n_alarms=int(alarm_indices.size),
        n_matched_switches=matched,
    )


__all__ = ["ChangePointMetrics", "change_point_metrics"]
