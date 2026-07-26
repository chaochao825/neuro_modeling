"""Metrics for controlled prefix-routing falsification streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class RoutingStressMetrics:
    accuracy: float
    wrong_lock_fraction: float
    post_switch_accuracy: float
    switch_latency: float
    n_frames: int


def _first_stable_correct(
    correct: np.ndarray, *, start: int, stability_frames: int
) -> float:
    horizon = correct.size - start
    if horizon <= 0:
        raise ValueError("switch_index must leave at least one post-switch frame")
    width = min(stability_frames, horizon)
    for index in range(start, correct.size - width + 1):
        if bool(np.all(correct[index : index + width])):
            return float(index - start)
    return float(horizon)


def routing_stress_metrics(
    labels: ArrayLike,
    predictions: ArrayLike,
    *,
    stable_wrong_predictions: ArrayLike | None = None,
    switch_index: int | None = None,
    stability_frames: int = 3,
) -> RoutingStressMetrics:
    """Measure accuracy, wrong lock, and recovery on one independent stream."""

    target = np.asarray(labels, dtype=np.int64)
    output = np.asarray(predictions, dtype=np.int64)
    if target.ndim != 1 or target.size == 0 or output.shape != target.shape:
        raise ValueError("labels and predictions must be aligned non-empty vectors")
    if np.any(target < 0) or np.any(output < 0):
        raise ValueError("labels and predictions must be non-negative")
    if isinstance(stability_frames, bool) or int(stability_frames) < 1:
        raise ValueError("stability_frames must be a positive integer")
    stability_frames = int(stability_frames)
    if stable_wrong_predictions is None:
        wrong_lock = float("nan")
    else:
        wrong = np.asarray(stable_wrong_predictions, dtype=np.int64)
        if wrong.shape != target.shape or np.any(wrong < 0):
            raise ValueError("stable_wrong_predictions must align with labels")
        eligible = wrong != target
        wrong_lock = (
            float(np.mean(output[eligible] == wrong[eligible]))
            if np.any(eligible)
            else float("nan")
        )
    correct = output == target
    if switch_index is None:
        post_accuracy = float("nan")
        latency = float("nan")
    else:
        if (
            isinstance(switch_index, bool)
            or not isinstance(switch_index, (int, np.integer))
            or not 0 < int(switch_index) < target.size
        ):
            raise ValueError("switch_index must be an interior integer frame")
        switch_index = int(switch_index)
        post_accuracy = float(np.mean(correct[switch_index:]))
        latency = _first_stable_correct(
            correct, start=switch_index, stability_frames=stability_frames
        )
    return RoutingStressMetrics(
        accuracy=float(np.mean(correct)),
        wrong_lock_fraction=wrong_lock,
        post_switch_accuracy=post_accuracy,
        switch_latency=latency,
        n_frames=int(target.size),
    )


__all__ = ["RoutingStressMetrics", "routing_stress_metrics"]
