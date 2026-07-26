"""Independent-video metrics and fail-closed memory-demand qualification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.metrics import roc_auc_score


def _aligned_vector(value: ArrayLike, *, name: str, n_frames: int) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (n_frames,):
        raise ValueError(f"{name} must align with frames")
    return result


def source_video_accuracy(
    predictions_value: ArrayLike,
    labels_value: ArrayLike,
    *,
    source_video_ids: ArrayLike,
    switch_flags: ArrayLike,
    post_switch_window: int,
) -> pd.DataFrame:
    """Aggregate frames to source video; frames are never repetitions."""

    predictions = np.asarray(predictions_value)
    labels = np.asarray(labels_value)
    if predictions.dtype.kind not in {"i", "u"} or predictions.ndim != 1:
        raise ValueError("predictions must be an integer vector")
    if labels.dtype.kind not in {"i", "u"} or labels.shape != predictions.shape:
        raise ValueError("labels must be an aligned integer vector")
    if predictions.size == 0:
        raise ValueError("prediction vectors must be non-empty")
    sources = _aligned_vector(
        source_video_ids, name="source_video_ids", n_frames=predictions.size
    ).astype(str)
    switches = _aligned_vector(
        switch_flags, name="switch_flags", n_frames=predictions.size
    ).astype(bool)
    if np.any(np.char.str_len(sources) == 0):
        raise ValueError("source video identifiers must be non-empty")
    if isinstance(post_switch_window, bool) or not isinstance(
        post_switch_window, (int, np.integer)
    ):
        raise TypeError("post_switch_window must be an integer")
    post_switch_window = int(post_switch_window)
    if post_switch_window < 1:
        raise ValueError("post_switch_window must be positive")
    post_mask = np.zeros(predictions.size, dtype=np.bool_)
    for switch_index in np.flatnonzero(switches):
        source = sources[switch_index]
        end = switch_index
        while (
            end < predictions.size
            and sources[end] == source
            and end < switch_index + post_switch_window
        ):
            post_mask[end] = True
            end += 1
    rows: list[dict[str, object]] = []
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        source_post = mask & post_mask
        rows.append(
            {
                "source_video_id": source,
                "n_frames": int(np.sum(mask)),
                "accuracy": float(np.mean(predictions[mask] == labels[mask])),
                "n_post_switch_frames": int(np.sum(source_post)),
                "post_switch_accuracy": (
                    float(np.mean(predictions[source_post] == labels[source_post]))
                    if np.any(source_post)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class SwitchReachability:
    auc: float
    recall: float
    false_alarms_per_1000: float
    median_delay: float
    n_switches: int
    n_alarms: int
    n_matched_switches: int


def switch_reachability(
    score_value: ArrayLike,
    *,
    switch_flags: ArrayLike,
    stream_ids: ArrayLike,
    threshold: float,
    detection_tolerance: int,
    refractory_frames: int,
    min_run_frames: int,
) -> SwitchReachability:
    """Evaluate whether a causal statistic can actually actuate forgetting."""

    score = np.asarray(score_value, dtype=np.float64)
    if score.ndim != 1 or score.size == 0 or not np.all(np.isfinite(score)):
        raise ValueError("score must be a non-empty finite vector")
    switches = _aligned_vector(
        switch_flags, name="switch_flags", n_frames=score.size
    ).astype(bool)
    streams = _aligned_vector(
        stream_ids, name="stream_ids", n_frames=score.size
    ).astype(str)
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    for name, value in (
        ("detection_tolerance", detection_tolerance),
        ("refractory_frames", refractory_frames),
        ("min_run_frames", min_run_frames),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    tolerance = int(detection_tolerance)
    refractory = int(refractory_frames)
    minimum_run = int(min_run_frames)

    eligible = np.zeros(score.size, dtype=np.bool_)
    alarms = np.zeros(score.size, dtype=np.bool_)
    previous_stream = ""
    run = 0
    last_alarm = -10**9
    above = False
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        if stream != previous_stream:
            run = 0
            last_alarm = -10**9
            above = False
        run += 1
        eligible[index] = run >= minimum_run
        current_above = bool(score[index] >= threshold)
        rising = current_above and not above
        if eligible[index] and rising and index - last_alarm >= refractory:
            alarms[index] = True
            last_alarm = index
        above = current_above
        previous_stream = stream

    positive = switches & eligible
    if np.any(positive) and np.any(eligible & ~switches):
        auc = float(roc_auc_score(switches[eligible].astype(int), score[eligible]))
    else:
        auc = float("nan")
    switch_indices = np.flatnonzero(positive)
    alarm_indices = np.flatnonzero(alarms)
    used = np.zeros(alarm_indices.size, dtype=np.bool_)
    delays: list[int] = []
    for switch_index in switch_indices:
        same_stream = streams[alarm_indices] == streams[switch_index]
        candidates = np.flatnonzero(
            (~used)
            & same_stream
            & (alarm_indices >= switch_index)
            & (alarm_indices <= switch_index + tolerance)
        )
        if candidates.size:
            match = int(candidates[0])
            used[match] = True
            delays.append(int(alarm_indices[match] - switch_index))
    matched = len(delays)
    false_alarms = int(alarm_indices.size - matched)
    eligible_count = max(int(np.sum(eligible)), 1)
    return SwitchReachability(
        auc=auc,
        recall=float(matched / len(switch_indices)) if len(switch_indices) else float("nan"),
        false_alarms_per_1000=1000.0 * false_alarms / eligible_count,
        median_delay=float(np.median(delays)) if delays else float("nan"),
        n_switches=int(len(switch_indices)),
        n_alarms=int(len(alarm_indices)),
        n_matched_switches=matched,
    )


@dataclass(frozen=True, slots=True)
class MemoryDemandQualification:
    stable_accumulation_gain: float
    oracle_adaptation_headroom: float
    cumulative_post_switch_harm: float
    reachability_auc: float
    reachability_recall: float
    reachability_false_alarms_per_1000: float
    reachability_median_delay: float
    stable_accumulation_gate: bool
    oracle_headroom_gate: bool
    cumulative_harm_gate: bool
    reachability_gate: bool
    passed: bool


def qualify_memory_demand(
    *,
    current_frame_natural_accuracy: float,
    best_accumulator_natural_accuracy: float,
    best_fixed_hidden_accuracy: float,
    oracle_hidden_accuracy: float,
    best_fixed_post_switch_accuracy: float,
    cumulative_post_switch_accuracy: float,
    reachability: SwitchReachability,
    stable_gain_mcid: float,
    oracle_headroom_mcid: float,
    cumulative_harm_mcid: float,
    min_reachability_auc: float,
    min_reachability_recall: float,
    max_false_alarms_per_1000: float,
    max_median_delay: float,
) -> MemoryDemandQualification:
    """Apply all preregistered gates with conjunction, never an OR rule."""

    scalar_inputs = {
        "current_frame_natural_accuracy": current_frame_natural_accuracy,
        "best_accumulator_natural_accuracy": best_accumulator_natural_accuracy,
        "best_fixed_hidden_accuracy": best_fixed_hidden_accuracy,
        "oracle_hidden_accuracy": oracle_hidden_accuracy,
        "best_fixed_post_switch_accuracy": best_fixed_post_switch_accuracy,
        "cumulative_post_switch_accuracy": cumulative_post_switch_accuracy,
        "stable_gain_mcid": stable_gain_mcid,
        "oracle_headroom_mcid": oracle_headroom_mcid,
        "cumulative_harm_mcid": cumulative_harm_mcid,
        "min_reachability_auc": min_reachability_auc,
        "min_reachability_recall": min_reachability_recall,
        "max_false_alarms_per_1000": max_false_alarms_per_1000,
        "max_median_delay": max_median_delay,
    }
    if not all(np.isfinite(float(value)) for value in scalar_inputs.values()):
        raise ValueError("qualification inputs must be finite")
    stable_gain = float(
        best_accumulator_natural_accuracy - current_frame_natural_accuracy
    )
    oracle_headroom = float(oracle_hidden_accuracy - best_fixed_hidden_accuracy)
    cumulative_harm = float(
        best_fixed_post_switch_accuracy - cumulative_post_switch_accuracy
    )
    stable_gate = stable_gain >= float(stable_gain_mcid)
    oracle_gate = oracle_headroom >= float(oracle_headroom_mcid)
    cumulative_gate = cumulative_harm >= float(cumulative_harm_mcid)
    finite_delay = np.isfinite(reachability.median_delay)
    reachability_gate = bool(
        np.isfinite(reachability.auc)
        and reachability.auc >= float(min_reachability_auc)
        and np.isfinite(reachability.recall)
        and reachability.recall >= float(min_reachability_recall)
        and reachability.false_alarms_per_1000
        <= float(max_false_alarms_per_1000)
        and finite_delay
        and reachability.median_delay <= float(max_median_delay)
    )
    passed = bool(stable_gate and oracle_gate and cumulative_gate and reachability_gate)
    return MemoryDemandQualification(
        stable_accumulation_gain=stable_gain,
        oracle_adaptation_headroom=oracle_headroom,
        cumulative_post_switch_harm=cumulative_harm,
        reachability_auc=float(reachability.auc),
        reachability_recall=float(reachability.recall),
        reachability_false_alarms_per_1000=float(
            reachability.false_alarms_per_1000
        ),
        reachability_median_delay=float(reachability.median_delay),
        stable_accumulation_gate=bool(stable_gate),
        oracle_headroom_gate=bool(oracle_gate),
        cumulative_harm_gate=bool(cumulative_gate),
        reachability_gate=reachability_gate,
        passed=passed,
    )


__all__ = [
    "MemoryDemandQualification",
    "SwitchReachability",
    "qualify_memory_demand",
    "source_video_accuracy",
    "switch_reachability",
]
