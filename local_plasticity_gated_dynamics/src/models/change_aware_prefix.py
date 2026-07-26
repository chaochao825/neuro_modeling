"""Causal class-evidence accumulators with explicit forgetting and resets.

The functions consume label-free class-probability evidence.  Stream IDs may
reset state at observable boundaries; hidden change labels are accepted only
by the evaluation-only scheduled-reset control.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _readonly(value: ArrayLike, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _evidence(value: ArrayLike) -> FloatArray:
    evidence = np.asarray(value, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[0] == 0 or evidence.shape[1] < 2:
        raise ValueError("evidence must have shape [frame, class>=2]")
    if not np.all(np.isfinite(evidence)) or np.any(evidence < 0.0):
        raise ValueError("evidence must be finite and non-negative")
    if not np.allclose(np.sum(evidence, axis=1), 1.0, atol=1e-6):
        raise ValueError("each evidence row must sum to one")
    return evidence


def _streams(value: ArrayLike, *, n_frames: int) -> NDArray[np.str_]:
    streams = np.asarray(value, dtype=str)
    if streams.shape != (n_frames,) or np.any(np.char.str_len(streams) == 0):
        raise ValueError("stream_ids must align with frames and be non-empty")
    return streams


def _retention(value: float, *, allow_one: bool = True) -> float:
    result = float(value)
    upper_ok = result <= 1.0 if allow_one else result < 1.0
    if not np.isfinite(result) or result < 0.0 or not upper_ok:
        bound = "[0, 1]" if allow_one else "[0, 1)"
        raise ValueError(f"retention must lie in {bound}")
    return result


def _normalise(value: FloatArray) -> FloatArray:
    total = float(np.sum(value))
    if total <= 0.0:
        raise ValueError("cannot normalise zero evidence")
    return np.asarray(value / total, dtype=np.float64)


def jensen_shannon_divergence(first: ArrayLike, second: ArrayLike) -> float:
    """Return natural-log Jensen-Shannon divergence between class vectors."""

    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    if p.ndim != 1 or p.shape != q.shape or p.size < 2:
        raise ValueError("JSD inputs must be aligned class vectors")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("JSD inputs must be finite")
    if np.any(p < 0.0) or np.any(q < 0.0):
        raise ValueError("JSD inputs must be non-negative")
    p = _normalise(p)
    q = _normalise(q)
    midpoint = 0.5 * (p + q)

    def kl(left: FloatArray, right: FloatArray) -> float:
        positive = left > 0.0
        return float(np.sum(left[positive] * np.log(left[positive] / right[positive])))

    return max(0.0, 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint))


@dataclass(frozen=True, slots=True, eq=False)
class AccumulatorTrace:
    """Framewise output and causal state for one accumulation condition."""

    predictions: IntArray
    class_state: FloatArray
    detector_scores: FloatArray
    alarm_flags: BoolArray
    reset_flags: BoolArray
    run_lengths: IntArray

    def __post_init__(self) -> None:
        predictions = np.asarray(self.predictions, dtype=np.int64)
        state = np.asarray(self.class_state, dtype=np.float64)
        scores = np.asarray(self.detector_scores, dtype=np.float64)
        alarms = np.asarray(self.alarm_flags, dtype=np.bool_)
        resets = np.asarray(self.reset_flags, dtype=np.bool_)
        run_lengths = np.asarray(self.run_lengths, dtype=np.int64)
        if predictions.ndim != 1 or predictions.size == 0:
            raise ValueError("predictions must be a non-empty vector")
        n_frames = predictions.size
        if state.ndim != 2 or state.shape[0] != n_frames or state.shape[1] < 2:
            raise ValueError("class_state must have shape [frame, class>=2]")
        for name, array in (
            ("detector_scores", scores),
            ("alarm_flags", alarms),
            ("reset_flags", resets),
            ("run_lengths", run_lengths),
        ):
            if array.shape != (n_frames,):
                raise ValueError(f"{name} must align with predictions")
        if not np.all(np.isfinite(state)) or np.any(state < 0.0):
            raise ValueError("class_state must be finite and non-negative")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
            raise ValueError("detector scores must be finite and non-negative")
        if np.any(run_lengths < 1):
            raise ValueError("run lengths must be positive")
        if np.any(predictions < 0) or np.any(predictions >= state.shape[1]):
            raise ValueError("predictions fall outside the class state")
        if np.any(resets & ~alarms):
            raise ValueError("every reset must have a corresponding alarm")
        object.__setattr__(self, "predictions", _readonly(predictions, dtype=np.int64))
        object.__setattr__(self, "class_state", _readonly(state, dtype=np.float64))
        object.__setattr__(self, "detector_scores", _readonly(scores, dtype=np.float64))
        object.__setattr__(self, "alarm_flags", _readonly(alarms, dtype=np.bool_))
        object.__setattr__(self, "reset_flags", _readonly(resets, dtype=np.bool_))
        object.__setattr__(self, "run_lengths", _readonly(run_lengths, dtype=np.int64))

    @property
    def state_l1(self) -> FloatArray:
        return _readonly(np.sum(np.abs(self.class_state), axis=1), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class JSDChangeConfig:
    """Frozen two-timescale detector parameters."""

    fast_retention: float
    jsd_threshold: float
    patience: int
    min_run_frames: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fast_retention",
            _retention(self.fast_retention, allow_one=False),
        )
        threshold = float(self.jsd_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= np.log(2.0):
            raise ValueError("jsd_threshold must lie in [0, log(2)]")
        object.__setattr__(self, "jsd_threshold", threshold)
        for name in ("patience", "min_run_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))


def _empty_trace_arrays(n_frames: int, n_classes: int) -> tuple[np.ndarray, ...]:
    return (
        np.empty((n_frames, n_classes), dtype=np.float64),
        np.zeros(n_frames, dtype=np.float64),
        np.zeros(n_frames, dtype=np.bool_),
        np.zeros(n_frames, dtype=np.bool_),
        np.empty(n_frames, dtype=np.int64),
    )


def fixed_forgetting_accumulator(
    evidence_value: ArrayLike, *, stream_ids: ArrayLike, retention: float
) -> AccumulatorTrace:
    """Accumulate evidence with fixed exponential forgetting."""

    evidence = _evidence(evidence_value)
    streams = _streams(stream_ids, n_frames=evidence.shape[0])
    keep = _retention(retention)
    states, scores, alarms, resets, runs = _empty_trace_arrays(*evidence.shape)
    state = np.zeros(evidence.shape[1], dtype=np.float64)
    previous = ""
    run = 0
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        if stream != previous:
            state.fill(0.0)
            run = 0
        state = keep * state + evidence[index]
        run += 1
        states[index] = state
        runs[index] = run
        previous = stream
    return AccumulatorTrace(
        predictions=np.argmax(states, axis=1),
        class_state=states,
        detector_scores=scores,
        alarm_flags=alarms,
        reset_flags=resets,
        run_lengths=runs,
    )


def sliding_window_accumulator(
    evidence_value: ArrayLike, *, stream_ids: ArrayLike, window_frames: int
) -> AccumulatorTrace:
    """Accumulate exactly the most recent ``window_frames`` evidence rows."""

    evidence = _evidence(evidence_value)
    streams = _streams(stream_ids, n_frames=evidence.shape[0])
    if isinstance(window_frames, bool) or not isinstance(
        window_frames, (int, np.integer)
    ):
        raise TypeError("window_frames must be an integer")
    window_frames = int(window_frames)
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    states, scores, alarms, resets, runs = _empty_trace_arrays(*evidence.shape)
    window: deque[FloatArray] = deque()
    state = np.zeros(evidence.shape[1], dtype=np.float64)
    previous = ""
    run = 0
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        if stream != previous:
            window.clear()
            state.fill(0.0)
            run = 0
        row = np.asarray(evidence[index], dtype=np.float64)
        window.append(row)
        state += row
        if len(window) > window_frames:
            state -= window.popleft()
        run += 1
        states[index] = state
        runs[index] = run
        previous = stream
    return AccumulatorTrace(
        predictions=np.argmax(states, axis=1),
        class_state=states,
        detector_scores=scores,
        alarm_flags=alarms,
        reset_flags=resets,
        run_lengths=runs,
    )


def jsd_change_accumulator(
    evidence_value: ArrayLike,
    *,
    stream_ids: ArrayLike,
    config: JSDChangeConfig,
    enable_reset: bool = True,
) -> AccumulatorTrace:
    """Detect a modal distribution shift and optionally reset task evidence."""

    evidence = _evidence(evidence_value)
    streams = _streams(stream_ids, n_frames=evidence.shape[0])
    states, scores, alarms, resets, runs = _empty_trace_arrays(*evidence.shape)
    state = np.zeros(evidence.shape[1], dtype=np.float64)
    fast = np.zeros(evidence.shape[1], dtype=np.float64)
    previous = ""
    run = 0
    streak = 0
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        if stream != previous:
            state.fill(0.0)
            fast.fill(0.0)
            run = 0
            streak = 0
        candidate_fast = config.fast_retention * fast + evidence[index]
        score = 0.0
        modal_change = False
        if run > 0:
            task_distribution = _normalise(state)
            fast_distribution = _normalise(candidate_fast)
            score = jensen_shannon_divergence(
                fast_distribution, task_distribution
            )
            modal_change = int(np.argmax(fast_distribution)) != int(
                np.argmax(task_distribution)
            )
        eligible = (
            run >= config.min_run_frames
            and modal_change
            and score >= config.jsd_threshold
        )
        streak = streak + 1 if eligible else 0
        alarm = streak >= config.patience
        if alarm and enable_reset:
            state = candidate_fast.copy()
            fast = state.copy()
            run = 1
            resets[index] = True
        else:
            state += evidence[index]
            fast = candidate_fast
            run += 1
        if alarm:
            alarms[index] = True
            streak = 0
        scores[index] = score
        states[index] = state
        runs[index] = run
        previous = stream
    return AccumulatorTrace(
        predictions=np.argmax(states, axis=1),
        class_state=states,
        detector_scores=scores,
        alarm_flags=alarms,
        reset_flags=resets,
        run_lengths=runs,
    )


def scheduled_reset_accumulator(
    evidence_value: ArrayLike, *, stream_ids: ArrayLike, reset_schedule: ArrayLike
) -> AccumulatorTrace:
    """Evaluation control that resets cumulative evidence on a fixed schedule."""

    evidence = _evidence(evidence_value)
    streams = _streams(stream_ids, n_frames=evidence.shape[0])
    schedule = np.asarray(reset_schedule, dtype=np.bool_)
    if schedule.shape != (evidence.shape[0],):
        raise ValueError("reset_schedule must align with frames")
    states, scores, alarms, resets, runs = _empty_trace_arrays(*evidence.shape)
    state = np.zeros(evidence.shape[1], dtype=np.float64)
    previous = ""
    run = 0
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        boundary = stream != previous
        if boundary or schedule[index]:
            state.fill(0.0)
            run = 0
        state += evidence[index]
        run += 1
        states[index] = state
        runs[index] = run
        if schedule[index]:
            alarms[index] = True
            resets[index] = True
        previous = stream
    return AccumulatorTrace(
        predictions=np.argmax(states, axis=1),
        class_state=states,
        detector_scores=scores,
        alarm_flags=alarms,
        reset_flags=resets,
        run_lengths=runs,
    )


def circularly_shift_resets(
    reset_schedule: ArrayLike, *, stream_ids: ArrayLike, offset: int
) -> BoolArray:
    """Shift reset times within each contiguous stream while preserving count."""

    schedule = np.asarray(reset_schedule, dtype=np.bool_)
    if schedule.ndim != 1 or schedule.size == 0:
        raise ValueError("reset_schedule must be a non-empty vector")
    streams = _streams(stream_ids, n_frames=schedule.size)
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise TypeError("offset must be an integer")
    if int(offset) == 0:
        raise ValueError("offset must be nonzero")
    shifted = np.zeros(schedule.size, dtype=np.bool_)
    start = 0
    while start < schedule.size:
        end = start + 1
        while end < schedule.size and streams[end] == streams[start]:
            end += 1
        length = end - start
        positions = np.flatnonzero(schedule[start:end])
        if positions.size:
            shifted[start + ((positions + int(offset)) % length)] = True
        start = end
    return _readonly(shifted, dtype=np.bool_)


__all__ = [
    "AccumulatorTrace",
    "JSDChangeConfig",
    "circularly_shift_resets",
    "fixed_forgetting_accumulator",
    "jensen_shannon_divergence",
    "jsd_change_accumulator",
    "scheduled_reset_accumulator",
    "sliding_window_accumulator",
]
