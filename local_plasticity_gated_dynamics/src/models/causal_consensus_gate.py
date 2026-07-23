"""Label-free causal belief over a bank of few-shot actuator predictions.

ORBIT query videos depict one personalized object per video.  The gate treats
the running concentration of each actuator's past class predictions as a
self-supervised reliability observation.  It updates a small action-by-class
count state and selects the currently most self-consistent actuator.  No query
label, future frame, autograd, or parameter update is available.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _readonly(value: ArrayLike, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def instantaneous_majority_predictions(
    action_predictions: ArrayLike,
    *,
    n_classes: int,
    tie_break_order: tuple[int, ...],
) -> tuple[IntArray, IntArray]:
    """Return a label-free, state-free full-bank ensemble control."""

    raw = np.asarray(action_predictions)
    if raw.dtype.kind not in {"i", "u"} or raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError("action_predictions must be a non-empty integer matrix")
    predictions = np.asarray(raw, dtype=np.int64)
    n_actions = predictions.shape[1]
    if n_actions < 2 or set(tie_break_order) != set(range(n_actions)):
        raise ValueError("tie_break_order must be a permutation of all actions")
    if n_classes < 2 or np.any(predictions < 0) or np.any(predictions >= n_classes):
        raise ValueError("action predictions fall outside the class range")
    output = np.empty(predictions.shape[0], dtype=np.int64)
    actions = np.empty(predictions.shape[0], dtype=np.int64)
    for index, row in enumerate(predictions):
        counts = np.bincount(row, minlength=n_classes)
        tied_classes = set(np.flatnonzero(counts == counts.max()).tolist())
        action = next(
            candidate
            for candidate in tie_break_order
            if int(row[candidate]) in tied_classes
        )
        actions[index] = action
        output[index] = row[action]
    return _readonly(output, dtype=np.int64), _readonly(actions, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class CausalConsensusConfig:
    retention: float = 1.0
    prior_count: float = 0.0
    delay_frames: int = 0
    reset_each_frame: bool = False
    tie_break_order: tuple[int, ...] = (3, 1, 0, 2)

    def __post_init__(self) -> None:
        retention = float(self.retention)
        prior = float(self.prior_count)
        if not np.isfinite(retention) or not 0.0 < retention <= 1.0:
            raise ValueError("retention must lie in (0, 1]")
        if not np.isfinite(prior) or prior < 0.0:
            raise ValueError("prior_count must be finite and non-negative")
        if (
            isinstance(self.delay_frames, bool)
            or not isinstance(self.delay_frames, (int, np.integer))
            or int(self.delay_frames) < 0
        ):
            raise ValueError("delay_frames must be a non-negative integer")
        order = tuple(int(value) for value in self.tie_break_order)
        if not order or len(order) != len(set(order)) or min(order) < 0:
            raise ValueError("tie_break_order must contain unique action indices")
        object.__setattr__(self, "retention", retention)
        object.__setattr__(self, "prior_count", prior)
        object.__setattr__(self, "delay_frames", int(self.delay_frames))
        object.__setattr__(self, "tie_break_order", order)


@dataclass(frozen=True, slots=True, eq=False)
class CausalConsensusTrace:
    actions: IntArray
    predictions: IntArray
    beliefs: FloatArray
    count_state_l1: FloatArray
    full_bank_event_l1: FloatArray
    video_ids: NDArray[np.str_]
    used_query_labels: bool = False
    used_future_frames: bool = False
    used_autograd: bool = False
    used_bptt: bool = False

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.int64)
        predictions = np.asarray(self.predictions, dtype=np.int64)
        beliefs = np.asarray(self.beliefs, dtype=np.float64)
        state = np.asarray(self.count_state_l1, dtype=np.float64)
        costs = np.asarray(self.full_bank_event_l1, dtype=np.float64)
        videos = np.asarray(self.video_ids, dtype=str)
        if actions.ndim != 1 or actions.size == 0:
            raise ValueError("actions must be a non-empty vector")
        n = actions.size
        if predictions.shape != (n,) or state.shape != (n,) or costs.shape != (n,):
            raise ValueError("consensus trace arrays must share the frame dimension")
        if beliefs.ndim != 2 or beliefs.shape[0] != n or videos.shape != (n,):
            raise ValueError("consensus belief metadata has the wrong shape")
        if np.any(actions < 0) or np.any(actions >= beliefs.shape[1]):
            raise ValueError("consensus action is outside the action range")
        if not np.isfinite(beliefs).all() or not np.isfinite(state).all():
            raise ValueError("consensus trace must be finite")
        if np.any((beliefs < 0.0) | (beliefs > 1.0)):
            raise ValueError("consensus beliefs must lie in [0, 1]")
        if np.any(state < 0.0) or np.any(costs < 0.0):
            raise ValueError("consensus state and costs must be non-negative")
        object.__setattr__(self, "actions", _readonly(actions, dtype=np.int64))
        object.__setattr__(self, "predictions", _readonly(predictions, dtype=np.int64))
        object.__setattr__(self, "beliefs", _readonly(beliefs, dtype=np.float64))
        object.__setattr__(self, "count_state_l1", _readonly(state, dtype=np.float64))
        object.__setattr__(
            self, "full_bank_event_l1", _readonly(costs, dtype=np.float64)
        )
        object.__setattr__(self, "video_ids", _readonly(videos, dtype=str))


class CausalConsensusGate:
    """Backpropagation-free action belief updated from prediction persistence."""

    def __init__(
        self,
        n_actions: int,
        n_classes: int,
        *,
        config: CausalConsensusConfig | None = None,
    ) -> None:
        for name, value, minimum in (
            ("n_actions", n_actions, 2),
            ("n_classes", n_classes, 2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.n_actions = int(n_actions)
        self.n_classes = int(n_classes)
        self.config = config or CausalConsensusConfig()
        if set(self.config.tie_break_order) != set(range(self.n_actions)):
            raise ValueError("tie_break_order must be a permutation of all actions")
        self.counts = np.zeros((self.n_actions, self.n_classes), dtype=np.float64)
        self.pending: deque[np.ndarray] = deque()

    def reset_sequence(self) -> None:
        self.counts.fill(0.0)
        self.pending.clear()

    def _update(self, predictions: IntArray) -> None:
        self.counts *= self.config.retention
        self.counts[np.arange(self.n_actions), predictions] += 1.0

    def _belief(self) -> FloatArray:
        prior = self.config.prior_count
        numerator = np.max(self.counts, axis=1) + prior
        denominator = np.sum(self.counts, axis=1) + self.n_classes * prior
        # Before a delayed observation arrives, all actions are equally
        # uninformative and the registered tie order supplies the safe prior.
        return np.divide(
            numerator,
            denominator,
            out=np.ones(self.n_actions, dtype=np.float64),
            where=denominator > 0.0,
        )

    def _select(self, belief: FloatArray) -> int:
        maximum = float(np.max(belief))
        tied = set(
            np.flatnonzero(np.isclose(belief, maximum, rtol=0.0, atol=1e-12)).tolist()
        )
        return next(action for action in self.config.tie_break_order if action in tied)

    def trace(
        self,
        action_predictions: ArrayLike,
        *,
        video_ids: ArrayLike,
        action_event_l1: ArrayLike | None = None,
    ) -> CausalConsensusTrace:
        raw = np.asarray(action_predictions)
        if raw.dtype.kind not in {"i", "u"}:
            raise TypeError("action_predictions must be an integer matrix")
        predictions = np.asarray(raw, dtype=np.int64)
        videos = np.asarray(video_ids, dtype=str)
        if predictions.ndim != 2 or predictions.shape[1] != self.n_actions:
            raise ValueError("action_predictions must have shape [frame, action]")
        n = predictions.shape[0]
        if n == 0 or videos.shape != (n,) or np.any(np.char.str_len(videos) == 0):
            raise ValueError("video_ids must align with a non-empty prediction tape")
        if np.any(predictions < 0) or np.any(predictions >= self.n_classes):
            raise ValueError("action predictions fall outside the class range")
        if action_event_l1 is None:
            frame_cost = np.zeros(n, dtype=np.float64)
        else:
            costs = np.asarray(action_event_l1, dtype=np.float64)
            if costs.shape != predictions.shape or np.any(costs < 0.0):
                raise ValueError("action_event_l1 must align with predictions")
            frame_cost = np.sum(costs, axis=1)

        actions = np.empty(n, dtype=np.int64)
        output = np.empty(n, dtype=np.int64)
        beliefs = np.empty((n, self.n_actions), dtype=np.float64)
        state_l1 = np.empty(n, dtype=np.float64)
        previous_video = ""
        for index in range(n):
            video_id = str(videos[index])
            if video_id != previous_video or self.config.reset_each_frame:
                self.reset_sequence()
            current = predictions[index].copy()
            self.pending.append(current)
            if len(self.pending) > self.config.delay_frames:
                self._update(self.pending.popleft())
            belief = self._belief()
            action = self._select(belief)
            actions[index] = action
            output[index] = current[action]
            beliefs[index] = belief
            state_l1[index] = float(np.sum(np.abs(self.counts)))
            previous_video = video_id
        return CausalConsensusTrace(
            actions=actions,
            predictions=output,
            beliefs=beliefs,
            count_state_l1=state_l1,
            full_bank_event_l1=frame_cost,
            video_ids=videos,
        )
