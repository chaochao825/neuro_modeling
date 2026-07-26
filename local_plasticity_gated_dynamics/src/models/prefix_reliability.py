"""Causal prefix ensembles and validation-only action calibration.

The functions in this module operate on already-computed action scores.  They
never consume evaluation labels.  Labels enter only through
``fit_action_calibration`` on a caller-declared development split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _readonly(value: ArrayLike, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _video_ids(value: ArrayLike, *, n_frames: int) -> NDArray[np.str_]:
    videos = np.asarray(value, dtype=str)
    if videos.shape != (n_frames,) or np.any(np.char.str_len(videos) == 0):
        raise ValueError("video_ids must align with frames and be non-empty")
    return videos


def _weights(value: ArrayLike | None, *, n_actions: int) -> FloatArray:
    if value is None:
        result = np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
    else:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (n_actions,) or not np.all(np.isfinite(result)):
            raise ValueError("action_weights must be a finite action vector")
        if np.any(result < 0.0) or float(np.sum(result)) <= 0.0:
            raise ValueError("action_weights must be non-negative with positive sum")
        result = result / np.sum(result)
    return np.asarray(result, dtype=np.float64)


def _retention(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("retention must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class PrefixEnsembleTrace:
    """Predictions and causal state emitted by a prefix ensemble."""

    predictions: IntArray
    class_state: FloatArray
    state_l1: FloatArray

    def __post_init__(self) -> None:
        predictions = np.asarray(self.predictions, dtype=np.int64)
        state = np.asarray(self.class_state, dtype=np.float64)
        state_l1 = np.asarray(self.state_l1, dtype=np.float64)
        if predictions.ndim != 1 or predictions.size == 0:
            raise ValueError("predictions must be a non-empty vector")
        if state.ndim != 2 or state.shape[0] != predictions.size:
            raise ValueError("class_state must have shape [frame, class]")
        if state_l1.shape != predictions.shape:
            raise ValueError("state_l1 must align with predictions")
        if np.any(predictions < 0) or np.any(predictions >= state.shape[1]):
            raise ValueError("predictions fall outside class_state")
        if not np.all(np.isfinite(state)) or np.any(state < 0.0):
            raise ValueError("class_state must be finite and non-negative")
        if not np.allclose(np.sum(np.abs(state), axis=1), state_l1):
            raise ValueError("state_l1 does not match class_state")
        object.__setattr__(self, "predictions", _readonly(predictions, dtype=np.int64))
        object.__setattr__(self, "class_state", _readonly(state, dtype=np.float64))
        object.__setattr__(self, "state_l1", _readonly(state_l1, dtype=np.float64))


@dataclass(frozen=True, slots=True, eq=False)
class ActionCalibration:
    """Validation-fitted temperature scaling and convex stacking weights."""

    temperatures: FloatArray
    stacking_weights: FloatArray
    action_nll: FloatArray
    ensemble_nll: float
    n_frames: int

    def __post_init__(self) -> None:
        temperatures = np.asarray(self.temperatures, dtype=np.float64)
        weights = np.asarray(self.stacking_weights, dtype=np.float64)
        action_nll = np.asarray(self.action_nll, dtype=np.float64)
        if temperatures.ndim != 1 or temperatures.size < 2:
            raise ValueError("calibration requires at least two actions")
        if weights.shape != temperatures.shape or action_nll.shape != temperatures.shape:
            raise ValueError("calibration vectors must have the same action dimension")
        if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
            raise ValueError("temperatures must be finite and positive")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("stacking weights must be finite and non-negative")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-8):
            raise ValueError("stacking weights must sum to one")
        if not np.all(np.isfinite(action_nll)) or np.any(action_nll < 0.0):
            raise ValueError("action NLL values must be finite and non-negative")
        if not np.isfinite(self.ensemble_nll) or self.ensemble_nll < 0.0:
            raise ValueError("ensemble_nll must be finite and non-negative")
        if isinstance(self.n_frames, bool) or int(self.n_frames) < 1:
            raise ValueError("n_frames must be positive")
        object.__setattr__(self, "temperatures", _readonly(temperatures, dtype=np.float64))
        object.__setattr__(self, "stacking_weights", _readonly(weights, dtype=np.float64))
        object.__setattr__(self, "action_nll", _readonly(action_nll, dtype=np.float64))
        object.__setattr__(self, "ensemble_nll", float(self.ensemble_nll))
        object.__setattr__(self, "n_frames", int(self.n_frames))


def action_probabilities(
    action_scores: ArrayLike, *, temperatures: ArrayLike | None = None
) -> FloatArray:
    """Convert ``[frame, action, class]`` scores to per-action probabilities."""

    scores = np.asarray(action_scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[0] == 0 or scores.shape[1] < 2 or scores.shape[2] < 2:
        raise ValueError("action_scores must have shape [frame, action>=2, class>=2]")
    if not np.all(np.isfinite(scores)):
        raise ValueError("action_scores must be finite")
    if temperatures is None:
        scale = np.ones(scores.shape[1], dtype=np.float64)
    else:
        scale = np.asarray(temperatures, dtype=np.float64)
        if scale.shape != (scores.shape[1],) or not np.all(np.isfinite(scale)):
            raise ValueError("temperatures must be a finite action vector")
        if np.any(scale <= 0.0):
            raise ValueError("temperatures must be positive")
    logits = scores / scale[None, :, None]
    logits = logits - np.max(logits, axis=2, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=2, keepdims=True)
    return _readonly(probabilities, dtype=np.float64)


def prefix_probability_ensemble(
    action_probabilities_value: ArrayLike,
    *,
    video_ids: ArrayLike,
    action_weights: ArrayLike | None = None,
    retention: float = 1.0,
    confidence_weighted: bool = False,
) -> PrefixEnsembleTrace:
    """Average action probabilities and integrate them causally within video."""

    probabilities = np.asarray(action_probabilities_value, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] == 0:
        raise ValueError("action probabilities must have shape [frame, action, class]")
    if probabilities.shape[1] < 2 or probabilities.shape[2] < 2:
        raise ValueError("at least two actions and classes are required")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("action probabilities must be finite and non-negative")
    if not np.allclose(np.sum(probabilities, axis=2), 1.0, atol=1e-6):
        raise ValueError("each action probability vector must sum to one")
    videos = _video_ids(video_ids, n_frames=probabilities.shape[0])
    weights = _weights(action_weights, n_actions=probabilities.shape[1])
    keep = _retention(retention)
    state = np.zeros(probabilities.shape[2], dtype=np.float64)
    states = np.empty((probabilities.shape[0], probabilities.shape[2]), dtype=np.float64)
    previous_video = ""
    for index, video_id_raw in enumerate(videos):
        video_id = str(video_id_raw)
        if video_id != previous_video:
            state.fill(0.0)
        frame_weights = weights
        if confidence_weighted:
            frame_weights = weights * np.max(probabilities[index], axis=1)
            frame_weights = frame_weights / np.sum(frame_weights)
        evidence = np.sum(probabilities[index] * frame_weights[:, None], axis=0)
        state = keep * state + evidence
        states[index] = state
        previous_video = video_id
    predictions = np.argmax(states, axis=1).astype(np.int64)
    return PrefixEnsembleTrace(
        predictions=predictions,
        class_state=states,
        state_l1=np.sum(np.abs(states), axis=1),
    )


def prefix_class_vote(
    action_predictions: ArrayLike,
    *,
    n_classes: int,
    video_ids: ArrayLike,
    action_weights: ArrayLike | None = None,
    retention: float = 1.0,
) -> PrefixEnsembleTrace:
    """Accumulate weighted class votes from the action bank within each video."""

    raw = np.asarray(action_predictions)
    if raw.dtype.kind not in {"i", "u"} or raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError("action_predictions must be a non-empty integer matrix")
    predictions = np.asarray(raw, dtype=np.int64)
    if predictions.shape[1] < 2 or isinstance(n_classes, bool) or int(n_classes) < 2:
        raise ValueError("at least two actions and classes are required")
    n_classes = int(n_classes)
    if np.any(predictions < 0) or np.any(predictions >= n_classes):
        raise ValueError("action predictions fall outside the class range")
    videos = _video_ids(video_ids, n_frames=predictions.shape[0])
    weights = _weights(action_weights, n_actions=predictions.shape[1])
    keep = _retention(retention)
    state = np.zeros(n_classes, dtype=np.float64)
    states = np.empty((predictions.shape[0], n_classes), dtype=np.float64)
    previous_video = ""
    for index, video_id_raw in enumerate(videos):
        video_id = str(video_id_raw)
        if video_id != previous_video:
            state.fill(0.0)
        evidence = np.bincount(
            predictions[index], weights=weights, minlength=n_classes
        ).astype(np.float64)
        state = keep * state + evidence
        states[index] = state
        previous_video = video_id
    output = np.argmax(states, axis=1).astype(np.int64)
    return PrefixEnsembleTrace(
        predictions=output,
        class_state=states,
        state_l1=np.sum(np.abs(states), axis=1),
    )


def _validate_calibration_batches(
    score_batches: Sequence[ArrayLike], label_batches: Sequence[ArrayLike]
) -> tuple[list[FloatArray], list[IntArray], int, int]:
    if len(score_batches) == 0 or len(score_batches) != len(label_batches):
        raise ValueError("score and label batches must be non-empty and aligned")
    scores_out: list[FloatArray] = []
    labels_out: list[IntArray] = []
    n_actions = -1
    n_frames = 0
    for scores_value, labels_value in zip(score_batches, label_batches, strict=True):
        scores = np.asarray(scores_value, dtype=np.float64)
        labels = np.asarray(labels_value, dtype=np.int64)
        if scores.ndim != 3 or scores.shape[0] == 0 or scores.shape[2] < 2:
            raise ValueError("each score batch must have shape [frame, action, class>=2]")
        if labels.shape != (scores.shape[0],):
            raise ValueError("each label batch must align with its frames")
        if not np.all(np.isfinite(scores)):
            raise ValueError("calibration scores must be finite")
        if np.any(labels < 0) or np.any(labels >= scores.shape[2]):
            raise ValueError("calibration labels fall outside their class range")
        if n_actions < 0:
            n_actions = scores.shape[1]
        if scores.shape[1] != n_actions or n_actions < 2:
            raise ValueError("all batches need the same action dimension >= 2")
        scores_out.append(scores)
        labels_out.append(labels)
        n_frames += labels.size
    return scores_out, labels_out, n_actions, n_frames


def _mean_action_nll(
    score_batches: Sequence[FloatArray],
    label_batches: Sequence[IntArray],
    *,
    action: int,
    temperature: float,
) -> float:
    total = 0.0
    frames = 0
    for scores, labels in zip(score_batches, label_batches, strict=True):
        logits = scores[:, action, :] / float(temperature)
        total += float(
            np.sum(logsumexp(logits, axis=1) - logits[np.arange(labels.size), labels])
        )
        frames += labels.size
    return total / frames


def fit_action_calibration(
    score_batches: Sequence[ArrayLike],
    label_batches: Sequence[ArrayLike],
    *,
    temperature_bounds: tuple[float, float] = (0.05, 20.0),
    stacking_l2: float = 1e-3,
) -> ActionCalibration:
    """Fit scalar temperatures and convex stacking weights on development labels."""

    scores, labels, n_actions, n_frames = _validate_calibration_batches(
        score_batches, label_batches
    )
    lower, upper = map(float, temperature_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0.0 < lower < upper:
        raise ValueError("temperature_bounds must satisfy 0 < lower < upper")
    stacking_l2 = float(stacking_l2)
    if not np.isfinite(stacking_l2) or stacking_l2 < 0.0:
        raise ValueError("stacking_l2 must be finite and non-negative")
    temperatures = np.empty(n_actions, dtype=np.float64)
    action_nll = np.empty(n_actions, dtype=np.float64)
    for action in range(n_actions):
        fit = minimize_scalar(
            lambda value: _mean_action_nll(
                scores, labels, action=action, temperature=float(value)
            ),
            method="bounded",
            bounds=(lower, upper),
            options={"xatol": 1e-5},
        )
        if not fit.success or not np.isfinite(fit.fun):
            raise RuntimeError(f"temperature fit failed for action {action}: {fit.message}")
        temperatures[action] = float(fit.x)
        action_nll[action] = float(fit.fun)
    probability_batches = [
        np.asarray(action_probabilities(batch, temperatures=temperatures))
        for batch in scores
    ]
    uniform = np.full(n_actions, 1.0 / n_actions, dtype=np.float64)

    def objective(weights_value: FloatArray) -> float:
        total = 0.0
        frames = 0
        for probabilities, target in zip(probability_batches, labels, strict=True):
            correct = probabilities[
                np.arange(target.size)[:, None],
                np.arange(n_actions)[None, :],
                target[:, None],
            ]
            mixture = np.clip(correct @ weights_value, 1e-12, 1.0)
            total -= float(np.sum(np.log(mixture)))
            frames += target.size
        return total / frames + stacking_l2 * float(np.sum((weights_value - uniform) ** 2))

    fit = minimize(
        objective,
        uniform,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_actions,
        constraints={"type": "eq", "fun": lambda value: float(np.sum(value) - 1.0)},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"stacking fit failed: {fit.message}")
    weights = np.clip(np.asarray(fit.x, dtype=np.float64), 0.0, 1.0)
    weights /= np.sum(weights)
    unpenalized_nll = float(fit.fun - stacking_l2 * np.sum((weights - uniform) ** 2))
    return ActionCalibration(
        temperatures=temperatures,
        stacking_weights=weights,
        action_nll=action_nll,
        ensemble_nll=max(0.0, unpenalized_nll),
        n_frames=n_frames,
    )


__all__ = [
    "ActionCalibration",
    "PrefixEnsembleTrace",
    "action_probabilities",
    "fit_action_calibration",
    "prefix_class_vote",
    "prefix_probability_ensemble",
]
