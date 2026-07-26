"""Causal Bayesian change-point control for class-probability prefixes.

The detector is a truncated categorical Bayesian online change-point filter.
Input probability vectors act as fractional categorical observations under a
symmetric Dirichlet model.  No labels or future observations are consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.models.change_aware_prefix import AccumulatorTrace


FloatArray = NDArray[np.float64]
Mode = Literal["hard_reset", "score_only", "posterior"]


def _evidence(value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError("evidence must have shape [frame, class>=2]")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("evidence must be finite and non-negative")
    if not np.allclose(np.sum(result, axis=1), 1.0, atol=1e-6):
        raise ValueError("each evidence row must sum to one")
    return result


def _streams(value: ArrayLike, *, n_frames: int) -> NDArray[np.str_]:
    result = np.asarray(value, dtype=str)
    if result.shape != (n_frames,) or np.any(np.char.str_len(result) == 0):
        raise ValueError("stream_ids must align with frames and be non-empty")
    return result


@dataclass(frozen=True, slots=True)
class BOCPDConfig:
    """Frozen categorical BOCPD and hard-alarm parameters."""

    hazard: float
    prior_concentration: float
    alarm_threshold: float
    min_run_frames: int
    max_run_length: int = 128

    def __post_init__(self) -> None:
        for name in ("hazard", "prior_concentration", "alarm_threshold"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 < self.hazard < 1.0:
            raise ValueError("hazard must lie in (0, 1)")
        if self.prior_concentration <= 0.0:
            raise ValueError("prior_concentration must be positive")
        if not 0.0 < self.alarm_threshold <= 1.0:
            raise ValueError("alarm_threshold must lie in (0, 1]")
        for name in ("min_run_frames", "max_run_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        if self.max_run_length < self.min_run_frames:
            raise ValueError("max_run_length must cover min_run_frames")


class _CategoricalBOCPD:
    def __init__(self, *, n_classes: int, config: BOCPDConfig) -> None:
        self.config = config
        self.prior = np.full(
            n_classes,
            config.prior_concentration / n_classes,
            dtype=np.float64,
        )
        self.probabilities = np.empty(0, dtype=np.float64)
        self.alpha = np.empty((0, n_classes), dtype=np.float64)

    def reset(self) -> None:
        self.probabilities = np.empty(0, dtype=np.float64)
        self.alpha = np.empty((0, self.prior.size), dtype=np.float64)

    @staticmethod
    def _log_likelihood(evidence: FloatArray, predictive: FloatArray) -> FloatArray:
        return np.sum(evidence[None, :] * np.log(np.maximum(predictive, 1e-300)), axis=1)

    @staticmethod
    def _normalise_log_weights(log_weights: FloatArray) -> FloatArray:
        maximum = float(np.max(log_weights))
        weights = np.exp(log_weights - maximum)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("BOCPD posterior normalization failed")
        return weights / total

    def update(self, evidence: FloatArray) -> tuple[float, FloatArray, float]:
        if self.probabilities.size == 0:
            self.probabilities = np.ones(1, dtype=np.float64)
            self.alpha = (self.prior + evidence)[None, :]
            return 0.0, evidence.copy(), 1.0

        predictive = self.alpha / np.sum(self.alpha, axis=1, keepdims=True)
        growth_log_likelihood = self._log_likelihood(evidence, predictive)
        prior_predictive = self.prior / np.sum(self.prior)
        cp_log_likelihood = float(
            np.sum(evidence * np.log(np.maximum(prior_predictive, 1e-300)))
        )
        log_previous = np.log(np.maximum(self.probabilities, 1e-300))
        log_weights = np.concatenate(
            [
                np.asarray([np.log(self.config.hazard) + cp_log_likelihood]),
                log_previous
                + np.log1p(-self.config.hazard)
                + growth_log_likelihood,
            ]
        )
        probabilities = self._normalise_log_weights(log_weights)
        alpha = np.vstack([self.prior + evidence, self.alpha + evidence])

        if probabilities.size > self.config.max_run_length:
            keep = self.config.max_run_length - 1
            tail_probability = float(np.sum(probabilities[keep:]))
            tail_alpha = np.average(
                alpha[keep:], axis=0, weights=probabilities[keep:]
            )
            probabilities = np.concatenate(
                [probabilities[:keep], np.asarray([tail_probability])]
            )
            alpha = np.vstack([alpha[:keep], tail_alpha])
            probabilities /= np.sum(probabilities)

        self.probabilities = probabilities
        self.alpha = alpha
        within_run_counts = np.maximum(alpha - self.prior[None, :], 0.0)
        expected_counts = np.sum(probabilities[:, None] * within_run_counts, axis=0)
        expected_run = float(np.sum(expected_counts))
        return float(probabilities[0]), expected_counts, expected_run


def bocpd_prefix_accumulator(
    evidence_value: ArrayLike,
    *,
    stream_ids: ArrayLike,
    config: BOCPDConfig,
    mode: Mode = "hard_reset",
) -> AccumulatorTrace:
    """Run categorical BOCPD with hard-reset, score-only, or posterior state."""

    if mode not in {"hard_reset", "score_only", "posterior"}:
        raise ValueError("mode must be hard_reset, score_only, or posterior")
    evidence = _evidence(evidence_value)
    streams = _streams(stream_ids, n_frames=evidence.shape[0])
    n_frames, n_classes = evidence.shape
    states = np.empty((n_frames, n_classes), dtype=np.float64)
    scores = np.zeros(n_frames, dtype=np.float64)
    alarms = np.zeros(n_frames, dtype=np.bool_)
    resets = np.zeros(n_frames, dtype=np.bool_)
    runs = np.empty(n_frames, dtype=np.int64)
    task_state = np.zeros(n_classes, dtype=np.float64)
    detector = _CategoricalBOCPD(n_classes=n_classes, config=config)
    previous = ""
    decision_run = 0
    detector_run = 0

    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        if stream != previous:
            detector.reset()
            task_state.fill(0.0)
            decision_run = 0
            detector_run = 0
        score, posterior_state, expected_run = detector.update(evidence[index])
        eligible = detector_run >= config.min_run_frames and score >= config.alarm_threshold
        alarm = bool(eligible and mode != "posterior")
        if mode == "posterior":
            state = posterior_state
            decision_run = max(1, int(round(expected_run)))
            detector_run += 1
        else:
            if alarm and mode == "hard_reset":
                task_state = evidence[index].copy()
                resets[index] = True
                decision_run = 1
            else:
                task_state += evidence[index]
                decision_run += 1
            state = task_state
            if alarm:
                # Both hard-reset and score-only conditions reset the detector,
                # so their alarm sequence is exactly matched and only the task
                # state intervention differs.
                detector.reset()
                detector.update(evidence[index])
                detector_run = 1
            else:
                detector_run += 1
        scores[index] = score
        alarms[index] = alarm
        states[index] = state
        runs[index] = max(1, decision_run)
        previous = stream

    return AccumulatorTrace(
        predictions=np.argmax(states, axis=1),
        class_state=states,
        detector_scores=scores,
        alarm_flags=alarms,
        reset_flags=resets,
        run_lengths=runs,
    )


__all__ = ["BOCPDConfig", "bocpd_prefix_accumulator"]
