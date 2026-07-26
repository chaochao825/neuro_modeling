"""Causal run-length observer for the IBL hidden-block task.

The observer receives only the binary stimulus side.  The initial unbiased
period and the biased-block duration family are task-structure assumptions;
per-trial ``probabilityLeft`` labels are never accepted by this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.ibl_behavior import (
    IBLBehaviorDataError,
    IBLBehaviorObservations,
)


Array = np.ndarray


def _readonly(value: object, *, dtype: type, ndim: int = 1) -> Array:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.ndim != ndim:
        raise IBLBehaviorDataError(f"expected a {ndim}-dimensional array")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise IBLBehaviorDataError("observer arrays must be finite")
    result.setflags(write=False)
    return result


def _fingerprint(*arrays: Array, tag: str) -> str:
    digest = hashlib.sha256(tag.encode("utf-8"))
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class RunLengthCandidate:
    """One task-structure candidate selected without block labels."""

    min_run: int
    max_run: int
    hazard_scale: float

    def __post_init__(self) -> None:
        if isinstance(self.min_run, bool) or isinstance(self.max_run, bool):
            raise TypeError("run limits must be integers")
        if int(self.min_run) < 1 or int(self.max_run) <= int(self.min_run):
            raise ValueError("require 1 <= min_run < max_run")
        if not np.isfinite(self.hazard_scale) or float(self.hazard_scale) <= 0.0:
            raise ValueError("hazard_scale must be positive")


@dataclass(frozen=True)
class RunLengthPrediction:
    """Predictive states available before each trial's stimulus."""

    beliefs: Array
    release_probability: Array
    expected_run_length: Array
    run_length_sd: Array
    run_length_ess: Array
    belief_entropy: Array
    trial_ids: Array
    fit_trial_ids: Array
    candidate: RunLengthCandidate
    train_stimulus_nll: float
    burn_in_trials: int
    release_window: int
    emission_left: tuple[float, float]

    def __post_init__(self) -> None:
        trial_ids = _readonly(self.trial_ids, dtype=int)
        fit_ids = _readonly(self.fit_trial_ids, dtype=int)
        beliefs = _readonly(self.beliefs, dtype=float, ndim=2)
        vectors = {
            "release_probability": _readonly(self.release_probability, dtype=float),
            "expected_run_length": _readonly(self.expected_run_length, dtype=float),
            "run_length_sd": _readonly(self.run_length_sd, dtype=float),
            "run_length_ess": _readonly(self.run_length_ess, dtype=float),
            "belief_entropy": _readonly(self.belief_entropy, dtype=float),
        }
        if beliefs.shape != (trial_ids.size, 2):
            raise IBLBehaviorDataError("beliefs must have shape (n_trials, 2)")
        if any(value.shape != trial_ids.shape for value in vectors.values()):
            raise IBLBehaviorDataError("every observer state must align with trials")
        if np.any(beliefs < 0.0) or not np.allclose(beliefs.sum(axis=1), 1.0):
            raise IBLBehaviorDataError("belief rows must be probabilities")
        if np.any(
            (vectors["release_probability"] < 0.0)
            | (vectors["release_probability"] > 1.0)
        ):
            raise IBLBehaviorDataError("release probabilities must lie in [0,1]")
        if np.any(vectors["run_length_ess"] < 1.0 - 1e-10):
            raise IBLBehaviorDataError("run-length ESS cannot be below one")
        if not np.isfinite(self.train_stimulus_nll):
            raise IBLBehaviorDataError("train_stimulus_nll must be finite")
        object.__setattr__(self, "trial_ids", trial_ids)
        object.__setattr__(self, "fit_trial_ids", fit_ids)
        object.__setattr__(self, "beliefs", beliefs)
        for name, value in vectors.items():
            object.__setattr__(self, name, value)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.beliefs,
            self.release_probability,
            self.expected_run_length,
            self.run_length_sd,
            self.run_length_ess,
            self.trial_ids,
            tag="ibl-run-length-observer-v1",
        )


def _validate_training_prefix(indices: object, n_trials: int) -> Array:
    train = np.asarray(indices, dtype=int)
    if train.ndim != 1 or train.size < 3 or np.any(train < 0):
        raise IBLBehaviorDataError("train_indices must be a non-empty prefix")
    if np.any(train >= n_trials) or not np.array_equal(train, np.arange(train.size)):
        raise IBLBehaviorDataError("train_indices must be the chronological prefix")
    return train


def _hazard(completed_run: int, candidate: RunLengthCandidate) -> float:
    if completed_run < candidate.min_run:
        return 0.0
    if completed_run >= candidate.max_run:
        return 1.0
    return float(-np.expm1(-1.0 / candidate.hazard_scale))


def _run_filter(
    observations: IBLBehaviorObservations,
    candidate: RunLengthCandidate,
    *,
    burn_in_trials: int,
    release_window: int,
    emission_left: tuple[float, float],
) -> tuple[dict[str, Array], Array]:
    sides = observations.stimulus_side
    n_trials = sides.size
    beliefs = np.full((n_trials, 2), 0.5, dtype=float)
    release = np.zeros(n_trials, dtype=float)
    expected = np.zeros(n_trials, dtype=float)
    run_sd = np.zeros(n_trials, dtype=float)
    run_ess = np.ones(n_trials, dtype=float)
    entropy = np.full(n_trials, np.log(2.0), dtype=float)
    predictive_left = np.full(n_trials, 0.5, dtype=float)
    if burn_in_trials >= n_trials:
        raise IBLBehaviorDataError("burn_in_trials must leave biased trials")

    emission = np.asarray(emission_left, dtype=float)
    weights = np.zeros((2, candidate.max_run), dtype=float)
    weights[:, 0] = 0.5
    run_axis = np.arange(candidate.max_run, dtype=float)
    recent_change_probability = 0.0
    for trial in range(burn_in_trials, n_trials):
        state_mass = weights.sum(axis=1)
        state_mass /= state_mass.sum()
        run_mass = weights.sum(axis=0)
        run_mass /= run_mass.sum()
        beliefs[trial] = state_mass
        release[trial] = recent_change_probability
        expected[trial] = float(run_mass @ run_axis)
        variance = float(run_mass @ (run_axis - expected[trial]) ** 2)
        run_sd[trial] = np.sqrt(max(variance, 0.0))
        run_ess[trial] = 1.0 / float(np.sum(run_mass**2))
        entropy[trial] = -float(
            np.sum(state_mass * np.log(np.clip(state_mass, 1e-12, 1.0)))
        )
        predictive_left[trial] = float(state_mass @ emission)

        likelihood = emission if int(sides[trial]) == 1 else 1.0 - emission
        posterior = weights * likelihood[:, None]
        normalizer = float(posterior.sum())
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            raise RuntimeError("observer likelihood collapsed")
        posterior /= normalizer
        recent_change_probability = (
            0.0
            if trial == burn_in_trials
            else float(posterior[:, :release_window].sum())
        )
        next_weights = np.zeros_like(weights)
        for state in (0, 1):
            other = 1 - state
            for run in range(candidate.max_run):
                mass = float(posterior[state, run])
                if mass == 0.0:
                    continue
                switch = _hazard(run + 1, candidate)
                if switch < 1.0:
                    next_weights[state, run + 1] += mass * (1.0 - switch)
                if switch > 0.0:
                    next_weights[other, 0] += mass * switch
        weights = next_weights / next_weights.sum()
    states = {
        "beliefs": beliefs,
        "release_probability": release,
        "expected_run_length": expected,
        "run_length_sd": run_sd,
        "run_length_ess": run_ess,
        "belief_entropy": entropy,
    }
    return states, predictive_left


class SemiMarkovBlockObserver:
    """Select a causal run-length model by train-only stimulus NLL."""

    def __init__(
        self,
        *,
        candidates: Sequence[RunLengthCandidate] = (
            RunLengthCandidate(20, 100, 30.0),
            RunLengthCandidate(20, 100, 60.0),
            RunLengthCandidate(20, 100, 90.0),
        ),
        burn_in_trials: int = 90,
        release_window: int = 5,
        emission_left: tuple[float, float] = (0.2, 0.8),
    ) -> None:
        self.candidates = tuple(candidates)
        if not self.candidates or len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidates must be non-empty and unique")
        self.burn_in_trials = int(burn_in_trials)
        if self.burn_in_trials < 0:
            raise ValueError("burn_in_trials cannot be negative")
        self.release_window = int(release_window)
        if self.release_window < 1:
            raise ValueError("release_window must be positive")
        low, high = (float(value) for value in emission_left)
        if not 0.0 < low < high < 1.0:
            raise ValueError("emission_left must be ordered probabilities")
        self.emission_left = (low, high)
        self._fitted = False

    def fit(
        self,
        observations: IBLBehaviorObservations,
        train_indices: object,
    ) -> "SemiMarkovBlockObserver":
        if not isinstance(observations, IBLBehaviorObservations):
            raise TypeError("observations must be IBLBehaviorObservations")
        train = _validate_training_prefix(train_indices, observations.trial_ids.size)
        scored = train[train >= self.burn_in_trials]
        if scored.size < 10:
            raise IBLBehaviorDataError("training prefix has too few biased trials")
        targets = observations.stimulus_side[scored].astype(float)
        scores: list[float] = []
        for candidate in self.candidates:
            _, probability = _run_filter(
                observations,
                candidate,
                burn_in_trials=self.burn_in_trials,
                release_window=self.release_window,
                emission_left=self.emission_left,
            )
            selected = np.clip(probability[scored], 1e-9, 1.0 - 1e-9)
            nll = -float(
                np.mean(
                    targets * np.log(selected)
                    + (1.0 - targets) * np.log(1.0 - selected)
                )
            )
            scores.append(nll)
        best = min(range(len(scores)), key=lambda index: (scores[index], index))
        self.candidate_ = self.candidates[best]
        self.train_stimulus_nll_ = scores[best]
        self.fit_trial_ids_ = observations.trial_ids[train].copy()
        self.fit_trial_ids_.setflags(write=False)
        self._fitted = True
        return self

    def predict(self, observations: IBLBehaviorObservations) -> RunLengthPrediction:
        if not self._fitted:
            raise RuntimeError("SemiMarkovBlockObserver must be fit first")
        if not isinstance(observations, IBLBehaviorObservations):
            raise TypeError("observations must be IBLBehaviorObservations")
        states, _ = _run_filter(
            observations,
            self.candidate_,
            burn_in_trials=self.burn_in_trials,
            release_window=self.release_window,
            emission_left=self.emission_left,
        )
        return RunLengthPrediction(
            **states,
            trial_ids=observations.trial_ids,
            fit_trial_ids=self.fit_trial_ids_,
            candidate=self.candidate_,
            train_stimulus_nll=self.train_stimulus_nll_,
            burn_in_trials=self.burn_in_trials,
            release_window=self.release_window,
            emission_left=self.emission_left,
        )


__all__ = [
    "RunLengthCandidate",
    "RunLengthPrediction",
    "SemiMarkovBlockObserver",
]
