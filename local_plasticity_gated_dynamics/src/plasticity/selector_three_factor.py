"""Pure NumPy three-factor learning for the fixed Exp27 actuator dictionary.

The update is intentionally local: a presynaptic cue/"pre-belief" eligibility
trace is multiplied by a three-dimensional candidate-wise modulatory
advantage.  This module does not import or call a gradient engine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.actuator_selector_dataset import CANDIDATE_MODES


def _positive_float(value: object, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not np.isfinite(resolved) or (resolved < 0.0 if allow_zero else resolved <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return resolved


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be positive")
    return resolved


def _readonly(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    shifted = values / temperature
    shifted = shifted - np.max(shifted, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


@dataclass(frozen=True)
class LocalSelectorTrainingReceipt:
    """Auditable receipt for a completed local selector fit."""

    algorithm: str
    objective: str
    eligibility_definition: str
    used_bptt: bool
    used_autograd: bool
    candidate_modes: tuple[str, ...]
    input_dim: int
    modulatory_dimension: int
    learning_rate: float
    epochs: int
    temperature: float
    teacher_temperature: float
    l2: float
    eligibility_decay: float
    belief_retention: float
    shuffle_seed: int
    shuffle_fingerprint: str
    n_samples: int
    final_loss: float
    cumulative_update_l1: float
    cumulative_update_l2: float
    weights: np.ndarray
    train_probabilities: np.ndarray

    def __post_init__(self) -> None:
        if self.algorithm != "local_three_factor":
            raise ValueError("algorithm must be local_three_factor")
        if self.used_bptt or self.used_autograd:
            raise ValueError("local selector cannot use BPTT or autograd")
        if tuple(self.candidate_modes) != CANDIDATE_MODES:
            raise ValueError("receipt candidate dictionary is invalid")
        if self.modulatory_dimension != len(CANDIDATE_MODES):
            raise ValueError("modulatory signal must have K=3 dimensions")
        if (
            not isinstance(self.shuffle_fingerprint, str)
            or len(self.shuffle_fingerprint) != 64
        ):
            raise ValueError("shuffle_fingerprint must be a SHA-256 digest")
        weights = _readonly(self.weights, name="weights", ndim=2)
        probabilities = _readonly(
            self.train_probabilities, name="train_probabilities", ndim=2
        )
        if weights.shape != (len(CANDIDATE_MODES), int(self.input_dim)):
            raise ValueError("receipt weights have an invalid shape")
        if probabilities.shape != (int(self.n_samples), len(CANDIDATE_MODES)):
            raise ValueError("receipt probabilities have an invalid shape")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("receipt probabilities must sum to one")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "train_probabilities", probabilities)
        scalar_values = (
            self.learning_rate,
            self.temperature,
            self.teacher_temperature,
            self.l2,
            self.eligibility_decay,
            self.belief_retention,
            self.final_loss,
            self.cumulative_update_l1,
            self.cumulative_update_l2,
        )
        if not np.all(np.isfinite(scalar_values)) or min(scalar_values) < 0.0:
            raise ValueError("receipt scalar metrics must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a directly JSON-serializable receipt."""

        return {
            field: (value.tolist() if isinstance(value, np.ndarray) else value)
            for field, value in self.__dict__.items()
        }


class LocalThreeFactorSelector:
    """Sequential local selector over routing, gain, and low-rank actuators."""

    used_bptt = False
    used_autograd = False

    def __init__(
        self,
        input_dim: int = 8,
        *,
        learning_rate: float = 0.05,
        epochs: int = 200,
        temperature: float = 1.0,
        teacher_temperature: float = 0.05,
        l2: float = 1e-4,
        eligibility_decay: float = 0.8,
        belief_retention: float = 0.8,
        shuffle_seed: int = 0,
    ) -> None:
        self.input_dim = _positive_int(input_dim, name="input_dim")
        self.learning_rate = _positive_float(learning_rate, name="learning_rate")
        self.epochs = _positive_int(epochs, name="epochs")
        self.temperature = _positive_float(temperature, name="temperature")
        self.teacher_temperature = _positive_float(
            teacher_temperature, name="teacher_temperature"
        )
        self.l2 = _positive_float(l2, name="l2", allow_zero=True)
        self.eligibility_decay = _positive_float(
            eligibility_decay, name="eligibility_decay", allow_zero=True
        )
        self.belief_retention = _positive_float(
            belief_retention, name="belief_retention", allow_zero=True
        )
        if self.eligibility_decay > 1.0 or self.belief_retention > 1.0:
            raise ValueError(
                "eligibility_decay and belief_retention must lie in [0, 1]"
            )
        if isinstance(shuffle_seed, (bool, np.bool_)) or not isinstance(
            shuffle_seed, (int, np.integer)
        ):
            raise TypeError("shuffle_seed must be an integer")
        if int(shuffle_seed) < 0:
            raise ValueError("shuffle_seed must be non-negative")
        self.shuffle_seed = int(shuffle_seed)
        self._weights: np.ndarray | None = None

    @property
    def weights(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("selector has not been fitted")
        result = np.array(self._weights, copy=True)
        result.setflags(write=False)
        return result

    def _validate_inputs(
        self, cues: object, utilities: object | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        cue_array = _readonly(cues, name="cues", ndim=3)
        if cue_array.shape[0] < 1 or cue_array.shape[1] != 3:
            raise ValueError("cues must have shape [sample, 3, input_dim]")
        if cue_array.shape[2] != self.input_dim:
            raise ValueError("cue input dimension does not match selector")
        if utilities is None:
            return cue_array, None
        utility_array = _readonly(utilities, name="utilities", ndim=2)
        if utility_array.shape != (cue_array.shape[0], len(CANDIDATE_MODES)):
            raise ValueError("utilities must have shape [sample, 3]")
        if np.any((utility_array < 0.0) | (utility_array > 1.0)):
            raise ValueError("utility values must lie in [0, 1]")
        return cue_array, utility_array

    def _belief_and_eligibility(
        self, cue_sequence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        belief = np.zeros(self.input_dim, dtype=np.float64)
        eligibility = np.zeros(self.input_dim, dtype=np.float64)
        for pre_belief in cue_sequence:
            eligibility = self.eligibility_decay * eligibility + pre_belief
            belief = self.belief_retention * belief + pre_belief
        return belief, eligibility

    def fit(self, cues: object, utilities: object) -> LocalSelectorTrainingReceipt:
        cue_array, utility_array = self._validate_inputs(cues, utilities)
        assert utility_array is not None
        teacher = _softmax(utility_array, temperature=self.teacher_temperature)
        weights = np.zeros((len(CANDIDATE_MODES), self.input_dim), dtype=np.float64)
        rng = np.random.default_rng(self.shuffle_seed)
        shuffle_digest = hashlib.sha256()
        cumulative_l1 = 0.0
        cumulative_squared_l2 = 0.0
        for _ in range(self.epochs):
            order = rng.permutation(cue_array.shape[0])
            shuffle_digest.update(order.astype("<i8", copy=False).tobytes())
            for index in order:
                belief, eligibility = self._belief_and_eligibility(cue_array[index])
                probability = _softmax(
                    weights @ belief,
                    temperature=self.temperature,
                )
                modulatory_advantage = teacher[index] - probability
                if modulatory_advantage.shape != (len(CANDIDATE_MODES),):
                    raise RuntimeError("modulatory advantage lost its K=3 contract")
                local_term = np.outer(modulatory_advantage, eligibility)
                update = self.learning_rate * (local_term - self.l2 * weights)
                if not np.all(np.isfinite(update)):
                    raise FloatingPointError("local selector update became non-finite")
                weights += update
                cumulative_l1 += float(np.sum(np.abs(update)))
                cumulative_squared_l2 += float(np.sum(np.square(update)))
        self._weights = np.array(weights, copy=True)
        probabilities = self.predict_proba(cue_array)
        final_loss = float(
            -np.mean(
                np.sum(teacher * np.log(np.maximum(probabilities, 1e-300)), axis=1)
            )
        )
        frozen_weights = self.weights
        return LocalSelectorTrainingReceipt(
            algorithm="local_three_factor",
            objective="soft_teacher_cross_entropy",
            eligibility_definition=(
                "pre_belief_cue_trace_times_three_dimensional_modulatory_advantage"
            ),
            used_bptt=False,
            used_autograd=False,
            candidate_modes=CANDIDATE_MODES,
            input_dim=self.input_dim,
            modulatory_dimension=len(CANDIDATE_MODES),
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            temperature=self.temperature,
            teacher_temperature=self.teacher_temperature,
            l2=self.l2,
            eligibility_decay=self.eligibility_decay,
            belief_retention=self.belief_retention,
            shuffle_seed=self.shuffle_seed,
            shuffle_fingerprint=shuffle_digest.hexdigest(),
            n_samples=cue_array.shape[0],
            final_loss=final_loss,
            cumulative_update_l1=cumulative_l1,
            cumulative_update_l2=float(np.sqrt(cumulative_squared_l2)),
            weights=frozen_weights,
            train_probabilities=probabilities,
        )

    def predict_proba(self, cues: object) -> np.ndarray:
        cue_array, _ = self._validate_inputs(cues)
        if self._weights is None:
            raise RuntimeError("selector has not been fitted")
        beliefs = np.vstack(
            [self._belief_and_eligibility(sequence)[0] for sequence in cue_array]
        )
        probabilities = _softmax(
            beliefs @ self._weights.T,
            temperature=self.temperature,
        )
        if not np.all(np.isfinite(probabilities)) or not np.allclose(
            np.sum(probabilities, axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise FloatingPointError("local selector probabilities are invalid")
        probabilities.setflags(write=False)
        return probabilities

    def predict(self, cues: object) -> np.ndarray:
        choices = np.argmax(self.predict_proba(cues), axis=1).astype(np.int64)
        choices.setflags(write=False)
        return choices


__all__ = ["LocalSelectorTrainingReceipt", "LocalThreeFactorSelector"]
