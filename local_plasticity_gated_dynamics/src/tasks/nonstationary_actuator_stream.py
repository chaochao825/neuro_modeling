"""Paired hidden-state and sparse-feedback tapes for Exp32.

The hidden demand is a symmetric two-state HMM.  Switches occur at unknown
trial times and no boundary signal is exposed to a controller.  Feedback
schedules are exact, nested fractions of one common random ranking and exclude
the common terminal delay horizon so every registered reward can arrive.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils.reproducibility import derive_seed


IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _probability(value: object, *, name: str, closed_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    lower_ok = result >= 0.0 if closed_zero else result > 0.0
    if not np.isfinite(result) or not lower_ok or result > 1.0:
        interval = "[0, 1]" if closed_zero else "(0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _readonly(array: object, *, dtype: np.dtype, ndim: int) -> np.ndarray:
    value = np.asarray(array, dtype=dtype)
    if value.ndim != ndim:
        raise ValueError(f"array must be {ndim}-dimensional")
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _fingerprint(*arrays: np.ndarray, labels: tuple[object, ...]) -> str:
    digest = hashlib.sha256(repr(labels).encode("utf-8"))
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NonstationaryActuatorStreamConfig:
    n_trials: int = 2048
    key_dim: int = 16
    direct_reliabilities: tuple[float, float] = (0.55, 0.95)
    load_values: tuple[int, int] = (8, 24)
    distractor_write_values: tuple[int, int] = (8, 48)
    distractor_strength: float = 1.0
    hazards: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    feedback_fractions: tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625)
    feedback_delays: tuple[int, ...] = (0, 4, 16)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_trials", _positive_int(self.n_trials, name="n_trials", minimum=64)
        )
        object.__setattr__(
            self, "key_dim", _positive_int(self.key_dim, name="key_dim", minimum=2)
        )
        reliabilities = tuple(float(value) for value in self.direct_reliabilities)
        if (
            len(reliabilities) != 2
            or not 0.5 < reliabilities[0] < reliabilities[1] <= 1.0
        ):
            raise ValueError(
                "direct_reliabilities must be two increasing values in (0.5, 1]"
            )
        loads = tuple(
            _positive_int(value, name="load_values element", minimum=2)
            for value in self.load_values
        )
        distractors = tuple(
            _positive_int(value, name="distractor_write_values element", minimum=0)
            for value in self.distractor_write_values
        )
        if len(loads) != 2 or loads[0] >= loads[1]:
            raise ValueError("load_values must contain two increasing values")
        if len(distractors) != 2 or distractors[0] >= distractors[1]:
            raise ValueError(
                "distractor_write_values must contain two increasing values"
            )
        hazards = tuple(_probability(value, name="hazard") for value in self.hazards)
        fractions = tuple(
            _probability(value, name="feedback fraction")
            for value in self.feedback_fractions
        )
        delays = tuple(
            _positive_int(value, name="feedback delay", minimum=0)
            for value in self.feedback_delays
        )
        if len(set(hazards)) != len(hazards) or hazards != tuple(sorted(hazards)):
            raise ValueError("hazards must be unique and increasing")
        if len(set(fractions)) != len(fractions) or fractions != tuple(
            sorted(fractions, reverse=True)
        ):
            raise ValueError("feedback_fractions must be unique and decreasing")
        if len(set(delays)) != len(delays) or delays != tuple(sorted(delays)):
            raise ValueError("feedback_delays must be unique and increasing")
        strength = float(self.distractor_strength)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("distractor_strength must be finite and non-negative")
        if max(delays) + 2 >= self.n_trials:
            raise ValueError(
                "terminal delay horizon leaves no feedback-eligible trials"
            )
        object.__setattr__(self, "direct_reliabilities", reliabilities)
        object.__setattr__(self, "load_values", loads)
        object.__setattr__(self, "distractor_write_values", distractors)
        object.__setattr__(self, "hazards", hazards)
        object.__setattr__(self, "feedback_fractions", fractions)
        object.__setattr__(self, "feedback_delays", delays)
        object.__setattr__(self, "distractor_strength", strength)

    @property
    def feedback_eligible_trials(self) -> int:
        return self.n_trials - max(self.feedback_delays) - 1


@dataclass(frozen=True, slots=True, eq=False)
class NonstationaryStreamTape:
    hidden_states: IntArray
    switch_mask: BoolArray
    feedback_available: BoolArray
    hazard: float
    feedback_fraction: float
    feedback_delay: int
    state_fingerprint: str
    feedback_fingerprint: str

    def __post_init__(self) -> None:
        states = _readonly(self.hidden_states, dtype=np.dtype(np.int64), ndim=1)
        switches = _readonly(self.switch_mask, dtype=np.dtype(bool), ndim=1)
        feedback = _readonly(self.feedback_available, dtype=np.dtype(bool), ndim=1)
        if switches.shape != states.shape or feedback.shape != states.shape:
            raise ValueError("stream arrays must have identical shape")
        if np.any((states < 0) | (states > 1)):
            raise ValueError("hidden states must be binary")
        if switches.size and bool(switches[0]):
            raise ValueError("the first trial cannot be a switch")
        if not np.array_equal(switches[1:], states[1:] != states[:-1]):
            raise ValueError("switch mask is inconsistent with hidden states")
        for name in ("state_fingerprint", "feedback_fingerprint"):
            if len(str(getattr(self, name))) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        object.__setattr__(self, "hidden_states", states)
        object.__setattr__(self, "switch_mask", switches)
        object.__setattr__(self, "feedback_available", feedback)


def make_hidden_state_tape(
    config: NonstationaryActuatorStreamConfig,
    *,
    seed: int,
    hazard: float,
) -> tuple[IntArray, BoolArray, str]:
    """Generate one symmetric HMM stream without exposing switch times."""

    if not isinstance(config, NonstationaryActuatorStreamConfig):
        raise TypeError("config must be a NonstationaryActuatorStreamConfig")
    seed = _positive_int(seed, name="seed", minimum=0)
    hazard = float(hazard)
    if hazard not in config.hazards:
        raise ValueError("hazard is not registered by the stream config")
    rng = np.random.default_rng(derive_seed(seed, "exp32", "hidden", hazard))
    states = np.empty(config.n_trials, dtype=np.int64)
    states[0] = int(rng.integers(2))
    transitions = rng.random(config.n_trials - 1) < hazard
    for trial, switch in enumerate(transitions, start=1):
        states[trial] = states[trial - 1] ^ int(switch)
    switches = np.zeros(config.n_trials, dtype=bool)
    switches[1:] = transitions
    digest = _fingerprint(states, switches, labels=(seed, hazard, "hidden"))
    states.setflags(write=False)
    switches.setflags(write=False)
    return states, switches, digest


def make_nested_feedback_tapes(
    config: NonstationaryActuatorStreamConfig,
    *,
    seed: int,
) -> dict[float, tuple[BoolArray, str]]:
    """Return exact nested feedback schedules on one seed-level ranking."""

    if not isinstance(config, NonstationaryActuatorStreamConfig):
        raise TypeError("config must be a NonstationaryActuatorStreamConfig")
    seed = _positive_int(seed, name="seed", minimum=0)
    eligible = config.feedback_eligible_trials
    rng = np.random.default_rng(derive_seed(seed, "exp32", "feedback-ranking"))
    ranking = rng.permutation(eligible)
    tapes: dict[float, tuple[BoolArray, str]] = {}
    previous: np.ndarray | None = None
    for fraction in config.feedback_fractions:
        count = int(round(fraction * eligible))
        if count < 1:
            raise ValueError("feedback fraction rounds to zero observations")
        schedule = np.zeros(config.n_trials, dtype=bool)
        schedule[ranking[:count]] = True
        if previous is not None and np.any(schedule & ~previous):
            raise RuntimeError("feedback schedules are not nested")
        digest = _fingerprint(schedule, labels=(seed, fraction, eligible, "feedback"))
        schedule.setflags(write=False)
        tapes[float(fraction)] = (schedule, digest)
        previous = schedule
    return tapes


def make_stream_tape(
    config: NonstationaryActuatorStreamConfig,
    *,
    seed: int,
    hazard: float,
    feedback_fraction: float,
    feedback_delay: int,
) -> NonstationaryStreamTape:
    states, switches, state_digest = make_hidden_state_tape(
        config, seed=seed, hazard=hazard
    )
    feedback = make_nested_feedback_tapes(config, seed=seed)
    feedback_fraction = float(feedback_fraction)
    feedback_delay = int(feedback_delay)
    if feedback_fraction not in feedback:
        raise ValueError("feedback fraction is not registered")
    if feedback_delay not in config.feedback_delays:
        raise ValueError("feedback delay is not registered")
    schedule, schedule_digest = feedback[feedback_fraction]
    return NonstationaryStreamTape(
        hidden_states=states,
        switch_mask=switches,
        feedback_available=schedule,
        hazard=float(hazard),
        feedback_fraction=feedback_fraction,
        feedback_delay=feedback_delay,
        state_fingerprint=state_digest,
        feedback_fingerprint=schedule_digest,
    )


__all__ = [
    "NonstationaryActuatorStreamConfig",
    "NonstationaryStreamTape",
    "make_hidden_state_tape",
    "make_nested_feedback_tapes",
    "make_stream_tape",
]
