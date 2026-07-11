"""Leakage-safe hidden-context integration task and capability views.

The task owner retains the latent HMM state, but gate implementations receive
only :class:`GateObservationBatch`.  Task learning and evaluation truth are
separate immutable objects so a gate cannot acquire true context merely by
accepting the generator's complete batch.

All stochastic sources are generated as independent, labelled streams.  In
particular, cue correctness is obtained by thresholding a shared uniform tape.
The same tape can therefore be reused across cue-reliability sweeps without
changing latent states, coherences, or sensory noise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

import numpy as np

from src.utils.reproducibility import derive_seed, make_rng
from src.utils.splits import grouped_train_test_split


Array = np.ndarray


def _validated_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validated_probability(value: object, *, name: str, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar probability")
    result = float(value)
    if not np.isfinite(result) or not minimum <= result <= 1.0:
        raise ValueError(f"{name} must lie in [{minimum}, 1]")
    return result


def _validated_real(
    value: object, *, name: str, minimum: float, strict: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    valid = result > minimum if strict else result >= minimum
    if not np.isfinite(result) or not valid:
        comparison = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be finite and {comparison} {minimum}")
    return result


def _array_copy(
    value: object,
    *,
    name: str,
    dtype: Any,
    shape: tuple[int, ...] | None = None,
) -> Array:
    raw = np.asarray(value)
    if dtype is int and (
        np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise TypeError(f"{name} must contain integers")
    if dtype is bool and raw.dtype != bool and not np.isin(raw, [0, 1]).all():
        raise TypeError(f"{name} must contain booleans")
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _fingerprint(*values: object) -> str:
    """Stable SHA-256 including array dtype and shape."""

    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        elif hasattr(value, "__dataclass_fields__"):
            digest.update(
                json.dumps(asdict(value), sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validated_trial_metadata(
    *,
    n_trials: int,
    trial_ids: object,
    episode_ids: object,
    episode_trial_indices: object,
    episode_start: object,
) -> tuple[Array, Array, Array, Array]:
    trial = _array_copy(trial_ids, name="trial_ids", dtype=int, shape=(n_trials,))
    episode = _array_copy(episode_ids, name="episode_ids", dtype=int, shape=(n_trials,))
    within = _array_copy(
        episode_trial_indices,
        name="episode_trial_indices",
        dtype=int,
        shape=(n_trials,),
    )
    starts = _array_copy(
        episode_start, name="episode_start", dtype=bool, shape=(n_trials,)
    )
    if np.unique(trial).size != n_trials:
        raise ValueError("trial_ids must be unique")
    if np.any(episode < 0) or np.any(within < 0):
        raise ValueError("episode metadata must be non-negative")
    if not n_trials:
        raise ValueError("a batch must contain at least one trial")
    for episode_id in list(dict.fromkeys(episode.tolist())):
        indices = np.flatnonzero(episode == episode_id)
        if not np.array_equal(within[indices], np.arange(indices.size)):
            raise ValueError("episode_trial_indices must be contiguous and zero based")
        expected_start = np.zeros(indices.size, dtype=bool)
        expected_start[0] = True
        if not np.array_equal(starts[indices], expected_start):
            raise ValueError(
                "episode_start must mark exactly the first trial per episode"
            )
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("trials from each episode must remain contiguous")
    return trial, episode, within, starts


def _validated_subset_indices(indices: object, *, length: int) -> Array:
    raw = np.asarray(indices)
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("trial_indices must contain integers")
    result = raw.astype(int, copy=False)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("trial_indices must be a non-empty vector")
    if np.any(result < 0) or np.any(result >= length):
        raise ValueError("trial_indices are out of range")
    if np.unique(result).size != result.size:
        raise ValueError("trial_indices must not contain duplicates")
    return result


@dataclass(frozen=True)
class HiddenContextConfig:
    """Configuration for independent binary-HMM task episodes."""

    n_episodes: int = 20
    trials_per_episode: int = 40
    context_hazard: float = 0.05
    cue_reliability: float = 0.85
    dt_ms: int = 20
    cue_ms: int = 200
    sensory_ms: int = 800
    delay_ms: int = 500
    response_ms: int = 200
    coherence_values: tuple[float, ...] = (-0.5, -0.25, 0.25, 0.5)
    sensory_noise_std: float = 1.0
    input_scale: float = 1.0
    response_target: Literal["choice", "evidence"] = "choice"

    def __post_init__(self) -> None:
        for name, minimum in (
            ("n_episodes", 2),
            ("trials_per_episode", 2),
            ("dt_ms", 1),
            ("cue_ms", 1),
            ("sensory_ms", 1),
            ("delay_ms", 1),
            ("response_ms", 1),
        ):
            object.__setattr__(
                self,
                name,
                _validated_integer(getattr(self, name), name=name, minimum=minimum),
            )
        for name in ("cue_ms", "sensory_ms", "delay_ms", "response_ms"):
            if getattr(self, name) % self.dt_ms:
                raise ValueError(f"{name} must be a positive multiple of dt_ms")
        object.__setattr__(
            self,
            "context_hazard",
            _validated_probability(
                self.context_hazard, name="context_hazard", minimum=0.0
            ),
        )
        object.__setattr__(
            self,
            "cue_reliability",
            _validated_probability(
                self.cue_reliability, name="cue_reliability", minimum=0.5
            ),
        )
        if isinstance(self.coherence_values, (str, bytes)):
            raise TypeError("coherence_values must be a finite numeric sequence")
        try:
            coherence_values = tuple(float(value) for value in self.coherence_values)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "coherence_values must be a finite numeric sequence"
            ) from error
        if (
            not coherence_values
            or not np.isfinite(coherence_values).all()
            or any(value == 0.0 for value in coherence_values)
        ):
            raise ValueError("coherence_values must contain finite non-zero values")
        object.__setattr__(self, "coherence_values", coherence_values)
        object.__setattr__(
            self,
            "sensory_noise_std",
            _validated_real(
                self.sensory_noise_std,
                name="sensory_noise_std",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "input_scale",
            _validated_real(
                self.input_scale, name="input_scale", minimum=0.0, strict=True
            ),
        )
        if self.response_target not in {"choice", "evidence"}:
            raise ValueError("response_target must be 'choice' or 'evidence'")

    @property
    def n_trials(self) -> int:
        return self.n_episodes * self.trials_per_episode

    @property
    def epoch_steps(self) -> dict[str, int]:
        return {
            "cue": self.cue_ms // self.dt_ms,
            "sensory": self.sensory_ms // self.dt_ms,
            "delay": self.delay_ms // self.dt_ms,
            "response": self.response_ms // self.dt_ms,
        }

    @property
    def n_steps(self) -> int:
        return sum(self.epoch_steps.values())


@dataclass(frozen=True)
class HiddenContextRandomTape:
    """Reliability- and hazard-independent random variates for paired sweeps."""

    seed: int
    initial_state_uniform: Array
    transition_uniform: Array
    cue_uniform: Array
    coherence_uniform: Array
    sensory_noise_standard: Array

    def __post_init__(self) -> None:
        seed = _validated_integer(self.seed, name="seed", minimum=0)
        initial = _array_copy(
            self.initial_state_uniform,
            name="initial_state_uniform",
            dtype=float,
        )
        if initial.ndim != 1 or initial.size < 2:
            raise ValueError("initial_state_uniform must contain at least two episodes")
        n_episodes = initial.size
        transition = np.asarray(self.transition_uniform)
        if transition.ndim != 2 or transition.shape[0] != n_episodes:
            raise ValueError("transition_uniform must have shape [episode, trial-1]")
        n_trials = transition.shape[1] + 1
        transition = _array_copy(
            transition,
            name="transition_uniform",
            dtype=float,
            shape=(n_episodes, n_trials - 1),
        )
        cue = _array_copy(
            self.cue_uniform,
            name="cue_uniform",
            dtype=float,
            shape=(n_episodes, n_trials),
        )
        coherence = _array_copy(
            self.coherence_uniform,
            name="coherence_uniform",
            dtype=float,
            shape=(n_episodes, n_trials, 2),
        )
        sensory = np.asarray(self.sensory_noise_standard)
        if sensory.ndim != 4 or sensory.shape[:2] != (n_episodes, n_trials):
            raise ValueError(
                "sensory_noise_standard must have shape [episode, trial, time, 2]"
            )
        sensory = _array_copy(
            sensory,
            name="sensory_noise_standard",
            dtype=float,
            shape=(n_episodes, n_trials, sensory.shape[2], 2),
        )
        for name, array in (
            ("initial_state_uniform", initial),
            ("transition_uniform", transition),
            ("cue_uniform", cue),
            ("coherence_uniform", coherence),
        ):
            if np.any((array < 0.0) | (array >= 1.0)):
                raise ValueError(f"{name} must lie in [0, 1)")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "initial_state_uniform", initial)
        object.__setattr__(self, "transition_uniform", transition)
        object.__setattr__(self, "cue_uniform", cue)
        object.__setattr__(self, "coherence_uniform", coherence)
        object.__setattr__(self, "sensory_noise_standard", sensory)

    @classmethod
    def create(
        cls, config: HiddenContextConfig, *, seed: int
    ) -> "HiddenContextRandomTape":
        if not isinstance(config, HiddenContextConfig):
            raise TypeError("config must be a HiddenContextConfig")
        seed = _validated_integer(seed, name="seed", minimum=0)
        shape = (config.n_episodes, config.trials_per_episode)
        return cls(
            seed=seed,
            initial_state_uniform=make_rng(
                seed, "hidden-context", "initial-state"
            ).random(config.n_episodes),
            transition_uniform=make_rng(seed, "hidden-context", "transition").random(
                (config.n_episodes, config.trials_per_episode - 1)
            ),
            cue_uniform=make_rng(seed, "hidden-context", "cue").random(shape),
            coherence_uniform=make_rng(seed, "hidden-context", "coherence").random(
                (*shape, 2)
            ),
            sensory_noise_standard=make_rng(
                seed, "hidden-context", "sensory-noise"
            ).normal(size=(*shape, config.epoch_steps["sensory"], 2)),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.seed,
            self.initial_state_uniform,
            self.transition_uniform,
            self.cue_uniform,
            self.coherence_uniform,
            self.sensory_noise_standard,
        )

    @property
    def stream_seeds(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (name, derive_seed(self.seed, "hidden-context", label))
            for name, label in (
                ("initial_state", "initial-state"),
                ("transition", "transition"),
                ("cue", "cue"),
                ("coherence", "coherence"),
                ("sensory_noise", "sensory-noise"),
            )
        )

    @property
    def stream_fingerprints(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, _fingerprint(seed, array))
            for (name, seed), array in zip(
                self.stream_seeds,
                (
                    self.initial_state_uniform,
                    self.transition_uniform,
                    self.cue_uniform,
                    self.coherence_uniform,
                    self.sensory_noise_standard,
                ),
                strict=True,
            )
        )


@dataclass(frozen=True)
class GateObservationBatch:
    """The complete and exclusive capability supplied to hidden gate code."""

    cue_observations: Array
    trial_ids: Array
    episode_ids: Array
    episode_trial_indices: Array
    episode_start: Array

    def __post_init__(self) -> None:
        raw = np.asarray(self.cue_observations)
        if raw.ndim != 1:
            raise ValueError("cue_observations must be a one-dimensional vector")
        cue = _array_copy(
            raw,
            name="cue_observations",
            dtype=int,
            shape=(raw.size,),
        )
        if not np.isin(cue, [0, 1]).all():
            raise ValueError("cue_observations must be binary")
        metadata = _validated_trial_metadata(
            n_trials=cue.size,
            trial_ids=self.trial_ids,
            episode_ids=self.episode_ids,
            episode_trial_indices=self.episode_trial_indices,
            episode_start=self.episode_start,
        )
        object.__setattr__(self, "cue_observations", cue)
        for field, value in zip(
            ("trial_ids", "episode_ids", "episode_trial_indices", "episode_start"),
            metadata,
            strict=True,
        ):
            object.__setattr__(self, field, value)

    def subset(self, trial_indices: object) -> "GateObservationBatch":
        indices = _validated_subset_indices(
            trial_indices, length=self.cue_observations.size
        )
        return replace(
            self,
            cue_observations=self.cue_observations[indices],
            trial_ids=self.trial_ids[indices],
            episode_ids=self.episode_ids[indices],
            episode_trial_indices=self.episode_trial_indices[indices],
            episode_start=self.episode_start[indices],
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(*(getattr(self, field.name) for field in fields(self)))


@dataclass(frozen=True)
class TaskLearningBatch:
    """Task-facing observations and teaching targets without latent labels."""

    inputs: Array
    targets: Array
    loss_mask: Array
    trial_ids: Array
    episode_ids: Array
    episode_trial_indices: Array
    episode_start: Array
    epoch: Array
    time_ms: Array
    config: HiddenContextConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, HiddenContextConfig):
            raise TypeError("config must be a HiddenContextConfig")
        inputs = np.asarray(self.inputs)
        if inputs.ndim != 3 or inputs.shape[2] != 4:
            raise ValueError("inputs must have shape [trial, time, 4]")
        n_trials, n_steps, _ = inputs.shape
        inputs = _array_copy(
            inputs, name="inputs", dtype=float, shape=(n_trials, n_steps, 4)
        )
        targets = _array_copy(
            self.targets,
            name="targets",
            dtype=float,
            shape=(n_trials, n_steps, 1),
        )
        loss_mask = _array_copy(
            self.loss_mask,
            name="loss_mask",
            dtype=bool,
            shape=(n_trials, n_steps),
        )
        if n_steps != self.config.n_steps:
            raise ValueError("trial time dimension does not match config")
        metadata = _validated_trial_metadata(
            n_trials=n_trials,
            trial_ids=self.trial_ids,
            episode_ids=self.episode_ids,
            episode_trial_indices=self.episode_trial_indices,
            episode_start=self.episode_start,
        )
        epoch = np.array(self.epoch, dtype="U8", order="C", copy=True)
        time_ms = _array_copy(
            self.time_ms, name="time_ms", dtype=float, shape=(n_steps,)
        )
        if (
            epoch.shape != (n_steps,)
            or not np.isin(epoch, ["cue", "sensory", "delay", "response"]).all()
        ):
            raise ValueError("epoch metadata does not match trial length")
        expected_time = np.arange(n_steps, dtype=float) * self.config.dt_ms
        if not np.array_equal(time_ms, expected_time):
            raise ValueError("time_ms must be the config-defined regular time axis")
        epoch.setflags(write=False)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "loss_mask", loss_mask)
        for field, value in zip(
            ("trial_ids", "episode_ids", "episode_trial_indices", "episode_start"),
            metadata,
            strict=True,
        ):
            object.__setattr__(self, field, value)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "time_ms", time_ms)

    def subset(self, trial_indices: object) -> "TaskLearningBatch":
        indices = _validated_subset_indices(trial_indices, length=self.inputs.shape[0])
        return replace(
            self,
            inputs=self.inputs[indices],
            targets=self.targets[indices],
            loss_mask=self.loss_mask[indices],
            trial_ids=self.trial_ids[indices],
            episode_ids=self.episode_ids[indices],
            episode_trial_indices=self.episode_trial_indices[indices],
            episode_start=self.episode_start[indices],
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(*(getattr(self, field.name) for field in fields(self)))


@dataclass(frozen=True)
class EvaluationTruth:
    """Latent labels available only after gate and behavior outputs are frozen."""

    hidden_states: Array
    choices: Array
    coherences: Array
    switch_mask: Array
    trial_ids: Array
    episode_ids: Array
    episode_trial_indices: Array
    episode_start: Array

    def __post_init__(self) -> None:
        states = np.asarray(self.hidden_states)
        if states.ndim != 1:
            raise ValueError("hidden_states must be one dimensional")
        n_trials = states.size
        states = _array_copy(states, name="hidden_states", dtype=int, shape=(n_trials,))
        choices = _array_copy(
            self.choices, name="choices", dtype=int, shape=(n_trials,)
        )
        coherences = _array_copy(
            self.coherences,
            name="coherences",
            dtype=float,
            shape=(n_trials, 2),
        )
        switches = _array_copy(
            self.switch_mask,
            name="switch_mask",
            dtype=bool,
            shape=(n_trials,),
        )
        if not np.isin(states, [0, 1]).all():
            raise ValueError("hidden_states must be binary")
        if not np.isin(choices, [-1, 1]).all():
            raise ValueError("choices must contain only -1 and +1")
        metadata = _validated_trial_metadata(
            n_trials=n_trials,
            trial_ids=self.trial_ids,
            episode_ids=self.episode_ids,
            episode_trial_indices=self.episode_trial_indices,
            episode_start=self.episode_start,
        )
        expected_switch = np.zeros(n_trials, dtype=bool)
        if n_trials > 1:
            expected_switch[1:] = (metadata[1][1:] == metadata[1][:-1]) & (
                states[1:] != states[:-1]
            )
        if not np.array_equal(switches, expected_switch):
            raise ValueError(
                "switch_mask must mark within-episode hidden-state changes"
            )
        object.__setattr__(self, "hidden_states", states)
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "coherences", coherences)
        object.__setattr__(self, "switch_mask", switches)
        for field, value in zip(
            ("trial_ids", "episode_ids", "episode_trial_indices", "episode_start"),
            metadata,
            strict=True,
        ):
            object.__setattr__(self, field, value)

    def subset(self, trial_indices: object) -> "EvaluationTruth":
        indices = _validated_subset_indices(
            trial_indices, length=self.hidden_states.size
        )
        return replace(
            self,
            hidden_states=self.hidden_states[indices],
            choices=self.choices[indices],
            coherences=self.coherences[indices],
            switch_mask=self.switch_mask[indices],
            trial_ids=self.trial_ids[indices],
            episode_ids=self.episode_ids[indices],
            episode_trial_indices=self.episode_trial_indices[indices],
            episode_start=self.episode_start[indices],
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(*(getattr(self, field.name) for field in fields(self)))


@dataclass(frozen=True)
class HiddenContextDataset:
    """Task, gate, and truth capabilities with shared provenance."""

    task: TaskLearningBatch
    gate: GateObservationBatch
    truth: EvaluationTruth
    source_seed: int
    random_tape_fingerprint: str
    random_stream_seeds: tuple[tuple[str, int], ...]
    random_stream_fingerprints: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskLearningBatch):
            raise TypeError("task must be a TaskLearningBatch")
        if not isinstance(self.gate, GateObservationBatch):
            raise TypeError("gate must be a GateObservationBatch")
        if not isinstance(self.truth, EvaluationTruth):
            raise TypeError("truth must be an EvaluationTruth")
        source_seed = _validated_integer(
            self.source_seed, name="source_seed", minimum=0
        )
        for view in (self.gate, self.truth):
            for name in (
                "trial_ids",
                "episode_ids",
                "episode_trial_indices",
                "episode_start",
            ):
                if not np.array_equal(getattr(self.task, name), getattr(view, name)):
                    raise ValueError(f"capability views disagree on {name}")
        cue_steps = self.task.config.epoch_steps["cue"]
        expected = np.zeros((self.gate.cue_observations.size, cue_steps, 2))
        expected[
            np.arange(expected.shape[0])[:, None],
            np.arange(cue_steps)[None, :],
            self.gate.cue_observations[:, None],
        ] = self.task.config.input_scale
        if not np.array_equal(self.task.inputs[:, :cue_steps, 2:], expected):
            raise ValueError("task cue channels disagree with gate observations")
        if (
            not isinstance(self.random_tape_fingerprint, str)
            or not self.random_tape_fingerprint
        ):
            raise ValueError("random_tape_fingerprint must be non-empty")
        stream_names = [name for name, _ in self.random_stream_seeds]
        fingerprint_names = [name for name, _ in self.random_stream_fingerprints]
        if (
            stream_names != fingerprint_names
            or len(stream_names) != len(set(stream_names))
            or not stream_names
        ):
            raise ValueError("random stream provenance is inconsistent")
        object.__setattr__(self, "source_seed", source_seed)

    @property
    def config(self) -> HiddenContextConfig:
        return self.task.config

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.task.fingerprint,
            self.gate.fingerprint,
            self.truth.fingerprint,
            self.source_seed,
            self.random_tape_fingerprint,
        )

    def subset(self, trial_indices: object) -> "HiddenContextDataset":
        indices = _validated_subset_indices(
            trial_indices, length=self.task.inputs.shape[0]
        )
        selected = set(indices.tolist())
        for episode_id in np.unique(self.task.episode_ids):
            episode_indices = set(
                np.flatnonzero(self.task.episode_ids == episode_id).tolist()
            )
            overlap = selected & episode_indices
            if overlap and overlap != episode_indices:
                raise ValueError("dataset subsets must retain complete episodes")
        return replace(
            self,
            task=self.task.subset(indices),
            gate=self.gate.subset(indices),
            truth=self.truth.subset(indices),
        )

    def train_test_split(
        self, *, test_fraction: float = 0.2, seed: int
    ) -> tuple["HiddenContextDataset", "HiddenContextDataset"]:
        train, test = grouped_train_test_split(
            self.task.episode_ids, test_fraction=test_fraction, seed=seed
        )
        train_dataset = self.subset(train)
        test_dataset = self.subset(test)
        if set(train_dataset.task.episode_ids) & set(test_dataset.task.episode_ids):
            raise RuntimeError("episode leakage detected")
        return train_dataset, test_dataset


def make_hidden_context_random_tape(
    config: HiddenContextConfig, *, seed: int
) -> HiddenContextRandomTape:
    """Create a reusable tape for paired reliability and hazard sweeps."""

    return HiddenContextRandomTape.create(config, seed=seed)


def _validate_tape_shape(
    tape: HiddenContextRandomTape, config: HiddenContextConfig, *, seed: int
) -> None:
    if tape.seed != seed:
        raise ValueError("random tape seed differs from the requested dataset seed")
    expected_trials = (config.n_episodes, config.trials_per_episode)
    if tape.initial_state_uniform.shape != (config.n_episodes,):
        raise ValueError("random tape episode count differs from config")
    if tape.transition_uniform.shape != (
        config.n_episodes,
        config.trials_per_episode - 1,
    ):
        raise ValueError("random tape trial count differs from config")
    if tape.cue_uniform.shape != expected_trials or tape.coherence_uniform.shape != (
        *expected_trials,
        2,
    ):
        raise ValueError("random tape trial shape differs from config")
    if tape.sensory_noise_standard.shape != (
        *expected_trials,
        config.epoch_steps["sensory"],
        2,
    ):
        raise ValueError("random tape sensory time dimension differs from config")


def generate_hidden_context(
    config: HiddenContextConfig | None = None,
    *,
    seed: int,
    random_tape: HiddenContextRandomTape | None = None,
) -> HiddenContextDataset:
    """Generate HMM episodes and return strictly separated capabilities."""

    cfg = config or HiddenContextConfig()
    if not isinstance(cfg, HiddenContextConfig):
        raise TypeError("config must be a HiddenContextConfig")
    seed = _validated_integer(seed, name="seed", minimum=0)
    tape = random_tape or make_hidden_context_random_tape(cfg, seed=seed)
    if not isinstance(tape, HiddenContextRandomTape):
        raise TypeError("random_tape must be a HiddenContextRandomTape")
    _validate_tape_shape(tape, cfg, seed=seed)

    shape = (cfg.n_episodes, cfg.trials_per_episode)
    hidden = np.empty(shape, dtype=int)
    hidden[:, 0] = (tape.initial_state_uniform >= 0.5).astype(int)
    switches = tape.transition_uniform < cfg.context_hazard
    for trial in range(1, cfg.trials_per_episode):
        hidden[:, trial] = np.where(
            switches[:, trial - 1], 1 - hidden[:, trial - 1], hidden[:, trial - 1]
        )
    cue_correct = tape.cue_uniform < cfg.cue_reliability
    cue = np.where(cue_correct, hidden, 1 - hidden).astype(int)

    values = np.asarray(cfg.coherence_values, dtype=float)
    coherence_indices = np.minimum(
        (tape.coherence_uniform * len(values)).astype(int), len(values) - 1
    )
    coherences = values[coherence_indices]
    sensory = (
        coherences[:, :, None, :] + cfg.sensory_noise_std * tape.sensory_noise_standard
    )

    n_trials = cfg.n_trials
    steps = cfg.epoch_steps
    cue_end = steps["cue"]
    sensory_end = cue_end + steps["sensory"]
    delay_end = sensory_end + steps["delay"]
    inputs = np.zeros((n_trials, cfg.n_steps, 4), dtype=float)
    targets = np.zeros((n_trials, cfg.n_steps, 1), dtype=float)
    loss_mask = np.zeros((n_trials, cfg.n_steps), dtype=bool)
    flat_hidden = hidden.reshape(-1)
    flat_cue = cue.reshape(-1)
    flat_sensory = sensory.reshape(n_trials, steps["sensory"], 2)
    for trial in range(n_trials):
        inputs[trial, :cue_end, 2 + flat_cue[trial]] = cfg.input_scale
        inputs[trial, cue_end:sensory_end, :2] = cfg.input_scale * flat_sensory[trial]
        relevant = flat_sensory[trial, :, flat_hidden[trial]]
        trace = np.cumsum(relevant) / np.sqrt(np.arange(1, relevant.size + 1))
        normalized = np.tanh(trace)
        targets[trial, cue_end:sensory_end, 0] = normalized
        final_evidence = float(normalized[-1])
        targets[trial, sensory_end:delay_end, 0] = final_evidence
        choice = 1.0 if relevant.sum() >= 0.0 else -1.0
        targets[trial, delay_end:, 0] = (
            choice if cfg.response_target == "choice" else final_evidence
        )
        loss_mask[trial, cue_end:] = True

    relevant_sensory = flat_sensory[
        np.arange(n_trials)[:, None],
        np.arange(steps["sensory"])[None, :],
        flat_hidden[:, None],
    ]
    choices = np.where(
        relevant_sensory.sum(axis=1) >= 0.0,
        1,
        -1,
    )
    episode_ids = np.repeat(np.arange(cfg.n_episodes), cfg.trials_per_episode)
    episode_trial_indices = np.tile(np.arange(cfg.trials_per_episode), cfg.n_episodes)
    episode_start = episode_trial_indices == 0
    trial_ids = np.arange(n_trials)
    flat_switch = np.zeros(n_trials, dtype=bool)
    flat_switch[~episode_start] = switches.reshape(-1)
    epoch = np.empty(cfg.n_steps, dtype="U8")
    epoch[:cue_end] = "cue"
    epoch[cue_end:sensory_end] = "sensory"
    epoch[sensory_end:delay_end] = "delay"
    epoch[delay_end:] = "response"

    task = TaskLearningBatch(
        inputs=inputs,
        targets=targets,
        loss_mask=loss_mask,
        trial_ids=trial_ids,
        episode_ids=episode_ids,
        episode_trial_indices=episode_trial_indices,
        episode_start=episode_start,
        epoch=epoch,
        time_ms=np.arange(cfg.n_steps, dtype=float) * cfg.dt_ms,
        config=cfg,
    )
    gate = GateObservationBatch(
        cue_observations=flat_cue,
        trial_ids=trial_ids,
        episode_ids=episode_ids,
        episode_trial_indices=episode_trial_indices,
        episode_start=episode_start,
    )
    truth = EvaluationTruth(
        hidden_states=flat_hidden,
        choices=choices,
        coherences=coherences.reshape(n_trials, 2),
        switch_mask=flat_switch,
        trial_ids=trial_ids,
        episode_ids=episode_ids,
        episode_trial_indices=episode_trial_indices,
        episode_start=episode_start,
    )
    return HiddenContextDataset(
        task=task,
        gate=gate,
        truth=truth,
        source_seed=seed,
        random_tape_fingerprint=tape.fingerprint,
        random_stream_seeds=tape.stream_seeds,
        random_stream_fingerprints=tape.stream_fingerprints,
    )


__all__ = [
    "EvaluationTruth",
    "GateObservationBatch",
    "HiddenContextConfig",
    "HiddenContextDataset",
    "HiddenContextRandomTape",
    "TaskLearningBatch",
    "generate_hidden_context",
    "make_hidden_context_random_tape",
]
