"""Mante-style context-dependent evidence integration task.

Trials are generated independently and carry explicit block identifiers so
downstream code can split whole blocks without ever shuffling time points.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from src.utils.reproducibility import make_rng
from src.utils.splits import grouped_train_test_split


@dataclass(frozen=True)
class ContextIntegrationConfig:
    n_trials: int = 400
    dt_ms: int = 20
    cue_ms: int = 200
    sensory_ms: int = 800
    delay_ms: int = 500
    response_ms: int = 200
    context_block_trials: int = 20
    trial_by_trial_after: int | None = None
    coherence_values: tuple[float, ...] = (-0.5, -0.25, 0.25, 0.5)
    sensory_noise_std: float = 1.0
    input_scale: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        integer_fields = (
            "n_trials",
            "dt_ms",
            "cue_ms",
            "sensory_ms",
            "delay_ms",
            "response_ms",
            "context_block_trials",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
        if self.n_trials < 4:
            raise ValueError("n_trials must be at least four")
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        for name in ("cue_ms", "sensory_ms", "delay_ms", "response_ms"):
            value = getattr(self, name)
            if value <= 0 or value % self.dt_ms:
                raise ValueError(f"{name} must be a positive multiple of dt_ms")
        if self.context_block_trials < 1:
            raise ValueError("context_block_trials must be positive")
        if self.trial_by_trial_after is not None:
            if isinstance(self.trial_by_trial_after, (bool, np.bool_)) or not isinstance(
                self.trial_by_trial_after, (int, np.integer)
            ):
                raise TypeError("trial_by_trial_after must be an integer or None")
            if not 1 <= self.trial_by_trial_after < self.n_trials:
                raise ValueError("trial_by_trial_after must lie inside the trial range")
        if isinstance(self.coherence_values, (str, bytes)):
            raise TypeError("coherence_values must be a finite numeric sequence")
        try:
            coherences = tuple(float(value) for value in self.coherence_values)
        except (TypeError, ValueError) as error:
            raise TypeError("coherence_values must be a finite numeric sequence") from error
        if not coherences or not np.isfinite(coherences).all() or any(
            value == 0 for value in coherences
        ):
            raise ValueError("coherence_values must contain non-zero values")
        object.__setattr__(self, "coherence_values", coherences)
        try:
            sensory_noise_std = float(self.sensory_noise_std)
            input_scale = float(self.input_scale)
        except (TypeError, ValueError) as error:
            raise TypeError("sensory_noise_std and input_scale must be numeric") from error
        if not np.isfinite(sensory_noise_std) or sensory_noise_std < 0:
            raise ValueError("sensory_noise_std must be non-negative")
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("input_scale must be positive and finite")
        object.__setattr__(self, "sensory_noise_std", sensory_noise_std)
        object.__setattr__(self, "input_scale", input_scale)

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
class ContextIntegrationBatch:
    """Complete trials plus metadata required for leakage-safe evaluation."""

    inputs: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    contexts: np.ndarray
    choices: np.ndarray
    coherences: np.ndarray
    trial_ids: np.ndarray
    block_ids: np.ndarray
    epoch: np.ndarray
    time_ms: np.ndarray
    config: ContextIntegrationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, ContextIntegrationConfig):
            raise TypeError("config must be a ContextIntegrationConfig")
        inputs = np.array(self.inputs, dtype=float, order="C", copy=True)
        targets = np.array(self.targets, dtype=float, order="C", copy=True)
        raw_loss_mask = np.asarray(self.loss_mask)
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [trial, time, input]")
        n_trials, n_steps, n_inputs = inputs.shape
        if n_inputs != 4:
            raise ValueError("inputs must have channels [sensory1, sensory2, cue1, cue2]")
        if targets.shape != (n_trials, n_steps, 1):
            raise ValueError("targets must have shape [trial, time, 1]")
        if raw_loss_mask.shape != (n_trials, n_steps) or (
            raw_loss_mask.dtype != bool and not np.isin(raw_loss_mask, [0, 1]).all()
        ):
            raise ValueError("loss_mask shape does not match trials")
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
            raise ValueError("inputs and targets must contain only finite values")
        loss_mask = np.array(raw_loss_mask, dtype=bool, copy=True)
        contexts = np.asarray(self.contexts)
        choices = np.asarray(self.choices)
        trial_ids = np.asarray(self.trial_ids)
        block_ids = np.asarray(self.block_ids)
        if (
            contexts.shape != (n_trials,)
            or np.issubdtype(contexts.dtype, np.bool_)
            or not np.issubdtype(contexts.dtype, np.integer)
            or not np.isin(contexts, [0, 1]).all()
        ):
            raise ValueError("contexts must be a binary vector")
        for name, values in (
            ("choices", choices),
            ("trial_ids", trial_ids),
            ("block_ids", block_ids),
        ):
            if values.shape != (n_trials,) or np.issubdtype(
                values.dtype, np.bool_
            ) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} must be an integer vector matching trials")
        if not np.isin(choices, [-1, 1]).all():
            raise ValueError("choices must contain only -1 and +1")
        if np.unique(trial_ids).size != n_trials:
            raise ValueError("trial_ids must be unique")
        if np.any(block_ids < 0):
            raise ValueError("block_ids must be non-negative")
        coherences = np.array(self.coherences, dtype=float, order="C", copy=True)
        if coherences.shape != (n_trials, 2) or not np.isfinite(coherences).all():
            raise ValueError("coherences must have shape [trial, 2]")
        epoch = np.array(self.epoch, dtype="U8", order="C", copy=True)
        time_ms = np.array(self.time_ms, dtype=float, order="C", copy=True)
        if epoch.shape != (n_steps,) or time_ms.shape != (n_steps,):
            raise ValueError("epoch/time metadata does not match trial length")
        if n_steps != self.config.n_steps or not np.isin(
            epoch, ["cue", "sensory", "delay", "response"]
        ).all():
            raise ValueError("epoch metadata does not match config")
        expected_time = np.arange(n_steps, dtype=float) * self.config.dt_ms
        if not np.isfinite(time_ms).all() or not np.array_equal(time_ms, expected_time):
            raise ValueError("time_ms must be the config-defined regular time axis")
        replacements = {
            "inputs": inputs,
            "targets": targets,
            "loss_mask": loss_mask,
            "contexts": np.array(contexts, dtype=int, copy=True),
            "choices": np.array(choices, dtype=int, copy=True),
            "coherences": coherences,
            "trial_ids": np.array(trial_ids, dtype=int, copy=True),
            "block_ids": np.array(block_ids, dtype=int, copy=True),
            "epoch": epoch,
            "time_ms": time_ms,
        }
        for name, values in replacements.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def subset(self, trial_indices: np.ndarray) -> "ContextIntegrationBatch":
        raw_indices = np.asarray(trial_indices)
        if np.issubdtype(raw_indices.dtype, np.bool_) or not np.issubdtype(
            raw_indices.dtype, np.integer
        ):
            raise TypeError("trial_indices must contain integers")
        indices = raw_indices.astype(int, copy=False)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.inputs.shape[0]):
            raise ValueError("trial_indices are out of range")
        if np.unique(indices).size != indices.size:
            raise ValueError("trial_indices must not contain duplicates")
        return replace(
            self,
            inputs=self.inputs[indices],
            targets=self.targets[indices],
            loss_mask=self.loss_mask[indices],
            contexts=self.contexts[indices],
            choices=self.choices[indices],
            coherences=self.coherences[indices],
            trial_ids=self.trial_ids[indices],
            block_ids=self.block_ids[indices],
        )

    def train_test_split(
        self,
        *,
        test_fraction: float = 0.2,
        seed: int,
        require_context_coverage: bool = True,
    ) -> tuple["ContextIntegrationBatch", "ContextIntegrationBatch"]:
        if not isinstance(require_context_coverage, (bool, np.bool_)):
            raise TypeError("require_context_coverage must be boolean")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        all_contexts = set(self.contexts.tolist())
        for attempt in range(100):
            train, test = grouped_train_test_split(
                self.block_ids,
                test_fraction=test_fraction,
                seed=int(seed) + attempt,
            )
            if not require_context_coverage or (
                set(self.contexts[train].tolist()) == all_contexts
                and set(self.contexts[test].tolist()) == all_contexts
            ):
                return self.subset(train), self.subset(test)
        raise ValueError(
            "could not split whole blocks while retaining both contexts in train and test"
        )


def _context_schedule(config: ContextIntegrationConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    contexts = np.empty(config.n_trials, dtype=int)
    block_ids = np.empty(config.n_trials, dtype=int)
    block_id = 0
    first_context = int(rng.integers(0, 2))
    previous: int | None = None
    for trial in range(config.n_trials):
        if config.trial_by_trial_after is not None and trial >= config.trial_by_trial_after:
            context = int(rng.integers(0, 2))
        else:
            context = (first_context + trial // config.context_block_trials) % 2
        if previous is not None:
            if config.trial_by_trial_after is not None and trial >= config.trial_by_trial_after:
                # In the late regime every trial is its own scheduling block,
                # even if two independently sampled contexts happen to match.
                block_id += 1
            elif context != previous:
                block_id += 1
        contexts[trial] = context
        block_ids[trial] = block_id
        previous = context
    return contexts, block_ids


def generate_context_integration(
    config: ContextIntegrationConfig | None = None,
    *,
    seed: int,
    response_target: Literal["choice", "evidence"] = "choice",
) -> ContextIntegrationBatch:
    """Generate independent complete trials with block-structured contexts.

    The continuous target follows normalized accumulated relevant evidence in
    the sensory epoch, is held through the delay, and becomes either its sign
    or its value in the response epoch. This supplies a causal low-dimensional
    teaching signal for local three-factor updates.
    """

    cfg = config or ContextIntegrationConfig()
    if not isinstance(cfg, ContextIntegrationConfig):
        raise TypeError("config must be a ContextIntegrationConfig")
    cfg.validate()
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not isinstance(response_target, str) or response_target not in {
        "choice",
        "evidence",
    }:
        raise ValueError("response_target must be 'choice' or 'evidence'")

    schedule_rng = make_rng(seed, "context_schedule")
    evidence_rng = make_rng(seed, "coherences")
    noise_rng = make_rng(seed, "sensory_noise")
    contexts, block_ids = _context_schedule(cfg, schedule_rng)
    coherences = evidence_rng.choice(
        np.asarray(cfg.coherence_values, dtype=float), size=(cfg.n_trials, 2), replace=True
    )

    epoch_steps = cfg.epoch_steps
    cue_end = epoch_steps["cue"]
    sensory_end = cue_end + epoch_steps["sensory"]
    delay_end = sensory_end + epoch_steps["delay"]
    n_steps = cfg.n_steps
    inputs = np.zeros((cfg.n_trials, n_steps, 4), dtype=float)
    targets = np.zeros((cfg.n_trials, n_steps, 1), dtype=float)
    loss_mask = np.zeros((cfg.n_trials, n_steps), dtype=bool)

    epoch = np.empty(n_steps, dtype="U8")
    epoch[:cue_end] = "cue"
    epoch[cue_end:sensory_end] = "sensory"
    epoch[sensory_end:delay_end] = "delay"
    epoch[delay_end:] = "response"

    for trial in range(cfg.n_trials):
        context = contexts[trial]
        inputs[trial, :cue_end, 2 + context] = cfg.input_scale
        sensory = coherences[trial][None, :] + noise_rng.normal(
            0.0, cfg.sensory_noise_std, size=(epoch_steps["sensory"], 2)
        )
        inputs[trial, cue_end:sensory_end, :2] = cfg.input_scale * sensory
        relevant = sensory[:, context]
        trace = np.cumsum(relevant) / np.sqrt(np.arange(1, relevant.size + 1))
        # A fixed causal nonlinearity avoids using the future maximum of this
        # trial to rescale earlier teaching targets.
        normalized = np.tanh(trace)
        targets[trial, cue_end:sensory_end, 0] = normalized
        final_evidence = float(normalized[-1])
        targets[trial, sensory_end:delay_end, 0] = final_evidence
        choices_value = 1.0 if relevant.sum() >= 0.0 else -1.0
        response_value = choices_value if response_target == "choice" else final_evidence
        targets[trial, delay_end:, 0] = response_value
        loss_mask[trial, cue_end:] = True

    choices = np.where(targets[:, -1, 0] >= 0.0, 1, -1).astype(int)
    return ContextIntegrationBatch(
        inputs=inputs,
        targets=targets,
        loss_mask=loss_mask,
        contexts=contexts,
        choices=choices,
        coherences=coherences,
        trial_ids=np.arange(cfg.n_trials, dtype=int),
        block_ids=block_ids,
        epoch=epoch,
        time_ms=np.arange(n_steps, dtype=float) * cfg.dt_ms,
        config=cfg,
    )
