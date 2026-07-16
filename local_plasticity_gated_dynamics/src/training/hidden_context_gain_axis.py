"""Train/dev/test-safe local gain-axis proposal tapes for hidden context.

The train split fits the belief gate and neutral readout elsewhere.  This
module consumes only a frozen neutral-gain dev trajectory and constructs
neuron-local eligibility traces.  A scalar task error and scalar third factor
then produce trial-wise gain-axis proposals without BPTT or recurrent-weight
updates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.models.ei_rate_network import EIRateNetwork
from src.plasticity.gain_axis import GainAxisThreeFactorRule
from src.tasks.hidden_context import TaskLearningBatch
from src.training.hidden_context_ei import ReceiverReadout, ReceiverSimulation


FloatArray = NDArray[np.float64]


def _readonly(value: ArrayLike, *, dtype: object = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _fingerprint(label: str, *values: object) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _targets(task: TaskLearningBatch) -> FloatArray:
    response = np.flatnonzero(np.asarray(task.epoch) == "response")
    if response.size == 0 or not np.all(task.loss_mask[:, response]):
        raise ValueError("response epoch must be fully supervised")
    values = np.asarray(task.targets[:, response, 0], dtype=np.float64)
    if not np.all(values == values[:, :1]):
        raise ValueError("response target must be constant within each trial")
    targets = values[:, 0]
    if not np.isin(targets, (-1.0, 1.0)).all():
        raise ValueError("gain-axis learning requires binary -1/+1 targets")
    return targets


@dataclass(frozen=True, slots=True)
class GainAxisLocalTape:
    """Frozen dev-only eligibility and scalar-error tape."""

    eligibility: FloatArray
    feedback_coefficients: FloatArray
    feedback_schedule: FloatArray
    feedback_policy: str
    task_error: FloatArray
    neutral_scores: FloatArray
    targets: FloatArray
    trial_ids: NDArray[np.int64]
    episode_ids: NDArray[np.int64]
    tau_eligibility_steps: float
    local_feedback_scope: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class GainAxisProposalTape:
    """Trial-wise vector proposals for one scalar third-factor trajectory."""

    proposals: FloatArray
    third_factor: FloatArray
    clipped_task_error: FloatArray
    learning_rate: float
    error_clip: float | None
    local_tape_fingerprint: str
    fingerprint: str


def build_gain_axis_local_tape(
    network: EIRateNetwork,
    simulation: ReceiverSimulation,
    task: TaskLearningBatch,
    readout: ReceiverReadout,
    *,
    integration_substeps: int,
    tau_eligibility_steps: float,
    feedback_coefficients: ArrayLike | None = None,
    feedback_policy: str = "readout_aligned",
) -> GainAxisLocalTape:
    """Build dev eligibility from a neutral population-gain trajectory only."""

    if not isinstance(network, EIRateNetwork):
        raise TypeError("network must be an EIRateNetwork")
    if not isinstance(simulation, ReceiverSimulation):
        raise TypeError("simulation must be a ReceiverSimulation")
    if not isinstance(readout, ReceiverReadout):
        raise TypeError("readout must be a ReceiverReadout")
    if simulation.trajectory_sequence_scope != "trial_reset_state":
        raise ValueError("gain-axis local tape requires per-trial reset trajectories")
    if simulation.full_x_trajectory is None:
        raise ValueError("simulation must record every integration substep")
    if simulation.population_gain and not np.allclose(
        simulation.gain.gains, 1.0, atol=0.0, rtol=0.0
    ):
        raise ValueError("local tape must be generated at neutral population gain")
    if (
        isinstance(integration_substeps, (bool, np.bool_))
        or not isinstance(integration_substeps, (int, np.integer))
        or int(integration_substeps) < 1
    ):
        raise ValueError("integration_substeps must be a positive integer")
    substeps = int(integration_substeps)
    tau = _positive(tau_eligibility_steps, name="tau_eligibility_steps")
    n_trials = task.trial_ids.size
    n_steps = task.inputs.shape[1]
    expected = (n_trials, n_steps * substeps + 1, network.n_units)
    if simulation.full_x_trajectory.shape != expected:
        raise ValueError("recorded state trajectory has the wrong shape")
    if simulation.features.shape != (n_trials, 3 * network.n_units):
        raise ValueError("neutral receiver features have the wrong shape")
    if readout.weights.shape != (3 * network.n_units,):
        raise ValueError("readout weights do not match three receiver epochs")

    coarse_indices = np.arange(
        substeps,
        n_steps * substeps + 1,
        substeps,
    )
    coarse_x = np.asarray(
        simulation.full_x_trajectory[:, coarse_indices], dtype=np.float64
    )
    aligned_beta = (readout.weights / readout.scale).reshape(3, network.n_units)
    if feedback_coefficients is None:
        beta = np.asarray(aligned_beta, dtype=np.float64)
    else:
        raw_feedback = np.asarray(feedback_coefficients)
        if raw_feedback.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
            raise TypeError("feedback_coefficients must be real numeric")
        beta = np.asarray(raw_feedback, dtype=np.float64)
        if beta.shape != aligned_beta.shape or not np.all(np.isfinite(beta)):
            raise ValueError(
                f"feedback_coefficients must be finite with shape {aligned_beta.shape}"
            )
    if not isinstance(feedback_policy, str) or not feedback_policy:
        raise ValueError("feedback_policy must be a non-empty string")
    epochs = np.asarray(task.epoch, dtype="U8")
    active_names = ("sensory", "delay", "response")
    feedback = np.zeros((n_steps, network.n_units), dtype=np.float64)
    for index, name in enumerate(active_names):
        mask = epochs == name
        count = int(np.count_nonzero(mask))
        if count < 1:
            raise ValueError(f"task has no {name} epoch")
        feedback[mask] = beta[index] / count

    eligibility = np.zeros((n_trials, network.n_units), dtype=np.float64)
    eligibility_rule = GainAxisThreeFactorRule(
        learning_rate=1.0,
        tau_eligibility=tau,
        dt=1.0,
    )
    for trial in range(n_trials):
        eligibility_rule.reset(network.n_units)
        for time in range(n_steps):
            activated = np.tanh(coarse_x[trial, time])
            derivative = 1.0 - activated * activated
            if network.activation_name == "rectified_tanh":
                derivative = np.where(activated > 0.0, derivative, 0.0)
            local_drive = feedback[time] * coarse_x[trial, time] * derivative
            eligibility_rule.update_eligibility(local_drive)
        trace = eligibility_rule.eligibility_trace
        if trace is None:
            raise RuntimeError("gain-axis eligibility rule did not initialize")
        eligibility[trial] = trace

    targets = _targets(task)
    scores = readout.scores(simulation.features)
    task_error = targets - scores
    fingerprint = _fingerprint(
        "hidden-context-gain-axis-local-tape-v1",
        simulation.trajectory_fingerprint,
        readout.checkpoint_id,
        task.fingerprint,
        coarse_x,
        feedback,
        beta,
        feedback_policy,
        eligibility,
        task_error,
        tau,
    )
    return GainAxisLocalTape(
        eligibility=_readonly(eligibility),
        feedback_coefficients=_readonly(beta),
        feedback_schedule=_readonly(feedback),
        feedback_policy=feedback_policy,
        task_error=_readonly(task_error),
        neutral_scores=_readonly(scores),
        targets=_readonly(targets),
        trial_ids=_readonly(task.trial_ids, dtype=np.int64),
        episode_ids=_readonly(task.episode_ids, dtype=np.int64),
        tau_eligibility_steps=tau,
        local_feedback_scope=(
            f"{feedback_policy}_gain_axis_three_factor_rule_eligibility_with_"
            "fixed_feedback_coefficient_times_same_neuron_local_state_and_"
            "activation_derivative_no_weight_transport_free_claim"
        ),
        fingerprint=fingerprint,
    )


def make_gain_axis_proposal_tape(
    local_tape: GainAxisLocalTape,
    third_factor: ArrayLike,
    *,
    learning_rate: float,
    error_clip: float | None,
) -> GainAxisProposalTape:
    """Combine frozen local eligibility with scalar error and third factor."""

    if not isinstance(local_tape, GainAxisLocalTape):
        raise TypeError("local_tape must be a GainAxisLocalTape")
    rate = _positive(learning_rate, name="learning_rate")
    third_raw = np.asarray(third_factor)
    if third_raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("third_factor must be real numeric")
    third = np.asarray(third_raw, dtype=np.float64)
    if (
        third.shape != local_tape.task_error.shape
        or not np.all(np.isfinite(third))
        or np.any((third < -1.0) | (third > 1.0))
    ):
        raise ValueError("third_factor must align with trials and lie in [-1, 1]")
    clip = None if error_clip is None else _positive(error_clip, name="error_clip")
    clipped = (
        np.asarray(local_tape.task_error, dtype=np.float64)
        if clip is None
        else np.clip(local_tape.task_error, -clip, clip)
    )
    proposals = rate * local_tape.eligibility * clipped[:, None] * third[:, None]
    fingerprint = _fingerprint(
        "hidden-context-gain-axis-proposal-tape-v1",
        local_tape.fingerprint,
        third,
        clipped,
        proposals,
        rate,
        clip,
    )
    return GainAxisProposalTape(
        proposals=_readonly(proposals),
        third_factor=_readonly(third),
        clipped_task_error=_readonly(clipped),
        learning_rate=rate,
        error_clip=clip,
        local_tape_fingerprint=local_tape.fingerprint,
        fingerprint=fingerprint,
    )


__all__ = [
    "GainAxisLocalTape",
    "GainAxisProposalTape",
    "build_gain_axis_local_tape",
    "make_gain_axis_proposal_tape",
]
