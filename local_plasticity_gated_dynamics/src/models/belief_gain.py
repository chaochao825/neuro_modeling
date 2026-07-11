"""Low-dimensional belief-to-gain control for E/I receiver networks.

This module is intentionally incapable of receiving hidden-context labels.  It
maps a frozen binary posterior to a rank-one population gain perturbation.  The
axis is centered separately within excitatory and inhibitory populations so
the manipulation does not introduce a trivial mean E/I gain shift.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _finite_vector(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric vector")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _fingerprint(*arrays: ArrayLike | object) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def balanced_gain_axis(excitatory_mask: ArrayLike, *, seed: int) -> FloatArray:
    """Create a deterministic axis with zero mean inside both E and I groups."""

    raw_mask = np.asarray(excitatory_mask)
    if raw_mask.ndim != 1 or raw_mask.size < 2 or raw_mask.dtype.kind != "b":
        raise ValueError("excitatory_mask must be a one-dimensional boolean mask")
    if np.all(raw_mask) or not np.any(raw_mask):
        raise ValueError("excitatory_mask must contain both E and I units")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    axis = np.random.default_rng(int(seed)).normal(size=raw_mask.size)
    for population in (raw_mask, ~raw_mask):
        if np.count_nonzero(population) == 1:
            axis[population] = 0.0
        else:
            axis[population] -= np.mean(axis[population])
    scale = float(np.max(np.abs(axis)))
    if scale == 0.0:
        raise ValueError("gain axis is degenerate for the requested populations")
    return _readonly(axis / scale)


def gain_control_rank(gains: ArrayLike, *, atol: float = 1e-12) -> int:
    """Rank of gain deviations from neutral across all trial/time samples."""

    raw = np.asarray(gains)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("gains must be real numeric")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 3 or 0 in array.shape or not np.all(np.isfinite(array)):
        raise ValueError("gains must have finite shape [trial, time, unit]")
    if np.any(array <= 0.0):
        raise ValueError("all gains must be positive")
    if not np.isfinite(float(atol)) or float(atol) < 0.0:
        raise ValueError("atol must be non-negative and finite")
    deviations = (array - 1.0).reshape(-1, array.shape[-1])
    return int(np.linalg.matrix_rank(deviations, tol=float(atol)))


@dataclass(frozen=True)
class BeliefGainTrajectory:
    """Frozen gain tensor and audit metadata for one dataset split."""

    gains: FloatArray
    axis: FloatArray
    posterior_state1: FloatArray
    active_time_mask: NDArray[np.bool_]
    strength: float
    control_rank: int
    fingerprint: str


def belief_gain_trajectory(
    posterior_state1: ArrayLike,
    epoch_labels: ArrayLike,
    axis: ArrayLike,
    *,
    strength: float,
    neutral_epochs: tuple[str, ...] = ("cue",),
) -> BeliefGainTrajectory:
    """Map causal posterior ``p(z=1)`` to ``1 + alpha*a*(2p-1)`` gains.

    The posterior is the only trial-varying input.  Hidden-state truth is not a
    parameter of this function.  ``strength < 1`` together with ``|axis|<=1``
    guarantees strictly positive gains.
    """

    posterior = _finite_vector(posterior_state1, name="posterior_state1")
    if np.any((posterior < 0.0) | (posterior > 1.0)):
        raise ValueError("posterior_state1 must lie in [0, 1]")
    population_axis = _finite_vector(axis, name="axis")
    if np.max(np.abs(population_axis)) > 1.0 + 1e-12:
        raise ValueError("axis maximum absolute value must not exceed one")
    if (
        isinstance(strength, (bool, np.bool_))
        or not np.isscalar(strength)
        or not np.isfinite(float(strength))
        or not 0.0 <= float(strength) < 1.0
    ):
        raise ValueError("strength must be a finite scalar in [0, 1)")
    raw_epoch = np.asarray(epoch_labels)
    if raw_epoch.ndim != 1 or raw_epoch.size == 0:
        raise ValueError("epoch_labels must be a non-empty vector")
    epochs = np.asarray(raw_epoch, dtype="U32")
    if not isinstance(neutral_epochs, tuple) or any(
        not isinstance(item, str) or not item for item in neutral_epochs
    ):
        raise ValueError("neutral_epochs must be a tuple of non-empty strings")
    active = ~np.isin(epochs, neutral_epochs)
    signed_belief = 2.0 * posterior - 1.0
    modulation = (
        float(strength)
        * signed_belief[:, np.newaxis, np.newaxis]
        * active[np.newaxis, :, np.newaxis]
        * population_axis[np.newaxis, np.newaxis, :]
    )
    gains = 1.0 + modulation
    if np.any(gains <= 0.0):
        raise RuntimeError("validated belief-to-gain map produced a non-positive gain")
    frozen_gains = _readonly(gains)
    frozen_active = np.array(active, dtype=bool, copy=True)
    frozen_active.setflags(write=False)
    rank = gain_control_rank(frozen_gains)
    return BeliefGainTrajectory(
        gains=frozen_gains,
        axis=_readonly(population_axis),
        posterior_state1=_readonly(posterior),
        active_time_mask=frozen_active,
        strength=float(strength),
        control_rank=rank,
        fingerprint=_fingerprint(
            "belief-gain-v1", frozen_gains, population_axis, posterior, epochs
        ),
    )


__all__ = [
    "BeliefGainTrajectory",
    "balanced_gain_axis",
    "belief_gain_trajectory",
    "gain_control_rank",
]
