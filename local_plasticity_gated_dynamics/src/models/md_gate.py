"""Competitive Hebbian PFC-to-MD inference and MD-to-PFC gain gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class MDGateOutput:
    """One causal gate decision and optional local Hebbian update."""

    scores: Array
    md_activity: Array
    winner: int
    pfc_trace: Array
    pfc_gain: Array
    hebbian_update: Array


def _finite_vector(value: Array, *, name: str, length: int) -> Array:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


class MDGate:
    """Small MD-like competitive layer trained by local Hebbian plasticity.

    PFC activity first updates a causal presynaptic trace.  A hard winner-take-
    all decision selects exactly one MD unit.  When ``learn=True``, only that
    unit's incoming row receives ``eta * modulator * outer(md, trace)`` and is
    then L2-normalized.  A fixed non-negative MD-to-PFC matrix converts the
    winner into positive multiplicative gains for the recurrent network.
    """

    def __init__(
        self,
        n_pfc: int,
        *,
        n_md: int = 4,
        learning_rate: float = 0.01,
        tau_trace: float = 100.0,
        dt: float = 1.0,
        gain_strength: float = 0.5,
        min_gain: float = 1e-6,
        gain_weights: Array | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(n_pfc, (int, np.integer)) or isinstance(n_pfc, bool) or n_pfc <= 0:
            raise ValueError("n_pfc must be a positive integer")
        if not isinstance(n_md, (int, np.integer)) or isinstance(n_md, bool) or not 4 <= n_md <= 8:
            raise ValueError("n_md must be an integer from 4 through 8")
        if not np.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative and finite")
        if not np.isfinite(tau_trace) or tau_trace <= 0.0:
            raise ValueError("tau_trace must be positive and finite")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(gain_strength) or gain_strength < 0.0:
            raise ValueError("gain_strength must be non-negative and finite")
        if not np.isfinite(min_gain) or min_gain <= 0.0:
            raise ValueError("min_gain must be positive and finite")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        self.n_pfc = int(n_pfc)
        self.n_md = int(n_md)
        self.learning_rate = float(learning_rate)
        self.tau_trace = float(tau_trace)
        self.dt = float(dt)
        self.gain_strength = float(gain_strength)
        self.min_gain = float(min_gain)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        pfc_to_md = rng.uniform(0.05, 1.0, size=(self.n_md, self.n_pfc))
        self.pfc_to_md = self._normalize_rows(pfc_to_md)
        if gain_weights is None:
            generated = rng.uniform(0.0, 1.0, size=(self.n_pfc, self.n_md))
            maxima = np.maximum(np.max(generated, axis=0, keepdims=True), 1e-12)
            self.md_to_pfc = generated / maxima
        else:
            provided = np.asarray(gain_weights, dtype=float)
            if provided.shape != (self.n_pfc, self.n_md):
                raise ValueError(f"gain_weights must have shape ({self.n_pfc}, {self.n_md})")
            if not np.all(np.isfinite(provided)) or np.any(provided < 0.0):
                raise ValueError("gain_weights must be finite and non-negative")
            self.md_to_pfc = provided.copy()
        self._pfc_trace = np.zeros(self.n_pfc, dtype=float)

    @staticmethod
    def _normalize_rows(weights: Array) -> Array:
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        return weights / np.maximum(norms, 1e-12)

    @property
    def pfc_trace(self) -> Array:
        return self._pfc_trace.copy()

    @property
    def hebbian_weights(self) -> Array:
        return self.pfc_to_md.copy()

    @property
    def gain_weights(self) -> Array:
        return self.md_to_pfc.copy()

    def reset(self) -> None:
        """Clear only the fast presynaptic trace, retaining learned weights."""

        self._pfc_trace.fill(0.0)

    def update_trace(self, pfc_activity: Array) -> Array:
        """Integrate a causal presynaptic trace exactly for one time step."""

        pfc = _finite_vector(pfc_activity, name="pfc_activity", length=self.n_pfc)
        retention = np.exp(-self.dt / self.tau_trace)
        self._pfc_trace = retention * self._pfc_trace + (1.0 - retention) * pfc
        return self._pfc_trace.copy()

    def _gain_from_md(self, md_activity: Array) -> Array:
        gain = 1.0 + self.gain_strength * (self.md_to_pfc @ md_activity)
        return np.maximum(gain, self.min_gain)

    def step(
        self,
        pfc_activity: Array,
        *,
        learn: bool = False,
        modulatory_signal: float = 1.0,
        md_bias: Array | None = None,
    ) -> MDGateOutput:
        """Infer a WTA context, optionally learn, and return PFC gains."""

        if not isinstance(learn, (bool, np.bool_)):
            raise ValueError("learn must be boolean")
        if (
            isinstance(modulatory_signal, (bool, np.bool_))
            or not np.isscalar(modulatory_signal)
            or not np.isfinite(modulatory_signal)
        ):
            raise ValueError("modulatory_signal must be a finite scalar")
        pfc = _finite_vector(pfc_activity, name="pfc_activity", length=self.n_pfc)
        if md_bias is None:
            bias = np.zeros(self.n_md, dtype=float)
        else:
            bias = _finite_vector(md_bias, name="md_bias", length=self.n_md)
        trace = self.update_trace(pfc)
        scores = self.pfc_to_md @ trace + bias
        winner = int(np.argmax(scores))
        md_activity = np.zeros(self.n_md, dtype=float)
        md_activity[winner] = 1.0

        update = np.zeros_like(self.pfc_to_md)
        if learn and self.learning_rate > 0.0 and modulatory_signal != 0.0:
            update = (
                self.learning_rate
                * self.dt
                * float(modulatory_signal)
                * np.outer(md_activity, trace)
            )
            candidate = self.pfc_to_md[winner] + update[winner]
            norm = np.linalg.norm(candidate)
            if norm > 1e-12:
                normalized = candidate / norm
                update[winner] = normalized - self.pfc_to_md[winner]
                self.pfc_to_md[winner] = normalized
            else:
                update[winner].fill(0.0)

        gain = self._gain_from_md(md_activity)
        return MDGateOutput(
            scores=scores.copy(),
            md_activity=md_activity,
            winner=winner,
            pfc_trace=trace,
            pfc_gain=gain,
            hebbian_update=update,
        )

    observe = step
