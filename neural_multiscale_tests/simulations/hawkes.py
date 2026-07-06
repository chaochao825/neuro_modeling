"""Discrete-time GLM/Hawkes-like simulator."""

from __future__ import annotations

from typing import Dict

import numpy as np

from analyses.metrics import sigmoid, summarize_activity
from models.glm import compare_nested_glms


def _local_weight(n_units: int, radius: int = 3) -> np.ndarray:
    w = np.zeros((n_units, n_units), dtype=float)
    for i in range(n_units):
        for d in range(1, radius + 1):
            w[i, (i - d) % n_units] = 1.0 / d
            w[i, (i + d) % n_units] = 1.0 / d
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1.0)
    return w


def simulate_hawkes(
    n_units: int = 48,
    t_steps: int = 1200,
    history_strength: float = 1.0,
    local_strength: float = 6.0,
    common_strength: float = 0.08,
    stimulus_strength: float = 0.35,
    seed: int = 1,
) -> Dict[str, object]:
    gen = np.random.default_rng(seed)
    w = _local_weight(n_units)
    spikes = np.zeros((t_steps, n_units), dtype=float)
    tuning = gen.normal(size=n_units)
    stimulus = np.sin(np.linspace(0.0, 8.0 * np.pi, t_steps))
    latent = np.zeros(t_steps)
    for t in range(1, t_steps):
        latent[t] = 0.96 * latent[t - 1] + 0.18 * gen.normal()
        prev = spikes[t - 1]
        eta = (
            -3.55
            + history_strength * prev
            + local_strength * (w @ prev)
            + common_strength * latent[t]
            + stimulus_strength * stimulus[t] * tuning / (np.std(tuning) + 1e-12)
        )
        p = np.clip(sigmoid(eta), 0.0, 0.35)
        spikes[t] = gen.random(n_units) < p
    glm = compare_nested_glms(spikes, stimulus=stimulus, seed=seed)
    return {
        "name": "glm_hawkes_history_local",
        "parameters": {
            "n_units": n_units,
            "t_steps": t_steps,
            "history_strength": history_strength,
            "local_strength": local_strength,
            "common_strength": common_strength,
            "stimulus_strength": stimulus_strength,
        },
        "spikes": spikes,
        "stimulus": stimulus,
        "metrics": summarize_activity(spikes),
        "glm_comparison": glm,
    }
