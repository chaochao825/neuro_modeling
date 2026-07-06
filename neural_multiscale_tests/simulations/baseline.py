"""Independent Bernoulli baseline."""

from __future__ import annotations

from typing import Dict

import numpy as np

from analyses.metrics import summarize_activity


def run_baseline(n_units: int = 48, t_steps: int = 1200, rate: float = 0.025, seed: int = 0) -> Dict[str, object]:
    gen = np.random.default_rng(seed)
    spikes = (gen.random((t_steps, n_units)) < rate).astype(float)
    return {
        "name": "baseline_independent_bernoulli",
        "parameters": {"n_units": n_units, "t_steps": t_steps, "rate": rate},
        "spikes": spikes,
        "metrics": summarize_activity(spikes),
    }

