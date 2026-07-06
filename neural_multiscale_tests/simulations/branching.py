"""Branching-process avalanche simulations."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from analyses.metrics import branching_ratio_counts, compare_tail_models, extract_avalanches


def simulate_branching_counts(
    m: float,
    t_steps: int = 4000,
    drive: float = 0.015,
    max_count: int = 700,
    seed: int = 0,
) -> np.ndarray:
    gen = np.random.default_rng(seed)
    counts = np.zeros(t_steps, dtype=float)
    counts[0] = 1
    for t in range(1, t_steps):
        if counts[t - 1] <= 0:
            counts[t] = 1 if gen.random() < drive else 0
        else:
            lam = min(max_count, m * counts[t - 1])
            counts[t] = min(max_count, gen.poisson(lam))
    return counts


def _dynamic_range_proxy(m: float, seed: int) -> Dict[str, float]:
    drives = np.geomspace(0.002, 0.18, 10)
    responses = []
    for i, d in enumerate(drives):
        c = simulate_branching_counts(m=m, t_steps=1600, drive=float(d), seed=seed + i, max_count=400)
        responses.append(float(np.mean(c)))
    resp = np.asarray(responses)
    lo = resp.min()
    hi = resp.max()
    if hi <= lo + 1e-12:
        return {"dynamic_range_db": 0.0, "mean_response": float(resp.mean())}
    r10 = lo + 0.1 * (hi - lo)
    r90 = lo + 0.9 * (hi - lo)
    d10 = float(np.interp(r10, resp, drives))
    d90 = float(np.interp(r90, resp, drives))
    return {"dynamic_range_db": float(10.0 * np.log10((d90 + 1e-12) / (d10 + 1e-12))), "mean_response": float(resp.mean())}


def run_branching_suite(t_steps: int = 4500, seed: int = 3) -> Dict[str, object]:
    cases: List[Dict[str, object]] = []
    for i, m in enumerate([0.75, 0.9, 1.0, 1.08]):
        counts = simulate_branching_counts(m=m, t_steps=t_steps, seed=seed + i)
        sizes, durations = extract_avalanches(counts)
        cases.append(
            {
                "m": m,
                "estimated_branching_ratio": branching_ratio_counts(counts),
                "n_avalanches": int(sizes.size),
                "size_tail": compare_tail_models(sizes),
                "duration_tail": compare_tail_models(durations),
                "dynamic_range": _dynamic_range_proxy(m, seed + 50 + i),
            }
        )
    return {"name": "branching_avalanche", "cases": cases}

