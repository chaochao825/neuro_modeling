"""Linear stochastic dynamics and critical-initialization checks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from analyses.metrics import (
    covariance_agreement,
    covariance_eigenspectrum,
    dmd_summary,
    fit_power_law_slope,
    lyapunov_covariance,
)


def _orthogonal(n: int, gen: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(gen.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def make_symmetric_critical_a(n: int, rho: float, alpha: float, gen: np.random.Generator) -> np.ndarray:
    q = _orthogonal(n, gen)
    ranks = np.arange(1, n + 1, dtype=float)
    raw = 1.0 - (1.0 - rho**2) * ranks**alpha
    eig = np.sqrt(np.clip(raw, 0.0, rho**2))
    return q @ np.diag(eig) @ q.T


def make_random_a(n: int, rho: float, symmetry: float, gen: np.random.Generator) -> np.ndarray:
    m = gen.normal(size=(n, n)) / np.sqrt(n)
    sym = 0.5 * (m + m.T)
    asym = m
    a = symmetry * sym + (1.0 - symmetry) * asym
    current = np.max(np.abs(np.linalg.eigvals(a)))
    return a * (rho / max(float(current), 1e-12))


def simulate_linear(a: np.ndarray, t_steps: int, noise_scale: float, seed: int) -> np.ndarray:
    gen = np.random.default_rng(seed)
    n = a.shape[0]
    x = np.zeros((t_steps, n), dtype=float)
    for t in range(1, t_steps):
        x[t] = a @ x[t - 1] + noise_scale * gen.normal(size=n)
    return x


def run_linear_suite(n_units: int = 48, t_steps: int = 1600, seed: int = 2) -> Dict[str, object]:
    gen = np.random.default_rng(seed)
    cases: List[Dict[str, object]] = []
    specs = [
        ("critical_symmetric_powerlaw", 0.985, 1.0, "designed"),
        ("subcritical_symmetric", 0.65, 1.0, "random"),
        ("nearcritical_mixed", 0.96, 0.45, "random"),
        ("nearcritical_asymmetric", 0.96, 0.0, "random"),
    ]
    for idx, (name, rho, symmetry, mode) in enumerate(specs):
        if mode == "designed":
            a = make_symmetric_critical_a(n_units, rho=rho, alpha=2.0 / 3.0, gen=gen)
        else:
            a = make_random_a(n_units, rho=rho, symmetry=symmetry, gen=gen)
        x = simulate_linear(a, t_steps=t_steps, noise_scale=1.0, seed=seed + 100 + idx)
        emp_cov = np.cov(x, rowvar=False)
        pred_cov = lyapunov_covariance(a, np.eye(n_units))
        eig = covariance_eigenspectrum(x)
        cases.append(
            {
                "name": name,
                "target_rho": rho,
                "symmetry": symmetry,
                "empirical_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(a)))),
                "eigenspectrum_power_law": fit_power_law_slope(eig, stop_rank=min(40, eig.size)),
                "dmd": dmd_summary(x),
                "lyapunov_agreement": covariance_agreement(emp_cov, pred_cov),
            }
        )
    return {"name": "linear_stochastic_dynamics", "cases": cases}

