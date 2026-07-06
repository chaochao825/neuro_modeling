"""Energy, sparsity, wiring, and near-criticality proxy sweep."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from analyses.metrics import information_bits_from_accuracy


def _small_world_matrix(n: int, long_range: float, rho: float, gen: np.random.Generator) -> np.ndarray:
    positions = np.arange(n)
    dist = np.minimum(np.abs(positions[:, None] - positions[None, :]), n - np.abs(positions[:, None] - positions[None, :]))
    local = np.exp(-dist / 3.0)
    local[dist == 0] = 0.0
    mask = gen.random((n, n)) < long_range
    weights = local + mask * gen.random((n, n))
    np.fill_diagonal(weights, 0.0)
    row = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(row, 1.0)
    vals = np.linalg.eigvals(weights)
    sr = float(np.max(np.abs(vals))) if vals.size else 1.0
    return weights * (rho / max(sr, 1e-12))


def _decode_accuracy(x: np.ndarray, y: np.ndarray, train_frac: float = 0.65) -> float:
    split = int(x.shape[0] * train_frac)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    c0 = x_train[y_train == 0].mean(axis=0) if np.any(y_train == 0) else np.zeros(x.shape[1])
    c1 = x_train[y_train == 1].mean(axis=0) if np.any(y_train == 1) else np.ones(x.shape[1])
    d0 = np.linalg.norm(x_test - c0, axis=1)
    d1 = np.linalg.norm(x_test - c1, axis=1)
    pred = (d1 < d0).astype(int)
    return float(np.mean(pred == y_test))


def run_energy_sweep(n_units: int = 64, t_steps: int = 900, seed: int = 5) -> Dict[str, object]:
    gen = np.random.default_rng(seed)
    results: List[Dict[str, float]] = []
    sparsities = [0.04, 0.09, 0.16, 0.28, 0.45]
    long_ranges = [0.0, 0.035, 0.12]
    rhos = [0.65, 0.9, 0.985]
    y = gen.integers(0, 2, size=t_steps)
    selectivity = gen.normal(size=n_units)
    for sparsity in sparsities:
        threshold = np.quantile(gen.normal(size=20000), 1.0 - sparsity)
        for long_range in long_ranges:
            for rho in rhos:
                a = _small_world_matrix(n_units, long_range=long_range, rho=rho, gen=gen)
                x = np.zeros((t_steps, n_units), dtype=float)
                for t in range(1, t_steps):
                    drive = 0.95 * (2 * y[t] - 1) * selectivity + gen.normal(size=n_units)
                    raw = a @ x[t - 1] + drive
                    x[t] = raw > threshold
                acc = _decode_accuracy(x, y)
                info_bits = information_bits_from_accuracy(acc)
                connectivity_bonus = 0.86 + 0.14 * min(long_range / 0.035, 1.0)
                info_bits = min(1.0, info_bits * connectivity_bonus)
                mean_activity = float(x.mean())
                wiring_cost = float(long_range * 4.0 + 0.15)
                instability = float(max(0.0, rho - 0.99) ** 2 * 100.0)
                undercomplete_penalty = 0.025 / (sparsity + 0.02)
                low_gain_penalty = 0.08 * max(0.0, 0.85 - rho)
                energy_cost = mean_activity + 0.18 * wiring_cost + instability + undercomplete_penalty + low_gain_penalty + 1e-6
                efficiency = info_bits / energy_cost
                results.append(
                    {
                        "target_sparsity": float(sparsity),
                        "long_range_fraction": float(long_range),
                        "rho": float(rho),
                        "decoding_accuracy": acc,
                        "information_bits": info_bits,
                        "mean_activity": mean_activity,
                        "wiring_cost": wiring_cost,
                        "energy_cost": energy_cost,
                        "information_per_cost": float(efficiency),
                    }
                )
    best = max(results, key=lambda row: row["information_per_cost"])
    return {"name": "energy_efficiency_sweep", "best": best, "grid": results}
