"""Small dependency-free logistic GLM comparisons for spike matrices."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from analyses.metrics import EPS, sigmoid


def _standardize_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.shape[1] == 0:
        return x, np.zeros(0), np.ones(0)
    mu = x.mean(axis=0)
    sig = x.std(axis=0) + EPS
    return (x - mu) / sig, mu, sig


def fit_logistic_ridge(x: np.ndarray, y: np.ndarray, steps: int = 160, lr: float = 0.12, l2: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.zeros(x.shape[1], dtype=float)
    for step in range(steps):
        p = sigmoid(x @ w)
        grad = x.T @ (p - y) / max(1, y.size)
        grad[1:] += l2 * w[1:]
        w -= (lr / (1.0 + 0.01 * step)) * grad
    return w


def log_likelihood(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(sigmoid(x @ w), EPS, 1.0 - EPS)
    y = y.ravel()
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _ring_local_sum(prev: np.ndarray, radius: int = 3) -> np.ndarray:
    out = np.zeros_like(prev, dtype=float)
    for shift in range(1, radius + 1):
        out += np.roll(prev, shift, axis=1) + np.roll(prev, -shift, axis=1)
    return out / max(1, 2 * radius)


def _features_for_unit(
    spikes: np.ndarray,
    unit: int,
    stimulus: np.ndarray | None,
    model: str,
    local_radius: int,
    *,
    include_index_local: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    prev = spikes[:-1]
    y = spikes[1:, unit]
    cols: List[np.ndarray] = [np.ones(prev.shape[0])]
    if model in {"history", "history_local", "history_global", "full"}:
        cols.append(prev[:, unit])
    if include_index_local and model in {"history_local", "full"}:
        cols.append(_ring_local_sum(prev, radius=local_radius)[:, unit])
    if model in {"history_global", "full"}:
        global_prev = (prev.sum(axis=1) - prev[:, unit]) / max(1, prev.shape[1] - 1)
        cols.append(global_prev)
    if stimulus is not None and model == "full":
        cols.append(np.asarray(stimulus[:-1], dtype=float).ravel())
    x = np.column_stack(cols)
    return x, y


def _split_standardize_features(
    x: np.ndarray, split: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit feature scaling on the training prefix and apply it to both splits."""

    if not 0 < split < x.shape[0]:
        raise ValueError("split must leave non-empty train and test prefixes")
    train = np.asarray(x[:split], dtype=float).copy()
    test = np.asarray(x[split:], dtype=float).copy()
    if x.shape[1] > 1:
        train_scaled, mu, sig = _standardize_features(train[:, 1:])
        test_scaled = (test[:, 1:] - mu) / sig
        train = np.column_stack([np.ones(train.shape[0]), train_scaled])
        test = np.column_stack([np.ones(test.shape[0]), test_scaled])
    return train, test


def compare_nested_glms(
    spikes: np.ndarray,
    stimulus: np.ndarray | None = None,
    train_frac: float = 0.7,
    max_units: int = 16,
    local_radius: int = 3,
    seed: int = 0,
    use_index_ring: bool = True,
) -> Dict[str, object]:
    arr = np.asarray(spikes, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 20:
        raise ValueError("spikes must be time x units with at least 20 bins")
    gen = np.random.default_rng(seed)
    units = np.arange(arr.shape[1])
    if units.size > max_units:
        units = np.sort(gen.choice(units, size=max_units, replace=False))
    models = ["bias", "history", "history_global", "full"]
    if use_index_ring:
        models.insert(2, "history_local")
    per_model: Dict[str, List[float]] = {m: [] for m in models}
    per_model_aic: Dict[str, List[float]] = {m: [] for m in models}
    split = int((arr.shape[0] - 1) * train_frac)
    for unit in units:
        for name in models:
            x, y = _features_for_unit(
                arr,
                int(unit),
                stimulus,
                name,
                local_radius,
                include_index_local=use_index_ring,
            )
            x_train, x_test = _split_standardize_features(x, split)
            y_train, y_test = y[:split], y[split:]
            w = fit_logistic_ridge(x_train, y_train)
            ll = log_likelihood(x_test, y_test, w)
            per_model[name].append(ll / max(1, y_test.size))
            per_model_aic[name].append(2 * x.shape[1] - 2 * log_likelihood(x_train, y_train, w))
    mean_ll = {name: float(np.mean(vals)) for name, vals in per_model.items()}
    mean_aic = {name: float(np.mean(vals)) for name, vals in per_model_aic.items()}
    base = mean_ll["bias"]
    bits_gain = {name: float((ll - base) / np.log(2.0)) for name, ll in mean_ll.items()}
    return {
        "units_fit": int(units.size),
        "test_loglik_per_bin": mean_ll,
        "bits_per_bin_gain_vs_bias": bits_gain,
        "mean_train_aic": mean_aic,
        "history_delta_bits": float(bits_gain["history"] - bits_gain["bias"]),
        "local_delta_bits": (
            float(bits_gain["history_local"] - bits_gain["history"])
            if use_index_ring
            else None
        ),
        "global_delta_bits": float(bits_gain["history_global"] - bits_gain["history"]),
        "full_delta_bits": float(
            bits_gain["full"]
            - (
                max(bits_gain["history_local"], bits_gain["history_global"])
                if use_index_ring
                else bits_gain["history_global"]
            )
        ),
        "local_structure": (
            "index_ring" if use_index_ring else "not_evaluated_missing_coordinates"
        ),
    }

