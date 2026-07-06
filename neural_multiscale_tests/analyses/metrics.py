"""Shared metrics for simulated and recorded neural population activity."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def ensure_time_by_unit(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array shaped time x units.")
    return arr


def autocorrelation_mean(x: np.ndarray, max_lag: int = 40) -> Dict[str, object]:
    arr = ensure_time_by_unit(x)
    centered = arr - arr.mean(axis=0, keepdims=True)
    var = np.mean(centered * centered, axis=0) + EPS
    vals: List[float] = []
    for lag in range(1, max_lag + 1):
        prod = centered[:-lag] * centered[lag:]
        vals.append(float(np.mean(np.mean(prod, axis=0) / var)))
    return {
        "max_lag": max_lag,
        "mean_abs": float(np.mean(np.abs(vals))) if vals else 0.0,
        "lag_values": vals,
    }


def cross_correlation_summary(x: np.ndarray) -> Dict[str, float]:
    arr = ensure_time_by_unit(x)
    if arr.shape[1] < 2:
        return {"mean_abs_offdiag": 0.0, "max_abs_offdiag": 0.0}
    centered = arr - arr.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True)
    z = np.divide(centered, std + EPS, out=np.zeros_like(centered), where=std > EPS)
    corr = (z.T @ z) / max(1, arr.shape[0] - 1)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    off = np.abs(corr[mask])
    return {
        "mean_abs_offdiag": float(off.mean()),
        "max_abs_offdiag": float(off.max(initial=0.0)),
    }


def covariance_eigenspectrum(x: np.ndarray) -> np.ndarray:
    arr = ensure_time_by_unit(x)
    cov = np.cov(arr, rowvar=False)
    cov = np.atleast_2d(np.nan_to_num(cov))
    eigvals = np.linalg.eigvalsh(cov)
    return np.sort(np.maximum(eigvals, 0.0))[::-1]


def fit_power_law_slope(values: Iterable[float], start_rank: int = 1, stop_rank: int | None = None) -> Dict[str, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals) & (vals > EPS)]
    if vals.size < 4:
        return {"alpha": 0.0, "r2": 0.0, "n": int(vals.size)}
    vals = np.sort(vals)[::-1]
    stop = stop_rank if stop_rank is not None else vals.size
    stop = max(start_rank + 3, min(stop, vals.size))
    ranks = np.arange(1, vals.size + 1, dtype=float)[start_rank - 1 : stop]
    y = np.log(vals[start_rank - 1 : stop])
    x = np.log(ranks)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2) + EPS)
    return {"alpha": float(-slope), "r2": float(1.0 - ss_res / ss_tot), "n": int(stop - start_rank + 1)}


def psd_summary(signal: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
    y = np.asarray(signal, dtype=float).ravel()
    if y.size < 8:
        return {"peak_freq": 0.0, "peak_ratio": 0.0, "spectral_slope": 0.0}
    y = y - y.mean()
    power = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(y.size, d=dt)
    if power.size <= 2:
        return {"peak_freq": 0.0, "peak_ratio": 0.0, "spectral_slope": 0.0}
    p = power[1:]
    f = freqs[1:]
    peak_idx = int(np.argmax(p))
    bg = float(np.median(p) + EPS)
    valid = (f > 0) & (p > EPS)
    if valid.sum() >= 4:
        slope, _ = np.polyfit(np.log(f[valid]), np.log(p[valid]), 1)
    else:
        slope = 0.0
    return {
        "peak_freq": float(f[peak_idx]),
        "peak_ratio": float(p[peak_idx] / bg),
        "spectral_slope": float(slope),
    }


def fit_linear_dynamics(x: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    arr = ensure_time_by_unit(x)
    x0 = arr[:-1].T
    x1 = arr[1:].T
    gram = x0 @ x0.T
    return (x1 @ x0.T) @ np.linalg.pinv(gram + ridge * np.eye(gram.shape[0]))


def dmd_summary(x: np.ndarray, ridge: float = 1e-5) -> Dict[str, object]:
    arr = ensure_time_by_unit(x)
    if arr.shape[0] < 3:
        return {"spectral_radius": 0.0, "complex_fraction": 0.0, "near_unit_complex": 0, "eigenvalues": []}
    a = fit_linear_dynamics(arr, ridge=ridge)
    eigvals = np.linalg.eigvals(a)
    radii = np.abs(eigvals)
    complex_mask = np.abs(np.imag(eigvals)) > 1e-4
    near_unit_complex = complex_mask & (radii > 0.8) & (radii < 1.1)
    packed = [[float(np.real(z)), float(np.imag(z))] for z in eigvals[: min(16, eigvals.size)]]
    return {
        "spectral_radius": float(radii.max(initial=0.0)),
        "complex_fraction": float(complex_mask.mean()) if eigvals.size else 0.0,
        "near_unit_complex": int(near_unit_complex.sum()),
        "eigenvalues": packed,
    }


def lyapunov_covariance(a: np.ndarray, q: np.ndarray, max_iter: int = 4000, tol: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    q = np.asarray(q, dtype=float)
    n = a.shape[0]
    try:
        mat = np.eye(n * n) - np.kron(a, a)
        sol = np.linalg.solve(mat, q.reshape(-1))
        cov = sol.reshape(n, n)
        return 0.5 * (cov + cov.T)
    except np.linalg.LinAlgError:
        cov = q.copy()
        for _ in range(max_iter):
            nxt = a @ cov @ a.T + q
            if np.linalg.norm(nxt - cov) / (np.linalg.norm(cov) + EPS) < tol:
                return 0.5 * (nxt + nxt.T)
            cov = nxt
        return 0.5 * (cov + cov.T)


def covariance_agreement(empirical: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    e = np.sort(np.maximum(np.linalg.eigvalsh(empirical), EPS))[::-1]
    p = np.sort(np.maximum(np.linalg.eigvalsh(predicted), EPS))[::-1]
    k = min(e.size, p.size)
    le = np.log(e[:k])
    lp = np.log(p[:k])
    if k < 3 or le.std() < EPS or lp.std() < EPS:
        corr = 0.0
    else:
        corr = float(np.mean(((le - le.mean()) / le.std()) * ((lp - lp.mean()) / lp.std())))
    rel = float(np.linalg.norm(empirical - predicted) / (np.linalg.norm(empirical) + EPS))
    return {"log_eigenspectrum_corr": corr, "relative_fro_error": rel}


def extract_avalanches(counts: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(list(counts), dtype=float).ravel()
    active = c > 0
    sizes: List[float] = []
    durations: List[int] = []
    i = 0
    while i < c.size:
        if not active[i]:
            i += 1
            continue
        j = i
        total = 0.0
        while j < c.size and active[j]:
            total += c[j]
            j += 1
        sizes.append(total)
        durations.append(j - i)
        i = j
    return np.asarray(sizes, dtype=float), np.asarray(durations, dtype=float)


def branching_ratio_counts(counts: Iterable[float]) -> float:
    c = np.asarray(list(counts), dtype=float).ravel()
    parents = c[:-1]
    children = c[1:]
    mask = parents > 0
    if mask.sum() == 0:
        return 0.0
    return float(children[mask].sum() / (parents[mask].sum() + EPS))


def _safe_aic(loglik: float, k: int) -> float:
    return float(2 * k - 2 * loglik)


def compare_tail_models(samples: Iterable[float], xmin: float = 1.0) -> Dict[str, object]:
    x = np.asarray(list(samples), dtype=float)
    x = x[np.isfinite(x) & (x >= xmin)]
    if x.size < 8:
        return {"n": int(x.size), "best_model": "insufficient", "models": {}}
    shifted = x / xmin
    alpha = 1.0 + x.size / (np.sum(np.log(np.maximum(shifted, 1.0 + EPS))) + EPS)
    ll_power = float(x.size * math.log(max(alpha - 1.0, EPS)) - x.size * math.log(xmin) - alpha * np.sum(np.log(shifted)))
    lam = 1.0 / (float(np.mean(x - xmin)) + EPS)
    ll_exp = float(x.size * math.log(lam + EPS) - lam * np.sum(x - xmin))
    lx = np.log(x)
    mu = float(lx.mean())
    sig = float(lx.std() + EPS)
    ll_logn = float(np.sum(-np.log(x * sig * math.sqrt(2 * math.pi)) - 0.5 * ((lx - mu) / sig) ** 2))
    best_stretched = (-np.inf, 1.0, 1.0)
    y = np.maximum(x - xmin + 1.0, EPS)
    for beta in np.linspace(0.35, 1.8, 24):
        scale = float(np.mean(y**beta) ** (1.0 / beta) + EPS)
        ll = float(np.sum(np.log(beta / scale) + (beta - 1.0) * np.log(y / scale) - (y / scale) ** beta))
        if ll > best_stretched[0]:
            best_stretched = (ll, float(beta), scale)
    models = {
        "power_law": {"aic": _safe_aic(ll_power, 1), "loglik": ll_power, "alpha": float(alpha)},
        "exponential": {"aic": _safe_aic(ll_exp, 1), "loglik": ll_exp, "lambda": float(lam)},
        "lognormal": {"aic": _safe_aic(ll_logn, 2), "loglik": ll_logn, "mu": mu, "sigma": sig},
        "stretched_exponential": {
            "aic": _safe_aic(best_stretched[0], 2),
            "loglik": best_stretched[0],
            "beta": best_stretched[1],
            "scale": best_stretched[2],
        },
    }
    best = min(models, key=lambda name: models[name]["aic"])
    return {"n": int(x.size), "best_model": best, "models": models}


def analytic_phase(signal: np.ndarray) -> np.ndarray:
    y = np.asarray(signal, dtype=float).ravel()
    n = y.size
    if n == 0:
        return y
    spectrum = np.fft.fft(y - y.mean())
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    analytic = np.fft.ifft(spectrum * h)
    return np.angle(analytic)


def spike_phase_locking(spikes: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    s = ensure_time_by_unit(spikes)
    phase = analytic_phase(reference)
    vals: List[float] = []
    for j in range(s.shape[1]):
        idx = s[:, j] > 0
        if idx.sum() >= 3:
            vals.append(float(np.abs(np.mean(np.exp(1j * phase[idx])))))
    if not vals:
        return {"mean_plv": 0.0, "unit_count": 0}
    return {"mean_plv": float(np.mean(vals)), "unit_count": int(len(vals))}


def information_bits_from_accuracy(acc: float) -> float:
    acc = float(np.clip(acc, 0.5, 1.0))
    err = 1.0 - acc
    if err <= EPS or err >= 1.0 - EPS:
        entropy = 0.0
    else:
        entropy = -(err * math.log(err, 2) + (1.0 - err) * math.log(1.0 - err, 2))
    return float(max(0.0, 1.0 - entropy))


def summarize_activity(x: np.ndarray, max_lag: int = 40) -> Dict[str, object]:
    arr = ensure_time_by_unit(x)
    eig = covariance_eigenspectrum(arr)
    pop = arr.sum(axis=1)
    sizes, durations = extract_avalanches(pop)
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "mean_rate": float(arr.mean()),
        "autocorrelation": autocorrelation_mean(arr, max_lag=max_lag),
        "cross_correlation": cross_correlation_summary(arr),
        "eigenspectrum": {
            "top5": [float(v) for v in eig[:5]],
            "power_law": fit_power_law_slope(eig, stop_rank=min(40, eig.size)),
        },
        "psd": psd_summary(pop),
        "dmd": dmd_summary(arr),
        "avalanche": {
            "n": int(sizes.size),
            "size_tail": compare_tail_models(sizes),
            "duration_tail": compare_tail_models(durations),
            "branching_ratio": branching_ratio_counts(pop),
        },
    }
