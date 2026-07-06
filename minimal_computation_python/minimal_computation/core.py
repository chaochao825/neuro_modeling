"""Core maximum-entropy and minimax input selection routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_entropy(p: np.ndarray | float) -> np.ndarray | float:
    p_arr = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    h = -(p_arr * np.log2(p_arr) + (1.0 - p_arr) * np.log2(1.0 - p_arr))
    if np.isscalar(p):
        return float(h)
    return h


def mutual_information(m: np.ndarray, c: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float)
    mm = np.outer(m, m)
    mi = np.zeros_like(c, dtype=float)

    terms = [
        (c, mm),
        (m[:, None] - c, m[:, None] - mm),
        (m[None, :] - c, m[None, :] - mm),
        (1.0 - m[:, None] - m[None, :] + c, 1.0 - m[:, None] - m[None, :] + mm),
    ]
    for p, q in terms:
        mask = p > 0
        mi[mask] += p[mask] * np.log2(np.maximum(p[mask], EPS) / np.maximum(q[mask], EPS))
    np.fill_diagonal(mi, binary_entropy(m))
    return mi


def pairwise_mi_with_output(m_all: np.ndarray, c_y: np.ndarray, y_mean: float) -> np.ndarray:
    """Mutual information between each binary input and one binary output."""
    mx = np.asarray(m_all, dtype=float)
    my = float(y_mean)
    c = np.asarray(c_y, dtype=float)
    out = np.zeros_like(mx, dtype=float)
    terms = [
        (c, mx * my),
        (mx - c, mx * (1.0 - my)),
        (my - c, (1.0 - mx) * my),
        (1.0 - mx - my + c, (1.0 - mx) * (1.0 - my)),
    ]
    for p, q in terms:
        mask = p > 0
        out[mask] += p[mask] * np.log2(np.maximum(p[mask], EPS) / np.maximum(q[mask], EPS))
    return out


def unique_columns_distribution(x_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_inputs.size == 0:
        return np.zeros((0, 1), dtype=float), np.ones(1, dtype=float)
    states, inverse, counts = np.unique(x_inputs.T, axis=0, return_inverse=True, return_counts=True)
    del inverse
    return states.T.astype(float), counts.astype(float) / x_inputs.shape[1]


@dataclass
class MaxEntFit:
    bias: float
    weights: np.ndarray
    complete: bool
    iterations: int
    error: float


def fit_maxent_neuron(
    y_obs: float,
    corr_obs: np.ndarray,
    x_states: np.ndarray,
    p_states: np.ndarray,
    exponent: float = 1.0,
    b0: float | None = None,
    w0: np.ndarray | None = None,
    threshold: float = 1e-6,
    max_steps: int = 10000,
    steps_check: int = 1000,
    inertia: float = 0.99,
) -> MaxEntFit:
    n_inputs = x_states.shape[0]
    b = float(np.log((y_obs + EPS) / (1.0 - y_obs + EPS)) if b0 is None else b0)
    w = np.zeros(n_inputs, dtype=float) if w0 is None else np.asarray(w0, dtype=float).copy()
    if w.size != n_inputs:
        w = np.zeros(n_inputs, dtype=float)
    corr_obs = np.asarray(corr_obs, dtype=float).reshape(-1)
    p_states = np.asarray(p_states, dtype=float).reshape(-1)
    db_new = 0.0
    dw_new = np.zeros(n_inputs, dtype=float)
    errs = np.zeros(max_steps, dtype=float)
    err = np.inf

    for t in range(1, max_steps + 1):
        db_old = db_new
        dw_old = dw_new
        py = sigmoid(b + x_states.T @ w)
        y_avg = float(p_states @ py)
        corr = x_states @ (p_states * py)
        dy_avg = y_obs - y_avg
        dcorr = corr_obs - corr
        err = float(np.max(np.abs(np.r_[dy_avg, dcorr]))) if dcorr.size else abs(dy_avg)
        if err < threshold:
            return MaxEntFit(b, w, True, t, err)
        errs[t - 1] = err
        if t > steps_check and err > errs[t - steps_check - 1]:
            return MaxEntFit(b, w, False, t, err)
        scale = t**exponent
        db_new = scale * dy_avg + inertia * db_old
        dw_new = scale * dcorr + inertia * dw_old
        b += db_new
        w += dw_new
    return MaxEntFit(b, w, False, max_steps, float(err))


def choose_best_fit(
    y_obs: float,
    corr_obs: np.ndarray,
    x_states: np.ndarray,
    p_states: np.ndarray,
    b0: float,
    w0: np.ndarray,
    threshold: float,
    exponents: Iterable[float] = (4 / 3, 1, 3 / 4, 1 / 2, 1 / 4, 0, -1 / 4),
) -> MaxEntFit:
    best: MaxEntFit | None = None
    for exponent in exponents:
        fit = fit_maxent_neuron(
            y_obs,
            corr_obs,
            x_states,
            p_states,
            exponent=exponent,
            b0=b0,
            w0=w0,
            threshold=threshold,
        )
        if fit.complete:
            return fit
        if best is None or fit.error < best.error:
            best = fit
    assert best is not None
    return best


def model_entropy(activity: np.ndarray, y: int, inputs: np.ndarray, fit: MaxEntFit) -> float:
    x_inputs = activity[inputs, :] if inputs.size else np.zeros((0, activity.shape[1]))
    py = sigmoid(fit.bias + fit.weights @ x_inputs)
    return float(np.mean(binary_entropy(py)))


def residual_correlation_error(activity: np.ndarray, y: int, inputs: np.ndarray, fit: MaxEntFit) -> float:
    t = activity.shape[1]
    activity32 = activity.astype(np.float32, copy=False)
    y_vec = activity32[y, :]
    c_y = activity32 @ y_vec / np.float32(t)
    positive = np.flatnonzero(c_y > 0)
    remaining = np.setdiff1d(positive, np.r_[inputs, y], assume_unique=False)
    if remaining.size == 0:
        return 0.0
    x_inputs = activity32[inputs, :] if inputs.size else np.zeros((0, t), dtype=np.float32)
    py = sigmoid(fit.bias + fit.weights @ x_inputs).astype(np.float32, copy=False)
    c_pred = activity32[remaining, :] @ py / np.float32(t)
    denom = np.sqrt(np.maximum(c_y[remaining], EPS) / t)
    return float(np.max(np.abs(c_y[remaining] - c_pred) / denom))


@dataclass
class MinimaxResult:
    neuron: int
    n_neurons: int
    n_time: int
    nums_in: List[int]
    entropies: List[float]
    independent_entropy: float
    residual_errors: List[float]
    complete_flags: List[bool]
    iterations: List[int]
    inputs_order: List[int]
    complete_num_inputs: int | None
    complete_fraction: float | None
    selected_inputs_complete: List[int]


def greedy_minimax_entropy(
    activity: np.ndarray,
    neuron_id_matlab: int,
    nums_in: Iterable[int],
    threshold: float = 1e-6,
    corr_error_threshold: float = 2.0,
) -> MinimaxResult:
    x_raw = np.asarray(activity)
    x = x_raw.astype(np.float32, copy=False)
    y = int(neuron_id_matlab) - 1
    n, t = x.shape
    m = x.mean(axis=1)
    y_vec = x[y, :]
    c_y = x @ y_vec / t
    mi_y = pairwise_mi_with_output(m, c_y, float(m[y]))
    s_ind = float(binary_entropy(m[y]))
    possible = np.setdiff1d(np.flatnonzero(c_y > 0), np.array([y]))
    if possible.size == 0:
        raise ValueError(f"Neuron {neuron_id_matlab} has no co-active candidate inputs")
    nums = sorted({int(v) for v in nums_in if 0 < int(v) <= possible.size})
    if not nums:
        raise ValueError("No valid input counts remain after applying the candidate-input limit")

    inputs: list[int] = []
    nums_done: list[int] = []
    entropies: list[float] = []
    residuals: list[float] = []
    completes: list[bool] = []
    iterations: list[int] = []
    b = float(np.log((m[y] + EPS) / (1.0 - m[y] + EPS)))
    w = np.zeros(0, dtype=float)

    for target in nums:
        remaining = np.setdiff1d(possible, np.asarray(inputs, dtype=int), assume_unique=False)
        need = target - len(inputs)
        if need <= 0 or remaining.size == 0:
            continue
        if not inputs:
            scores = mi_y[remaining]
        else:
            # Fast Python approximation to MATLAB's entropy-drop criterion:
            # rank remaining inputs by their current normalized residual
            # correlation after the fitted model with existing inputs.
            x_inputs_existing = x[np.asarray(inputs, dtype=int), :]
            py_existing = sigmoid(b + w @ x_inputs_existing).astype(np.float32, copy=False)
            c_pred = x[remaining, :] @ py_existing / np.float32(t)
            denom = np.sqrt(np.maximum(c_y[remaining], EPS) / t)
            scores = np.abs(c_y[remaining] - c_pred) / denom
        chosen = remaining[np.argsort(scores)[-need:][::-1]]
        inputs.extend(int(v) for v in chosen)
        inputs_arr = np.asarray(inputs, dtype=int)
        states, probs = unique_columns_distribution(x[inputs_arr, :])
        fit = choose_best_fit(
            float(m[y]),
            c_y[inputs_arr],
            states,
            probs,
            b0=b,
            w0=np.r_[w, np.zeros(len(chosen))],
            threshold=threshold,
        )
        b, w = fit.bias, fit.weights
        nums_done.append(len(inputs))
        entropies.append(model_entropy(x, y, inputs_arr, fit))
        residuals.append(residual_correlation_error(x, y, inputs_arr, fit))
        completes.append(bool(fit.complete))
        iterations.append(int(fit.iterations))

    complete_num = None
    selected_complete: list[int] = []
    for num, err in zip(nums_done, residuals):
        if err < corr_error_threshold:
            complete_num = int(num)
            selected_complete = inputs[:complete_num]
            break

    return MinimaxResult(
        neuron=neuron_id_matlab,
        n_neurons=n,
        n_time=t,
        nums_in=nums_done,
        entropies=entropies,
        independent_entropy=s_ind,
        residual_errors=residuals,
        complete_flags=completes,
        iterations=iterations,
        inputs_order=[int(v) + 1 for v in inputs],
        complete_num_inputs=complete_num,
        complete_fraction=(complete_num / (n - 1) if complete_num is not None else None),
        selected_inputs_complete=[int(v) + 1 for v in selected_complete],
    )
