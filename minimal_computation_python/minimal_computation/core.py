"""Core maximum-entropy and minimax input selection routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence

import numpy as np


EPS = 1e-12
Selector = Literal["schur_entropy_drop", "residual_approximation"]
Initialization = Literal["matlab_reset", "warm_start"]
CompletionMode = Literal["matlab_residual", "strict_optimizer_and_residual"]
FailureSelection = Literal["matlab_last", "best_error"]


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
    b0: float | None = None,
    w0: np.ndarray | None = None,
    threshold: float = 1e-6,
    exponents: Iterable[float] = (4 / 3, 1, 3 / 4, 1 / 2, 1 / 4, 0, -1 / 4),
    initialization: Initialization = "matlab_reset",
    max_steps: int = 10000,
    steps_check: int = 1000,
    inertia: float = 0.99,
    failure_selection: FailureSelection = "matlab_last",
) -> MaxEntFit:
    """Try the MATLAB learning-rate schedule and return the first converged fit.

    ``matlab_reset`` reproduces the upstream implementation: every exponent
    starts from the independent-model bias and zero input weights.  The former
    Python behavior is retained explicitly as ``warm_start``.
    """
    if initialization not in ("matlab_reset", "warm_start"):
        raise ValueError(f"Unknown initialization: {initialization}")
    if failure_selection not in ("matlab_last", "best_error"):
        raise ValueError(f"Unknown failure_selection: {failure_selection}")
    n_inputs = int(x_states.shape[0])
    if initialization == "matlab_reset":
        start_b = float(np.log((y_obs + EPS) / (1.0 - y_obs + EPS)))
        start_w = np.zeros(n_inputs, dtype=float)
    else:
        start_b = (
            float(np.log((y_obs + EPS) / (1.0 - y_obs + EPS)))
            if b0 is None
            else float(b0)
        )
        start_w = (
            np.zeros(n_inputs, dtype=float)
            if w0 is None
            else np.asarray(w0, dtype=float).reshape(-1)
        )
        if start_w.size != n_inputs:
            raise ValueError("warm_start w0 must have one value per selected input")

    best: MaxEntFit | None = None
    last: MaxEntFit | None = None
    for exponent in exponents:
        fit = fit_maxent_neuron(
            y_obs,
            corr_obs,
            x_states,
            p_states,
            exponent=exponent,
            b0=start_b,
            w0=start_w,
            threshold=threshold,
            max_steps=max_steps,
            steps_check=steps_check,
            inertia=inertia,
        )
        last = fit
        if fit.complete:
            return fit
        if best is None or fit.error < best.error:
            best = fit
    assert best is not None
    assert last is not None
    return last if failure_selection == "matlab_last" else best


def schur_entropy_drop_scores(
    activity: np.ndarray,
    y: int,
    selected_inputs: Sequence[int] | np.ndarray,
    candidate_inputs: Sequence[int] | np.ndarray,
    fit: MaxEntFit,
    *,
    candidate_block_size: int = 256,
    ridge: float = 0.0,
) -> np.ndarray:
    """MATLAB entropy-drop scores without constructing its full Hessian.

    The upstream code forms an ``(N + 1) x (N + 1)`` covariance/Hessian and
    takes a Schur complement.  This implementation computes the same selected
    block and candidate diagonals in chunks, requiring ``O(T (k + B) + B k)``
    working memory for ``k`` selected inputs and candidate block size ``B``
    instead of ``O(N^2)`` Hessian storage.
    """
    # Keep the source matrix in its compact dtype; only selected rows and the
    # current candidate block are promoted to float64.
    x = np.asarray(activity)
    if x.ndim != 2:
        raise ValueError("activity must be a neurons x time matrix")
    selected = np.asarray(selected_inputs, dtype=int).reshape(-1)
    candidates = np.asarray(candidate_inputs, dtype=int).reshape(-1)
    if selected.size == 0:
        raise ValueError("Schur entropy-drop scores require at least one selected input")
    if fit.weights.size != selected.size:
        raise ValueError("fit weights must correspond to selected_inputs")
    if candidate_block_size <= 0:
        raise ValueError("candidate_block_size must be positive")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if candidates.size == 0:
        return np.zeros(0, dtype=float)

    _, t = x.shape
    x_selected = np.asarray(x[selected, :], dtype=float)
    py = sigmoid(fit.bias + fit.weights @ x_selected)
    # MATLAB A = E[xx' p] - E[xx' p^2] = E[xx' p(1-p)].
    curvature_weight = py * (1.0 - py) / float(t)
    design = np.vstack((np.ones(t, dtype=float), x_selected))
    a_ss = (design * curvature_weight) @ design.T
    if ridge:
        a_ss = a_ss + ridge * np.eye(a_ss.shape[0])

    y_vec = np.asarray(x[y, :], dtype=float)
    scores = np.empty(candidates.size, dtype=float)
    for start in range(0, candidates.size, candidate_block_size):
        stop = min(start + candidate_block_size, candidates.size)
        x_candidate = np.asarray(x[candidates[start:stop], :], dtype=float)
        a_sr = (design * curvature_weight) @ x_candidate.T
        try:
            projected = np.linalg.solve(a_ss, a_sr)
        except np.linalg.LinAlgError:
            projected = np.linalg.pinv(a_ss) @ a_sr
        diag_rr = (x_candidate * x_candidate) @ curvature_weight
        schur_diag = diag_rr - np.sum(a_sr * projected, axis=0)

        observed_corr = x_candidate @ y_vec / float(t)
        predicted_corr = x_candidate @ py / float(t)
        residual = observed_corr - predicted_corr
        scores[start:stop] = 0.5 * residual * residual / np.maximum(schur_diag, EPS)
    return scores


def residual_approximation_scores(
    activity: np.ndarray,
    y: int,
    selected_inputs: Sequence[int] | np.ndarray,
    candidate_inputs: Sequence[int] | np.ndarray,
    fit: MaxEntFit,
) -> np.ndarray:
    """Former Python normalized-residual selector, retained as a baseline."""
    x = np.asarray(activity)
    selected = np.asarray(selected_inputs, dtype=int).reshape(-1)
    candidates = np.asarray(candidate_inputs, dtype=int).reshape(-1)
    if candidates.size == 0:
        return np.zeros(0, dtype=float)
    if fit.weights.size != selected.size:
        raise ValueError("fit weights must correspond to selected_inputs")
    t = x.shape[1]
    x_selected = np.asarray(x[selected, :], dtype=float)
    x_candidates = np.asarray(x[candidates, :], dtype=float)
    py = sigmoid(fit.bias + fit.weights @ x_selected)
    observed_corr = x_candidates @ np.asarray(x[y, :], dtype=float) / float(t)
    predicted_corr = x_candidates @ py / float(t)
    denom = np.sqrt(np.maximum(observed_corr, EPS) / float(t))
    return np.abs(observed_corr - predicted_corr) / denom


def candidate_input_scores(
    activity: np.ndarray,
    y: int,
    selected_inputs: Sequence[int] | np.ndarray,
    candidate_inputs: Sequence[int] | np.ndarray,
    fit: MaxEntFit,
    *,
    selector: Selector = "schur_entropy_drop",
    candidate_block_size: int = 256,
) -> np.ndarray:
    """Score candidates using exact first-step MI then the requested method."""
    x = np.asarray(activity)
    selected = np.asarray(selected_inputs, dtype=int).reshape(-1)
    candidates = np.asarray(candidate_inputs, dtype=int).reshape(-1)
    if selected.size == 0:
        x_candidates = np.asarray(x[candidates, :], dtype=float)
        m_candidates = x_candidates.mean(axis=1)
        c_y = x_candidates @ np.asarray(x[y, :], dtype=float) / float(x.shape[1])
        return pairwise_mi_with_output(m_candidates, c_y, float(x[y, :].mean()))
    if selector == "schur_entropy_drop":
        return schur_entropy_drop_scores(
            x,
            y,
            selected,
            candidates,
            fit,
            candidate_block_size=candidate_block_size,
        )
    if selector == "residual_approximation":
        return residual_approximation_scores(x, y, selected, candidates, fit)
    raise ValueError(f"Unknown selector: {selector}")


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
    # Optimizer convergence alone; kept separate from the model criterion.
    complete_flags: List[bool]
    criterion_complete_flags: List[bool]
    iterations: List[int]
    evaluation_phases: List[str]
    inputs_order: List[int]
    complete_num_inputs: int | None
    complete_fraction: float | None
    selected_inputs_complete: List[int]
    coarse_nums_in: List[int]
    n_candidates: int
    selector: str
    initialization: str
    optimizer_threshold: float
    corr_error_threshold: float
    candidate_block_size: int
    binary_search_performed: bool
    completion_mode: str
    failure_selection: str


@dataclass
class _ModelEvaluation:
    inputs: np.ndarray
    fit: MaxEntFit
    entropy: float
    residual_error: float
    criterion_complete: bool
    phase: str


def _independent_fit(y_mean: float) -> MaxEntFit:
    return MaxEntFit(
        bias=float(np.log((y_mean + EPS) / (1.0 - y_mean + EPS))),
        weights=np.zeros(0, dtype=float),
        complete=True,
        iterations=0,
        error=0.0,
    )


def _select_top_candidates(candidates: np.ndarray, scores: np.ndarray, count: int) -> np.ndarray:
    """Deterministic equivalent of MATLAB ``maxk`` (score, then index)."""
    finite_scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = np.lexsort((candidates, -finite_scores))
    return candidates[order[:count]]


def _evaluate_model(
    x: np.ndarray,
    y: int,
    inputs: np.ndarray,
    y_mean: float,
    c_y: np.ndarray,
    *,
    threshold: float,
    corr_error_threshold: float,
    initialization: Initialization,
    previous_fit: MaxEntFit,
    added_count: int,
    exponents: Iterable[float],
    max_steps: int,
    steps_check: int,
    inertia: float,
    phase: str,
    completion_mode: CompletionMode,
    failure_selection: FailureSelection,
) -> _ModelEvaluation:
    states, probs = unique_columns_distribution(x[inputs, :])
    warm_weights = np.r_[previous_fit.weights, np.zeros(added_count, dtype=float)]
    fit = choose_best_fit(
        y_mean,
        c_y[inputs],
        states,
        probs,
        b0=previous_fit.bias,
        w0=warm_weights,
        threshold=threshold,
        exponents=exponents,
        initialization=initialization,
        max_steps=max_steps,
        steps_check=steps_check,
        inertia=inertia,
        failure_selection=failure_selection,
    )
    residual = residual_correlation_error(x, y, inputs, fit)
    if completion_mode == "matlab_residual":
        # Upstream coarse/binary decisions use Cerr only; optimizer convergence
        # is recorded separately and does not enter the n* threshold.
        criterion_complete = bool(residual < corr_error_threshold)
    else:
        criterion_complete = bool(
            fit.complete and residual < corr_error_threshold
        )
    return _ModelEvaluation(
        inputs=inputs.copy(),
        fit=fit,
        entropy=model_entropy(x, y, inputs, fit),
        residual_error=residual,
        criterion_complete=criterion_complete,
        phase=phase,
    )


def greedy_minimax_entropy(
    activity: np.ndarray,
    neuron_id_matlab: int,
    nums_in: Iterable[int],
    threshold: float = 1e-6,
    corr_error_threshold: float = 2.0,
    *,
    selector: Selector = "schur_entropy_drop",
    initialization: Initialization = "matlab_reset",
    candidate_block_size: int = 256,
    refine_complete: bool = True,
    exponents: Iterable[float] = (4 / 3, 1, 3 / 4, 1 / 2, 1 / 4, 0, -1 / 4),
    optimizer_max_steps: int = 10000,
    optimizer_steps_check: int = 1000,
    optimizer_inertia: float = 0.99,
    completion_mode: CompletionMode = "matlab_residual",
    failure_selection: FailureSelection = "matlab_last",
) -> MinimaxResult:
    """Run coarse greedy selection and optionally refine the first complete set.

    The default path follows the upstream MATLAB method: mutual information
    for the first batch, analytic entropy-drop selection thereafter, reset
    initialization for every fit, and bisection after the coarse sweep.
    """
    if selector not in ("schur_entropy_drop", "residual_approximation"):
        raise ValueError(f"Unknown selector: {selector}")
    if initialization not in ("matlab_reset", "warm_start"):
        raise ValueError(f"Unknown initialization: {initialization}")
    if completion_mode not in (
        "matlab_residual",
        "strict_optimizer_and_residual",
    ):
        raise ValueError(f"Unknown completion_mode: {completion_mode}")
    if failure_selection not in ("matlab_last", "best_error"):
        raise ValueError(f"Unknown failure_selection: {failure_selection}")
    if threshold <= 0 or corr_error_threshold <= 0:
        raise ValueError("thresholds must be positive")
    if candidate_block_size <= 0:
        raise ValueError("candidate_block_size must be positive")
    exponents = tuple(float(v) for v in exponents)
    if not exponents:
        raise ValueError("exponents must not be empty")

    x = np.asarray(activity, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("activity must be a neurons x time matrix")
    y = int(neuron_id_matlab) - 1
    n, t = x.shape
    if y < 0 or y >= n:
        raise IndexError("neuron_id_matlab is outside the activity matrix")
    m = x.mean(axis=1)
    y_vec = x[y, :]
    c_y = x @ y_vec / t
    s_ind = float(binary_entropy(m[y]))
    possible = np.setdiff1d(np.flatnonzero(c_y > 0), np.array([y]))
    if possible.size == 0:
        raise ValueError(f"Neuron {neuron_id_matlab} has no co-active candidate inputs")
    nums = sorted({int(v) for v in nums_in if 0 < int(v) <= possible.size})
    if not nums:
        raise ValueError("No valid input counts remain after applying the candidate-input limit")

    inputs = np.zeros(0, dtype=int)
    current_fit = _independent_fit(float(m[y]))
    coarse_evaluations: list[_ModelEvaluation] = []

    for target in nums:
        remaining = np.setdiff1d(possible, inputs, assume_unique=False)
        need = target - inputs.size
        if need <= 0 or remaining.size == 0:
            continue
        scores = candidate_input_scores(
            x,
            y,
            inputs,
            remaining,
            current_fit,
            selector=selector,
            candidate_block_size=candidate_block_size,
        )
        chosen = _select_top_candidates(remaining, scores, need)
        inputs = np.r_[inputs, chosen].astype(int, copy=False)
        evaluation = _evaluate_model(
            x,
            y,
            inputs,
            float(m[y]),
            c_y,
            threshold=threshold,
            corr_error_threshold=corr_error_threshold,
            initialization=initialization,
            previous_fit=current_fit,
            added_count=chosen.size,
            exponents=exponents,
            max_steps=optimizer_max_steps,
            steps_check=optimizer_steps_check,
            inertia=optimizer_inertia,
            phase="coarse",
            completion_mode=completion_mode,
            failure_selection=failure_selection,
        )
        coarse_evaluations.append(evaluation)
        current_fit = evaluation.fit

    all_evaluations = list(coarse_evaluations)
    complete_num: int | None = None
    selected_complete = np.zeros(0, dtype=int)
    binary_search_performed = False
    first_good_index = next(
        (idx for idx, evaluation in enumerate(coarse_evaluations) if evaluation.criterion_complete),
        None,
    )
    if first_good_index is not None:
        good = coarse_evaluations[first_good_index]
        if first_good_index == 0:
            bad_inputs = np.zeros(0, dtype=int)
            bad_fit = _independent_fit(float(m[y]))
        else:
            bad = coarse_evaluations[first_good_index - 1]
            bad_inputs = bad.inputs.copy()
            bad_fit = bad.fit
        good_inputs = good.inputs.copy()

        while refine_complete and good_inputs.size - bad_inputs.size > 1:
            binary_search_performed = True
            gap = int(good_inputs.size - bad_inputs.size)
            # MATLAB round is half-away-from-zero for this positive quantity.
            add_count = int(np.floor(gap / 2.0 + 0.5))
            remaining = np.setdiff1d(possible, bad_inputs, assume_unique=False)
            scores = candidate_input_scores(
                x,
                y,
                bad_inputs,
                remaining,
                bad_fit,
                selector=selector,
                candidate_block_size=candidate_block_size,
            )
            chosen = _select_top_candidates(remaining, scores, add_count)
            trial_inputs = np.r_[bad_inputs, chosen].astype(int, copy=False)
            trial = _evaluate_model(
                x,
                y,
                trial_inputs,
                float(m[y]),
                c_y,
                threshold=threshold,
                corr_error_threshold=corr_error_threshold,
                initialization=initialization,
                previous_fit=bad_fit,
                added_count=chosen.size,
                exponents=exponents,
                max_steps=optimizer_max_steps,
                steps_check=optimizer_steps_check,
                inertia=optimizer_inertia,
                phase="binary_search",
                completion_mode=completion_mode,
                failure_selection=failure_selection,
            )
            all_evaluations.append(trial)
            if trial.criterion_complete:
                good_inputs = trial.inputs.copy()
            else:
                bad_inputs = trial.inputs.copy()
                bad_fit = trial.fit
        complete_num = int(good_inputs.size)
        selected_complete = good_inputs

    all_evaluations.sort(key=lambda evaluation: evaluation.inputs.size)

    return MinimaxResult(
        neuron=neuron_id_matlab,
        n_neurons=n,
        n_time=t,
        nums_in=[int(evaluation.inputs.size) for evaluation in all_evaluations],
        entropies=[float(evaluation.entropy) for evaluation in all_evaluations],
        independent_entropy=s_ind,
        residual_errors=[float(evaluation.residual_error) for evaluation in all_evaluations],
        complete_flags=[bool(evaluation.fit.complete) for evaluation in all_evaluations],
        criterion_complete_flags=[evaluation.criterion_complete for evaluation in all_evaluations],
        iterations=[int(evaluation.fit.iterations) for evaluation in all_evaluations],
        evaluation_phases=[evaluation.phase for evaluation in all_evaluations],
        inputs_order=[int(v) + 1 for v in inputs.tolist()],
        complete_num_inputs=complete_num,
        complete_fraction=(complete_num / (n - 1) if complete_num is not None else None),
        selected_inputs_complete=[int(v) + 1 for v in selected_complete.tolist()],
        coarse_nums_in=[int(evaluation.inputs.size) for evaluation in coarse_evaluations],
        n_candidates=int(possible.size),
        selector=selector,
        initialization=initialization,
        optimizer_threshold=float(threshold),
        corr_error_threshold=float(corr_error_threshold),
        candidate_block_size=int(candidate_block_size),
        binary_search_performed=binary_search_performed,
        completion_mode=completion_mode,
        failure_selection=failure_selection,
    )
