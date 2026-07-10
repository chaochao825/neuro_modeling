"""Context-switching, forgetting, and local-stability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _context_vector(context: ArrayLike, *, length: int) -> NDArray[np.object_]:
    array = np.asarray(context, dtype=object)
    if array.ndim != 1 or array.size != length:
        raise ValueError("context must be one-dimensional and match performance length")
    for item in array.tolist():
        if item is None:
            raise ValueError("context cannot contain missing labels")
        try:
            if bool(item != item):
                raise ValueError("context cannot contain missing labels")
        except (TypeError, ValueError):
            # Non-scalar equality is not a valid trial label.
            raise ValueError("context labels must be scalar and non-missing") from None
    return array


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True)
class SwitchCostSummary:
    """Performance loss immediately after valid context switches."""

    mean_cost: float
    median_cost: float
    per_switch_cost: FloatArray
    switch_indices: NDArray[np.int64]
    excluded_switch_indices: NDArray[np.int64]


def switch_cost_summary(
    performance: ArrayLike,
    context: ArrayLike,
    *,
    pre_window: int = 5,
    post_window: int = 5,
    higher_is_better: bool = True,
) -> SwitchCostSummary:
    """Compare stable pre-switch and immediate post-switch trial windows.

    A switch is included only when both windows fit in the recording and each
    window belongs wholly to the appropriate context.  Thus nearby switches
    cannot contaminate one another.  Positive cost always denotes worse
    post-switch performance.
    """

    values = _finite_vector(performance, name="performance")
    labels = _context_vector(context, length=values.size)
    pre = _positive_integer(pre_window, name="pre_window")
    post = _positive_integer(post_window, name="post_window")
    if not isinstance(higher_is_better, (bool, np.bool_)):
        raise TypeError("higher_is_better must be boolean")

    change = np.fromiter(
        (labels[index] != labels[index - 1] for index in range(1, labels.size)),
        dtype=bool,
        count=max(0, labels.size - 1),
    )
    candidates = np.flatnonzero(change).astype(np.int64) + 1
    if candidates.size == 0:
        raise ValueError("context contains no switches")

    valid: list[int] = []
    excluded: list[int] = []
    costs: list[float] = []
    for index in candidates.tolist():
        if index < pre or index + post > values.size:
            excluded.append(index)
            continue
        before_labels = labels[index - pre : index]
        after_labels = labels[index : index + post]
        if not all(item == labels[index - 1] for item in before_labels.tolist()):
            excluded.append(index)
            continue
        if not all(item == labels[index] for item in after_labels.tolist()):
            excluded.append(index)
            continue
        before = float(np.mean(values[index - pre : index]))
        after = float(np.mean(values[index : index + post]))
        costs.append(before - after if higher_is_better else after - before)
        valid.append(index)
    if not costs:
        raise ValueError("no switch has complete uncontaminated pre/post windows")
    per_switch = np.asarray(costs, dtype=np.float64)
    return SwitchCostSummary(
        mean_cost=float(np.mean(per_switch)),
        median_cost=float(np.median(per_switch)),
        per_switch_cost=per_switch,
        switch_indices=np.asarray(valid, dtype=np.int64),
        excluded_switch_indices=np.asarray(excluded, dtype=np.int64),
    )


def switch_cost(
    performance: ArrayLike,
    context: ArrayLike,
    *,
    pre_window: int = 5,
    post_window: int = 5,
    higher_is_better: bool = True,
) -> float:
    """Return mean context-switch cost; positive values indicate impairment."""

    return switch_cost_summary(
        performance,
        context,
        pre_window=pre_window,
        post_window=post_window,
        higher_is_better=higher_is_better,
    ).mean_cost


@dataclass(frozen=True)
class ForgettingSummary:
    """Paired performance change after intervening learning."""

    mean_forgetting: float
    median_forgetting: float
    per_unit_forgetting: FloatArray


def forgetting_summary(
    retained_performance_before: ArrayLike,
    retained_performance_after: ArrayLike,
    *,
    higher_is_better: bool = True,
) -> ForgettingSummary:
    """Measure loss on previously learned contexts after an intervention."""

    before = _finite_vector(retained_performance_before, name="retained_performance_before")
    after = _finite_vector(retained_performance_after, name="retained_performance_after")
    if before.shape != after.shape:
        raise ValueError("before and after performance must have identical shapes")
    if not isinstance(higher_is_better, (bool, np.bool_)):
        raise TypeError("higher_is_better must be boolean")
    per_unit = before - after if higher_is_better else after - before
    return ForgettingSummary(
        mean_forgetting=float(np.mean(per_unit)),
        median_forgetting=float(np.median(per_unit)),
        per_unit_forgetting=np.asarray(per_unit, dtype=np.float64),
    )


def forgetting(
    retained_performance_before: ArrayLike,
    retained_performance_after: ArrayLike,
    *,
    higher_is_better: bool = True,
) -> float:
    """Return mean forgetting; positive values indicate deterioration."""

    return forgetting_summary(
        retained_performance_before,
        retained_performance_after,
        higher_is_better=higher_is_better,
    ).mean_forgetting


@dataclass(frozen=True)
class JacobianSpectrumSummary:
    """Eigenvalue summary for one continuous- or discrete-time Jacobian."""

    eigenvalues: ComplexArray
    spectral_radius: float
    max_real_part: float
    stability_margin: float
    unstable_count: int
    unstable_fraction: float
    complex_fraction: float
    dynamics: str


def jacobian_spectrum_summary(
    jacobian: ArrayLike,
    *,
    dynamics: Literal["continuous", "discrete"] = "continuous",
    tolerance: float = 1e-9,
) -> JacobianSpectrumSummary:
    """Summarize local stability without conflating time conventions.

    Continuous-time modes are unstable when ``Re(lambda) > tolerance``;
    discrete-time modes are unstable when ``abs(lambda) > 1 + tolerance``.
    """

    raw = np.asarray(jacobian)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("jacobian must be a real numeric array")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("jacobian must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must contain only finite values")
    if dynamics not in {"continuous", "discrete"}:
        raise ValueError("dynamics must be 'continuous' or 'discrete'")
    if isinstance(tolerance, bool) or not np.isscalar(tolerance):
        raise TypeError("tolerance must be a non-negative finite scalar")
    tolerance_value = float(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be a non-negative finite scalar")

    eigenvalues = np.asarray(np.linalg.eigvals(matrix), dtype=np.complex128)
    magnitudes = np.abs(eigenvalues)
    real_parts = np.real(eigenvalues)
    spectral_radius = float(np.max(magnitudes))
    max_real_part = float(np.max(real_parts))
    if dynamics == "continuous":
        unstable = real_parts > tolerance_value
        stability_margin = -max_real_part
    else:
        unstable = magnitudes > 1.0 + tolerance_value
        stability_margin = 1.0 - spectral_radius
    complex_modes = np.abs(np.imag(eigenvalues)) > tolerance_value
    return JacobianSpectrumSummary(
        eigenvalues=eigenvalues,
        spectral_radius=spectral_radius,
        max_real_part=max_real_part,
        stability_margin=float(stability_margin),
        unstable_count=int(np.count_nonzero(unstable)),
        unstable_fraction=float(np.mean(unstable)),
        complex_fraction=float(np.mean(complex_modes)),
        dynamics=dynamics,
    )
