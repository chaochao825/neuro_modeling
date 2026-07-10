"""Train-fitted label-marginal subspaces for item/rank/rule/operation effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.analysis.manifold_metrics import principal_angles, subspace_overlap


@dataclass(frozen=True)
class FittedConditionSubspace:
    basis: np.ndarray
    grand_mean: np.ndarray
    labels: tuple[object, ...]
    fit_sample_ids: tuple[object, ...] | None

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float).copy()
        mean = np.asarray(self.grand_mean, dtype=float).copy()
        if basis.ndim != 2 or mean.shape != (basis.shape[0],):
            raise ValueError("basis/mean dimensions are inconsistent")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10):
            raise ValueError("basis must have orthonormal columns")
        basis.setflags(write=False)
        mean.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "grand_mean", mean)

    def transform(self, activity: np.ndarray) -> np.ndarray:
        values = np.asarray(activity, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.basis.shape[0]:
            raise ValueError("activity feature dimension differs from fitted subspace")
        return (values - self.grand_mean) @ self.basis


def fit_condition_subspace(
    activity_train: np.ndarray,
    labels_train: Sequence[object],
    *,
    n_components: int,
    sample_ids: Sequence[object] | None = None,
) -> FittedConditionSubspace:
    """Fit the SVD span of centered condition means using training rows only."""

    activity = np.asarray(activity_train, dtype=float)
    labels = np.asarray(labels_train, dtype=object)
    if activity.ndim != 2 or activity.shape[0] < 2 or not np.isfinite(activity).all():
        raise ValueError("activity_train must be finite [sample, feature] data")
    if labels.shape != (activity.shape[0],):
        raise ValueError("labels_train must match training samples")
    unique = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    if len(unique) < 2:
        raise ValueError("at least two condition labels are required")
    if not 1 <= n_components <= min(activity.shape[1], len(unique) - 1):
        raise ValueError("n_components exceeds the condition-mean rank bound")
    grand_mean = activity.mean(axis=0)
    marginal = np.stack([activity[labels == label].mean(axis=0) - grand_mean for label in unique])
    _, singular, vt = np.linalg.svd(marginal, full_matrices=False)
    numerical_rank = int(np.sum(singular > singular[0] * np.finfo(float).eps * max(marginal.shape)))
    if n_components > numerical_rank:
        raise ValueError("requested subspace exceeds numerical condition rank")
    ids = None if sample_ids is None else tuple(np.asarray(sample_ids, dtype=object).tolist())
    if ids is not None and len(ids) != activity.shape[0]:
        raise ValueError("sample_ids must match training samples")
    return FittedConditionSubspace(vt[:n_components].T, grand_mean, tuple(unique), ids)


def compare_condition_subspaces(
    first: FittedConditionSubspace, second: FittedConditionSubspace
) -> dict[str, object]:
    return {
        "principal_angles_degrees": principal_angles(
            first.basis, second.basis, degrees=True
        ).tolist(),
        "overlap": subspace_overlap(first.basis, second.basis),
    }


def fit_demixed_condition_subspace(
    activity_train: np.ndarray,
    target_labels: Sequence[object],
    *,
    nuisance_labels: dict[str, Sequence[object]],
    n_components: int,
    sample_ids: Sequence[object] | None = None,
) -> FittedConditionSubspace:
    """Fit a target marginal after OLS-demixing other categorical factors.

    The complete multivariate design is fit only on training samples. The
    returned basis spans the target-factor contribution with nuisance-factor
    coefficients held at zero, providing a compact dPCA-style marginalization.
    """

    activity = np.asarray(activity_train, dtype=float)
    target = np.asarray(target_labels, dtype=object)
    if activity.ndim != 2 or target.shape != (activity.shape[0],):
        raise ValueError("activity and target labels must share the sample dimension")

    def sum_contrasts(labels: np.ndarray) -> tuple[np.ndarray, list[object]]:
        levels: list[object] = []
        for label in labels:
            if label not in levels:
                levels.append(label)
        if len(levels) < 2:
            return np.empty((labels.size, 0), dtype=float), levels
        one_hot = np.column_stack([labels == level for level in levels]).astype(float)
        # Effect coding avoids the intercept + complete-one-hot rank defect.  The
        # omitted level receives -1 in every contrast, so factor effects sum to
        # zero while retaining exactly k - 1 estimable degrees of freedom.
        matrix = one_hot[:, :-1] - one_hot[:, [-1]]
        return matrix, levels

    target_design, levels = sum_contrasts(target)
    if len(levels) < 2:
        raise ValueError("at least two target levels are required")
    nuisance_designs = []
    for name, values in nuisance_labels.items():
        labels = np.asarray(values, dtype=object)
        if labels.shape != target.shape:
            raise ValueError(f"nuisance factor {name!r} does not match samples")
        design, nuisance_levels = sum_contrasts(labels)
        if len(nuisance_levels) < 2:
            raise ValueError(f"nuisance factor {name!r} has fewer than two levels")
        nuisance_designs.append(design)

    # Frisch-Waugh-Lovell partial regression makes the estimability audit
    # explicit: after projecting out nuisance factors, every target contrast
    # must retain an independent direction.  Exact target/nuisance confounding
    # is therefore a failed/inconclusive analysis, not an arbitrary pinv split.
    nuisance_design = np.column_stack(
        [np.ones(activity.shape[0]), *nuisance_designs]
    )
    nuisance_projection = nuisance_design @ np.linalg.pinv(nuisance_design)
    residualizer = np.eye(activity.shape[0]) - nuisance_projection
    residual_target = residualizer @ target_design
    target_scale = float(np.linalg.svd(target_design, compute_uv=False)[0])
    estimability_tolerance = (
        target_scale
        * np.finfo(float).eps
        * max(target_design.shape)
        * 10.0
    )
    target_singular = np.linalg.svd(residual_target, compute_uv=False)
    target_rank = int(np.sum(target_singular > estimability_tolerance))
    if target_rank != target_design.shape[1]:
        raise ValueError(
            "target contrasts are not estimable after controlling nuisance factors"
        )
    residual_activity = residualizer @ activity
    coefficients, _, solved_rank, _ = np.linalg.lstsq(
        residual_target, residual_activity, rcond=None
    )
    if int(solved_rank) != target_design.shape[1]:
        raise ValueError("demixed target regression is numerically rank-deficient")
    target_effect = target_design @ coefficients
    target_effect -= target_effect.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(target_effect, full_matrices=False)
    tolerance = singular[0] * np.finfo(float).eps * max(target_effect.shape)
    rank = int(np.sum(singular > tolerance))
    if not 1 <= n_components <= min(rank, activity.shape[1], len(levels) - 1):
        raise ValueError("requested demixed subspace exceeds target marginal rank")
    ids = None if sample_ids is None else tuple(np.asarray(sample_ids, dtype=object).tolist())
    if ids is not None and len(ids) != activity.shape[0]:
        raise ValueError("sample_ids must match training samples")
    return FittedConditionSubspace(
        vt[:n_components].T,
        activity.mean(axis=0),
        tuple(levels),
        ids,
    )
