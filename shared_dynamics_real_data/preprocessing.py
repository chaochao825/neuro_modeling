"""Train-only unit selection, normalization, and controlled PCA bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from sklearn.decomposition import PCA

from .splits import TimeSegment


BasisControl = Literal["aligned", "random", "orthogonal", "shuffled"]


def _canonical_qr(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def fit_controlled_basis(
    x_train: np.ndarray,
    latent_dim: int,
    *,
    control: BasisControl = "aligned",
    random_state: int = 0,
) -> np.ndarray:
    """Fit an orthonormal basis using training rows only.

    ``aligned`` is the leading PCA basis; ``orthogonal`` uses the next PCA
    directions when available (otherwise a projected random complement);
    ``random`` samples a Haar-like QR basis; ``shuffled`` permutes aligned basis
    rows, preserving singular values but destroying neuron alignment.
    """

    values = np.asarray(x_train, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("x_train must be a finite 2-D matrix with at least two rows")
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, (int, np.integer)):
        raise TypeError("latent_dim must be an integer")
    latent_dim = int(latent_dim)
    if latent_dim < 1 or latent_dim > min(values.shape):
        raise ValueError("latent_dim exceeds the train matrix rank bound")
    if control not in {"aligned", "random", "orthogonal", "shuffled"}:
        raise ValueError("unknown basis control")
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError("random_state must be an integer")
    rng = np.random.default_rng(int(random_state))
    if control == "random":
        return _canonical_qr(rng.standard_normal((values.shape[1], latent_dim)))

    n_pca = min(2 * latent_dim, min(values.shape))
    # Exact SVD is useful for tiny tests; randomized SVD keeps the real
    # 11k-unit visual matrices tractable and remains deterministic via seed.
    solver = "randomized" if min(values.shape) > 128 and n_pca < min(values.shape) else "full"
    pca = PCA(
        n_components=n_pca,
        svd_solver=solver,
        random_state=int(random_state) if solver == "randomized" else None,
    )
    aligned_full = pca.fit(values).components_.T
    aligned = aligned_full[:, :latent_dim]
    if control == "aligned":
        return aligned
    if control == "shuffled":
        return aligned[rng.permutation(values.shape[1])]
    if n_pca >= 2 * latent_dim:
        return aligned_full[:, latent_dim : 2 * latent_dim]
    if values.shape[1] < 2 * latent_dim:
        raise ValueError("orthogonal control requires at least 2*latent_dim features")
    candidate = rng.standard_normal((values.shape[1], latent_dim))
    candidate -= aligned @ (aligned.T @ candidate)
    return _canonical_qr(candidate)


@dataclass
class TrainOnlyPreprocessor:
    """Select and scale units using only explicitly supplied train segments."""

    max_units: int | None = None
    variance_floor: float = 1e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.variance_floor) or self.variance_floor <= 0:
            raise ValueError("variance_floor must be finite and positive")

    def fit(self, segments: Sequence[TimeSegment]) -> "TrainOnlyPreprocessor":
        if not segments:
            raise ValueError("training segments cannot be empty")
        matrices = [segment.values for segment in segments]
        n_features = {matrix.shape[1] for matrix in matrices}
        if len(n_features) != 1:
            raise ValueError("training segments have inconsistent feature counts")
        pooled = np.vstack(matrices)
        raw_variance = pooled.var(axis=0)
        n_total = pooled.shape[1]
        if self.max_units is None:
            n_keep = n_total
        else:
            if isinstance(self.max_units, bool) or not isinstance(
                self.max_units, (int, np.integer)
            ):
                raise TypeError("max_units must be an integer or None")
            n_keep = int(self.max_units)
            if n_keep < 1:
                raise ValueError("max_units must be positive")
            n_keep = min(n_keep, n_total)
        # Stable index tie-break makes selection deterministic across BLAS builds.
        order = np.lexsort((np.arange(n_total), -raw_variance))
        selected = np.sort(order[:n_keep])
        selected_train = np.asarray(pooled[:, selected], dtype=np.float64)
        mean = selected_train.mean(axis=0)
        scale = selected_train.std(axis=0)
        scale[scale < self.variance_floor] = 1.0
        self.n_input_features_ = n_total
        self.unit_indices_ = selected
        self.mean_ = mean
        self.scale_ = scale
        self.fit_timepoints_ = int(pooled.shape[0])
        return self

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "unit_indices_"):
            raise RuntimeError("preprocessor must be fitted first")
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != self.n_input_features_:
            raise ValueError("values have the wrong feature count")
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise TypeError("values must be real numeric")
        if not np.isfinite(array).all():
            raise ValueError("values contain non-finite entries")
        selected = np.asarray(array[:, self.unit_indices_], dtype=np.float64)
        return (selected - self.mean_) / self.scale_

    def transform_segments(
        self, segments: Sequence[TimeSegment]
    ) -> tuple[TimeSegment, ...]:
        return tuple(
            TimeSegment(
                segment.context,
                self.transform_array(segment.values),
                segment.indices,
            )
            for segment in segments
        )
