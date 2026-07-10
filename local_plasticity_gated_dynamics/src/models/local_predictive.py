"""Pure NumPy local predictive learning with low-dimensional feedback.

This module intentionally contains no autograd, PyTorch, or BPTT path.  For a
row-major batch ``X`` and next-step targets ``Y``, each context-local update is

``eta * P P.T (Y - X W.T).T X / batch_size``.

The update uses only the presynaptic activity and a projected postsynaptic
prediction error.  ``fit_fixed_point`` precomputes the required input and
target cross-moments so repeated full-batch iterations do not repeatedly scan
all samples.  A temporal-shuffle control may source both terms of the feedback
error from another time in the same trajectory block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.utils.reproducibility import make_rng


def _validate_integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_float(
    name: str,
    value: object,
    *,
    lower: float | None = None,
    strict_lower: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if lower is not None and (result <= lower if strict_lower else result < lower):
        relation = ">" if strict_lower else ">="
        raise ValueError(f"{name} must be {relation} {lower}")
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=float, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_int(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=int, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LocalPredictiveConfig:
    """Optimization settings for local predictive plasticity."""

    learning_rate: float = 0.05
    weight_decay: float = 0.0
    batch_size: int | None = 256
    max_epochs: int = 100
    tolerance: float = 1e-8
    shuffle_batches: bool = True
    max_update_fro_norm: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        _validate_float("learning_rate", self.learning_rate, lower=0.0, strict_lower=True)
        _validate_float("weight_decay", self.weight_decay, lower=0.0)
        if self.batch_size is not None:
            _validate_integer("batch_size", self.batch_size, minimum=1)
        _validate_integer("max_epochs", self.max_epochs, minimum=1)
        _validate_float("tolerance", self.tolerance, lower=0.0)
        if not isinstance(self.shuffle_batches, (bool, np.bool_)):
            raise TypeError("shuffle_batches must be boolean")
        if self.max_update_fro_norm is not None:
            _validate_float(
                "max_update_fro_norm",
                self.max_update_fro_norm,
                lower=0.0,
                strict_lower=True,
            )
        _validate_integer("seed", self.seed, minimum=0)


@dataclass(frozen=True)
class UpdateSummary:
    """Scalar audit record; full matrices remain available for the last update."""

    update_index: int
    n_samples: int
    contexts: tuple[int, ...]
    raw_l1: float
    applied_l1: float
    applied_fro: float
    weight_fro: float


def _extract_feedback_basis(feedback_basis: object) -> tuple[np.ndarray, str | None]:
    candidate = getattr(feedback_basis, "basis", feedback_basis)
    mode = getattr(feedback_basis, "mode", None)
    if mode is not None and mode not in {"aligned", "random", "orthogonal", "shuffled"}:
        raise ValueError(f"unknown feedback mode: {mode!r}")
    basis = np.asarray(candidate, dtype=float)
    if basis.ndim != 2:
        raise ValueError("feedback_basis must be a 2-D array or expose a 2-D .basis")
    n_features, feedback_dim = basis.shape
    if n_features < 1 or feedback_dim < 1 or feedback_dim > n_features:
        raise ValueError("feedback_basis must have shape (N, d_m) with 1 <= d_m <= N")
    if not np.all(np.isfinite(basis)):
        raise ValueError("feedback_basis contains non-finite values")
    if not np.allclose(basis.T @ basis, np.eye(feedback_dim), atol=1e-10):
        raise ValueError("feedback_basis columns must be orthonormal")
    return _readonly(basis), mode


class LocalPredictiveModel:
    """Context-gated recurrent predictor trained by a local three-factor rule.

    Parameters
    ----------
    feedback_basis:
        An ``(N, d_m)`` orthonormal array, or an object such as
        :class:`src.tasks.latent_dynamics.FeedbackSubspace` exposing ``.basis``.
    n_contexts:
        Number of separately gated recurrent matrices.  Context labels select
        a matrix; they never alter the local plasticity equation.
    initial_weights:
        Optional ``(N, N)`` matrix shared across contexts or a full
        ``(n_contexts, N, N)`` array.  Zero initialization is the default and
        makes the learned plastic component's rank constraint explicit.  A
        nonzero initialization is treated as a fixed bulk reference: weight
        decay regularizes only the learned difference from that reference.
    """

    def __init__(
        self,
        feedback_basis: object,
        *,
        n_contexts: int = 1,
        config: LocalPredictiveConfig | None = None,
        initial_weights: np.ndarray | None = None,
    ) -> None:
        self.feedback_basis_, self.feedback_mode_ = _extract_feedback_basis(feedback_basis)
        self.n_features = self.feedback_basis_.shape[0]
        self.feedback_dim = self.feedback_basis_.shape[1]
        self.n_contexts = _validate_integer("n_contexts", n_contexts, minimum=1)
        self.config = LocalPredictiveConfig() if config is None else config
        if not isinstance(self.config, LocalPredictiveConfig):
            raise TypeError("config must be a LocalPredictiveConfig")
        self.initial_weights_ = self._validate_initial_weights(initial_weights)
        self.reset()

    def _validate_initial_weights(self, values: np.ndarray | None) -> np.ndarray:
        expected = (self.n_contexts, self.n_features, self.n_features)
        if values is None:
            weights = np.zeros(expected, dtype=float)
        else:
            weights = np.asarray(values, dtype=float)
            if weights.shape == (self.n_features, self.n_features):
                weights = np.broadcast_to(weights, expected).copy()
            elif weights.shape != expected:
                raise ValueError(
                    "initial_weights must have shape "
                    f"({self.n_features}, {self.n_features}) or {expected}"
                )
            if not np.all(np.isfinite(weights)):
                raise ValueError("initial_weights contains non-finite values")
        return _readonly(weights)

    def reset(self) -> "LocalPredictiveModel":
        """Restore initial weights and clear all plasticity accounting."""

        self.weights_ = np.array(self.initial_weights_, copy=True)
        zeros = np.zeros_like(self.weights_)
        self.last_raw_plastic_update_ = _readonly(zeros)
        self.last_applied_plastic_update_ = _readonly(zeros)
        self.raw_plasticity_cost_ = 0.0
        self.plasticity_cost_ = 0.0
        self.plasticity_cost_by_context_ = np.zeros(self.n_contexts, dtype=float)
        self.update_history_: list[UpdateSummary] = []
        self.n_updates_ = 0
        self.n_epochs_ = 0
        self.converged_ = False
        self.feedback_permutation_: np.ndarray | None = None
        self.feedback_block_ids_: np.ndarray | None = None
        return self

    @property
    def feedback_projector(self) -> np.ndarray:
        """Dense projector for diagnostics; training uses the factored basis."""

        return self.feedback_basis_ @ self.feedback_basis_.T

    @property
    def plastic_component(self) -> np.ndarray:
        return self.weights_ - self.initial_weights_

    @property
    def raw_plastic_update(self) -> np.ndarray:
        return np.array(self.last_raw_plastic_update_, copy=True)

    @property
    def applied_plastic_update(self) -> np.ndarray:
        return np.array(self.last_applied_plastic_update_, copy=True)

    @property
    def plasticity_cost(self) -> float:
        """Cumulative L1 norm of actual applied recurrent weight changes."""

        return float(self.plasticity_cost_)

    @property
    def raw_plasticity_cost(self) -> float:
        """Cumulative L1 norm before decay and optional update clipping."""

        return float(self.raw_plasticity_cost_)

    @property
    def temporal_feedback_permutation(self) -> np.ndarray | None:
        """Recorded source-time ordering used by the most recent fit call."""

        if self.feedback_permutation_ is None:
            return None
        return np.array(self.feedback_permutation_, copy=True)

    def _validate_samples(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(inputs, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("inputs and targets must be 2-D arrays")
        if x.shape != y.shape or x.shape[0] < 1 or x.shape[1] != self.n_features:
            raise ValueError(
                f"inputs and targets must have equal shape (n_samples, {self.n_features})"
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("inputs and targets must contain only finite values")
        return x, y

    def _validate_contexts(
        self,
        contexts: int | Iterable[int] | np.ndarray | None,
        n_samples: int,
    ) -> np.ndarray:
        if contexts is None:
            if self.n_contexts != 1 and n_samples:
                raise ValueError("contexts are required when n_contexts > 1")
            return np.zeros(n_samples, dtype=int)
        if isinstance(contexts, (bool, np.bool_)):
            raise TypeError("context labels must be integers, not booleans")
        if isinstance(contexts, (int, np.integer)):
            labels = np.full(n_samples, int(contexts), dtype=int)
        else:
            if isinstance(contexts, np.ndarray):
                raw = contexts
            else:
                try:
                    raw = np.asarray(list(contexts))
                except TypeError as exc:
                    raise TypeError("contexts must be an integer or one-dimensional iterable") from exc
            if raw.ndim != 1 or raw.shape[0] != n_samples:
                raise ValueError("contexts must have one label per sample")
            if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
                raw.dtype, np.integer
            ):
                raise TypeError("all context labels must be integers")
            labels = raw.astype(int, copy=False)
        if np.any(labels < 0) or np.any(labels >= self.n_contexts):
            raise ValueError(f"context labels must lie in [0, {self.n_contexts - 1}]")
        return labels

    def _prepare_feedback_order(
        self,
        feedback_permutation: Iterable[int] | np.ndarray | None,
        block_ids: Iterable[int] | np.ndarray | None,
        contexts: np.ndarray,
    ) -> np.ndarray:
        """Validate and record a no-leakage temporal feedback permutation."""

        n_samples = contexts.shape[0]
        identity = np.arange(n_samples, dtype=int)
        if feedback_permutation is None:
            if self.feedback_mode_ == "shuffled":
                raise ValueError(
                    "shuffled feedback requires a block-local feedback_permutation"
                )
            self.feedback_permutation_ = None
            self.feedback_block_ids_ = None
            return identity
        if self.feedback_mode_ not in (None, "shuffled"):
            raise ValueError(
                "feedback_permutation is only valid for shuffled feedback controls"
            )
        if block_ids is None:
            raise ValueError("block_ids are required with feedback_permutation")

        if isinstance(feedback_permutation, np.ndarray):
            permutation = feedback_permutation
        else:
            try:
                permutation = np.asarray(list(feedback_permutation))
            except TypeError as exc:
                raise TypeError("feedback_permutation must be an integer sequence") from exc
        if permutation.ndim != 1 or permutation.shape[0] != n_samples:
            raise ValueError("feedback_permutation must contain one index per sample")
        if np.issubdtype(permutation.dtype, np.bool_) or not np.issubdtype(
            permutation.dtype, np.integer
        ):
            raise TypeError("feedback_permutation must contain integers")
        permutation = permutation.astype(int, copy=False)
        if not np.array_equal(np.sort(permutation), identity):
            raise ValueError("feedback_permutation must contain every sample index once")
        if np.any(permutation == identity):
            raise ValueError("feedback_permutation must not contain fixed time points")

        if isinstance(block_ids, np.ndarray):
            blocks = block_ids
        else:
            try:
                blocks = np.asarray(list(block_ids))
            except TypeError as exc:
                raise TypeError("block_ids must be an integer sequence") from exc
        if blocks.ndim != 1 or blocks.shape[0] != n_samples:
            raise ValueError("block_ids must contain one identifier per sample")
        if np.issubdtype(blocks.dtype, np.bool_) or not np.issubdtype(
            blocks.dtype, np.integer
        ):
            raise TypeError("block_ids must contain integers")
        blocks = blocks.astype(int, copy=False)
        if np.any(blocks != blocks[permutation]):
            raise ValueError("feedback_permutation cannot cross trajectory/block boundaries")
        if np.any(contexts != contexts[permutation]):
            raise ValueError("feedback_permutation cannot cross context boundaries")

        self.feedback_permutation_ = _readonly_int(permutation)
        self.feedback_block_ids_ = _readonly_int(blocks)
        return permutation

    def _limit_update(self, update: np.ndarray) -> np.ndarray:
        maximum = self.config.max_update_fro_norm
        if maximum is None:
            return update
        norm = float(np.linalg.norm(update))
        if norm <= maximum:
            return update
        return update * (maximum / norm)

    def _apply_hebbian_updates(
        self,
        hebbian_by_context: dict[int, np.ndarray],
        *,
        n_samples: int,
    ) -> None:
        raw_all = np.zeros_like(self.weights_)
        applied_all = np.zeros_like(self.weights_)
        for context, hebbian in hebbian_by_context.items():
            raw = self.config.learning_rate * hebbian
            plastic_component = (
                self.weights_[context] - self.initial_weights_[context]
            )
            proposed = raw - (
                self.config.learning_rate
                * self.config.weight_decay
                * plastic_component
            )
            applied = self._limit_update(proposed)
            raw_all[context] = raw
            applied_all[context] = applied
            self.weights_[context] += applied
            self.plasticity_cost_by_context_[context] += float(np.abs(applied).sum())

        raw_l1 = float(np.abs(raw_all).sum())
        applied_l1 = float(np.abs(applied_all).sum())
        self.raw_plasticity_cost_ += raw_l1
        self.plasticity_cost_ += applied_l1
        self.last_raw_plastic_update_ = _readonly(raw_all)
        self.last_applied_plastic_update_ = _readonly(applied_all)
        self.n_updates_ += 1
        active_contexts = tuple(sorted(hebbian_by_context))
        self.update_history_.append(
            UpdateSummary(
                update_index=self.n_updates_,
                n_samples=n_samples,
                contexts=active_contexts,
                raw_l1=raw_l1,
                applied_l1=applied_l1,
                applied_fro=float(np.linalg.norm(applied_all)),
                weight_fro=float(np.linalg.norm(self.weights_)),
            )
        )

    def _update_batch(
        self,
        presynaptic: np.ndarray,
        feedback_inputs: np.ndarray,
        feedback_targets: np.ndarray,
        contexts: np.ndarray,
    ) -> None:
        hebbian_by_context: dict[int, np.ndarray] = {}
        for context in np.unique(contexts):
            selected = contexts == context
            presynaptic_context = presynaptic[selected]
            feedback_input_context = feedback_inputs[selected]
            feedback_target_context = feedback_targets[selected]
            error = (
                feedback_target_context
                - feedback_input_context @ self.weights_[context].T
            )
            # q = P P.T error in column notation.  Row-major evaluation avoids
            # ever allocating the dense feedback projector.
            projected_error = (
                error @ self.feedback_basis_
            ) @ self.feedback_basis_.T
            hebbian_by_context[int(context)] = (
                projected_error.T @ presynaptic_context
                / presynaptic_context.shape[0]
            )
        self._apply_hebbian_updates(
            hebbian_by_context, n_samples=presynaptic.shape[0]
        )

    def partial_fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        contexts: int | Iterable[int] | np.ndarray | None = None,
        *,
        feedback_permutation: Iterable[int] | np.ndarray | None = None,
        block_ids: Iterable[int] | np.ndarray | None = None,
    ) -> "LocalPredictiveModel":
        """Apply one local mini-batch update without resetting model state."""

        x, y = self._validate_samples(inputs, targets)
        labels = self._validate_contexts(contexts, x.shape[0])
        feedback_order = self._prepare_feedback_order(
            feedback_permutation, block_ids, labels
        )
        self._update_batch(x, x[feedback_order], y[feedback_order], labels)
        return self

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        contexts: int | Iterable[int] | np.ndarray | None = None,
        *,
        feedback_permutation: Iterable[int] | np.ndarray | None = None,
        block_ids: Iterable[int] | np.ndarray | None = None,
        reset: bool = True,
    ) -> "LocalPredictiveModel":
        """Run deterministic local mini-batch epochs until convergence."""

        x, y = self._validate_samples(inputs, targets)
        labels = self._validate_contexts(contexts, x.shape[0])
        if not isinstance(reset, (bool, np.bool_)):
            raise TypeError("reset must be boolean")
        if reset:
            self.reset()
        else:
            self.converged_ = False
            self.n_epochs_ = 0
        feedback_order = self._prepare_feedback_order(
            feedback_permutation, block_ids, labels
        )
        batch_size = x.shape[0] if self.config.batch_size is None else self.config.batch_size
        rng = make_rng(self.config.seed, "local-predictive", "mini-batches")
        base_order = np.arange(x.shape[0])

        for epoch in range(self.config.max_epochs):
            before = self.weights_.copy()
            if self.config.shuffle_batches:
                order = rng.permutation(base_order)
            else:
                order = base_order
            for start in range(0, x.shape[0], batch_size):
                batch = order[start : start + batch_size]
                feedback_batch = feedback_order[batch]
                self._update_batch(
                    x[batch],
                    x[feedback_batch],
                    y[feedback_batch],
                    labels[batch],
                )
            self.n_epochs_ = epoch + 1
            change = float(np.linalg.norm(self.weights_ - before))
            threshold = self.config.tolerance * (1.0 + float(np.linalg.norm(before)))
            if change <= threshold:
                self.converged_ = True
                break
        return self

    def fit_fixed_point(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        contexts: int | Iterable[int] | np.ndarray | None = None,
        *,
        feedback_permutation: Iterable[int] | np.ndarray | None = None,
        block_ids: Iterable[int] | np.ndarray | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        reset: bool = True,
    ) -> "LocalPredictiveModel":
        """Iterate the exact full-batch local rule using sufficient statistics.

        For each context, the input and target cross-moments are computed once.
        In the ordinary condition these are ``X.T X / n`` and ``Y.T X / n``;
        temporal shuffling uses source-time inputs/targets against current-time
        presynaptic activity.  Every subsequent iteration is independent of
        sample count.  This is an iterative local-rule fixed point, not a
        least-squares solve and not a gradient-through-time method.
        """

        x, y = self._validate_samples(inputs, targets)
        labels = self._validate_contexts(contexts, x.shape[0])
        if max_iterations is None:
            max_iterations = self.config.max_epochs
        else:
            max_iterations = _validate_integer(
                "max_iterations", max_iterations, minimum=1
            )
        if tolerance is None:
            tolerance = self.config.tolerance
        else:
            tolerance = _validate_float("tolerance", tolerance, lower=0.0)
        if not isinstance(reset, (bool, np.bool_)):
            raise TypeError("reset must be boolean")
        if reset:
            self.reset()
        else:
            self.converged_ = False
            self.n_epochs_ = 0
        feedback_order = self._prepare_feedback_order(
            feedback_permutation, block_ids, labels
        )

        statistics: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        counts: dict[int, int] = {}
        for context in np.unique(labels):
            selected = labels == context
            x_context = x[selected]
            source = feedback_order[selected]
            feedback_inputs = x[source]
            feedback_targets = y[source]
            count = x_context.shape[0]
            statistics[int(context)] = (
                feedback_targets.T @ x_context / count,
                feedback_inputs.T @ x_context / count,
            )
            counts[int(context)] = count

        for iteration in range(max_iterations):
            before = self.weights_.copy()
            hebbian_by_context: dict[int, np.ndarray] = {}
            for context, (target_cross, input_covariance) in statistics.items():
                residual_cross = (
                    target_cross - self.weights_[context] @ input_covariance
                )
                hebbian_by_context[context] = self.feedback_basis_ @ (
                    self.feedback_basis_.T @ residual_cross
                )
            self._apply_hebbian_updates(
                hebbian_by_context,
                n_samples=sum(counts.values()),
            )
            self.n_epochs_ = iteration + 1
            change = float(np.linalg.norm(self.weights_ - before))
            threshold = tolerance * (1.0 + float(np.linalg.norm(before)))
            if change <= threshold:
                self.converged_ = True
                break
        return self

    def predict(
        self,
        inputs: np.ndarray,
        contexts: int | Iterable[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict one step with the context-selected recurrent matrix."""

        array = np.asarray(inputs, dtype=float)
        was_vector = array.ndim == 1
        if was_vector:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != self.n_features:
            raise ValueError(
                f"inputs must have shape ({self.n_features},) or "
                f"(n_samples, {self.n_features})"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("inputs contain non-finite values")
        labels = self._validate_contexts(contexts, array.shape[0])
        selected_weights = self.weights_[labels]
        prediction = np.einsum("bij,bj->bi", selected_weights, array, optimize=True)
        return prediction[0] if was_vector else prediction

    def rollout(
        self,
        initial_activity: np.ndarray,
        steps: int,
        contexts: int | Iterable[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Autonomously roll out predictions, including the initial state."""

        steps = _validate_integer("steps", steps, minimum=0)
        initial = np.asarray(initial_activity, dtype=float)
        if initial.shape != (self.n_features,):
            raise ValueError(f"initial_activity must have shape ({self.n_features},)")
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial_activity contains non-finite values")
        if steps == 0:
            return initial[None, :].copy()
        labels = self._validate_contexts(contexts, steps)
        trajectory = np.empty((steps + 1, self.n_features), dtype=float)
        trajectory[0] = initial
        for step, context in enumerate(labels):
            trajectory[step + 1] = self.weights_[context] @ trajectory[step]
        return trajectory


__all__ = [
    "LocalPredictiveConfig",
    "LocalPredictiveModel",
    "UpdateSummary",
]
