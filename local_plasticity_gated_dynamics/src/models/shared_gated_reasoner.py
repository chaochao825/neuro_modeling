"""Non-spiking hierarchical candidate controller with local readout fitting.

The architecture borrows two narrow, declared abstractions: a fast/slow
hierarchy inspired by hierarchical reasoning models, and (in ``trace`` mode)
discounted bilinear activation traces inspired by Continuous Thought Machines.
It is neither an implementation of HRM nor CTM.  All recurrent weights are
frozen; only a task-balanced closed-form readout is fit, so no BPTT occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.models.structured_reasoner import (
    CandidateSet,
    ComputeBudget,
    ComputeReceipt,
    FitReceipt,
    SolverOutput,
    StructuredReasonerError,
    TrainingCandidateSet,
)


_MODES = frozenset({"hierarchical", "flat", "trace"})


def _scaled_recurrent_matrix(
    rng: np.random.Generator,
    size: int,
    radius: float,
) -> np.ndarray:
    matrix = rng.normal(0.0, 1.0 / np.sqrt(size), size=(size, size))
    spectral = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if spectral > 0.0:
        matrix *= radius / spectral
    return matrix


@dataclass(frozen=True)
class _Encoding:
    representations: np.ndarray
    state_trace: np.ndarray
    bilinear_trace: np.ndarray
    fast_updates: int
    slow_updates: int
    trace_updates: int
    exhausted: bool


class HierarchicalCandidateController:
    """Frozen fast/slow rate dynamics plus a local, closed-form readout."""

    used_bptt = False

    def __init__(
        self,
        *,
        feature_dim: int,
        fast_dim: int = 24,
        slow_dim: int = 12,
        control_dim: int = 4,
        mode: str = "hierarchical",
        cycles: int = 3,
        fast_steps_per_cycle: int = 2,
        trace_pairs: int = 8,
        trace_decay: float = 0.9,
        ridge: float = 1e-3,
        seed: int = 0,
    ) -> None:
        integer_args = {
            "feature_dim": feature_dim,
            "fast_dim": fast_dim,
            "slow_dim": slow_dim,
            "control_dim": control_dim,
            "cycles": cycles,
            "fast_steps_per_cycle": fast_steps_per_cycle,
            "trace_pairs": trace_pairs,
            "seed": seed,
        }
        for name, value in integer_args.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise StructuredReasonerError(f"{name} must be an integer")
        if min(
            feature_dim,
            fast_dim,
            slow_dim,
            control_dim,
            cycles,
            fast_steps_per_cycle,
            trace_pairs,
        ) < 1:
            raise StructuredReasonerError("model dimensions and step counts must be positive")
        if seed < 0:
            raise StructuredReasonerError("seed must be non-negative")
        if control_dim >= fast_dim:
            raise StructuredReasonerError(
                "control_dim must be smaller than fast_dim to form a bottleneck"
            )
        if mode not in _MODES:
            raise StructuredReasonerError(f"mode must be one of {sorted(_MODES)}")
        if not 0.0 <= trace_decay < 1.0:
            raise StructuredReasonerError("trace_decay must lie in [0, 1)")
        if not np.isfinite(ridge) or ridge <= 0.0:
            raise StructuredReasonerError("ridge must be finite and positive")

        self.feature_dim = int(feature_dim)
        self.fast_dim = int(fast_dim)
        self.slow_dim = int(slow_dim)
        self.control_dim = int(control_dim)
        self.mode = mode
        self.cycles = int(cycles)
        self.fast_steps_per_cycle = int(fast_steps_per_cycle)
        self.trace_pairs = int(trace_pairs)
        self.trace_decay = float(trace_decay)
        self.ridge = float(ridge)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self.input_to_fast = rng.normal(
            0.0,
            1.0 / np.sqrt(self.feature_dim),
            size=(self.fast_dim, self.feature_dim),
        )
        self.fast_recurrent = _scaled_recurrent_matrix(rng, self.fast_dim, 0.55)
        self.fast_to_slow = rng.normal(
            0.0,
            1.0 / np.sqrt(self.fast_dim),
            size=(self.slow_dim, self.fast_dim),
        )
        self.slow_recurrent = _scaled_recurrent_matrix(rng, self.slow_dim, 0.5)
        # This explicit factorization is the low-dimensional H -> L control path.
        self.slow_to_control = rng.normal(
            0.0,
            1.0 / np.sqrt(self.slow_dim),
            size=(self.control_dim, self.slow_dim),
        )
        self.control_to_fast = rng.normal(
            0.0,
            1.0 / np.sqrt(self.control_dim),
            size=(self.fast_dim, self.control_dim),
        )
        self.trace_pair_left = rng.integers(0, self.fast_dim, self.trace_pairs)
        self.trace_pair_right = rng.integers(0, self.fast_dim, self.trace_pairs)
        for frozen in (
            self.input_to_fast,
            self.fast_recurrent,
            self.fast_to_slow,
            self.slow_recurrent,
            self.slow_to_control,
            self.control_to_fast,
            self.trace_pair_left,
            self.trace_pair_right,
        ):
            frozen.setflags(write=False)

        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.readout_: np.ndarray | None = None
        self.fit_receipt_: FitReceipt | None = None
        self.task_sample_weights_: dict[str, np.ndarray] = {}

    @property
    def control_operator(self) -> np.ndarray:
        """Return the rank-at-most-``control_dim`` slow-to-fast operator."""

        operator = self.control_to_fast @ self.slow_to_control
        operator.setflags(write=False)
        return operator

    @property
    def representation_dim(self) -> int:
        base = self.fast_dim
        if self.mode != "flat":
            base += self.slow_dim + self.control_dim
        if self.mode == "trace":
            base += self.trace_pairs
        return base

    @property
    def required_internal_steps(self) -> int:
        slow = 0 if self.mode == "flat" else 1
        return self.cycles * (self.fast_steps_per_cycle + slow)

    def _check_candidates(self, candidates: CandidateSet) -> None:
        if not isinstance(candidates, CandidateSet):
            raise TypeError("solve accepts CandidateSet only")
        if candidates.feature_dim != self.feature_dim:
            raise StructuredReasonerError(
                f"expected feature_dim={self.feature_dim}, got {candidates.feature_dim}"
            )

    @staticmethod
    def _task_balanced_scaler(
        tasks: Sequence[TrainingCandidateSet],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        feature_dim = tasks[0].public.feature_dim
        total = np.zeros(feature_dim, dtype=float)
        second = np.zeros(feature_dim, dtype=float)
        weights: dict[str, np.ndarray] = {}
        task_weight = 1.0 / len(tasks)
        for task in tasks:
            count = task.public.n_candidates
            sample_weight = np.full(count, task_weight / count, dtype=float)
            weights[task.public.task_id] = sample_weight
            total += np.sum(task.public.features * sample_weight[:, None], axis=0)
            second += np.sum(
                np.square(task.public.features) * sample_weight[:, None], axis=0
            )
        variance = np.maximum(second - np.square(total), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-8] = 1.0
        return total, scale, weights

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("fit must be called before encoding or solving")
        return (features - self.feature_mean_) / self.feature_scale_

    def _encode(
        self,
        features: np.ndarray,
        *,
        max_internal_steps: int,
    ) -> _Encoding:
        x = self._standardize(features)
        count = x.shape[0]
        fast = np.zeros((count, self.fast_dim), dtype=float)
        slow = np.zeros(self.slow_dim, dtype=float)
        alpha = np.zeros((count, self.trace_pairs), dtype=float)
        beta = np.zeros((count, self.trace_pairs), dtype=float)
        state_rows: list[np.ndarray] = []
        trace_rows: list[np.ndarray] = []
        fast_updates = 0
        slow_updates = 0
        trace_updates = 0
        internal_steps = 0

        for _ in range(self.cycles):
            if internal_steps >= max_internal_steps:
                break
            if self.mode == "flat":
                control = np.zeros(self.control_dim, dtype=float)
            else:
                control = np.tanh(self.slow_to_control @ slow)
            for _ in range(self.fast_steps_per_cycle):
                if internal_steps >= max_internal_steps:
                    break
                drive = x @ self.input_to_fast.T + fast @ self.fast_recurrent.T
                if self.mode != "flat":
                    drive += control @ self.control_to_fast.T
                fast = np.tanh(drive)
                fast_updates += 1
                internal_steps += 1
                if self.mode == "trace":
                    products = (
                        fast[:, self.trace_pair_left]
                        * fast[:, self.trace_pair_right]
                    )
                    alpha = self.trace_decay * alpha + products
                    beta = self.trace_decay * beta + 1.0
                    trace_value = alpha / np.sqrt(beta)
                    trace_rows.append(np.mean(trace_value, axis=0))
                    trace_updates += 1
                monitor = [np.mean(fast, axis=0)]
                if self.mode != "flat":
                    monitor.extend((slow, control))
                state_rows.append(np.concatenate(monitor))

            if self.mode != "flat" and internal_steps < max_internal_steps:
                slow = np.tanh(
                    self.slow_recurrent @ slow
                    + self.fast_to_slow @ np.mean(fast, axis=0)
                )
                slow_updates += 1
                internal_steps += 1
                control = np.tanh(self.slow_to_control @ slow)
                state_rows.append(np.concatenate((np.mean(fast, axis=0), slow, control)))

        components = [fast]
        if self.mode != "flat":
            control = np.tanh(self.slow_to_control @ slow)
            components.extend(
                (
                    np.repeat(slow[None, :], count, axis=0),
                    np.repeat(control[None, :], count, axis=0),
                )
            )
        if self.mode == "trace":
            trace_value = alpha / np.sqrt(np.maximum(beta, 1.0))
            components.append(trace_value)
        representations = np.concatenate(components, axis=1)
        state_trace = (
            np.stack(state_rows)
            if state_rows
            else np.empty((0, self.representation_dim), dtype=float)
        )
        if self.mode == "trace" and trace_rows:
            bilinear_trace = np.stack(trace_rows)
        else:
            bilinear_trace = np.empty((0, self.trace_pairs), dtype=float)
        return _Encoding(
            representations=representations,
            state_trace=state_trace,
            bilinear_trace=bilinear_trace,
            fast_updates=fast_updates,
            slow_updates=slow_updates,
            trace_updates=trace_updates,
            exhausted=internal_steps < self.required_internal_steps,
        )

    def fit(self, tasks: Sequence[TrainingCandidateSet]) -> FitReceipt:
        """Fit only the final readout with task-balanced weighted ridge."""

        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
            raise TypeError("fit expects a sequence of TrainingCandidateSet objects")
        training = tuple(tasks)
        if not training or not all(
            isinstance(task, TrainingCandidateSet) for task in training
        ):
            raise TypeError("fit accepts TrainingCandidateSet objects only")
        task_ids = tuple(task.public.task_id for task in training)
        if len(set(task_ids)) != len(task_ids):
            raise StructuredReasonerError("training task_ids must be unique")
        for task in training:
            self._check_candidates(task.public)

        mean, scale, weights = self._task_balanced_scaler(training)
        self.feature_mean_ = mean.copy()
        self.feature_scale_ = scale.copy()
        self.task_sample_weights_ = {
            key: value.copy() for key, value in weights.items()
        }

        design_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for task in training:
            encoded = self._encode(
                task.public.features,
                max_internal_steps=self.required_internal_steps,
            )
            design_parts.append(encoded.representations)
            label_parts.append(task.labels)
            weight_parts.append(weights[task.public.task_id])
        design = np.concatenate(design_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)
        sample_weights = np.concatenate(weight_parts, axis=0)
        augmented = np.column_stack((design, np.ones(design.shape[0])))
        weighted = augmented * np.sqrt(sample_weights[:, None])
        weighted_labels = labels * np.sqrt(sample_weights)
        regularizer = self.ridge * np.eye(augmented.shape[1])
        regularizer[-1, -1] = 0.0
        gram = weighted.T @ weighted + regularizer
        rhs = weighted.T @ weighted_labels
        try:
            self.readout_ = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.readout_ = np.linalg.pinv(gram) @ rhs
        predictions = augmented @ self.readout_
        per_task_losses: list[float] = []
        offset = 0
        for task in training:
            count = task.public.n_candidates
            error = predictions[offset : offset + count] - task.labels
            per_task_losses.append(float(np.mean(np.square(error))))
            offset += count
        receipt = FitReceipt(
            task_ids=task_ids,
            used_bptt=False,
            task_balanced=True,
            optimization="closed_form_task_balanced_ridge_readout",
            epochs=1,
            final_loss=float(np.mean(per_task_losses)),
            seed=self.seed,
        )
        self.fit_receipt_ = receipt
        return receipt

    def solve(
        self,
        candidates: CandidateSet,
        budget: ComputeBudget | None = None,
    ) -> SolverOutput:
        """Select a candidate without accepting or inspecting any labels."""

        self._check_candidates(candidates)
        if self.readout_ is None:
            raise RuntimeError("fit must be called before solve")
        if budget is None:
            budget = ComputeBudget(
                max_candidate_evaluations=candidates.n_candidates,
                max_internal_steps=self.required_internal_steps,
            )
        if not isinstance(budget, ComputeBudget):
            raise TypeError("budget must be a ComputeBudget")
        if candidates.n_candidates > budget.max_candidate_evaluations:
            raise StructuredReasonerError(
                "candidate set exceeds max_candidate_evaluations; refusing partial selection"
            )
        encoding = self._encode(
            candidates.features,
            max_internal_steps=budget.max_internal_steps,
        )
        augmented = np.column_stack(
            (encoding.representations, np.ones(candidates.n_candidates))
        )
        scores = augmented @ self.readout_
        selected = int(np.argmax(scores))
        internal_steps = encoding.fast_updates + encoding.slow_updates
        receipt = ComputeReceipt(
            budget=budget,
            candidate_evaluations=candidates.n_candidates,
            internal_steps=internal_steps,
            fast_updates=encoding.fast_updates,
            slow_updates=encoding.slow_updates,
            trace_updates=encoding.trace_updates,
            charged_units=candidates.n_candidates + internal_steps,
            exhausted=encoding.exhausted,
        )
        return SolverOutput(
            task_id=candidates.task_id,
            selected_index=selected,
            selected_candidate_id=candidates.candidate_ids[selected],
            selected_output=candidates.candidate_outputs[selected],
            scores=scores,
            trace=encoding.state_trace,
            bilinear_trace=encoding.bilinear_trace,
            receipt=receipt,
        )
