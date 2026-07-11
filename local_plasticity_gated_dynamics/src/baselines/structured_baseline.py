"""Small global-gradient recurrent baseline for structured candidate ranking.

This module is intentionally outside ``src.models``: gradients propagate
through the complete candidate sequence, making this a BPTT baseline rather
than part of the local-learning main model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.models.structured_reasoner import (
    CandidateSet,
    ComputeBudget,
    ComputeReceipt,
    FitReceipt,
    SolverOutput,
    StructuredReasonerError,
    TrainingCandidateSet,
)


class _CandidateGRU(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.gru(features)
        return self.readout(hidden).squeeze(-1), hidden


class SmallGRUBPTTBaseline:
    """Task-balanced GRU trained by full sequence BPTT."""

    used_bptt = True

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 16,
        epochs: int = 30,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> None:
        integer_args = {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "seed": seed,
        }
        for name, value in integer_args.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise StructuredReasonerError(f"{name} must be an integer")
        if min(feature_dim, hidden_dim, epochs) < 1 or seed < 0:
            raise StructuredReasonerError(
                "feature_dim, hidden_dim, epochs must be positive and seed non-negative"
            )
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise StructuredReasonerError("learning_rate must be finite and positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        self.network = _CandidateGRU(self.feature_dim, self.hidden_dim).to(
            dtype=torch.float64
        )
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.fit_receipt_: FitReceipt | None = None
        self.task_loss_weights_: dict[str, float] = {}

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
    ) -> tuple[np.ndarray, np.ndarray]:
        dim = tasks[0].public.feature_dim
        mean = np.zeros(dim, dtype=float)
        second = np.zeros(dim, dtype=float)
        task_weight = 1.0 / len(tasks)
        for task in tasks:
            sample_weight = task_weight / task.public.n_candidates
            mean += sample_weight * np.sum(task.public.features, axis=0)
            second += sample_weight * np.sum(np.square(task.public.features), axis=0)
        scale = np.sqrt(np.maximum(second - np.square(mean), 0.0))
        scale[scale < 1e-8] = 1.0
        return mean, scale

    def _tensor(self, candidates: CandidateSet) -> torch.Tensor:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("fit must be called before solve")
        standardized = (
            candidates.features - self.feature_mean_
        ) / self.feature_scale_
        return torch.as_tensor(standardized[None, :, :], dtype=torch.float64)

    def fit(self, tasks: Sequence[TrainingCandidateSet]) -> FitReceipt:
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

        mean, scale = self._task_balanced_scaler(training)
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        equal_task_weight = 1.0 / len(training)
        self.task_loss_weights_ = {
            task.public.task_id: equal_task_weight for task in training
        }
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        final_loss = float("nan")
        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            task_losses: list[torch.Tensor] = []
            for task in training:
                logits, _ = self.network(self._tensor(task.public))
                labels = torch.tensor(task.labels[None, :], dtype=torch.float64)
                task_losses.append(
                    nn.functional.binary_cross_entropy_with_logits(logits, labels)
                )
            # Each task, not each candidate, is one equally weighted training unit.
            loss = torch.stack(task_losses).mean()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        receipt = FitReceipt(
            task_ids=task_ids,
            used_bptt=True,
            task_balanced=True,
            optimization="adam_full_candidate_sequence_bptt",
            epochs=self.epochs,
            final_loss=final_loss,
            seed=self.seed,
        )
        self.fit_receipt_ = receipt
        return receipt

    def solve(
        self,
        candidates: CandidateSet,
        budget: ComputeBudget | None = None,
    ) -> SolverOutput:
        self._check_candidates(candidates)
        if self.fit_receipt_ is None:
            raise RuntimeError("fit must be called before solve")
        if budget is None:
            budget = ComputeBudget(
                max_candidate_evaluations=candidates.n_candidates,
                max_internal_steps=candidates.n_candidates,
            )
        if not isinstance(budget, ComputeBudget):
            raise TypeError("budget must be a ComputeBudget")
        if candidates.n_candidates > budget.max_candidate_evaluations:
            raise StructuredReasonerError(
                "candidate set exceeds max_candidate_evaluations; refusing partial selection"
            )
        if candidates.n_candidates > budget.max_internal_steps:
            raise StructuredReasonerError(
                "GRU sequence exceeds max_internal_steps; refusing truncated inference"
            )
        self.network.eval()
        with torch.no_grad():
            logits, hidden = self.network(self._tensor(candidates))
        scores = logits[0].cpu().numpy()
        trace = hidden[0].cpu().numpy()
        selected = int(np.argmax(scores))
        receipt = ComputeReceipt(
            budget=budget,
            candidate_evaluations=candidates.n_candidates,
            internal_steps=candidates.n_candidates,
            fast_updates=candidates.n_candidates,
            slow_updates=0,
            trace_updates=0,
            charged_units=2 * candidates.n_candidates,
            exhausted=False,
        )
        return SolverOutput(
            task_id=candidates.task_id,
            selected_index=selected,
            selected_candidate_id=candidates.candidate_ids[selected],
            selected_output=candidates.candidate_outputs[selected],
            scores=scores,
            trace=trace,
            bilinear_trace=np.empty((0, 0), dtype=float),
            receipt=receipt,
        )
