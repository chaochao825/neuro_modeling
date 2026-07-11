"""Capability, dynamics, and BPTT-separation tests for exp13 reasoners."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.baselines.structured_baseline import SmallGRUBPTTBaseline
from src.models.shared_gated_reasoner import HierarchicalCandidateController
from src.models.structured_reasoner import (
    CandidateSet,
    ComputeBudget,
    StructuredReasoner,
    TrainingCandidateSet,
)


def _public(
    task_id: str,
    features: np.ndarray,
) -> CandidateSet:
    count = features.shape[0]
    outputs = tuple(np.array([[index]], dtype=int) for index in range(count))
    return CandidateSet(
        task_id=task_id,
        family="arc",
        candidate_ids=tuple(f"candidate-{index}" for index in range(count)),
        features=features,
        candidate_outputs=outputs,
        candidate_provenance=tuple(f"generator:{index}" for index in range(count)),
    )


def _training() -> tuple[TrainingCandidateSet, ...]:
    return (
        TrainingCandidateSet(
            _public("train-small", np.array([[2.0, 0.0], [-1.0, 0.0]])),
            np.array([1.0, 0.0]),
        ),
        TrainingCandidateSet(
            _public(
                "train-large",
                np.array(
                    [
                        [1.5, 0.2],
                        [-0.5, -0.1],
                        [-1.0, 0.2],
                        [-1.5, -0.2],
                        [-2.0, 0.1],
                    ]
                ),
            ),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    )


def test_candidate_set_is_label_free_and_defensively_immutable() -> None:
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    first_output = np.array([[3]])
    public = CandidateSet(
        task_id="held-out",
        family="ARC",
        candidate_ids=("a", "b"),
        features=features,
        candidate_outputs=(first_output, np.array([[4]])),
        candidate_provenance=("p:a", "p:b"),
    )
    features[0, 0] = 99.0
    first_output[0, 0] = 99
    assert not hasattr(public, "labels")
    assert not hasattr(public, "target")
    assert public.features[0, 0] == 1.0
    assert public.candidate_outputs[0][0, 0] == 3
    with pytest.raises(ValueError):
        public.features[0, 0] = 2.0
    with pytest.raises(ValueError):
        public.candidate_outputs[0][0, 0] = 2


def test_solve_accepts_only_public_capability_and_is_deterministic() -> None:
    training = _training()
    public = _public("held-out", np.array([[1.2, 0.1], [-0.8, -0.1]]))
    outputs = []
    for _ in range(2):
        model = HierarchicalCandidateController(
            feature_dim=2,
            fast_dim=8,
            slow_dim=5,
            control_dim=2,
            seed=7,
        )
        receipt = model.fit(training)
        assert not receipt.used_bptt
        assert isinstance(model, StructuredReasoner)
        outputs.append(model.solve(public))
        with pytest.raises(TypeError, match="CandidateSet only"):
            model.solve(training[0])  # type: ignore[arg-type]
    assert outputs[0].selected_candidate_id == outputs[1].selected_candidate_id
    np.testing.assert_allclose(outputs[0].scores, outputs[1].scores)
    np.testing.assert_allclose(outputs[0].trace, outputs[1].trace)


def test_low_dimensional_control_is_an_explicit_rank_bottleneck() -> None:
    model = HierarchicalCandidateController(
        feature_dim=2,
        fast_dim=11,
        slow_dim=7,
        control_dim=3,
        seed=3,
    )
    assert model.slow_to_control.shape == (3, 7)
    assert model.control_to_fast.shape == (11, 3)
    assert model.control_operator.shape == (11, 7)
    assert np.linalg.matrix_rank(model.control_operator) <= 3
    model.fit(_training())
    result = model.solve(_training()[0].public)
    assert result.trace.shape[1] == 11 + 7 + 3


def test_discounted_bilinear_trace_recurrence_is_rate_based() -> None:
    task = TrainingCandidateSet(
        _public("single", np.array([[1.0, -0.25], [-0.5, 0.5]])),
        np.array([1.0, 0.0]),
    )
    model = HierarchicalCandidateController(
        feature_dim=2,
        fast_dim=6,
        slow_dim=4,
        control_dim=2,
        mode="trace",
        cycles=2,
        fast_steps_per_cycle=2,
        trace_pairs=3,
        trace_decay=0.6,
        seed=5,
    )
    model.fit((task,))
    result = model.solve(task.public)
    assert result.receipt.trace_updates == result.receipt.fast_updates == 4
    assert result.bilinear_trace.shape == (4, 3)
    assert np.isfinite(result.bilinear_trace).all()
    assert np.any(np.abs(result.bilinear_trace) > 0.0)
    # beta_t = decay * beta_(t-1) + 1 makes the trace distinct from an
    # instantaneous product after the first internal tick.
    assert not np.allclose(result.bilinear_trace[1], result.bilinear_trace[0])


def test_closed_form_fit_weights_tasks_equally_not_candidates() -> None:
    training = _training()
    model = HierarchicalCandidateController(
        feature_dim=2,
        fast_dim=7,
        slow_dim=4,
        control_dim=2,
        seed=2,
    )
    receipt = model.fit(training)
    sums = {
        task_id: float(np.sum(weights))
        for task_id, weights in model.task_sample_weights_.items()
    }
    assert receipt.task_balanced
    assert sums == pytest.approx({"train-small": 0.5, "train-large": 0.5})


def test_compute_budget_fails_closed_instead_of_dropping_candidates() -> None:
    model = HierarchicalCandidateController(
        feature_dim=2,
        fast_dim=7,
        slow_dim=4,
        control_dim=2,
    )
    model.fit(_training())
    public = _training()[1].public
    with pytest.raises(ValueError, match="refusing partial selection"):
        model.solve(
            public,
            ComputeBudget(max_candidate_evaluations=4, max_internal_steps=20),
        )
    truncated = model.solve(
        public,
        ComputeBudget(max_candidate_evaluations=5, max_internal_steps=1),
    )
    assert truncated.receipt.exhausted
    assert truncated.receipt.within_budget


def test_gru_is_explicit_bptt_baseline_and_main_model_has_no_torch_dependency() -> None:
    training = _training()
    baseline = SmallGRUBPTTBaseline(
        feature_dim=2,
        hidden_dim=5,
        epochs=2,
        seed=11,
    )
    receipt = baseline.fit(training)
    assert baseline.used_bptt and receipt.used_bptt
    assert receipt.task_balanced
    assert baseline.task_loss_weights_ == pytest.approx(
        {"train-small": 0.5, "train-large": 0.5}
    )
    output = baseline.solve(training[0].public)
    assert output.trace.shape == (2, 5)
    source = inspect.getsource(HierarchicalCandidateController)
    assert "torch" not in source
    assert "backward(" not in source
