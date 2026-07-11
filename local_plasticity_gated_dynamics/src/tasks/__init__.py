"""Synthetic task generators."""

from src.tasks.hidden_context import (
    EvaluationTruth,
    GateObservationBatch,
    HiddenContextConfig,
    HiddenContextDataset,
    HiddenContextRandomTape,
    TaskLearningBatch,
    generate_hidden_context,
    make_hidden_context_random_tape,
)

__all__ = [
    "EvaluationTruth",
    "GateObservationBatch",
    "HiddenContextConfig",
    "HiddenContextDataset",
    "HiddenContextRandomTape",
    "TaskLearningBatch",
    "generate_hidden_context",
    "make_hidden_context_random_tape",
]
