"""Leakage-safe shared-basis dynamics for the bundled neural datasets.

The package is deliberately independent from the two historical workstreams in
this repository.  It exposes a small, auditable API for block cross-validation,
train-only preprocessing, and scalable linear-Gaussian state-space models.
"""

from .data import (
    BUNDLED_DATASETS,
    load_activity_mat,
    load_bundled_datasets,
    load_visual_context_pair,
)
from .lds import (
    LDSScores,
    SharedBasisLDS,
    effective_rank,
    woodbury_gaussian_logpdf,
)
from .pipeline import SharedDynamicsPipeline
from .preprocessing import TrainOnlyPreprocessor, fit_controlled_basis
from .splits import (
    BlockFold,
    TimeSegment,
    build_transitions,
    purged_contiguous_folds,
)

__all__ = [
    "BUNDLED_DATASETS",
    "BlockFold",
    "LDSScores",
    "SharedBasisLDS",
    "SharedDynamicsPipeline",
    "TimeSegment",
    "TrainOnlyPreprocessor",
    "build_transitions",
    "effective_rank",
    "fit_controlled_basis",
    "load_activity_mat",
    "load_bundled_datasets",
    "load_visual_context_pair",
    "purged_contiguous_folds",
    "woodbury_gaussian_logpdf",
]
