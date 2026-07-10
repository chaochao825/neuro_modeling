"""End-to-end train-only wrapper for raw observation-space folds."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Sequence

import numpy as np

from .lds import Family, LDSScores, SharedBasisLDS
from .preprocessing import BasisControl, TrainOnlyPreprocessor
from .splits import TimeSegment


@dataclass
class SharedDynamicsPipeline:
    family: Family
    latent_dim: int
    max_units: int | None = None
    basis_control: BasisControl = "aligned"
    random_state: int = 0
    ridge: float = 1e-4
    variance_floor: float = 1e-5

    def fit(self, train_segments: Sequence[TimeSegment]) -> "SharedDynamicsPipeline":
        self.preprocessor_ = TrainOnlyPreprocessor(max_units=self.max_units).fit(
            train_segments
        )
        transformed = self.preprocessor_.transform_segments(train_segments)
        self.model_ = SharedBasisLDS(
            self.family,
            self.latent_dim,
            basis_control=self.basis_control,
            random_state=self.random_state,
            ridge=self.ridge,
            variance_floor=self.variance_floor,
        ).fit(transformed)
        return self

    def score(self, test_segments: Sequence[TimeSegment]) -> LDSScores:
        if not hasattr(self, "model_"):
            raise RuntimeError("pipeline must be fitted first")
        standardized = self.model_.score(
            self.preprocessor_.transform_segments(test_segments)
        )
        # y=(x-mean)/scale gives log p_X = log p_Y - sum(log(scale))
        # per observation. This is the density on the selected original units;
        # non-selected neurons are outside the modeled observation vector.
        log_jacobian = standardized.n_observations * float(
            np.log(self.preprocessor_.scale_).sum()
        )
        original_log_likelihood = (
            standardized.standardized_marginal_log_likelihood - log_jacobian
        )
        original_nll = -original_log_likelihood / (
            standardized.n_observations * self.model_.n_features_
        )
        return replace(
            standardized,
            marginal_log_likelihood=float(original_log_likelihood),
            nll_per_scalar=float(original_nll),
            likelihood_coordinate="original_selected_units",
        )
