"""Feasibility-aware adapter for the preregistered Exp29 source panel.

The selector is still fitted once on Exp26 seeds 0--29.  Exp29 seeds 60--89
are evaluation-only.  All registered heldout cells enter inference.  Selecting
an infeasible active actuator deterministically receives the exact frozen
utility for that cell; no row is removed and no utility is statistically
imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.exp29_confirmatory_source_panel import (
    IMPLEMENTATION_HASH_SCHEME,
    EXPECTED_GENERATORS,
    EXPECTED_MODES,
    EXPECTED_PANEL_ROWS,
    EXPECTED_SEEDS,
    PROFILE,
    exp29_implementation_sha256,
)
from scripts.package_exp29_confirmatory_source_panel import SourcePanelPackage
from src.data.actuator_selector_dataset import (
    CANDIDATE_MODES,
    NORMALIZED_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    FrozenSelectorMetaTrainingSet,
    SelectorGeneratorSpec,
    _feature_row,
)


FROZEN_MODE = "frozen"
ORACLE_MODES = (FROZEN_MODE, *CANDIDATE_MODES)


def _readonly(value: object, *, dtype: Any, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_flag(value: object, expected: bool) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value) is expected


@dataclass(frozen=True)
class ConfirmatorySelectorSource:
    package_receipt_sha256: str
    package_receipt_file_sha256: str
    package_conclusion_file_sha256: str
    raw_metrics_sha256: str
    source_config_sha256: str
    source_manifest_sha256: str
    implementation_contract_sha256: str
    statistics_unit: str
    seeds: np.ndarray
    generator_ids: tuple[str, ...]
    generator_splits: tuple[str, ...]
    generator_schema: tuple[SelectorGeneratorSpec, ...]
    raw_features: np.ndarray
    candidate_feasible: np.ndarray
    validation_deployment_utilities: np.ndarray
    test_deployment_utilities: np.ndarray
    frozen_validation_utilities: np.ndarray
    frozen_test_utilities: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "package_receipt_sha256",
            "package_receipt_file_sha256",
            "package_conclusion_file_sha256",
            "raw_metrics_sha256",
            "source_config_sha256",
            "source_manifest_sha256",
            "implementation_contract_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.statistics_unit != "seed":
            raise ValueError("confirmatory statistics unit must be seed")
        seeds = _readonly(self.seeds, dtype=np.int64, ndim=1, name="seeds")
        n_samples = seeds.shape[0]
        if tuple(int(value) for value in np.unique(seeds)) != EXPECTED_SEEDS:
            raise ValueError("confirmatory source seeds must be exactly 60--89")
        if n_samples != len(EXPECTED_SEEDS) * EXPECTED_GENERATORS:
            raise ValueError("confirmatory source must contain 30*88 samples")
        ids = tuple(self.generator_ids)
        splits = tuple(self.generator_splits)
        if len(ids) != n_samples or len(splits) != n_samples:
            raise ValueError("confirmatory sample metadata has an invalid length")
        if set(splits) != {"discovery", "heldout"}:
            raise ValueError("confirmatory source must retain both generator splits")
        features = _readonly(
            self.raw_features,
            dtype=np.float64,
            ndim=2,
            name="raw_features",
        )
        feasible = _readonly(
            self.candidate_feasible,
            dtype=bool,
            ndim=2,
            name="candidate_feasible",
        )
        validation = _readonly(
            self.validation_deployment_utilities,
            dtype=np.float64,
            ndim=2,
            name="validation_deployment_utilities",
        )
        test = _readonly(
            self.test_deployment_utilities,
            dtype=np.float64,
            ndim=2,
            name="test_deployment_utilities",
        )
        frozen_validation = _readonly(
            self.frozen_validation_utilities,
            dtype=np.float64,
            ndim=1,
            name="frozen_validation_utilities",
        )
        frozen_test = _readonly(
            self.frozen_test_utilities,
            dtype=np.float64,
            ndim=1,
            name="frozen_test_utilities",
        )
        if features.shape != (n_samples, len(RAW_FEATURE_NAMES)):
            raise ValueError("confirmatory raw feature shape is invalid")
        expected_candidate_shape = (n_samples, len(CANDIDATE_MODES))
        if (
            feasible.shape != expected_candidate_shape
            or validation.shape != expected_candidate_shape
            or test.shape != expected_candidate_shape
            or frozen_validation.shape != (n_samples,)
            or frozen_test.shape != (n_samples,)
        ):
            raise ValueError("confirmatory utility/feasibility shape is invalid")
        if (
            np.any((validation < 0.0) | (validation > 1.0))
            or np.any((test < 0.0) | (test > 1.0))
            or np.any((frozen_validation < 0.0) | (frozen_validation > 1.0))
            or np.any((frozen_test < 0.0) | (frozen_test > 1.0))
        ):
            raise ValueError("confirmatory balanced accuracy must lie in [0, 1]")
        if not np.array_equal(
            validation[~feasible],
            np.broadcast_to(frozen_validation[:, None], feasible.shape)[~feasible],
        ) or not np.array_equal(
            test[~feasible],
            np.broadcast_to(frozen_test[:, None], feasible.shape)[~feasible],
        ):
            raise ValueError("infeasible deployment utility must equal frozen utility")
        schema = tuple(self.generator_schema)
        if (
            len(schema) != EXPECTED_GENERATORS
            or len({item.generator_id for item in schema}) != EXPECTED_GENERATORS
            or any(not isinstance(item, SelectorGeneratorSpec) for item in schema)
        ):
            raise ValueError("confirmatory generator schema is invalid")
        schema_ids = {item.generator_id for item in schema}
        for seed in EXPECTED_SEEDS:
            seed_ids = {ids[index] for index in np.flatnonzero(seeds == seed)}
            if seed_ids != schema_ids:
                raise ValueError("confirmatory generator panel differs across seeds")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "generator_ids", ids)
        object.__setattr__(self, "generator_splits", splits)
        object.__setattr__(self, "generator_schema", schema)
        object.__setattr__(self, "raw_features", features)
        object.__setattr__(self, "candidate_feasible", feasible)
        object.__setattr__(self, "validation_deployment_utilities", validation)
        object.__setattr__(self, "test_deployment_utilities", test)
        object.__setattr__(self, "frozen_validation_utilities", frozen_validation)
        object.__setattr__(self, "frozen_test_utilities", frozen_test)

    def infeasible_rate_by_seed_family(self) -> dict[int, dict[str, float]]:
        result: dict[int, dict[str, float]] = {}
        for seed in EXPECTED_SEEDS:
            mask = self.seeds == seed
            result[seed] = {
                mode: float(np.mean(~self.candidate_feasible[mask, index]))
                for index, mode in enumerate(CANDIDATE_MODES)
            }
        return result


@dataclass(frozen=True)
class ConfirmatorySelectorFold:
    meta_training: FrozenSelectorMetaTrainingSet
    test_seed: int
    source_receipt_sha256: str
    source_raw_metrics_sha256: str
    test_generator_ids: tuple[str, ...]
    test_raw_features: np.ndarray
    test_normalized_features: np.ndarray
    candidate_feasible: np.ndarray
    validation_deployment_utilities: np.ndarray
    test_deployment_utilities: np.ndarray
    frozen_validation_utilities: np.ndarray
    frozen_test_utilities: np.ndarray
    test_unseen_composition: np.ndarray

    def __post_init__(self) -> None:
        if self.test_seed not in EXPECTED_SEEDS:
            raise ValueError("confirmatory fold seed must be in 60--89")
        n_samples = len(self.test_generator_ids)
        raw = _readonly(
            self.test_raw_features, dtype=np.float64, ndim=2, name="test_raw_features"
        )
        normalized = _readonly(
            self.test_normalized_features,
            dtype=np.float64,
            ndim=2,
            name="test_normalized_features",
        )
        feasible = _readonly(
            self.candidate_feasible,
            dtype=bool,
            ndim=2,
            name="candidate_feasible",
        )
        validation = _readonly(
            self.validation_deployment_utilities,
            dtype=np.float64,
            ndim=2,
            name="validation_deployment_utilities",
        )
        test = _readonly(
            self.test_deployment_utilities,
            dtype=np.float64,
            ndim=2,
            name="test_deployment_utilities",
        )
        frozen_validation = _readonly(
            self.frozen_validation_utilities,
            dtype=np.float64,
            ndim=1,
            name="frozen_validation_utilities",
        )
        frozen_test = _readonly(
            self.frozen_test_utilities,
            dtype=np.float64,
            ndim=1,
            name="frozen_test_utilities",
        )
        unseen = _readonly(
            self.test_unseen_composition,
            dtype=bool,
            ndim=1,
            name="test_unseen_composition",
        )
        if raw.shape != (n_samples, len(RAW_FEATURE_NAMES)):
            raise ValueError("confirmatory fold raw features have invalid shape")
        if normalized.shape != (n_samples, len(NORMALIZED_FEATURE_NAMES)):
            raise ValueError("confirmatory fold normalized features have invalid shape")
        candidate_shape = (n_samples, len(CANDIDATE_MODES))
        if (
            feasible.shape != candidate_shape
            or validation.shape != candidate_shape
            or test.shape != candidate_shape
            or frozen_validation.shape != (n_samples,)
            or frozen_test.shape != (n_samples,)
            or unseen.shape != (n_samples,)
        ):
            raise ValueError("confirmatory fold arrays have invalid shapes")
        expected = self.meta_training.normalizer.transform(raw)
        if not np.array_equal(normalized, expected):
            raise ValueError("confirmatory normalization was not meta-train fitted")
        object.__setattr__(self, "test_raw_features", raw)
        object.__setattr__(self, "test_normalized_features", normalized)
        object.__setattr__(self, "candidate_feasible", feasible)
        object.__setattr__(self, "validation_deployment_utilities", validation)
        object.__setattr__(self, "test_deployment_utilities", test)
        object.__setattr__(self, "frozen_validation_utilities", frozen_validation)
        object.__setattr__(self, "frozen_test_utilities", frozen_test)
        object.__setattr__(self, "test_unseen_composition", unseen)

    @property
    def matched_budget_support_mask(self) -> np.ndarray:
        """Only feasible active rows may support a matched-budget claim."""

        return self.candidate_feasible

    def deployment_utility(
        self, candidate_indices: object, *, split: str = "test"
    ) -> np.ndarray:
        """Score candidate choices unconditionally with frozen fallback."""

        indices = np.asarray(candidate_indices)
        if indices.shape != (
            len(self.test_generator_ids),
        ) or indices.dtype.kind not in {
            "i",
            "u",
        }:
            raise ValueError("candidate_indices must be one integer per test cell")
        if np.any((indices < 0) | (indices >= len(CANDIDATE_MODES))):
            raise ValueError("candidate_indices are outside the registered modes")
        utilities = (
            self.test_deployment_utilities
            if split == "test"
            else self.validation_deployment_utilities
            if split == "validation"
            else None
        )
        if utilities is None:
            raise ValueError("split must be validation or test")
        rows = np.arange(indices.shape[0])
        result = np.array(utilities[rows, indices.astype(np.int64)], copy=True)
        result.setflags(write=False)
        return result

    def oracle(self, *, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
        """Choose only among feasible active modes and the frozen actuator."""

        candidate = (
            self.test_deployment_utilities
            if split == "test"
            else self.validation_deployment_utilities
            if split == "validation"
            else None
        )
        frozen = (
            self.frozen_test_utilities
            if split == "test"
            else self.frozen_validation_utilities
            if split == "validation"
            else None
        )
        if candidate is None or frozen is None:
            raise ValueError("split must be validation or test")
        active = np.where(self.candidate_feasible, candidate, -np.inf)
        choice_set = np.column_stack([frozen, active])
        indices = np.argmax(choice_set, axis=1).astype(np.int64)
        utilities = choice_set[np.arange(indices.size), indices]
        indices.setflags(write=False)
        utilities.setflags(write=False)
        return indices, utilities


def confirmatory_source_from_package(
    package: SourcePanelPackage,
) -> ConfirmatorySelectorSource:
    """Convert an already validated package without trusting declarations alone."""

    if not isinstance(package, SourcePanelPackage):
        raise TypeError("package must be a validated SourcePanelPackage")
    receipt = package.receipt
    run_provenance = receipt.get("run_provenance")
    observed_implementation = exp29_implementation_sha256()
    payload = {
        key: value for key, value in receipt.items() if key != "receipt_payload_sha256"
    }
    if (
        package.receipt_payload_sha256 != _sha256(payload)
        or package.receipt_payload_sha256 != receipt.get("receipt_payload_sha256")
        or package.raw_metrics_sha256 != _jsonl_sha256(package.rows)
        or package.raw_metrics_sha256 != receipt.get("raw_metrics_sha256")
        or receipt.get("profile") != PROFILE
        or receipt.get("statistics_unit") != "seed"
        or not isinstance(run_provenance, Mapping)
        or run_provenance.get("exp29_implementation_hash_scheme")
        != IMPLEMENTATION_HASH_SCHEME
        or run_provenance.get("exp29_implementation_sha256") != observed_implementation
        or run_provenance.get("exp29_implementation_contract_sha256")
        != _sha256(observed_implementation)
        or receipt.get("coverage", {}).get("source_panel_valid") is not True
        or int(receipt.get("raw_metrics_row_count", -1)) != EXPECTED_PANEL_ROWS
        or len(package.rows) != EXPECTED_PANEL_ROWS
    ):
        raise ValueError("confirmatory package integrity is invalid")
    frame = pd.DataFrame([dict(row) for row in package.rows])
    required = {
        "seed",
        "generator_id",
        "generator_split",
        "actuator_mode",
        "alpha",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
        "chi",
        "state_demand",
        "input_demand",
        "validation_balanced_accuracy",
        "test_balanced_accuracy",
        "status",
        "actuator_feasible",
        "deployment_mode",
        "deployment_fallback_applied",
        "matched_budget_support_eligible",
        "manifest_hash",
        "statistics_unit",
    }
    if required - set(frame):
        raise ValueError("confirmatory rows lack adapter fields")
    if tuple(sorted(int(value) for value in frame["seed"].unique())) != EXPECTED_SEEDS:
        raise ValueError("confirmatory package seeds are not exactly 60--89")
    if not frame["statistics_unit"].eq("seed").all():
        raise ValueError("confirmatory statistics unit must be seed")
    duplicate = frame.duplicated(["seed", "generator_id", "actuator_mode"])
    if bool(duplicate.any()):
        raise ValueError("confirmatory package contains duplicate cells")

    sample_rows: list[
        tuple[int, str, pd.Series, np.ndarray, np.ndarray, np.ndarray, float, float]
    ] = []
    schema_by_id: dict[str, SelectorGeneratorSpec] = {}
    grouped = frame.groupby(["seed", "generator_id"], sort=True, observed=True)
    for (seed_value, generator_id_value), group in grouped:
        if set(group["actuator_mode"].astype(str)) != set(EXPECTED_MODES):
            raise ValueError("confirmatory candidate family is incomplete")
        indexed = group.set_index(group["actuator_mode"].astype(str), drop=False)
        frozen = indexed.loc[FROZEN_MODE]
        if (
            frozen["status"] != "complete"
            or not _strict_flag(frozen["actuator_feasible"], True)
            or not _strict_flag(frozen["deployment_fallback_applied"], False)
            or not _strict_flag(frozen["matched_budget_support_eligible"], False)
            or frozen["deployment_mode"] != FROZEN_MODE
        ):
            raise ValueError("confirmatory frozen actuator is not deployable")
        candidate_rows = [indexed.loc[mode] for mode in CANDIDATE_MODES]
        feasible = np.asarray(
            [row["status"] == "complete" for row in candidate_rows], dtype=bool
        )
        for flag, row in zip(feasible, candidate_rows, strict=True):
            if not _strict_flag(row["actuator_feasible"], bool(flag)):
                raise ValueError("confirmatory active status/feasibility disagrees")
            if not _strict_flag(row["matched_budget_support_eligible"], bool(flag)):
                raise ValueError("matched-budget eligibility includes infeasible row")
            if not _strict_flag(row["deployment_fallback_applied"], not bool(flag)):
                raise ValueError(
                    "confirmatory fallback flag disagrees with feasibility"
                )
            expected_mode = str(row["actuator_mode"]) if flag else FROZEN_MODE
            if row["deployment_mode"] != expected_mode:
                raise ValueError(
                    "confirmatory deployment mode violates fallback policy"
                )
        validation = np.asarray(
            [float(row["validation_balanced_accuracy"]) for row in candidate_rows]
        )
        test = np.asarray(
            [float(row["test_balanced_accuracy"]) for row in candidate_rows]
        )
        frozen_validation = float(frozen["validation_balanced_accuracy"])
        frozen_test = float(frozen["test_balanced_accuracy"])
        if not np.array_equal(
            validation[~feasible], np.full(np.sum(~feasible), frozen_validation)
        ) or not np.array_equal(
            test[~feasible], np.full(np.sum(~feasible), frozen_test)
        ):
            raise ValueError("infeasible candidate does not carry frozen utility")
        generator_id = str(generator_id_value)
        spec = SelectorGeneratorSpec(
            generator_id=generator_id,
            generator_split=str(frozen["generator_split"]),
            alpha=float(frozen["alpha"]),
            transition_rank=int(frozen["transition_rank"]),
            input_rank=int(frozen["input_rank"]),
            delay=float(frozen["delay"]),
            noise_std=float(frozen["noise_std"]),
        )
        if generator_id in schema_by_id and schema_by_id[generator_id] != spec:
            raise ValueError("confirmatory generator schema differs across seeds")
        schema_by_id[generator_id] = spec
        sample_rows.append(
            (
                int(seed_value),
                generator_id,
                frozen,
                feasible,
                validation,
                test,
                frozen_validation,
                frozen_test,
            )
        )
    split_order = {"discovery": 0, "heldout": 1}
    sample_rows.sort(
        key=lambda item: (
            item[0],
            split_order[str(item[2]["generator_split"])],
            item[1],
        )
    )
    schema = tuple(
        sorted(
            schema_by_id.values(),
            key=lambda item: (split_order[item.generator_split], item.generator_id),
        )
    )
    manifest_values = set(frame["manifest_hash"].astype(str))
    if len(manifest_values) != 1:
        raise ValueError("confirmatory package manifest identity is inconsistent")
    return ConfirmatorySelectorSource(
        package_receipt_sha256=package.receipt_payload_sha256,
        package_receipt_file_sha256=package.receipt_file_sha256,
        package_conclusion_file_sha256=package.conclusion_file_sha256,
        raw_metrics_sha256=package.raw_metrics_sha256,
        source_config_sha256=str(receipt["registered_config_sha256"]),
        source_manifest_sha256=next(iter(manifest_values)),
        implementation_contract_sha256=_sha256(observed_implementation),
        statistics_unit="seed",
        seeds=np.asarray([item[0] for item in sample_rows]),
        generator_ids=tuple(item[1] for item in sample_rows),
        generator_splits=tuple(str(item[2]["generator_split"]) for item in sample_rows),
        generator_schema=schema,
        raw_features=np.vstack([_feature_row(item[2]) for item in sample_rows]),
        candidate_feasible=np.vstack([item[3] for item in sample_rows]),
        validation_deployment_utilities=np.vstack([item[4] for item in sample_rows]),
        test_deployment_utilities=np.vstack([item[5] for item in sample_rows]),
        frozen_validation_utilities=np.asarray([item[6] for item in sample_rows]),
        frozen_test_utilities=np.asarray([item[7] for item in sample_rows]),
    )


def build_confirmatory_selector_folds(
    meta_training: FrozenSelectorMetaTrainingSet,
    source: ConfirmatorySelectorSource,
) -> tuple[ConfirmatorySelectorFold, ...]:
    """Reuse one frozen seeds-0--29 fit for every seed-60--89 test fold."""

    if not isinstance(meta_training, FrozenSelectorMetaTrainingSet):
        raise TypeError("meta_training must be FrozenSelectorMetaTrainingSet")
    if not isinstance(source, ConfirmatorySelectorSource):
        raise TypeError("source must be ConfirmatorySelectorSource")
    if source.source_manifest_sha256 != meta_training.source_manifest_sha256:
        raise ValueError("meta and confirmatory manifest identities differ")
    if source.generator_schema != meta_training.generator_schema:
        raise ValueError("meta and confirmatory generator schemas differ")
    discovery_compositions = set(meta_training.discovery_compositions)
    spec_by_id = {item.generator_id: item for item in source.generator_schema}
    splits = np.asarray(source.generator_splits, dtype=object)
    folds: list[ConfirmatorySelectorFold] = []
    for seed in EXPECTED_SEEDS:
        indices = np.flatnonzero((source.seeds == seed) & (splits == "heldout"))
        if indices.size == 0:
            raise ValueError("confirmatory heldout fold is empty")
        unseen = np.asarray(
            [
                spec_by_id[source.generator_ids[row]].composition
                not in discovery_compositions
                for row in indices
            ],
            dtype=bool,
        )
        raw = source.raw_features[indices]
        folds.append(
            ConfirmatorySelectorFold(
                meta_training=meta_training,
                test_seed=seed,
                source_receipt_sha256=source.package_receipt_sha256,
                source_raw_metrics_sha256=source.raw_metrics_sha256,
                test_generator_ids=tuple(
                    source.generator_ids[index] for index in indices
                ),
                test_raw_features=raw,
                test_normalized_features=meta_training.normalizer.transform(raw),
                candidate_feasible=source.candidate_feasible[indices],
                validation_deployment_utilities=(
                    source.validation_deployment_utilities[indices]
                ),
                test_deployment_utilities=source.test_deployment_utilities[indices],
                frozen_validation_utilities=source.frozen_validation_utilities[indices],
                frozen_test_utilities=source.frozen_test_utilities[indices],
                test_unseen_composition=unseen,
            )
        )
    if len({id(fold.meta_training) for fold in folds}) != 1:
        raise RuntimeError("confirmatory folds did not share one frozen meta fit")
    return tuple(folds)


__all__ = [
    "FROZEN_MODE",
    "ORACLE_MODES",
    "ConfirmatorySelectorSource",
    "ConfirmatorySelectorFold",
    "confirmatory_source_from_package",
    "build_confirmatory_selector_folds",
]
