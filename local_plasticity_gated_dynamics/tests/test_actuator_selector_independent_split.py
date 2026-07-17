from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.actuator_selector_dataset import (
    CANDIDATE_MODES,
    LOCKED_EXP26_INDEPENDENT_GENERATOR_COUNT,
    LOCKED_EXP26_INDEPENDENT_ROW_COUNT,
    LOCKED_EXP26_INDEPENDENT_TEST_SEEDS,
    LOCKED_EXP26_META_SEEDS,
    EXPECTED_PANEL_MODES,
    Exp26SelectorSource,
    build_frozen_selector_meta_training,
    build_independent_selector_test_fold,
    build_independent_selector_test_folds,
    exp26_selector_source_from_independent_package,
)


MANIFEST_SHA = "d" * 64


def _source(
    seeds: tuple[int, ...],
    *,
    profile: str,
    conclusion: str,
    raw_sha_character: str,
    config_sha_character: str,
    manifest_sha: str = MANIFEST_SHA,
    independent_discovery_offset: float = 0.0,
    independent_discovery_utility: tuple[float, float, float] | None = None,
    heldout_delay_offset: float = 0.0,
) -> Exp26SelectorSource:
    generators = (
        ("d0", "discovery", 0.0, 1, 1, 0.0, 0.1),
        ("d1", "discovery", 1.0, 2, 1, 4.0, 0.3),
        ("h0", "heldout", 0.0, 1, 1, 2.0, 0.6),
        ("h1", "heldout", 0.5, 4, 2, 4.0, 0.3),
    )
    seed_values: list[int] = []
    generator_ids: list[str] = []
    generator_splits: list[str] = []
    alpha: list[float] = []
    transition_rank: list[int] = []
    input_rank: list[int] = []
    delay: list[float] = []
    noise_std: list[float] = []
    raw_features: list[list[float]] = []
    validation_utilities: list[list[float]] = []
    test_utilities: list[list[float]] = []
    for seed in seeds:
        for generator_index, (
            generator_id,
            split,
            generator_alpha,
            rank_a,
            rank_b,
            generator_delay,
            noise,
        ) in enumerate(generators):
            seed_values.append(seed)
            generator_ids.append(generator_id)
            generator_splits.append(split)
            alpha.append(generator_alpha)
            transition_rank.append(rank_a)
            input_rank.append(rank_b)
            delay.append(
                generator_delay
                + (heldout_delay_offset if generator_id == "h1" else 0.0)
            )
            noise_std.append(noise)
            feature_offset = (
                independent_discovery_offset if split == "discovery" else 0.0
            )
            raw_features.append(
                [
                    0.1 + 0.001 * seed + 0.01 * generator_index + feature_offset,
                    -2.0 + 0.01 * seed + feature_offset,
                    -1.0 + 0.02 * generator_index + feature_offset,
                    float(rank_a),
                    float(rank_b),
                    generator_delay / 4.0,
                    -0.5 + noise + feature_offset,
                ]
            )
            if split == "discovery":
                default = (
                    [0.90, 0.60, 0.50]
                    if generator_id == "d0"
                    else [0.80, 0.70, 0.60]
                )
                validation_utilities.append(
                    list(independent_discovery_utility or default)
                )
            else:
                validation_utilities.append([0.05, 0.10, 0.99])
            test_utilities.append(
                [
                    0.55 + 0.001 * seed,
                    0.65 + 0.001 * generator_index,
                    0.75 - 0.001 * seed,
                ]
            )
    return Exp26SelectorSource(
        profile=profile,
        conclusion=conclusion,
        raw_metrics_sha256=raw_sha_character * 64,
        conclusion_sha256="b" * 64,
        config_sha256=config_sha_character * 64,
        manifest_sha256=manifest_sha,
        candidate_modes=CANDIDATE_MODES,
        seeds=np.asarray(seed_values),
        generator_ids=tuple(generator_ids),
        generator_splits=tuple(generator_splits),
        alpha=np.asarray(alpha),
        transition_rank=np.asarray(transition_rank),
        input_rank=np.asarray(input_rank),
        delay=np.asarray(delay),
        noise_std=np.asarray(noise_std),
        raw_features=np.asarray(raw_features),
        validation_utilities=np.asarray(validation_utilities),
        test_utilities=np.asarray(test_utilities),
    )


def _meta_source() -> Exp26SelectorSource:
    return _source(
        LOCKED_EXP26_META_SEEDS,
        profile="formal",
        conclusion="support",
        raw_sha_character="a",
        config_sha_character="c",
    )


def _independent_source(**kwargs: object) -> Exp26SelectorSource:
    return _source(
        LOCKED_EXP26_INDEPENDENT_TEST_SEEDS,
        profile="independent_test",
        conclusion="inconclusive",
        raw_sha_character="1",
        config_sha_character="2",
        **kwargs,
    )


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_from_source(source: Exp26SelectorSource) -> SimpleNamespace:
    rows: list[dict[str, object]] = []
    for seed in LOCKED_EXP26_INDEPENDENT_TEST_SEEDS:
        seed_indices = np.flatnonzero(source.seeds == seed)
        discovery = [
            int(index)
            for index in seed_indices
            if source.generator_splits[int(index)] == "discovery"
        ]
        heldout = [
            int(index)
            for index in seed_indices
            if source.generator_splits[int(index)] == "heldout"
        ]
        assert discovery and heldout
        for generator_index in range(LOCKED_EXP26_INDEPENDENT_GENERATOR_COUNT):
            split = "discovery" if generator_index < 44 else "heldout"
            split_index = generator_index if split == "discovery" else generator_index - 44
            template_indices = discovery if split == "discovery" else heldout
            index = template_indices[split_index % len(template_indices)]
            generator_id = f"{split[0]}{split_index:03d}"
            state_demand = 0.01 * (1 + seed)
            input_demand = 0.02 * (1 + (generator_index % 4))
            chi = state_demand / (state_demand + input_demand)
            validation = dict(
                zip(CANDIDATE_MODES, source.validation_utilities[index])
            )
            test = dict(zip(CANDIDATE_MODES, source.test_utilities[index]))
            for mode in EXPECTED_PANEL_MODES:
                rows.append(
                    {
                        "seed": seed,
                        "generator_id": generator_id,
                        "generator_split": split,
                        "actuator_mode": mode,
                        "chi": chi,
                        "state_demand": state_demand,
                        "input_demand": input_demand,
                        "transition_rank": int(source.transition_rank[index]),
                        "input_rank": int(source.input_rank[index]),
                        "delay": float(source.delay[index]),
                        "noise_std": float(source.noise_std[index]),
                        "alpha": float(source.alpha[index]),
                        "validation_balanced_accuracy": float(
                            validation.get(mode, 0.4 if mode == "frozen" else 0.95)
                        ),
                        "test_balanced_accuracy": float(
                            test.get(mode, 0.4 if mode == "frozen" else 0.95)
                        ),
                        "status": "complete",
                        "_effective_status": "complete",
                        "functional_budget_valid": True,
                        "effective_dynamics_strictly_stable": True,
                        "profile": "independent_test",
                        "manifest_hash": MANIFEST_SHA,
                        "source_exp26_manifest_sha256": MANIFEST_SHA,
                    }
                )
    source_contract = {"source_manifest_sha256": MANIFEST_SHA}
    receipt_payload = {
        "profile": "independent_test",
        "conclusion": "inconclusive",
        "evidence_role": "independent_test_source_only",
        "standalone_inference_performed": False,
        "standalone_inference_permitted": False,
        "registered_config_sha256": "2" * 64,
        "source_contract": source_contract,
        "source_contract_sha256": _canonical_sha(source_contract),
        "raw_metrics_sha256": "1" * 64,
        "raw_metrics_row_count": len(rows),
        "coverage": {
            "source_panel_valid": True,
            "expected_seeds": list(LOCKED_EXP26_INDEPENDENT_TEST_SEEDS),
            "observed_seeds": list(LOCKED_EXP26_INDEPENDENT_TEST_SEEDS),
            "expected_generators_per_seed": 88,
            "expected_modes_per_generator": 5,
            "expected_rows_per_seed": 440,
            "expected_row_count": LOCKED_EXP26_INDEPENDENT_ROW_COUNT,
            "observed_row_count": LOCKED_EXP26_INDEPENDENT_ROW_COUNT,
        },
    }
    receipt = {
        **receipt_payload,
        "receipt_payload_sha256": _canonical_sha(receipt_payload),
    }
    return SimpleNamespace(receipt=receipt, rows=tuple(rows))


def test_frozen_meta_fit_is_global_immutable_and_reused_for_all_test_seeds() -> None:
    meta_source = _meta_source()
    meta = build_frozen_selector_meta_training(meta_source)
    folds = build_independent_selector_test_folds(meta, _independent_source())

    assert tuple(np.unique(meta.train_seeds)) == LOCKED_EXP26_META_SEEDS
    assert meta.train_raw_features.shape == (60, 7)
    assert meta.train_validation_utilities.shape == (60, 3)
    assert meta.normalizer.n_fit_samples == 60
    assert meta.fixed_best_mode == "routing"
    assert meta.fixed_best_index == 0
    assert len(folds) == 30
    assert tuple(fold.test_seed for fold in folds) == (
        LOCKED_EXP26_INDEPENDENT_TEST_SEEDS
    )
    assert all(fold.meta_training is meta for fold in folds)
    assert all(fold.normalizer is meta.normalizer for fold in folds)
    with pytest.raises(ValueError):
        meta.train_validation_utilities[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        meta.fixed_best_mode = "gain"  # type: ignore[misc]


def test_source_panel_package_adapter_builds_the_locked_independent_source() -> None:
    package = _package_from_source(_independent_source())
    source = exp26_selector_source_from_independent_package(package)
    meta = build_frozen_selector_meta_training(_meta_source())

    assert source.profile == "independent_test"
    assert source.conclusion == "inconclusive"
    assert source.unique_seeds == LOCKED_EXP26_INDEPENDENT_TEST_SEEDS
    assert source.manifest_sha256 == meta.source_manifest_sha256
    assert len(set(source.generator_ids)) == 88
    assert sum(
        generator_id == "h000" for generator_id in source.generator_ids
    ) == len(LOCKED_EXP26_INDEPENDENT_TEST_SEEDS)

    tampered_receipt = dict(package.receipt)
    tampered_receipt["standalone_inference_performed"] = True
    with pytest.raises(ValueError, match="source-only"):
        exp26_selector_source_from_independent_package(
            SimpleNamespace(receipt=tampered_receipt, rows=package.rows)
        )


def test_source_panel_adapter_rejects_self_consistent_truncated_generator_panel() -> None:
    package = _package_from_source(_independent_source())
    truncated_rows = tuple(
        row for row in package.rows if row["generator_id"] != "h043"
    )
    payload = {
        key: value
        for key, value in package.receipt.items()
        if key != "receipt_payload_sha256"
    }
    payload["raw_metrics_row_count"] = len(truncated_rows)
    payload["raw_metrics_sha256"] = "e" * 64
    payload["coverage"] = {
        **payload["coverage"],
        "expected_generators_per_seed": 87,
        "expected_rows_per_seed": 435,
        "expected_row_count": len(truncated_rows),
        "observed_row_count": len(truncated_rows),
    }
    receipt = {**payload, "receipt_payload_sha256": _canonical_sha(payload)}
    with pytest.raises(ValueError, match=r"30\*88\*5"):
        exp26_selector_source_from_independent_package(
            SimpleNamespace(receipt=receipt, rows=truncated_rows)
        )


def test_independent_discovery_rows_cannot_change_fit_normalizer_or_fixed_best() -> None:
    meta = build_frozen_selector_meta_training(_meta_source())
    ordinary = _independent_source()
    adversarial = _independent_source(
        independent_discovery_offset=1e9,
        independent_discovery_utility=(0.0, 0.0, 1.0),
    )
    ordinary_fold = build_independent_selector_test_fold(meta, ordinary, 30)
    adversarial_fold = build_independent_selector_test_fold(meta, adversarial, 30)

    assert ordinary_fold.meta_training is adversarial_fold.meta_training is meta
    assert ordinary_fold.fixed_best_mode == adversarial_fold.fixed_best_mode == "routing"
    np.testing.assert_array_equal(
        ordinary_fold.normalizer.mean, adversarial_fold.normalizer.mean
    )
    np.testing.assert_array_equal(
        ordinary_fold.normalizer.scale, adversarial_fold.normalizer.scale
    )
    np.testing.assert_array_equal(
        ordinary_fold.test_raw_features, adversarial_fold.test_raw_features
    )
    np.testing.assert_array_equal(
        ordinary_fold.test_normalized_features,
        adversarial_fold.test_normalized_features,
    )
    np.testing.assert_array_equal(
        ordinary_fold.test_utilities, adversarial_fold.test_utilities
    )
    assert not hasattr(ordinary_fold, "independent_discovery_utilities")


def test_test_fold_uses_only_independent_heldout_test_rows_and_meta_unseen_keys() -> None:
    meta = build_frozen_selector_meta_training(_meta_source())
    source = _independent_source()
    fold = build_independent_selector_test_fold(meta, source, 37)
    source_splits = np.asarray(source.generator_splits, dtype=object)
    expected_indices = np.flatnonzero(
        (source.seeds == 37) & (source_splits == "heldout")
    )

    assert tuple(fold.test_generator_ids) == ("h0", "h1")
    np.testing.assert_array_equal(fold.test_seeds, [37, 37])
    np.testing.assert_array_equal(
        fold.test_utilities, source.test_utilities[expected_indices]
    )
    np.testing.assert_array_equal(fold.test_unseen_composition, [False, True])
    np.testing.assert_array_equal(fold.test_composition_overlap, [True, False])
    assert not fold.test_raw_features.flags.writeable


@pytest.mark.parametrize("defect", ["overlap", "missing_seed", "manifest", "schema"])
def test_independent_split_fails_closed_on_incompatible_panels(defect: str) -> None:
    meta = build_frozen_selector_meta_training(_meta_source())
    if defect == "overlap":
        source = _source(
            LOCKED_EXP26_META_SEEDS,
            profile="independent_test",
            conclusion="inconclusive",
            raw_sha_character="1",
            config_sha_character="2",
        )
        pattern = "disjoint"
    elif defect == "missing_seed":
        source = _source(
            LOCKED_EXP26_INDEPENDENT_TEST_SEEDS[:-1],
            profile="independent_test",
            conclusion="inconclusive",
            raw_sha_character="1",
            config_sha_character="2",
        )
        pattern = "exactly 30--59"
    elif defect == "manifest":
        source = _independent_source(manifest_sha="e" * 64)
        pattern = "manifest"
    else:
        source = _independent_source(heldout_delay_offset=1.0)
        pattern = "generator schemas"
    with pytest.raises(ValueError, match=pattern):
        build_independent_selector_test_fold(meta, source, 30)


def test_meta_training_requires_locked_formal_support_source() -> None:
    wrong_profile = _source(
        LOCKED_EXP26_META_SEEDS,
        profile="smoke",
        conclusion="inconclusive",
        raw_sha_character="a",
        config_sha_character="c",
    )
    with pytest.raises(ValueError, match="formal"):
        build_frozen_selector_meta_training(wrong_profile)

    non_support = _source(
        LOCKED_EXP26_META_SEEDS,
        profile="formal",
        conclusion="inconclusive",
        raw_sha_character="a",
        config_sha_character="c",
    )
    with pytest.raises(ValueError, match="support"):
        build_frozen_selector_meta_training(non_support)
