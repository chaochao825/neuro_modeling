from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.actuator_selector_dataset import (
    CANDIDATE_MODES,
    EXPECTED_PANEL_MODES,
    RAW_FEATURE_NAMES,
    SelectorFeatureNormalizer,
    build_outer_seed_loso,
    build_three_step_cues,
    load_exp26_selector_source,
)


CONFIG_SHA = "a" * 64
MANIFEST_SHA = "b" * 64


def _panel() -> pd.DataFrame:
    generators = (
        ("d0", "discovery", 0.0, 1, 1, 0, 0.0),
        ("d1", "discovery", 1.0, 2, 1, 4, 0.3),
        ("h0", "heldout", 0.0, 1, 1, 2, 0.6),
        ("h1", "heldout", 0.5, 4, 2, 4, 0.3),
    )
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for generator_index, (
            generator_id,
            split,
            alpha,
            rank_a,
            rank_b,
            delay,
            noise,
        ) in enumerate(generators):
            state = 0.01 * (1 + seed + generator_index)
            input_value = 0.02 * (1 + 2 * seed + generator_index)
            chi = state / (state + input_value)
            for mode_index, mode in enumerate(EXPECTED_PANEL_MODES):
                rows.append(
                    {
                        "seed": seed,
                        "generator_id": generator_id,
                        "generator_split": split,
                        "actuator_mode": mode,
                        "chi": chi,
                        "state_demand": state,
                        "input_demand": input_value,
                        "transition_rank": rank_a,
                        "input_rank": rank_b,
                        "delay": delay,
                        "noise_std": noise,
                        "alpha": alpha,
                        "validation_balanced_accuracy": 0.5 + 0.02 * mode_index,
                        "test_balanced_accuracy": 0.45 + 0.03 * mode_index,
                        "status": "complete",
                        "functional_budget_valid": True,
                        "profile": "formal",
                        "manifest_hash": MANIFEST_SHA,
                        "formal_config_sha256": CONFIG_SHA,
                        "registered_manifest_sha256": MANIFEST_SHA,
                    }
                )
    return pd.DataFrame(rows)


def _write_source(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    conclusion_result: str = "support",
    profile: str = "formal",
) -> tuple[Path, Path, str]:
    raw = tmp_path / "raw_metrics.csv.gz"
    frame.to_csv(raw, index=False, compression="gzip")
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    seeds = sorted(int(value) for value in frame["seed"].unique())
    n_generators = int(frame["generator_id"].nunique())
    conclusion = {
        "profile": profile,
        "conclusion": conclusion_result,
        "complete_primary_coverage": True,
        "confirmatory_eligible": profile == "formal" and conclusion_result == "support",
        "dev_only": profile == "smoke",
        "n_seeds": len(seeds),
        "registered_config_sha256": CONFIG_SHA,
        "registered_manifest_sha256": MANIFEST_SHA,
        "raw_metrics_sha256": raw_sha,
        "coverage": {
            "expected_seeds": seeds,
            "raw_row_count": len(frame),
            "primary_row_count": len(seeds) * n_generators * 4,
            "rgl_ceiling_row_count": len(seeds) * n_generators,
        },
    }
    conclusion_path = tmp_path / "conclusion.json"
    conclusion_path.write_text(json.dumps(conclusion), encoding="utf-8")
    return raw, conclusion_path, raw_sha


def test_loader_and_outer_seed_loso_are_complete_immutable_and_leakage_safe(
    tmp_path: Path,
) -> None:
    raw, conclusion, raw_sha = _write_source(tmp_path, _panel())
    source = load_exp26_selector_source(
        raw,
        conclusion,
        expected_profile="formal",
        expected_raw_sha256=raw_sha,
    )

    assert source.candidate_modes == CANDIDATE_MODES
    assert source.unique_seeds == (0, 1)
    assert source.raw_features.shape == (8, 7)
    assert not source.raw_features.flags.writeable
    fold = build_outer_seed_loso(source, outer_seed=0)
    assert tuple(np.unique(fold.train_seeds)) == (1,)
    assert tuple(np.unique(fold.test_seeds)) == (0,)
    assert fold.train_raw_features.shape == (2, len(RAW_FEATURE_NAMES))
    assert fold.test_raw_features.shape == (2, len(RAW_FEATURE_NAMES))
    np.testing.assert_array_equal(fold.test_unseen_composition, [False, True])
    np.testing.assert_array_equal(fold.test_composition_overlap, [True, False])
    assert not fold.train_utilities.flags.writeable
    with pytest.raises(ValueError):
        fold.train_utilities[0, 0] = 99.0


def test_normalization_fits_only_training_discovery_and_bias_occurs_once() -> None:
    train = np.arange(28.0).reshape(4, 7)
    heldout = np.full((2, 7), 1e9)
    normalizer = SelectorFeatureNormalizer.fit(train)
    expected_scale = np.std(train, axis=0)
    np.testing.assert_allclose(normalizer.mean, np.mean(train, axis=0))
    np.testing.assert_allclose(normalizer.scale, expected_scale)
    assert normalizer.n_fit_samples == len(train)
    assert len(normalizer.fit_fingerprint) == 64

    transformed = normalizer.transform(heldout)
    cues = build_three_step_cues(transformed)
    np.testing.assert_array_equal(cues.sum(axis=1), transformed)
    np.testing.assert_array_equal(cues[:, :, -1], [[0.0, 0.0, 1.0]] * 2)
    assert not cues.flags.writeable


def test_loader_fails_closed_on_sha_profile_and_non_support(tmp_path: Path) -> None:
    raw, conclusion, raw_sha = _write_source(tmp_path, _panel())
    with pytest.raises(ValueError, match="hashes disagree"):
        load_exp26_selector_source(
            raw,
            conclusion,
            expected_profile="formal",
            expected_raw_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="profile"):
        load_exp26_selector_source(
            raw,
            conclusion,
            expected_profile="smoke",
            expected_raw_sha256=raw_sha,
            require_support=False,
        )

    opposed_dir = tmp_path / "opposed"
    opposed_dir.mkdir()
    opposed_raw, opposed_conclusion, opposed_sha = _write_source(
        opposed_dir, _panel(), conclusion_result="oppose"
    )
    with pytest.raises(ValueError, match="support"):
        load_exp26_selector_source(
            opposed_raw,
            opposed_conclusion,
            expected_profile="formal",
            expected_raw_sha256=opposed_sha,
        )


@pytest.mark.parametrize("defect", ["missing", "duplicate", "mode", "invariant"])
def test_loader_fails_closed_on_panel_defects(tmp_path: Path, defect: str) -> None:
    frame = _panel()
    if defect == "missing":
        frame = frame.iloc[:-1].copy()
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif defect == "mode":
        frame.loc[frame.index[-1], "actuator_mode"] = "full"
    else:
        mask = (
            (frame["seed"] == 0)
            & (frame["generator_id"] == "d0")
            & (frame["actuator_mode"] == "gain")
        )
        frame.loc[mask, "chi"] += 0.1
    raw, conclusion, raw_sha = _write_source(tmp_path, frame)
    with pytest.raises(ValueError):
        load_exp26_selector_source(
            raw,
            conclusion,
            expected_profile="formal",
            expected_raw_sha256=raw_sha,
        )


def test_explicit_hash_argument_is_required_when_conclusion_does_not_bind_it(
    tmp_path: Path,
) -> None:
    raw, conclusion, raw_sha = _write_source(tmp_path, _panel())
    value = json.loads(conclusion.read_text(encoding="utf-8"))
    value.pop("raw_metrics_sha256")
    conclusion.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        load_exp26_selector_source(raw, conclusion, expected_profile="formal")
    loaded = load_exp26_selector_source(
        raw,
        conclusion,
        expected_profile="formal",
        expected_raw_sha256=raw_sha,
    )
    assert loaded.raw_metrics_sha256 == raw_sha


def test_loader_rejects_fractional_task_ranks_without_truncation(
    tmp_path: Path,
) -> None:
    frame = _panel()
    frame["transition_rank"] = frame["transition_rank"].astype(float)
    frame.loc[frame["generator_id"] == "d0", "transition_rank"] = 1.5
    raw, conclusion, raw_sha = _write_source(tmp_path, frame)

    with pytest.raises(ValueError, match="exact integers"):
        load_exp26_selector_source(
            raw,
            conclusion,
            expected_profile="formal",
            expected_raw_sha256=raw_sha,
        )
