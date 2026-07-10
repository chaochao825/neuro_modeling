import numpy as np
import pandas as pd
import pytest

from src.analysis.model_comparison import (
    build_model_comparison_table,
    paired_bootstrap,
    paired_wilcoxon,
    validate_replicate_unit,
)


def test_paired_bootstrap_aggregates_rows_within_seed() -> None:
    candidate = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    reference = np.zeros(6)
    unit_ids = [0, 0, 1, 1, 2, 2]

    first = paired_bootstrap(
        candidate,
        reference,
        unit_ids=unit_ids,
        replicate_unit="seed",
        n_resamples=500,
        confidence=0.9,
        seed=7,
    )
    second = paired_bootstrap(
        candidate,
        reference,
        unit_ids=unit_ids,
        replicate_unit="seed",
        n_resamples=500,
        confidence=0.9,
        seed=7,
    )

    assert first.n_units == 3
    assert first.estimate == pytest.approx(2.0)
    assert np.array_equal(first.unit_differences, [1.0, 2.0, 3.0])
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_neurons_are_never_accepted_as_replicates() -> None:
    with pytest.raises(ValueError, match="neurons"):
        validate_replicate_unit("neuron")
    with pytest.raises(ValueError, match="neurons"):
        paired_bootstrap(
            [1.0, 2.0],
            [0.0, 0.0],
            unit_ids=[10, 11],
            replicate_unit="neuron",
            n_resamples=100,
            seed=0,
        )


def test_all_zero_wilcoxon_difference_is_well_defined() -> None:
    result = paired_wilcoxon(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        unit_ids=["a", "b", "c"],
        replicate_unit="animal",
    )
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.n_nonzero == 0


def _comparison_frame() -> pd.DataFrame:
    rows = []
    offsets = {"common": 0.0, "shared": 1.0, "full": 0.5}
    parameters = {"common": 10, "shared": 20, "full": 30}
    for seed in range(4):
        for model, offset in offsets.items():
            for fold in range(2):
                rows.append(
                    {
                        "seed_id": seed,
                        "fold": fold,
                        "model": model,
                        "heldout_likelihood": seed + 0.1 * fold + offset,
                        "parameter_count": parameters[model],
                    }
                )
    return pd.DataFrame(rows)


def test_common_shared_full_table_is_paired_at_seed_level() -> None:
    table = build_model_comparison_table(
        _comparison_frame(),
        metric_columns="heldout_likelihood",
        unit_column="seed_id",
        replicate_unit="seed",
        pair_columns="fold",
        n_resamples=500,
        confidence=0.9,
        seed=11,
    )

    assert list(table["model"]) == ["common", "shared", "full"]
    assert np.all(table["n_units"] == 4)
    shared = table.loc[table["model"] == "shared"].iloc[0]
    assert shared["difference_vs_baseline"] == pytest.approx(1.0)
    assert shared["improvement_vs_baseline"] == pytest.approx(1.0)
    assert shared["parameter_count_mean"] == pytest.approx(20.0)
    assert 0.0 <= shared["wilcoxon_p_holm"] <= 1.0
    assert shared["holm_family"] == "metric=heldout_likelihood;models_vs=common"


def test_model_table_rejects_incomplete_panel_and_neuron_column() -> None:
    frame = _comparison_frame()
    incomplete = frame.loc[
        ~((frame["seed_id"] == 3) & (frame["model"] == "full"))
    ]
    with pytest.raises(ValueError, match="complete paired"):
        build_model_comparison_table(
            incomplete,
            metric_columns="heldout_likelihood",
            unit_column="seed_id",
            replicate_unit="seed",
            pair_columns="fold",
            n_resamples=100,
            seed=0,
        )

    renamed = frame.rename(columns={"seed_id": "neuron_id"})
    with pytest.raises(ValueError, match="neuron"):
        build_model_comparison_table(
            renamed,
            metric_columns="heldout_likelihood",
            unit_column="neuron_id",
            replicate_unit="seed",
            n_resamples=100,
            seed=0,
        )


def test_model_table_rejects_a_single_missing_fold_and_unlabelled_repeats() -> None:
    frame = _comparison_frame()
    missing_fold = frame.loc[
        ~(
            (frame["seed_id"] == 3)
            & (frame["model"] == "full")
            & (frame["fold"] == 1)
        )
    ]
    with pytest.raises(ValueError, match="complete paired fold/block"):
        build_model_comparison_table(
            missing_fold,
            metric_columns="heldout_likelihood",
            unit_column="seed_id",
            replicate_unit="seed",
            pair_columns="fold",
            n_resamples=100,
            seed=0,
        )
    with pytest.raises(ValueError, match="require pair_columns"):
        build_model_comparison_table(
            frame,
            metric_columns="heldout_likelihood",
            unit_column="seed_id",
            replicate_unit="seed",
            n_resamples=100,
            seed=0,
        )


def test_animal_table_aggregates_folds_then_sessions_then_animals() -> None:
    rows = []
    session_values = {
        ("animal-a", "session-a1"): [0.0, 2.0],
        ("animal-a", "session-a2"): [10.0],
        ("animal-b", "session-b1"): [4.0],
    }
    offsets = {"common": 0.0, "shared": 1.0, "full": 0.5}
    for (animal, session), fold_values in session_values.items():
        for fold, base_value in enumerate(fold_values):
            for model, offset in offsets.items():
                rows.append(
                    {
                        "animal_id": animal,
                        "session_id": session,
                        "fold": fold,
                        "model": model,
                        "score": base_value + offset,
                        "parameter_count": 10,
                    }
                )
    table = build_model_comparison_table(
        pd.DataFrame(rows),
        metric_columns="score",
        unit_column="animal_id",
        replicate_unit="animal",
        session_column="session_id",
        pair_columns="fold",
        n_resamples=200,
        seed=5,
    )

    common = table.loc[table["model"] == "common"].iloc[0]
    shared = table.loc[table["model"] == "shared"].iloc[0]
    # animal-a: mean(mean([0, 2]), mean([10])) = 5.5; animal-b = 4.
    assert common["mean"] == pytest.approx((5.5 + 4.0) / 2.0)
    assert shared["difference_vs_baseline"] == pytest.approx(1.0)
