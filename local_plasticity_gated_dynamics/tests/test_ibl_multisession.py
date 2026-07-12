from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ibl_loader import IBLDataError, IBLTrialData
from src.data.ibl_multisession import (
    IBLMultiSessionError,
    TrainOnlyNuisanceResidualizer,
    chronological_outer_inner_splits,
    complete_case_trial_mask,
    full_trial_sensitivity_nuisance_table,
    load_frozen_ibl_neural_cohort,
    past_safe_nuisance_table,
    prepare_ibl_neural_session,
)


def _manifest(path: Path) -> pd.DataFrame:
    rows = []
    for rank in range(24):
        failed = rank in {3, 22}
        rows.append(
            {
                "candidate_rank": rank,
                "eid": f"eid-{rank}",
                "subject": f"mouse-{rank % 6}",
                "pids": "" if failed else f"pid-{rank};pidb-{rank}",
                "bwm_repository_commit": "a" * 40,
                "status": "failed" if failed else "eligible",
                "eligible": not failed,
                "error": "download_failed" if failed else "",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _trial_table(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stimulus": np.linspace(-1, 1, n),
            "choice": np.where(np.arange(n) % 2, 1, -1),
            "reward": np.where(np.arange(n) % 3, 1, -1),
            "reaction_time": np.linspace(0.2, 0.8, n),
            "wheel": np.linspace(1, 2, n),
            "motion_energy_proxy": np.linspace(2, 3, n),
            "probability_left": np.repeat([0.8, 0.2], n // 2),
            "stim_on": np.arange(n),
            "first_movement": np.arange(n) + 0.3,
            "timing_valid": True,
            "block_id": np.repeat(np.arange(5), 2),
        }
    )


def _trial_data() -> IBLTrialData:
    table = _trial_table()
    movement_table = table.copy(deep=True)
    movement_table["motion_energy_proxy"] += 1000.0
    counts = np.arange(10 * 3 * 2, dtype=float).reshape(10, 3, 2)
    return IBLTrialData(
        eid="eid-x",
        animal_id="mouse-x",
        covariates=table,
        view_covariates={
            "stimulus_pre": table,
            "movement_pre": movement_table,
        },
        activity={"stimulus_pre": counts, "movement_pre": counts + 1},
        valid_masks={
            "stimulus_pre": np.ones(10, bool),
            "movement_pre": np.ones(10, bool),
        },
        time_axes={
            "stimulus_pre": np.array([-0.3, -0.2, -0.1]),
            "movement_pre": np.array([-0.3, -0.2, -0.1]),
        },
        unit_ids=np.array(["u0", "u1"]),
        regions=np.array(["MOs", "VISp"]),
    )


def test_cohort_freeze_selects_ranked_sessions_and_preserves_every_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.csv"
    source = _manifest(path)
    cohort = load_frozen_ibl_neural_cohort(path)
    assert len(cohort.entries) == len(source)
    assert len(cohort.selected_entries) == 20
    assert len({entry.animal_id for entry in cohort.selected_entries}) >= 5
    assert len(set(cohort.selected_eids)) == 20
    assert (
        cohort.source_manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    failed = [entry for entry in cohort.entries if entry.source_status == "failed"]
    assert len(failed) == 2
    assert all(not entry.selected for entry in failed)
    assert all(entry.source_error == "download_failed" for entry in failed)
    assert cohort.selected_entries[0].pids == ("pid-0", "pidb-0")


def test_cohort_fails_closed_below_preregistered_threshold(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    _manifest(path).iloc[:10].to_csv(path, index=False)
    with pytest.raises(IBLMultiSessionError, match="cannot satisfy"):
        load_frozen_ibl_neural_cohort(path)


def test_adapter_freezes_counts_and_exposes_explicit_trial_ids() -> None:
    prepared = prepare_ibl_neural_session(
        _trial_data(), current_trial_ids=[f"trial-{index}" for index in range(10)]
    )
    assert set(prepared.count_views) == {"stimulus_pre", "movement_pre"}
    assert prepared.current_trial_ids[0] == "trial-0"
    assert prepared.count_views["stimulus_pre"].shape == (10, 3, 2)
    assert prepared.count_views["stimulus_pre"].dtype == np.int64
    assert prepared.trial_table("movement_pre").loc[
        0, "motion_energy_proxy"
    ] == pytest.approx(
        prepared.trial_table("stimulus_pre").loc[0, "motion_energy_proxy"] + 1000.0
    )
    assert not prepared.count_views["stimulus_pre"].flags.writeable
    assert not prepared.unit_ids.flags.writeable
    with pytest.raises(ValueError):
        prepared.count_views["stimulus_pre"][0, 0, 0] = 99


def test_adapter_rejects_nonbinary_masks_invalid_pre_event_axes_and_duplicate_units() -> (
    None
):
    prepared = prepare_ibl_neural_session(_trial_data())
    masks = {name: value.copy() for name, value in prepared.valid_masks.items()}
    masks["stimulus_pre"] = masks["stimulus_pre"].astype(int)
    masks["stimulus_pre"][0] = 2
    with pytest.raises(IBLMultiSessionError, match="valid mask must be binary"):
        replace(prepared, valid_masks=masks)

    for invalid_axis in (
        np.asarray([-0.3, np.nan, -0.1]),
        np.asarray([-0.3, -0.1, 0.0]),
        np.asarray([-0.3, -0.1, -0.2]),
    ):
        axes = {name: value.copy() for name, value in prepared.time_axes.items()}
        axes["stimulus_pre"] = invalid_axis
        with pytest.raises(IBLMultiSessionError, match="strictly pre-event"):
            replace(prepared, time_axes=axes)

    with pytest.raises(IBLMultiSessionError, match="unique/non-empty"):
        replace(prepared, unit_ids=np.asarray(["duplicate", "duplicate"]))


@pytest.mark.parametrize("invalid", [-1.0, 0.25, np.nan, np.inf, float(2**63)])
def test_adapter_rejects_invalid_count_values(invalid: float) -> None:
    data = _trial_data()
    activity = {name: values.copy() for name, values in data.activity.items()}
    activity["stimulus_pre"][0, 0, 0] = invalid
    if not np.isfinite(invalid):
        # IBLTrialData itself already rejects non-finite activity.
        with pytest.raises(IBLDataError, match="non-finite"):
            IBLTrialData(
                eid=data.eid,
                animal_id=data.animal_id,
                covariates=data.covariates,
                view_covariates=data.view_covariates,
                activity=activity,
                valid_masks=data.valid_masks,
                time_axes=data.time_axes,
                unit_ids=data.unit_ids,
                regions=data.regions,
            )
        return
    altered = IBLTrialData(
        eid=data.eid,
        animal_id=data.animal_id,
        covariates=data.covariates,
        view_covariates=data.view_covariates,
        activity=activity,
        valid_masks=data.valid_masks,
        time_axes=data.time_axes,
        unit_ids=data.unit_ids,
        regions=data.regions,
    )
    with pytest.raises(IBLMultiSessionError, match="counts must"):
        prepare_ibl_neural_session(altered)


def test_outer_inner_splits_are_chronological_whole_blocks_without_switch_resets() -> (
    None
):
    ids = tuple(f"t{index}" for index in range(10))
    blocks = tuple(np.repeat(np.arange(5), 2))
    outer, inner = chronological_outer_inner_splits(ids, blocks)
    assert not (set(outer.train_ids) & set(outer.heldout_ids))
    assert set(outer.train_ids) | set(outer.heldout_ids) == set(ids)
    assert max(outer.train_blocks) < min(outer.heldout_blocks)
    assert set(inner.train_ids) | set(inner.heldout_ids) == set(outer.train_ids)
    assert outer.reset_ids == ("t0",)
    assert inner.reset_ids == ("t0",)


def test_past_safe_and_full_trial_tables_have_explicit_timing_scope() -> None:
    table = _trial_table()
    safe = past_safe_nuisance_table(table, view="stimulus_pre")
    assert "choice" not in safe
    assert "choice_lag1" in safe
    assert "stimulus_current" not in safe
    assert np.isnan(safe.loc[0, "wheel_lag1"])
    assert safe.loc[1, "choice_lag1"] == table.loc[0, "choice"]
    assert safe.attrs["eligible_for_prestim_causal_timing"]
    full = full_trial_sensitivity_nuisance_table(table)
    assert {"choice", "reward", "reaction_time", "wheel"} <= set(full)
    assert not full.attrs["eligible_for_prestim_causal_timing"]


def test_complete_case_mask_is_frozen_and_excludes_lagged_missing_rows() -> None:
    table = past_safe_nuisance_table(_trial_table(), view="stimulus_pre")
    ids = tuple(f"t{index}" for index in range(len(table)))
    valid = np.ones(len(table), dtype=bool)
    valid[5] = False
    mask, receipt = complete_case_trial_mask(table, valid_mask=valid, trial_ids=ids)
    assert not mask.flags.writeable
    assert receipt.kept_trial_ids == tuple(
        trial_id for index, trial_id in enumerate(ids) if index not in {0, 5}
    )
    assert receipt.excluded_trial_ids == ("t0", "t5")
    assert receipt.nuisance_scope == "past_safe_pre_event"
    assert len(receipt.mask_sha256) == 64


def test_residualizer_fit_is_invariant_to_heldout_target_mutation() -> None:
    table = full_trial_sensitivity_nuisance_table(_trial_table())
    activity = np.arange(10 * 2, dtype=float).reshape(10, 2)
    ids = tuple(f"t{index}" for index in range(10))
    train_ids = ids[:6]
    first = TrainOnlyNuisanceResidualizer(table.columns).fit(
        table, activity, trial_ids=ids, train_ids=train_ids
    )
    mutated = activity.copy()
    mutated[6:] += 1e6
    second = TrainOnlyNuisanceResidualizer(table.columns).fit(
        table, mutated, trial_ids=ids, train_ids=train_ids
    )
    assert first == second
    assert first.fit_trial_ids == train_ids
