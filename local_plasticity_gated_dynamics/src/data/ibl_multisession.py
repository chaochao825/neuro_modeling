"""Offline, preregistered data protocol for bounded multi-session IBL analyses.

Evidence scope is deliberately narrow: these helpers freeze cohort identity,
prepare already-loaded neural counts, enforce chronological whole-block splits,
and separate past-safe nuisance regression from full-trial sensitivity analysis.
They never download data or establish a neural-dynamics result by themselves.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.ibl_loader import IBLTrialData, TrialNuisanceResidualizer


class IBLMultiSessionError(ValueError):
    """Raised when exp14 cohort or trial-level contracts are violated."""


_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IBLMultiSessionError(f"{name} must be a non-empty string")
    return value.strip()


def _truth(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "eligible"}


@dataclass(frozen=True, slots=True)
class IBLNeuralCohortEntry:
    candidate_rank: int
    eid: str
    animal_id: str
    pids: tuple[str, ...]
    bwm_repository_commit: str
    source_status: str
    source_eligible: bool
    selected: bool
    disposition: str
    source_error: str


@dataclass(frozen=True, slots=True)
class FrozenIBLNeuralCohort:
    entries: tuple[IBLNeuralCohortEntry, ...]
    source_manifest_sha256: str
    target_sessions: int
    minimum_animals: int
    evidence_scope: str = "offline_preregistered_neural_cohort_only"

    @property
    def selected_entries(self) -> tuple[IBLNeuralCohortEntry, ...]:
        return tuple(entry for entry in self.entries if entry.selected)

    @property
    def excluded_entries(self) -> tuple[IBLNeuralCohortEntry, ...]:
        return tuple(entry for entry in self.entries if not entry.selected)

    @property
    def selected_eids(self) -> tuple[str, ...]:
        return tuple(entry.eid for entry in self.selected_entries)


def load_frozen_ibl_neural_cohort(
    manifest_csv: str | Path,
    *,
    target_sessions: int = 20,
    minimum_animals: int = 5,
) -> FrozenIBLNeuralCohort:
    """Freeze a deterministic neural cohort while retaining every source row."""

    if target_sessions < 20 or minimum_animals < 5:
        raise IBLMultiSessionError("exp14 requires >=20 sessions and >=5 animals")
    path = Path(manifest_csv)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "candidate_rank",
        "eid",
        "subject",
        "pids",
        "bwm_repository_commit",
        "status",
        "eligible",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise IBLMultiSessionError(f"cohort manifest is missing columns: {missing}")
    ranks = pd.to_numeric(frame["candidate_rank"], errors="raise").astype(int)
    if ranks.duplicated().any() or (ranks < 0).any():
        raise IBLMultiSessionError(
            "candidate_rank values must be unique and non-negative"
        )
    ordered = frame.assign(candidate_rank=ranks).sort_values(
        "candidate_rank", kind="mergesort"
    )
    eligible_rows: list[tuple[int, pd.Series]] = []
    seen_eids: set[str] = set()
    for index, row in ordered.iterrows():
        eid = str(row["eid"]).strip()
        pids = tuple(item for item in str(row["pids"]).split(";") if item)
        eligible = _truth(row["eligible"]) and str(row["status"]).lower() == "eligible"
        if eligible and eid and pids and eid not in seen_eids:
            eligible_rows.append((index, row))
            seen_eids.add(eid)
    selected_indices = {index for index, _ in eligible_rows[:target_sessions]}
    animals = {
        str(row["subject"]).strip()
        for index, row in eligible_rows
        if index in selected_indices
    }
    if len(animals) < minimum_animals:
        for index, row in eligible_rows[target_sessions:]:
            animal = str(row["subject"]).strip()
            if animal not in animals:
                selected_indices.add(index)
                animals.add(animal)
            if len(animals) >= minimum_animals:
                break
    if len(selected_indices) < target_sessions or len(animals) < minimum_animals:
        raise IBLMultiSessionError(
            "manifest cannot satisfy preregistered session/animal thresholds"
        )
    entries: list[IBLNeuralCohortEntry] = []
    seen_for_disposition: set[str] = set()
    for index, row in ordered.iterrows():
        eid = str(row["eid"]).strip()
        animal = str(row["subject"]).strip()
        pids = tuple(sorted(item for item in str(row["pids"]).split(";") if item))
        source_eligible = _truth(row["eligible"])
        selected = index in selected_indices
        if selected:
            disposition = "selected_preregistered"
        elif eid in seen_for_disposition:
            disposition = "excluded_duplicate_eid"
        elif not source_eligible or str(row["status"]).lower() != "eligible":
            disposition = "excluded_source_ineligible_or_failed"
        elif not pids:
            disposition = "excluded_no_probe_pid"
        else:
            disposition = "excluded_after_target_reached"
        seen_for_disposition.add(eid)
        entries.append(
            IBLNeuralCohortEntry(
                candidate_rank=int(row["candidate_rank"]),
                eid=_text(eid, name="eid"),
                animal_id=_text(animal, name="subject"),
                pids=pids,
                bwm_repository_commit=str(row["bwm_repository_commit"]).strip(),
                source_status=str(row["status"]),
                source_eligible=source_eligible,
                selected=selected,
                disposition=disposition,
                source_error=str(row.get("error", "")),
            )
        )
    commits = {entry.bwm_repository_commit for entry in entries if entry.selected}
    if len(commits) != 1 or not _GIT_COMMIT.fullmatch(next(iter(commits), "")):
        raise IBLMultiSessionError("selected sessions must bind one 40-hex BWM commit")
    return FrozenIBLNeuralCohort(
        entries=tuple(entries),
        source_manifest_sha256=digest,
        target_sessions=target_sessions,
        minimum_animals=minimum_animals,
    )


@dataclass(frozen=True, slots=True)
class PreparedIBLNeuralSession:
    eid: str
    animal_id: str
    count_views: Mapping[str, np.ndarray]
    valid_masks: Mapping[str, np.ndarray]
    time_axes: Mapping[str, np.ndarray]
    regions: np.ndarray
    unit_ids: np.ndarray
    view_trial_tables: Mapping[str, pd.DataFrame]
    current_trial_ids: np.ndarray
    evidence_scope: str = "pre_event_neural_counts_no_download"

    def __post_init__(self) -> None:
        required = {"stimulus_pre", "movement_pre"}
        if set(self.count_views) != required:
            raise IBLMultiSessionError(
                f"count_views must contain exactly {sorted(required)}"
            )
        items = list(self.current_trial_ids)
        trial_ids = np.empty(len(items), dtype=object)
        trial_ids[:] = items
        try:
            unique_ids = len(set(items)) == len(items)
        except TypeError as error:
            raise IBLMultiSessionError("current_trial_ids must be hashable") from error
        if trial_ids.ndim != 1 or not unique_ids:
            raise IBLMultiSessionError("current_trial_ids must be a unique vector")
        n_trials = len(trial_ids)
        views: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        axes: dict[str, np.ndarray] = {}
        for name in required:
            raw_values = np.asarray(self.count_views[name])
            if raw_values.ndim != 3 or raw_values.shape[0] != n_trials:
                raise IBLMultiSessionError(f"{name} counts must be trial x time x unit")
            if not np.isfinite(raw_values).all() or np.any(raw_values < 0):
                raise IBLMultiSessionError(
                    f"{name} counts must be finite and non-negative"
                )
            # The largest int64 is not exactly representable in float64; use
            # the first out-of-range integer boundary for a safe comparison.
            if (
                np.any(raw_values >= float(2**63))
                or not np.equal(raw_values, np.floor(raw_values)).all()
            ):
                raise IBLMultiSessionError(
                    f"{name} counts must be exact integer values representable as int64"
                )
            values = np.array(raw_values, dtype=np.int64, copy=True)
            raw_mask = np.asarray(self.valid_masks[name])
            if raw_mask.dtype != bool and not np.isin(raw_mask, [0, 1]).all():
                raise IBLMultiSessionError(f"{name} valid mask must be binary")
            mask = np.array(raw_mask, dtype=bool, copy=True)
            axis = np.array(self.time_axes[name], dtype=float, copy=True)
            if mask.shape != (n_trials,) or axis.shape != (values.shape[1],):
                raise IBLMultiSessionError(f"{name} mask/time axis is inconsistent")
            if (
                not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0.0)
                or np.any(axis >= 0.0)
            ):
                raise IBLMultiSessionError(
                    f"{name} time axis must be finite, increasing, and strictly pre-event"
                )
            for array in (values, mask, axis):
                array.setflags(write=False)
            views[name], masks[name], axes[name] = values, mask, axis
        unit_ids = np.array(self.unit_ids, dtype=str, copy=True)
        regions = np.array(self.regions, dtype=str, copy=True)
        if unit_ids.ndim != 1 or regions.shape != unit_ids.shape:
            raise IBLMultiSessionError("unit_ids and regions must be matching vectors")
        if (
            len(set(unit_ids.tolist())) != len(unit_ids)
            or np.any(np.char.str_len(unit_ids) == 0)
            or np.any(np.char.str_len(regions) == 0)
        ):
            raise IBLMultiSessionError(
                "unit_ids must be unique/non-empty and regions non-empty"
            )
        if any(view.shape[2] != len(unit_ids) for view in views.values()):
            raise IBLMultiSessionError("count views and unit metadata disagree")
        if set(self.view_trial_tables) != required:
            raise IBLMultiSessionError(
                f"view_trial_tables must contain exactly {sorted(required)}"
            )
        tables: dict[str, pd.DataFrame] = {}
        for name in required:
            table = self.view_trial_tables[name]
            if not isinstance(table, pd.DataFrame) or len(table) != n_trials:
                raise IBLMultiSessionError(
                    f"{name} trial table must align with current_trial_ids"
                )
            tables[name] = table.copy(deep=True)
        for array in (trial_ids, unit_ids, regions):
            array.setflags(write=False)
        object.__setattr__(self, "eid", _text(self.eid, name="eid"))
        object.__setattr__(self, "animal_id", _text(self.animal_id, name="animal_id"))
        object.__setattr__(self, "count_views", MappingProxyType(views))
        object.__setattr__(self, "valid_masks", MappingProxyType(masks))
        object.__setattr__(self, "time_axes", MappingProxyType(axes))
        object.__setattr__(self, "current_trial_ids", trial_ids)
        object.__setattr__(self, "unit_ids", unit_ids)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "view_trial_tables", MappingProxyType(tables))

    def trial_table(self, view: str) -> pd.DataFrame:
        """Return a defensive copy of the event-specific per-trial table."""

        if view not in self.view_trial_tables:
            raise IBLMultiSessionError(f"unknown neural view {view!r}")
        return self.view_trial_tables[view].copy(deep=True)


def prepare_ibl_neural_session(
    data: IBLTrialData,
    *,
    current_trial_ids: Sequence[object] | None = None,
) -> PreparedIBLNeuralSession:
    """Adapt existing offline :class:`IBLTrialData`; no loader/network changes."""

    if not isinstance(data, IBLTrialData):
        raise TypeError("data must be IBLTrialData")
    ids = (
        np.arange(len(data.covariates))
        if current_trial_ids is None
        else current_trial_ids
    )
    view_tables: dict[str, pd.DataFrame] = {}
    for name in ("stimulus_pre", "movement_pre"):
        table = data.view_covariates[name].copy(deep=True)
        if "motion_energy_proxy" not in table and "pose" in table:
            table = table.rename(columns={"pose": "motion_energy_proxy"})
        view_tables[name] = table
    return PreparedIBLNeuralSession(
        eid=data.eid,
        animal_id=data.animal_id,
        count_views={
            name: data.activity[name] for name in ("stimulus_pre", "movement_pre")
        },
        valid_masks={
            name: data.valid_masks[name] for name in ("stimulus_pre", "movement_pre")
        },
        time_axes={
            name: data.time_axes[name] for name in ("stimulus_pre", "movement_pre")
        },
        regions=data.regions,
        unit_ids=data.unit_ids,
        view_trial_tables=view_tables,
        current_trial_ids=ids,
    )


@dataclass(frozen=True, slots=True)
class ChronologicalBlockSplit:
    train_ids: tuple[object, ...]
    heldout_ids: tuple[object, ...]
    train_blocks: tuple[object, ...]
    heldout_blocks: tuple[object, ...]
    reset_ids: tuple[object, ...]


def chronological_whole_block_split(
    trial_ids: Sequence[object],
    block_ids: Sequence[object],
    *,
    heldout_fraction: float = 0.2,
) -> ChronologicalBlockSplit:
    """Hold out a chronological suffix of complete blocks, never timepoints."""

    ids, blocks = tuple(trial_ids), tuple(block_ids)
    if len(ids) != len(blocks) or len(ids) < 2 or len(set(ids)) != len(ids):
        raise IBLMultiSessionError(
            "trial/block IDs must be aligned unique trial vectors"
        )
    if not 0.0 < heldout_fraction < 1.0:
        raise IBLMultiSessionError("heldout_fraction must lie in (0, 1)")
    runs: list[tuple[object, list[object]]] = []
    for trial_id, block in zip(ids, blocks, strict=True):
        if not runs or block != runs[-1][0]:
            if any(block == previous[0] for previous in runs):
                raise IBLMultiSessionError(
                    "block IDs must form contiguous whole blocks"
                )
            runs.append((block, []))
        runs[-1][1].append(trial_id)
    if len(runs) < 2:
        raise IBLMultiSessionError("at least two blocks are required")
    target = max(1, int(np.ceil(len(ids) * heldout_fraction)))
    cut = len(runs) - 1
    count = len(runs[-1][1])
    while cut > 1 and count < target:
        cut -= 1
        count += len(runs[cut][1])
    train = tuple(item for _, values in runs[:cut] for item in values)
    heldout = tuple(item for _, values in runs[cut:] for item in values)
    return ChronologicalBlockSplit(
        train_ids=train,
        heldout_ids=heldout,
        train_blocks=tuple(block for block, _ in runs[:cut]),
        heldout_blocks=tuple(block for block, _ in runs[cut:]),
        # Context switches are not reset points; only the session start is.
        reset_ids=(ids[0],),
    )


def chronological_outer_inner_splits(
    trial_ids: Sequence[object],
    block_ids: Sequence[object],
    *,
    outer_test_fraction: float = 0.2,
    inner_validation_fraction: float = 0.2,
) -> tuple[ChronologicalBlockSplit, ChronologicalBlockSplit]:
    outer = chronological_whole_block_split(
        trial_ids, block_ids, heldout_fraction=outer_test_fraction
    )
    block_by_id = dict(zip(trial_ids, block_ids, strict=True))
    inner = chronological_whole_block_split(
        outer.train_ids,
        [block_by_id[item] for item in outer.train_ids],
        heldout_fraction=inner_validation_fraction,
    )
    return outer, inner


def past_safe_nuisance_table(
    trial_table: pd.DataFrame, *, view: str = "stimulus_pre"
) -> pd.DataFrame:
    """Build covariates available before the analyzed event on trial *t*."""

    required = {
        "stimulus",
        "choice",
        "reward",
        "reaction_time",
        "wheel",
        "motion_energy_proxy",
    }
    missing = sorted(required - set(trial_table))
    if missing:
        raise IBLMultiSessionError(
            f"trial table is missing nuisance columns: {missing}"
        )
    if view not in {"stimulus_pre", "movement_pre"}:
        raise IBLMultiSessionError("view must be stimulus_pre or movement_pre")
    result = pd.DataFrame(index=trial_table.index)
    result["motion_energy_pre_current"] = pd.to_numeric(
        trial_table["motion_energy_proxy"], errors="coerce"
    )
    for column in ("stimulus", "choice", "reward", "reaction_time", "wheel"):
        result[f"{column}_lag1"] = pd.to_numeric(
            trial_table[column], errors="coerce"
        ).shift(1)
    if view == "movement_pre":
        result["stimulus_current"] = pd.to_numeric(
            trial_table["stimulus"], errors="coerce"
        )
    result.attrs.update(
        nuisance_scope="past_safe_pre_event",
        eligible_for_prestim_causal_timing=True,
        contains_current_choice_reward_rt=False,
        contains_stimulus_to_response_wheel=False,
    )
    return result


def full_trial_sensitivity_nuisance_table(trial_table: pd.DataFrame) -> pd.DataFrame:
    """Current-trial covariates for sensitivity analysis, never causal timing."""

    columns = (
        "stimulus",
        "choice",
        "reward",
        "reaction_time",
        "wheel",
        "motion_energy_proxy",
    )
    missing = [column for column in columns if column not in trial_table]
    if missing:
        raise IBLMultiSessionError(
            f"trial table is missing nuisance columns: {missing}"
        )
    result = trial_table.loc[:, columns].apply(pd.to_numeric, errors="coerce").copy()
    result.attrs.update(
        nuisance_scope="full_trial_sensitivity",
        eligible_for_prestim_causal_timing=False,
        contains_current_choice_reward_rt=True,
        contains_stimulus_to_response_wheel=True,
    )
    return result


@dataclass(frozen=True, slots=True)
class CompleteCaseReceipt:
    kept_trial_ids: tuple[object, ...]
    excluded_trial_ids: tuple[object, ...]
    columns: tuple[str, ...]
    nuisance_scope: str
    mask_sha256: str
    evidence_scope: str = "frozen_complete_case_trials"


def complete_case_trial_mask(
    nuisance_table: pd.DataFrame,
    *,
    valid_mask: Sequence[bool],
    trial_ids: Sequence[object],
) -> tuple[np.ndarray, CompleteCaseReceipt]:
    """Freeze the one trial mask used by counts, beliefs, controls, and splits.

    Lagged primary covariates intentionally make the first trial incomplete.
    This helper excludes it (and any other missing row) explicitly instead of
    silently imputing it or letting downstream fitters select different rows.
    """

    ids = tuple(trial_ids)
    raw_valid = np.asarray(valid_mask)
    if (
        len(ids) != len(nuisance_table)
        or raw_valid.shape != (len(ids),)
        or raw_valid.dtype != bool
    ):
        raise IBLMultiSessionError(
            "trial IDs, nuisance rows, and binary valid_mask must align"
        )
    try:
        if len(set(ids)) != len(ids):
            raise IBLMultiSessionError("trial_ids must be unique")
    except TypeError as error:
        raise IBLMultiSessionError("trial_ids must be hashable") from error
    numeric = nuisance_table.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric).all(axis=1)
    mask = np.array(raw_valid & finite, dtype=bool, copy=True)
    mask.setflags(write=False)
    kept = tuple(item for item, include in zip(ids, mask, strict=True) if include)
    excluded = tuple(
        item for item, include in zip(ids, mask, strict=True) if not include
    )
    if len(kept) < 2:
        raise IBLMultiSessionError("fewer than two complete trials remain")
    digest = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
    receipt = CompleteCaseReceipt(
        kept_trial_ids=kept,
        excluded_trial_ids=excluded,
        columns=tuple(str(column) for column in nuisance_table.columns),
        nuisance_scope=str(nuisance_table.attrs.get("nuisance_scope", "unspecified")),
        mask_sha256=digest,
    )
    return mask, receipt


@dataclass(frozen=True, slots=True)
class ResidualizerFitReceipt:
    fit_trial_ids: tuple[object, ...]
    columns: tuple[str, ...]
    coefficient_sha256: str
    evidence_scope: str = "train_only_nuisance_fit"


class TrainOnlyNuisanceResidualizer:
    """Select fit rows by explicit IDs before fitting the existing residualizer."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self.model = TrialNuisanceResidualizer(self.columns)
        self.receipt: ResidualizerFitReceipt | None = None

    def fit(
        self,
        nuisance_table: pd.DataFrame,
        activity: np.ndarray,
        *,
        trial_ids: Sequence[object],
        train_ids: Sequence[object],
    ) -> ResidualizerFitReceipt:
        ids = tuple(trial_ids)
        train = tuple(train_ids)
        if len(ids) != len(nuisance_table) or np.asarray(activity).shape[0] != len(ids):
            raise IBLMultiSessionError("trial IDs, nuisances, and activity must align")
        positions = {value: index for index, value in enumerate(ids)}
        if (
            len(positions) != len(ids)
            or not train
            or len(set(train)) != len(train)
            or any(item not in positions for item in train)
        ):
            raise IBLMultiSessionError(
                "train_ids must be a non-empty subset of unique trial_ids"
            )
        selected = np.asarray([positions[item] for item in train], dtype=int)
        if np.any(np.diff(selected) <= 0):
            raise IBLMultiSessionError("train_ids must retain chronological order")
        self.model.fit(
            nuisance_table.iloc[selected],
            np.asarray(activity)[selected],
            sample_ids=train,
        )
        digest = hashlib.sha256(self.model.coefficients_.tobytes()).hexdigest()
        self.receipt = ResidualizerFitReceipt(train, self.columns, digest)
        return self.receipt

    def transform(
        self, nuisance_table: pd.DataFrame, activity: np.ndarray
    ) -> np.ndarray:
        if self.receipt is None:
            raise RuntimeError("fit must be called before transform")
        return self.model.transform(nuisance_table, activity)


__all__ = [
    "ChronologicalBlockSplit",
    "CompleteCaseReceipt",
    "FrozenIBLNeuralCohort",
    "IBLMultiSessionError",
    "IBLNeuralCohortEntry",
    "PreparedIBLNeuralSession",
    "ResidualizerFitReceipt",
    "TrainOnlyNuisanceResidualizer",
    "chronological_outer_inner_splits",
    "chronological_whole_block_split",
    "complete_case_trial_mask",
    "full_trial_sensitivity_nuisance_table",
    "load_frozen_ibl_neural_cohort",
    "past_safe_nuisance_table",
    "prepare_ibl_neural_session",
]
