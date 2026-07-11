"""Deterministic selection and QC helpers for an IBL behavior cohort.

Network access is kept out of this module.  The command-line freezer injects a
ONE client and calls these pure helpers, which makes cohort selection and
eligibility testable without downloading public data during unit tests.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


TRIAL_TABLE_COLUMNS = (
    "contrastLeft",
    "contrastRight",
    "choice",
    "feedbackType",
    "probabilityLeft",
)


@dataclass(frozen=True)
class IBLBehaviorCohortCriteria:
    """Outcome-blind, session-level inclusion thresholds."""

    target_sessions: int = 30
    min_animals: int = 10
    max_sessions_per_animal: int = 3
    min_raw_trials: int = 400
    min_analysis_trials: int = 300
    min_valid_choices: int = 300
    min_valid_feedback: int = 300
    min_context_switches: int = 8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.min_animals > self.target_sessions:
            raise ValueError("min_animals cannot exceed target_sessions")


def balanced_session_order(bwm_query: pd.DataFrame) -> pd.DataFrame:
    """Return unique BWM sessions in deterministic animal-round-robin order."""

    if not isinstance(bwm_query, pd.DataFrame):
        raise TypeError("bwm_query must be a pandas DataFrame")
    required = {"eid", "subject", "date", "lab"}
    missing = sorted(required - set(bwm_query.columns))
    if missing:
        raise ValueError(f"bwm_query is missing columns: {missing}")
    frame = bwm_query.copy(deep=True)
    for column in ("eid", "subject", "date", "lab"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.len().eq(0).any():
            raise ValueError(f"bwm_query column {column} contains empty values")
    if "pid" not in frame:
        frame["pid"] = ""
    session_rows: list[dict[str, Any]] = []
    for (eid, subject, date, lab), group in frame.groupby(
        ["eid", "subject", "date", "lab"], sort=False
    ):
        session_rows.append(
            {
                "eid": eid,
                "subject": subject,
                "date": date,
                "lab": lab,
                "pids": ";".join(sorted(set(group["pid"].astype(str)) - {""})),
            }
        )
    sessions = pd.DataFrame(session_rows).sort_values(
        ["subject", "date", "eid"], kind="mergesort"
    )
    grouped = {
        subject: group.to_dict("records")
        for subject, group in sessions.groupby("subject", sort=True)
    }
    ordered: list[dict[str, Any]] = []
    for round_index in range(max(len(group) for group in grouped.values())):
        for subject in sorted(grouped):
            if round_index < len(grouped[subject]):
                row = dict(grouped[subject][round_index])
                row["animal_round"] = round_index
                ordered.append(row)
    result = pd.DataFrame(ordered)
    result.insert(0, "candidate_rank", np.arange(len(result), dtype=int))
    return result.reset_index(drop=True)


def compact_behavior_trials(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Retain auditable columns and mark trials valid for the gate analysis."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = sorted(set(TRIAL_TABLE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"trial table is missing columns: {missing}")
    compact = frame.loc[:, list(TRIAL_TABLE_COLUMNS)].copy(deep=True)
    for column in TRIAL_TABLE_COLUMNS:
        compact[column] = pd.to_numeric(compact[column], errors="coerce")
    left = compact["contrastLeft"].to_numpy(float)
    right = compact["contrastRight"].to_numpy(float)
    probability = compact["probabilityLeft"].to_numpy(float)
    exact_context = np.any(
        np.isclose(
            probability[:, np.newaxis],
            np.array([0.2, 0.5, 0.8])[np.newaxis, :],
            atol=1e-6,
            rtol=0.0,
        ),
        axis=1,
    )
    analysis_valid = (
        (np.isfinite(left) ^ np.isfinite(right))
        & np.isfinite(probability)
        & exact_context
    )
    compact.insert(0, "source_trial_index", np.arange(len(compact), dtype=int))
    compact["analysis_valid"] = analysis_valid
    return compact, analysis_valid


def behavior_session_qc(
    frame: pd.DataFrame,
    criteria: IBLBehaviorCohortCriteria,
    *,
    official_bwm_mask: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """Evaluate pre-registered QC without consulting model outcomes."""

    if not isinstance(criteria, IBLBehaviorCohortCriteria):
        raise TypeError("criteria must be IBLBehaviorCohortCriteria")
    compact, analysis_valid = compact_behavior_trials(frame)
    if official_bwm_mask is None:
        official_mask = np.ones(len(compact), dtype=bool)
        official_mask_present = False
    else:
        official_mask = np.asarray(official_bwm_mask)
        if official_mask.shape != (len(compact),) or official_mask.dtype.kind != "b":
            raise ValueError("official_bwm_mask must be a matching boolean vector")
        official_mask = official_mask.astype(bool, copy=False)
        official_mask_present = True
    compact["official_bwm_mask"] = official_mask
    analysis = compact.loc[analysis_valid]
    probability = analysis["probabilityLeft"].to_numpy(float)
    choice = analysis["choice"].to_numpy(float)
    feedback = analysis["feedbackType"].to_numpy(float)
    analysis_official_mask = official_mask[analysis_valid]
    binary_context = np.isclose(probability, 0.2, atol=1e-6) | np.isclose(
        probability, 0.8, atol=1e-6
    )
    changes = np.zeros(probability.size, dtype=bool)
    if probability.size > 1:
        changes[1:] = (
            binary_context[1:]
            & binary_context[:-1]
            & ~np.isclose(probability[1:], probability[:-1], atol=1e-6, rtol=0.0)
        )
    has_low_context = bool(np.any(np.isclose(probability, 0.2, atol=1e-6)))
    has_high_context = bool(np.any(np.isclose(probability, 0.8, atol=1e-6)))
    valid_choices = int(
        np.count_nonzero(np.isin(choice, [-1.0, 1.0]) & analysis_official_mask)
    )
    valid_feedback = int(
        np.count_nonzero(np.isin(feedback, [-1.0, 1.0]) & analysis_official_mask)
    )
    diagnostics: dict[str, Any] = {
        "raw_trial_count": int(len(frame)),
        "essential_trial_count": int(np.count_nonzero(analysis_valid)),
        "analysis_trial_count": int(np.count_nonzero(analysis_valid & official_mask)),
        "removed_trial_count": int(np.count_nonzero(~analysis_valid)),
        "valid_choice_count": valid_choices,
        "valid_feedback_count": valid_feedback,
        "official_bwm_mask_present": official_mask_present,
        "official_bwm_mask_trial_count": int(np.count_nonzero(official_mask)),
        "context_switch_count": int(np.count_nonzero(changes)),
        "has_low_and_high_context": has_low_context and has_high_context,
    }
    reasons: list[str] = []
    for violated, reason in (
        (diagnostics["raw_trial_count"] < criteria.min_raw_trials, "raw_trials"),
        (
            diagnostics["analysis_trial_count"] < criteria.min_analysis_trials,
            "analysis_trials",
        ),
        (valid_choices < criteria.min_valid_choices, "valid_choices"),
        (valid_feedback < criteria.min_valid_feedback, "valid_feedback"),
        (np.count_nonzero(~analysis_valid) > 0, "essential_trial_gaps"),
        (not official_mask_present, "official_bwm_mask_missing"),
        (
            diagnostics["context_switch_count"] < criteria.min_context_switches,
            "context_switches",
        ),
        (not diagnostics["has_low_and_high_context"], "context_coverage"),
    ):
        if violated:
            reasons.append(reason)
    diagnostics["eligible"] = not reasons
    diagnostics["exclusion_reason"] = ";".join(reasons)
    diagnostics["analysis_frame"] = compact.drop(columns="analysis_valid").reset_index(
        drop=True
    )
    return diagnostics


def default_trials_table_provenance(details: pd.DataFrame) -> dict[str, Any]:
    """Extract UUID, revision, hash, size, and QC for the default trials table."""

    if not isinstance(details, pd.DataFrame):
        raise TypeError("details must be a pandas DataFrame")
    frame = details.reset_index()
    if "rel_path" not in frame or "id" not in frame:
        raise ValueError("dataset details must provide id and rel_path")
    candidates = frame[
        frame["rel_path"].astype(str).str.endswith("_ibl_trials.table.pqt")
    ]
    if "default_revision" in candidates and candidates["default_revision"].any():
        candidates = candidates[candidates["default_revision"].astype(bool)]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one default _ibl_trials.table.pqt dataset, found {len(candidates)}"
        )
    row = candidates.iloc[0]
    rel_path = str(row["rel_path"])
    match = re.search(r"#([^#]+)#", rel_path)
    return {
        "dataset_uuid": str(row["id"]),
        "dataset_rel_path": rel_path,
        "dataset_revision": match.group(1) if match else "",
        "dataset_hash": str(row.get("hash", "")),
        "dataset_file_size": float(row.get("file_size", np.nan)),
        "dataset_qc": str(row.get("qc", "")),
    }


__all__ = [
    "IBLBehaviorCohortCriteria",
    "TRIAL_TABLE_COLUMNS",
    "balanced_session_order",
    "behavior_session_qc",
    "compact_behavior_trials",
    "default_trials_table_provenance",
]
