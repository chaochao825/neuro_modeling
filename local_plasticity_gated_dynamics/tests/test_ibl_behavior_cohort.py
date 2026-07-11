"""Pure tests for deterministic IBL behavior-cohort freezing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.ibl_behavior_cohort import (
    IBLBehaviorCohortCriteria,
    balanced_session_order,
    behavior_session_qc,
    default_trials_table_provenance,
)


def test_balanced_order_uses_one_session_per_animal_before_second_round() -> None:
    query = pd.DataFrame(
        {
            "eid": ["a2", "a1", "a1", "b1", "b2", "c1"],
            "pid": ["pa2", "pa1", "pa1b", "pb1", "pb2", "pc1"],
            "subject": ["A", "A", "A", "B", "B", "C"],
            "date": [
                "2024-02-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-02-01",
                "2024-01-01",
            ],
            "lab": ["lab"] * 6,
        }
    )
    ordered = balanced_session_order(query)
    assert ordered["eid"].tolist() == ["a1", "b1", "c1", "a2", "b2"]
    assert ordered.iloc[0]["pids"] == "pa1;pa1b"
    assert ordered["candidate_rank"].tolist() == list(range(5))


def test_behavior_qc_counts_binary_context_switches_and_preserves_analysis_rows() -> (
    None
):
    blocks = np.repeat([0.5, 0.2, 0.8, 0.2, 0.8], 4)
    left = np.arange(blocks.size) % 2 == 0
    table = pd.DataFrame(
        {
            "contrastLeft": np.where(left, 0.5, np.nan),
            "contrastRight": np.where(~left, 0.5, np.nan),
            "choice": np.where(left, 1, -1),
            "feedbackType": 1,
            "probabilityLeft": blocks,
        }
    )
    criteria = IBLBehaviorCohortCriteria(
        target_sessions=1,
        min_animals=1,
        max_sessions_per_animal=1,
        min_raw_trials=20,
        min_analysis_trials=20,
        min_valid_choices=20,
        min_valid_feedback=20,
        min_context_switches=3,
    )
    qc = behavior_session_qc(
        table, criteria, official_bwm_mask=np.ones(len(table), dtype=bool)
    )
    assert qc["eligible"]
    assert qc["context_switch_count"] == 3
    assert qc["analysis_trial_count"] == 20
    assert len(qc["analysis_frame"]) == 20


def test_default_table_provenance_uses_default_revision() -> None:
    details = pd.DataFrame(
        {
            "rel_path": [
                "alf/_ibl_trials.table.pqt",
                "alf/#2025-03-03#/_ibl_trials.table.pqt",
            ],
            "default_revision": [False, True],
            "hash": ["old", "new"],
            "file_size": [10.0, 12.0],
            "qc": ["NOT_SET", "PASS"],
        },
        index=pd.Index(["old-id", "new-id"], name="id"),
    )
    provenance = default_trials_table_provenance(details)
    assert provenance["dataset_uuid"] == "new-id"
    assert provenance["dataset_revision"] == "2025-03-03"
    assert provenance["dataset_hash"] == "new"
