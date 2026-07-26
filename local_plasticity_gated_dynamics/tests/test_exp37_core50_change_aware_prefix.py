from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp37_core50_change_aware_prefix import (
    CONDITIONS,
    _condition_traces,
    validate_config,
    validate_preregistration,
)
from src.data.core50_streaming import Core50Stream
from src.models.bocpd_prefix import BOCPDConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads(
        (PROJECT_ROOT / "configs/prospective/exp37_core50_change_aware_prefix.json").read_text(
            encoding="utf-8"
        )
    )


def test_exp37_frozen_config_and_receipt_validate() -> None:
    config = _config()
    validate_config(config)
    receipt = validate_preregistration(
        PROJECT_ROOT / "configs/prospective/exp37_core50_change_aware_prefix.json"
    )
    assert receipt["external_outcomes_inspected"] is False
    config["used_external_labels_for_fit"] = True
    with pytest.raises(ValueError, match="used_external_labels"):
        validate_config(config)


def test_exp37_condition_panel_is_paired_and_complete() -> None:
    first = np.tile([0.98, 0.01, 0.005, 0.005], (12, 1))
    second = np.tile([0.01, 0.98, 0.005, 0.005], (12, 1))
    evidence = np.vstack([first, second])
    labels = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
    switches = np.zeros(24, dtype=bool)
    switches[12] = True
    stream = Core50Stream(
        session_id="s3",
        task_index=0,
        panel="hidden_switch",
        evidence=evidence,
        labels=labels,
        stream_ids=np.repeat("hidden", 24),
        switch_flags=switches,
        object_ids=("o1", "o2", "o3", "o4"),
        source_cells=("s3/o1", "s3/o2"),
    )
    traces = _condition_traces(
        stream,
        fixed_retention=0.5,
        window_frames=4,
        detector=BOCPDConfig(
            hazard=0.05,
            prior_concentration=1.0,
            alarm_threshold=0.2,
            min_run_frames=2,
            max_run_length=32,
        ),
        seed=3,
    )
    assert tuple(traces) == CONDITIONS
    assert all(len(trace.predictions) == len(labels) for trace in traces.values())
    assert np.array_equal(
        traces["bocpd_change_reset"].alarm_flags,
        traces["bocpd_score_no_reset"].alarm_flags,
    )
    assert not traces["bocpd_score_no_reset"].reset_flags.any()
    assert traces["oracle_change_reset"].reset_flags.sum() == 1
