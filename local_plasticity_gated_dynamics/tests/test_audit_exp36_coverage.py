from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_exp36_coverage import OUTCOME_FIELDS, _failure_rows, collector_schema_coverage


def test_failure_reader_skips_completed_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "status": "complete",
                        "collector_id": "P1",
                        "accuracy": "MUST_NOT_BE_PARSED_AS_A_NUMBER",
                    }
                ),
                json.dumps(
                    {
                        "status": "failed",
                        "collector_id": "P2",
                        "task_index": 0,
                        "panel": "natural",
                        "condition": "cumulative",
                        "error_type": "ValueError",
                        "error": "too few classes",
                        "seed": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = list(_failure_rows(path))
    assert len(rows) == 1
    assert rows[0]["collector_id"] == "P2"
    assert OUTCOME_FIELDS.isdisjoint(rows[0])


def test_failure_reader_rejects_outcome_bearing_failed_record(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"status": "failed", "collector_id": "P1", "accuracy": 0.1})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="outcome fields"):
        list(_failure_rows(path))


def test_collector_schema_requires_four_paired_objects(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for collector, n_objects, n_paired in (("P1", 4, 4), ("P2", 4, 3), ("P3", 5, 4)):
        for object_index in range(n_objects):
            types = ("clean", "clutter") if object_index < n_paired else ("clean",)
            for video_type in types:
                rows.append(
                    {
                        "user_id": collector,
                        "object_name": f"object-{object_index}",
                        "video_type": video_type,
                        "video_id": f"{collector}-{object_index}-{video_type}",
                        "n_frames": 10,
                    }
                )
    path = tmp_path / "feature_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = collector_schema_coverage(path).set_index("collector_id")
    assert bool(result.loc["P1", "eligible_for_registered_sampling"])
    assert not bool(result.loc["P2", "eligible_for_registered_sampling"])
    assert not bool(result.loc["P3", "eligible_for_registered_sampling"])
    assert result.loc["P2", "n_incomplete_objects"] == 1
