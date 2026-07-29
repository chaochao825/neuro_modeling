from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.validate_exp44_artifacts import validate_exp44_artifacts


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "exp44_piray_daw_qr_behavior_development_v1"


def test_frozen_exp44_result_replays_and_remains_locked() -> None:
    receipt = validate_exp44_artifacts(RESULT)
    assert receipt["status"] == "pass"
    assert receipt["n_participants"] == 223
    assert receipt["participant_metric_rows"] == 223 * 6
    assert receipt["cell_metric_rows"] == 223 * 6 * 4
    assert receipt["conclusion"] == "oppose"
    assert not receipt["development_gate_passed"]
    assert receipt["experiment2_locked"]
    assert receipt["popgym_locked"]


def test_exp44_validator_rejects_a_changed_registered_decision(tmp_path: Path) -> None:
    copied = tmp_path / "result"
    shutil.copytree(RESULT, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["development_gate_passed"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Re-hash the edited summary so this test reaches semantic replay rather
    # than stopping at the lower-level byte-integrity guard.
    manifest["artifacts"]["summary.json"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="development gate does not replay"):
        validate_exp44_artifacts(copied)


def test_exp44_validator_rejects_artifact_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "result"
    shutil.copytree(RESULT, copied)
    comparisons = copied / "comparisons.csv"
    comparisons.write_text(
        comparisons.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_exp44_artifacts(copied)
