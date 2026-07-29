from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAILURE = (
    ROOT / "results" / "history" / "exp44_piray_daw_entrypoint_failure_20260730"
)


def test_exp44_preoutcome_launch_failure_is_hash_bound() -> None:
    manifest = json.loads((FAILURE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_stage"] == "module_import_before_data_loading"
    assert not manifest["outcomes_accessed"]
    assert manifest["frozen_tag"] == "exp44-dev-v1-preoutcome-20260730"
    assert (FAILURE / "exit_status.txt").read_text(encoding="utf-8").strip() == "1"
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((FAILURE / name).read_bytes()).hexdigest() == expected
