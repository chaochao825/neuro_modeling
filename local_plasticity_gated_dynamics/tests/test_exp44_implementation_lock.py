from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "provenance" / "exp44_development_implementation_lock_20260730.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exp44_preoutcome_lock_matches_protocol_config_and_sources() -> None:
    receipt = json.loads(LOCK.read_text(encoding="utf-8"))
    assert receipt["stage"] == "prospective_development_preoutcome"
    assert receipt["authorized_experiment"] == 1
    assert not receipt["development_outcomes_accessed_before_lock"]
    assert not receipt["confirmation_experiment_accessed"]
    assert not receipt["popgym_accessed"]
    assert not receipt["claim_upgrade_allowed"]
    assert not receipt["schema_only_access_before_lock"][
        "project_model_comparisons_computed"
    ]
    for relative, expected in receipt["sha256"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == expected, relative


def test_exp44_locked_config_cannot_access_confirmation() -> None:
    receipt = json.loads(LOCK.read_text(encoding="utf-8"))
    config_path = ROOT / "configs" / "development" / "exp44_piray_daw_qr_behavior_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["stage"] == "development"
    assert config["data"]["experiment"] == receipt["authorized_experiment"]
    assert config["confirmation_lock"]["experiment"] == 2
    assert not config["confirmation_lock"]["allow_confirmation"]
