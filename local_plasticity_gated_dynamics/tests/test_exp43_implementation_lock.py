from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "provenance" / "exp43_implementation_lock_20260728.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exp43_implementation_lock_matches_current_sources_and_protocol() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["status"] == "implementation_frozen_before_outcome"
    assert lock["development_only"] is True
    assert lock["claim_upgrade_allowed"] is False
    assert lock["outcome_exposed_at_lock"] is False
    assert lock["formal_seeds_accessed_at_lock"] is False
    assert lock["development_seeds"] == list(range(43000, 43008))
    assert lock["reserved_formal_seeds"] == list(range(43100, 43130))
    assert lock["config_sha256"] == _sha256(
        PROJECT_ROOT
        / "configs"
        / "development"
        / "exp43_fast_slow_causal_decomposition_probe_v1.json"
    )
    assert lock["protocol_sha256"] == _sha256(
        PROJECT_ROOT
        / "docs"
        / "exp43_fast_slow_causal_decomposition_protocol_20260728.md"
    )
    assert lock["source_sha256"] == {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in lock["source_sha256"]
    }
    encoded = json.dumps(lock["source_sha256"], sort_keys=True).encode("utf-8")
    assert lock["implementation_sha256"] == hashlib.sha256(encoded).hexdigest()
