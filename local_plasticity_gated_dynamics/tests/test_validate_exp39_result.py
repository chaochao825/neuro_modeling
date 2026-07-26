from __future__ import annotations

from pathlib import Path

from scripts.validate_exp39_result import validate_result


ROOT = Path(__file__).resolve().parents[1]


def test_formal_exp39_package_replays_when_materialized() -> None:
    result = ROOT / "results/exp39_factorized_uncertainty_prospective_v1"
    if not result.exists():
        return
    audit = validate_result(result)
    assert audit["audit_status"] == "passed"
    assert audit["n_seeds"] == 30
    assert audit["paired_tape_check"] == "passed"
