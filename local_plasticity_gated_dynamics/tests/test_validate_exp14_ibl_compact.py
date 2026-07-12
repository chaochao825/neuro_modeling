from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.validate_exp14_ibl_compact as validator


def _formal_config() -> dict[str, object]:
    return {
        "profile": "formal",
        "data_mode": "frozen_compact_cache",
        "compact_cache_manifest": "data/ibl_neural/exp14/compact_v1/compact_manifest.csv",
        "expected_source_manifest_sha256": "a" * 64,
        "expected_acquisition_bundle_sha256": "b" * 64,
        "expected_bwm_repository_commit": "c" * 40,
        "expected_compact_manifest_sha256": "d" * 64,
        "expected_compact_bundle_sha256": "e" * 64,
        "planned_sessions": 3,
        "planned_animals": 2,
    }


def _fake_session(
    eid: str, animal: str, counts: np.ndarray, valid: np.ndarray, regions: list[str]
) -> SimpleNamespace:
    return SimpleNamespace(
        eid=eid,
        animal_id=animal,
        count_views={"stimulus_pre": counts, "movement_pre": counts.copy()},
        valid_masks={"stimulus_pre": valid, "movement_pre": valid.copy()},
        unit_ids=np.asarray([f"{eid}-{index}" for index in range(counts.shape[2])]),
        regions=np.asarray(regions),
    )


def test_validate_compact_delegates_every_frozen_hash_and_reports_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "formal.json"
    config_path.write_text(json.dumps(_formal_config()), encoding="utf-8")
    manifest = tmp_path / "compact_manifest.csv"
    manifest.write_text("reviewed fixture\n", encoding="utf-8")
    first = _fake_session(
        "eid-1",
        "animal-1",
        np.ones((3, 2, 2), dtype=np.int64),
        np.array([1, 0, 1]),
        ["MOs", "MD"],
    )
    second = _fake_session(
        "eid-2",
        "animal-2",
        np.full((2, 2, 1), 2, dtype=np.int64),
        np.array([1, 1]),
        ["MOs"],
    )
    complete = (
        SimpleNamespace(
            status="complete", acquisition_validation_status="official_failed_retained"
        ),
        SimpleNamespace(
            status="complete", acquisition_validation_status="official_failed_retained"
        ),
    )
    cohort = SimpleNamespace(
        dispositions=complete + (SimpleNamespace(status="failed"),),
        complete_dispositions=complete,
        sessions=(first, second),
        compact_manifest_sha256="d" * 64,
        compact_bundle_sha256="e" * 64,
        evidence_scope="reviewed_offline_compact_ibl_counts",
    )
    call: dict[str, object] = {}

    def fake_loader(path: Path, **kwargs: object) -> SimpleNamespace:
        call.update(path=path, **kwargs)
        return cohort

    monkeypatch.setattr(validator, "load_compact_neural_cohort", fake_loader)
    report = validator.validate_compact(config_path, compact_manifest=manifest)

    assert call == {
        "path": manifest.resolve(),
        "expected_source_manifest_sha256": "a" * 64,
        "expected_acquisition_bundle_sha256": "b" * 64,
        "expected_bwm_repository_commit": "c" * 40,
        "expected_compact_manifest_sha256": "d" * 64,
        "expected_compact_bundle_sha256": "e" * 64,
        "expected_sessions": 3,
        "minimum_animals": 2,
    }
    assert report["offline_only"] is True
    observed = report["observed"]
    assert observed["status_counts"] == {"complete": 2, "failed": 1}
    assert observed["complete_sessions"] == 2
    assert observed["unique_animals"] == 2
    assert observed["total_units"] == 3
    for view in ("stimulus_pre", "movement_pre"):
        assert observed["views"][view] == {
            "trial_records": 5,
            "valid_trials": 4,
            "time_bins": [2],
            "total_binned_spikes": 20,
        }


def test_validator_rejects_nonformal_config_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _formal_config()
    values["profile"] = "smoke"
    config_path = tmp_path / "smoke.json"
    config_path.write_text(json.dumps(values), encoding="utf-8")

    def forbidden_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("cache loader must not run")

    monkeypatch.setattr(validator, "load_compact_neural_cohort", forbidden_loader)
    with pytest.raises(validator.CompactValidationError, match="only profile='formal'"):
        validator.validate_compact(config_path, compact_manifest=tmp_path / "missing")


def test_main_reports_invalid_config_only_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "invalid.json"
    config_path.write_text("[]", encoding="utf-8")
    assert validator.main(["--config", str(config_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["status"] == "invalid"
    assert payload["error_type"] == "CompactValidationError"


def test_frozen_audit_snapshot_hashes_are_byte_exact() -> None:
    root = Path("provenance/exp14_ibl_compact")
    expected = {
        "postprocess_compact.py": "951a516ae7eb2aa024f37e1890d6afd34dd19378905b90744dfadb32e7a76a17",
        "compact_schema.json": "f29c5506be93393499b90535c401bec1c82e0754737733897716cd4e2fade39d",
        "launch_postprocess.sh": "fe8a30ab705d7e0c8474e574fcae36ce68f510d2d53de81003edd3a3f05f2837",
        "bwm_loading.py": "c2e570c62cd0e047303c97d7999711b659a1c37eaa56dd2740ffff2c81f85321",
        "POSTPROCESS_READY_FOR_REVIEW.json": "e338ea6058782b486198bb12dc649f8416e864507221163862f0af1f8702ec2c",
        "POSTPROCESS_REVIEW_APPROVED.json": "9e8b7b0ec1c28029c82da19211f69ad8ed0c912c55f3b6c4eb519e8d15e28ff1",
        "compact_contract_audit_v2_20260712T0320Z.json": "5690e0611fa4931ba2f2f11735df5b5f5886bd2cd8638714bcb88a452827a7eb",
        "postprocess_execute_job_v2_20260712T0315Z.json": "77117600084dc694797231eaab7f8818fb8fcadeb5b88e52d9f44a6c282217e0",
        "postprocess_execute_exit_v2_20260712T0315Z.json": "e9e070ed306aa3bb4bbe891c1497191a7b150960652a4dc364c385092757083d",
        "postprocess_validate_v2_20260712T0310Z.jsonl": "d9208e4c061cb9483da8c58d0deea2f1c23e6cda6d9e7c0c1a1f09b9262df838",
        "postprocess_validate_v2_20260712T0311Z.jsonl": "78e548082556d247dd5ea7fb7536c47d03c11cc9a743f262f3c778194a083e96",
        "postprocess_validate_postapproval_v2_20260712T0314Z.jsonl": "78e548082556d247dd5ea7fb7536c47d03c11cc9a743f262f3c778194a083e96",
        "run_receipts.json": "55df22b11d0544ae328d6a2fa179d76a9a2c5e7db42e668536c5c1d6937dd22c",
        "BWM_LICENSE.txt": "150ac8fd0875e151e7d44ef9bb16deb632e0f56903d0cb8ffb8051f3978baf04",
    }
    observed = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in expected
    }
    assert observed == expected
    receipts = json.loads((root / "run_receipts.json").read_text(encoding="utf-8"))
    assert receipts["compact"]["manifest_sha256"] == (
        "a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09"
    )
    assert receipts["compact"]["bundle_sha256"] == (
        "f6cd351717986ede771a6bbbe755edeb3c30ef4bda48e86c8471bcf4364a41a4"
    )
    assert "MIT License" in (root / "BWM_LICENSE.txt").read_text(encoding="utf-8")
