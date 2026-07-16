from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import data.compositional_tasks_loader as loader
from data.compositional_tasks_loader import OfficialFileSpec
from experiments.exp25_compositional_tasks_real import (
    FAMILIES,
    PROTOCOLS,
    run_seed,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_official_stub(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, OfficialFileSpec]:
    raw_root = root / "raw"
    raw_root.mkdir(parents=True)
    specs = {}
    for index, (name, official) in enumerate(loader.OFFICIAL_FILE_SPECS.items()):
        payload = f"official-reviewed-fixture-{index}:{name}".encode()
        (raw_root / name).write_bytes(payload)
        specs[name] = OfficialFileSpec(
            name=name,
            file_id=official.file_id,
            size=len(payload),
            md5=hashlib.md5(payload).hexdigest(),
        )
    monkeypatch.setattr(
        loader,
        "OFFICIAL_FILE_SPECS",
        MappingProxyType(specs),
    )
    return specs


def _write_canonical_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    specs = _write_official_stub(root, monkeypatch)
    canonical = root / "canonical"
    canonical.mkdir()
    rng = np.random.default_rng(91)
    trial_rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    session_records: list[dict[str, object]] = []
    n_trials, n_time, n_units = 18, 5, 10
    latent_dim = 3
    operators = (
        np.array(
            [
                [0.66, 0.12, 0.00],
                [-0.06, 0.71, 0.10],
                [0.04, -0.08, 0.68],
            ]
        ),
        np.array(
            [
                [0.69, -0.13, 0.05],
                [0.12, 0.65, -0.06],
                [0.00, 0.10, 0.70],
            ]
        ),
        np.array(
            [
                [0.63, 0.02, -0.14],
                [0.05, 0.72, 0.08],
                [0.13, -0.04, 0.66],
            ]
        ),
    )
    input_operators = (
        np.array([[0.20, 0.04, -0.05], [-0.03, 0.16, 0.07]]),
        np.array([[-0.14, 0.08, 0.05], [0.06, 0.18, -0.04]]),
        np.array([[0.05, -0.16, 0.09], [0.12, 0.02, 0.14]]),
    )
    for session_index in range(2):
        session_id = f"session-{session_index}"
        animal_id = f"animal-{session_index}"
        trial_ids = np.asarray(
            [f"{session_id}:trial-{index:02d}" for index in range(n_trials)]
        )
        unit_ids = np.asarray(
            [f"{session_id}:unit-{index:02d}" for index in range(n_units)]
        )
        loading = rng.normal(0.0, 0.22, size=(latent_dim, n_units))
        bias = rng.normal(1.25, 0.08, size=n_units)
        counts = np.empty((n_trials, n_time, n_units), dtype=np.int64)
        inputs = np.empty((n_trials, n_time, 2), dtype=float)
        for trial in range(n_trials):
            composition = trial % 3
            inputs[trial, :, 0] = np.linspace(-0.8, 0.8, n_time)
            inputs[trial, :, 1] = (composition - 1) * 0.55
            state = rng.normal(0.0, 0.32, size=latent_dim)
            for time in range(n_time):
                rate = np.exp(np.clip(bias + state @ loading, -2.0, 3.0))
                counts[trial, time] = rng.poisson(rate)
                state = (
                    state @ operators[composition]
                    + inputs[trial, time] @ input_operators[composition]
                    + rng.normal(0.0, 0.03, latent_dim)
                )
            trial_rows.append(
                {
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "trial_id": trial_ids[trial],
                    "trial_order": trial,
                    "block_id": trial // 3,
                    "composition_id": f"composition-{composition}",
                    "cue": f"cue-{composition}",
                    "behavior": f"action-{composition}",
                    "stimulus_id": f"stimulus-{composition}",
                    "action_id": f"action-{composition}",
                }
            )
        for unit_index, unit_id in enumerate(unit_ids):
            unit_rows.append(
                {
                    "session_id": session_id,
                    "unit_id": unit_id,
                    "region": f"region-{unit_index % 2}",
                }
            )
        asset = canonical / f"{session_id}.npz"
        np.savez(
            asset,
            counts=counts,
            inputs=inputs,
            trial_ids=trial_ids,
            unit_ids=unit_ids,
        )
        session_records.append(
            {
                "session_id": session_id,
                "animal_id": animal_id,
                "asset": _descriptor(asset, root),
                "counts_key": "counts",
                "inputs_key": "inputs",
                "trial_ids_key": "trial_ids",
                "unit_ids_key": "unit_ids",
            }
        )
    trials_path = canonical / "trials.csv"
    units_path = canonical / "units.csv"
    conversion_path = canonical / "prepare_compositional_tasks.py"
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)
    pd.DataFrame(unit_rows).to_csv(units_path, index=False)
    conversion_path.write_text(
        "# hash-pinned canonical conversion test fixture\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": loader.MANIFEST_SCHEMA,
        "source": dict(loader.OFFICIAL_SOURCE_PROVENANCE),
        "official_files": [
            specs[name].manifest_record() for name in sorted(specs)
        ],
        "canonical": {
            "trials": _descriptor(trials_path, root),
            "units": _descriptor(units_path, root),
            "sessions": session_records,
            "conversion_code": _descriptor(conversion_path, root),
        },
    }
    manifest_path = root / "canonical_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return manifest_path, _sha256(manifest_path)


def _config(root: Path, manifest: Path, digest: str) -> dict[str, object]:
    return {
        "profile": "smoke",
        "data_mode": "official_canonical_only",
        "data_root": str(root),
        "manifest_path": str(manifest),
        "expected_manifest_sha256": digest,
        "minimum_sessions": 2,
        "minimum_animals": 2,
        "protocols": list(PROTOCOLS),
        "candidate_latent_dims": [2, 4, 8, 16],
        "max_outer_folds": 1,
        "max_inner_folds": 1,
        "belief": {
            "cue_columns": ["cue"],
            "behavior_columns": ["behavior"],
            "numeric_columns": [],
            "fit_label_column": "composition_id",
            "ridge": 0.1,
            "temperature": 1.0,
        },
        "model": {
            "gate_rank": 2,
            "ridge": 0.02,
            "poisson_ridge": 0.02,
            "max_irls": 4,
        },
    }


def _records(run_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (run_path / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def test_exp25_hash_pinned_multi_session_data_fails_unaligned_basis_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "canonical-data"
    manifest, digest = _write_canonical_fixture(data_root, monkeypatch)
    run_path = run_seed(
        _config(data_root, manifest, digest),
        7,
        tmp_path / "results",
    )
    planned = json.loads(
        (run_path / "planned_conditions.json").read_text(encoding="utf-8")
    )
    assert len(planned) == len(FAMILIES) * len(PROTOCOLS) == 20
    records = _records(run_path)
    invalid = [value for value in records if value["status"] == "invalid"]
    assert len(invalid) == len(FAMILIES) * len(PROTOCOLS)
    assert not [
        value
        for value in records
        if value.get("record_type") in {"outer_fold", "protocol_aggregate"}
    ]
    assert {value["model_family"] for value in invalid} == set(FAMILIES)
    assert {value["protocol"] for value in invalid} == set(PROTOCOLS)
    for value in invalid:
        assert value["official_source_verified"] is True
        assert value["canonical_conversion_verified"] is True
        assert value["canonical_manifest_sha256"] == digest
        assert value["stage"] == "latent_coordinate_identifiability"
        assert (
            value["latent_coordinate_system"]
            == "independent_session_train_only_pca"
        )
        assert value["session_basis_alignment"] == "not_implemented"
        assert value["shared_cross_session_dynamics_identifiable"] is False
        assert value["shared_cross_session_dynamics_claimed"] is False
        assert value["heldout_likelihood_comparison_valid"] is False
        assert value["train_only_shared_basis_implemented"] is False
        assert "independently fit train-only PCA" in value["reason"]
        assert value["independent_statistical_units"] == ["animal", "session"]
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
    assert status["condition_invalid"] == len(FAMILIES) * len(PROTOCOLS)


def test_exp25_official_processed_files_without_canonical_counts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "official-only"
    _write_official_stub(data_root, monkeypatch)
    missing_manifest = data_root / "canonical_manifest.json"
    run_path = run_seed(
        _config(data_root, missing_manifest, "1" * 64),
        3,
        tmp_path / "results",
    )
    records = _records(run_path)
    assert len(records) == len(FAMILIES) * len(PROTOCOLS)
    assert all(value["status"] == "failed" for value in records)
    assert all(
        "canonical trial-level neural counts are absent" in value["error"]
        for value in records
    )
    assert all("synthetic" in value["error"] for value in records)
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
    assert status["condition_failures"] == 20
