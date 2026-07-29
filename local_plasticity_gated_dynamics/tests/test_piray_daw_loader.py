from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.data.piray_daw import EXPECTED_PARTICIPANTS, load_piray_daw


ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = ROOT / "data" / "raw" / "piray_daw_v1"


def _write_fixture(root: Path, *, missing_field: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.zip").write_bytes(b"fixture archive")
    experiment = root / "experiment1"
    experiment.mkdir()
    trials, blocks = 50, 4
    participants = np.empty((EXPECTED_PARTICIPANTS[1], 1), dtype=object)
    for index in range(EXPECTED_PARTICIPANTS[1]):
        participant = {
            "bucket": np.full((trials, blocks), 60 + index % 2, dtype=np.uint8),
            "response_time": np.full((trials, blocks), 500, dtype=np.uint16),
            "randomization_order": np.array([0, 1, 2, 3], dtype=np.uint8),
            "age": 30,
            "gender": "x",
        }
        if missing_field:
            participant.pop("response_time")
        participants[index, 0] = participant
    metadata = {
        "true_vol": np.array([4, 49, 4, 49], dtype=np.uint8),
        "true_sto": np.array([16, 16, 64, 64], dtype=np.uint8),
        "bird": np.full((trials, blocks), 60.0),
        "bag": np.full((trials, blocks), 61.0),
    }
    savemat(experiment / "data.mat", {"meta_data": metadata, "data": participants})


def test_loader_builds_canonical_read_only_arrays(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    dataset = load_piray_daw(tmp_path, experiment=1, verify_hashes=False)
    assert dataset.bucket.shape == (223, 50, 4)
    assert dataset.bag.shape == (50, 4)
    assert dataset.true_process_variance.tolist() == [4.0, 49.0, 4.0, 49.0]
    assert not dataset.bucket.flags.writeable
    frame = dataset.long_frame()
    assert len(frame) == 223 * 50 * 4
    assert "bird" not in frame
    assert "bird" in dataset.long_frame(include_hidden_bird=True)


def test_confirmation_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="confirmation-locked"):
        load_piray_daw(tmp_path, experiment=2)


def test_loader_rejects_missing_participant_fields(tmp_path: Path) -> None:
    _write_fixture(tmp_path, missing_field=True)
    with pytest.raises(ValueError, match="response_time"):
        load_piray_daw(tmp_path, experiment=1, verify_hashes=False)


def test_loader_rejects_unverified_archive(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        load_piray_daw(tmp_path, experiment=1, verify_hashes=True)


def test_loader_rejects_invalid_experiment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="experiment must be 1 or 2"):
        load_piray_daw(tmp_path, experiment=3)


@pytest.mark.integration
def test_released_experiment1_bundle_passes_hash_and_schema_audit() -> None:
    if not (REAL_DATA / "experiment1" / "data.mat").is_file():
        pytest.skip("hash-bound Piray--Daw bundle is not installed")
    dataset = load_piray_daw(REAL_DATA, experiment=1, verify_hashes=True)
    assert dataset.n_participants == 223
    assert int(np.isnan(dataset.age).sum()) == 1
    assert np.array_equal(np.sort(dataset.randomization_order, axis=1)[0], np.arange(4))
    assert set(dataset.true_process_variance) == {4.0, 49.0}
    assert set(dataset.true_observation_variance) == {16.0, 64.0}
