from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from src.data.ibl_block_sidecar import (
    IBLBlockSidecarError,
    load_ibl_block_truth,
)
from src.data.ibl_multisession import PreparedIBLNeuralSession


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    probability = np.asarray([0.5, 0.5, 0.8, 0.8, 0.2, 0.2])
    stimulus_side = np.asarray([1, 0, 1, 1, 0, 0])
    choice = np.asarray([-1, 1, -1, 1, 1, -1])
    official = np.asarray([True, True, False, True, True, True])
    compact = pd.DataFrame(
        {
            "source_trial_index": np.arange(6),
            "contrastLeft": np.where(stimulus_side == 1, 0.5, np.nan),
            "contrastRight": np.where(stimulus_side == 0, 0.5, np.nan),
            "choice": choice,
            "feedbackType": np.asarray([1, 1, -1, 1, 1, -1]),
            "probabilityLeft": probability,
            "official_bwm_mask": official,
        }
    )
    neural = pd.DataFrame(
        {
            "trial_id": np.arange(6),
            "stimulus_side": stimulus_side,
            "choice": choice,
            "official_bwm_mask": official,
        }
    )
    return compact, neural


def _prepared(
    neural_mutation: Callable[[pd.DataFrame], None] | None = None,
) -> PreparedIBLNeuralSession:
    _, neural = _tables()
    stimulus = neural.copy(deep=True)
    movement = neural.copy(deep=True)
    if neural_mutation is not None:
        neural_mutation(stimulus)
    counts = np.ones((6, 3, 2), dtype=np.int64)
    return PreparedIBLNeuralSession(
        eid="eid-sidecar",
        animal_id="animal-sidecar",
        count_views={"stimulus_pre": counts, "movement_pre": counts + 1},
        valid_masks={
            "stimulus_pre": np.ones(6, dtype=bool),
            "movement_pre": np.ones(6, dtype=bool),
        },
        time_axes={
            "stimulus_pre": np.asarray([-0.3, -0.2, -0.1]),
            "movement_pre": np.asarray([-0.3, -0.2, -0.1]),
        },
        regions=np.asarray(["VISp", "LP"]),
        unit_ids=np.asarray(["unit-0", "unit-1"]),
        view_trial_tables={"stimulus_pre": stimulus, "movement_pre": movement},
        current_trial_ids=np.arange(6),
    )


def _fixture(
    tmp_path: Path,
    *,
    compact_mutation: Callable[[pd.DataFrame], None] | None = None,
    manifest_eid: str = "eid-sidecar",
    manifest_animal: str = "animal-sidecar",
) -> tuple[Path, str, Path]:
    compact, _ = _tables()
    if compact_mutation is not None:
        compact_mutation(compact)
    session_dir = tmp_path / "sessions" / "eid-sidecar"
    session_dir.mkdir(parents=True)
    compact_path = session_dir / "trials.csv"
    compact.to_csv(compact_path, index=False)
    manifest = pd.DataFrame(
        [
            {
                "eid": manifest_eid,
                "subject": manifest_animal,
                "status": "eligible",
                "eligible": True,
                "compact_table": "sessions/eid-sidecar/trials.csv",
                "compact_table_sha256": _sha256(compact_path),
                "cohort_id": "fixture-cohort",
                "dataset_uuid": "fixture-dataset",
                "dataset_revision": "fixture-revision",
                "dataset_hash": "fixture-hash",
                "dataset_qc": "PASS",
                "bwm_repository_commit": "a" * 40,
            }
        ]
    )
    manifest_path = tmp_path / "cohort_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path, _sha256(manifest_path), compact_path


def test_sidecar_is_hash_bound_readonly_and_evaluation_only(tmp_path: Path) -> None:
    manifest, digest, _ = _fixture(tmp_path)
    truth, provenance = load_ibl_block_truth(
        manifest,
        _prepared(),
        expected_manifest_sha256=digest,
    )

    np.testing.assert_array_equal(truth.trial_ids, np.arange(6))
    np.testing.assert_array_equal(
        truth.probability_left, np.asarray([0.5, 0.5, 0.8, 0.8, 0.2, 0.2])
    )
    np.testing.assert_array_equal(
        truth.block_switch, np.asarray([False, False, True, False, True, False])
    )
    assert all(
        not value.flags.writeable
        for value in (
            truth.trial_ids,
            truth.probability_left,
            truth.block_switch,
            truth.official_bwm_mask,
        )
    )
    assert not hasattr(truth, "cue_observations")
    assert not hasattr(truth, "observations")
    assert (
        provenance["access_scope"]
        == "whole_block_split_and_postfit_evaluation_only"
    )
    assert provenance["eligible_for_whole_block_split"] is True
    assert provenance["eligible_for_postfit_evaluation"] is True
    assert provenance["eligible_for_gate_input"] is False
    assert provenance["eligible_for_model_input"] is False
    assert provenance["cohort_manifest_sha256"] == digest
    assert provenance["truth_fingerprint"] == truth.fingerprint
    with pytest.raises(TypeError):
        provenance["access_scope"] = "model_input"  # type: ignore[index]


def test_sidecar_rejects_manifest_and_compact_hash_tampering(tmp_path: Path) -> None:
    manifest, digest, compact = _fixture(tmp_path)
    with pytest.raises(IBLBlockSidecarError, match="manifest SHA-256"):
        load_ibl_block_truth(
            manifest,
            _prepared(),
            expected_manifest_sha256="0" * 64,
        )

    compact.write_bytes(compact.read_bytes() + b"\n")
    with pytest.raises(IBLBlockSidecarError, match="compact table SHA-256"):
        load_ibl_block_truth(
            manifest,
            _prepared(),
            expected_manifest_sha256=digest,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda table: table.__setitem__(
                "source_trial_index", np.arange(6) + 1
            ),
            "source_trial_index",
        ),
    ],
)
def test_sidecar_rejects_source_trial_misalignment(
    tmp_path: Path,
    mutation: Callable[[pd.DataFrame], None],
    message: str,
) -> None:
    manifest, digest, _ = _fixture(tmp_path, compact_mutation=mutation)
    with pytest.raises(IBLBlockSidecarError, match=message):
        load_ibl_block_truth(
            manifest,
            _prepared(),
            expected_manifest_sha256=digest,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table.loc.__setitem__((0, "trial_id"), 99), "trial_id"),
        (
            lambda table: table.loc.__setitem__((0, "stimulus_side"), 0),
            "stimulus_side",
        ),
        (lambda table: table.loc.__setitem__((0, "choice"), 1), "choice"),
        (
            lambda table: table.loc.__setitem__((0, "official_bwm_mask"), False),
            "official_bwm_mask",
        ),
    ],
)
def test_sidecar_rejects_neural_trial_alignment_changes(
    tmp_path: Path,
    mutation: Callable[[pd.DataFrame], None],
    message: str,
) -> None:
    manifest, digest, _ = _fixture(tmp_path)
    with pytest.raises(IBLBlockSidecarError, match=message):
        load_ibl_block_truth(
            manifest,
            _prepared(mutation),
            expected_manifest_sha256=digest,
        )


@pytest.mark.parametrize(
    ("eid", "animal", "message"),
    [
        ("other-eid", "animal-sidecar", "exactly one row"),
        ("eid-sidecar", "other-animal", "animal"),
    ],
)
def test_sidecar_rejects_manifest_session_identity_mismatch(
    tmp_path: Path,
    eid: str,
    animal: str,
    message: str,
) -> None:
    manifest, digest, _ = _fixture(
        tmp_path,
        manifest_eid=eid,
        manifest_animal=animal,
    )
    with pytest.raises(IBLBlockSidecarError, match=message):
        load_ibl_block_truth(
            manifest,
            _prepared(),
            expected_manifest_sha256=digest,
        )
