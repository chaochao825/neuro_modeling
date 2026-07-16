from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import data.compositional_tasks_loader as loader
from data.compositional_tasks_loader import (
    CODE_DOI,
    FIGSHARE_ARTICLE_ID,
    FIGSHARE_DOI,
    FIGSHARE_LICENSE,
    MANIFEST_SCHEMA,
    OFFICIAL_FILE_SPECS,
    PAPER_URL,
    CompositionalTasksDataError,
    OfficialFileSpec,
    PastOnlyBeliefEstimator,
    leave_one_animal_out_splits,
    leave_one_block_out_splits,
    leave_one_composition_out_splits,
    leave_one_session_out_splits,
    load_compositional_tasks,
    validate_official_compositional_source,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str]:
    raw_root = root / "raw"
    canonical_root = root / "canonical"
    raw_root.mkdir(parents=True)
    canonical_root.mkdir()
    specs: dict[str, OfficialFileSpec] = {}
    for index, name in enumerate(OFFICIAL_FILE_SPECS):
        payload = f"reviewed-file-{index}:{name}".encode()
        path = raw_root / name
        path.write_bytes(payload)
        specs[name] = OfficialFileSpec(
            name=name,
            file_id=OFFICIAL_FILE_SPECS[name].file_id,
            size=len(payload),
            md5=hashlib.md5(payload).hexdigest(),
        )
    monkeypatch.setattr(loader, "OFFICIAL_FILE_SPECS", MappingProxyType(specs))

    trial_rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    session_records: list[dict[str, object]] = []
    for session_index in range(2):
        session_id = f"session-{session_index}"
        animal_id = f"animal-{session_index}"
        trial_ids = np.asarray(
            [f"{session_id}:trial-{index}" for index in range(4)]
        )
        unit_ids = np.asarray([f"{session_id}:unit-{index}" for index in range(3)])
        for trial_index, trial_id in enumerate(trial_ids):
            trial_rows.append(
                {
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "trial_order": trial_index,
                    "block_id": trial_index // 2,
                    "composition_id": trial_index % 2,
                    "cue": f"cue-{trial_index % 2}",
                    "behavior": f"choice-{trial_index % 2}",
                    "stimulus_id": f"stim-{trial_index % 2}",
                    "action_id": f"action-{trial_index % 2}",
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
        counts = (
            np.arange(4 * 5 * 3, dtype=np.int64).reshape(4, 5, 3) % 5
        )
        inputs = np.zeros((4, 5, 2), dtype=float)
        inputs[:, :, 0] = np.arange(4)[:, None]
        inputs[:, :, 1] = np.linspace(-1.0, 1.0, 5)
        asset = canonical_root / f"{session_id}.npz"
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
    trials_path = canonical_root / "trials.csv"
    units_path = canonical_root / "units.csv"
    conversion_path = canonical_root / "prepare_compositional_tasks.py"
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)
    pd.DataFrame(unit_rows).to_csv(units_path, index=False)
    conversion_path.write_text(
        "# reviewed canonical conversion fixture\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source": {
            "paper_url": PAPER_URL,
            "figshare_article_id": FIGSHARE_ARTICLE_ID,
            "figshare_doi": FIGSHARE_DOI,
            "license": FIGSHARE_LICENSE,
            "code_doi": CODE_DOI,
        },
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
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    return manifest_path, _sha256(manifest_path)


def test_official_release_constants_are_pinned() -> None:
    assert FIGSHARE_ARTICLE_ID == 30276238
    assert FIGSHARE_DOI == "10.6084/m9.figshare.30276238.v1"
    assert FIGSHARE_LICENSE == "CC BY 4.0"
    assert CODE_DOI == "10.5281/zenodo.17274345"
    assert OFFICIAL_FILE_SPECS["BhvData.mat"].file_id == 58487176
    assert OFFICIAL_FILE_SPECS["GLMdata.mat"].size == 288_180_371
    assert (
        OFFICIAL_FILE_SPECS["DynamicTransformationData.mat"].md5
        == "d73c9aeea6a61268ee880b192c38006b"
    )


def test_loader_verifies_source_manifest_schema_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, digest = _write_fixture(tmp_path, monkeypatch)
    dataset = load_compositional_tasks(
        tmp_path, manifest, expected_manifest_sha256=digest
    )
    assert len(dataset.sessions) == 2
    assert dataset.receipt.source_verified
    assert dataset.receipt.canonical_verified
    assert set(dataset.receipt.official_file_md5) == set(
        loader.OFFICIAL_FILE_SPECS
    )
    assert dataset.sessions[0].counts.shape == (4, 5, 3)
    assert not dataset.sessions[0].counts.flags.writeable
    assert not dataset.sessions[0].inputs.flags.writeable
    assert tuple(dataset.sessions[0].trial_ids) == tuple(
        dataset.trials.loc[
            dataset.trials["session_id"].eq("session-0"), "trial_id"
        ]
    )
    source = validate_official_compositional_source(tmp_path)
    assert source.source_verified
    assert source.source_provenance["figshare_article_id"] == 30276238


def test_loader_fails_closed_on_tampering_missing_schema_and_no_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, digest = _write_fixture(tmp_path, monkeypatch)
    first = next(iter(loader.OFFICIAL_FILE_SPECS))
    with (tmp_path / "raw" / first).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(CompositionalTasksDataError, match="size mismatch"):
        load_compositional_tasks(
            tmp_path, manifest, expected_manifest_sha256=digest
        )
    with pytest.raises(FileNotFoundError):
        load_compositional_tasks(
            tmp_path / "absent",
            manifest,
            expected_manifest_sha256=digest,
        )

    other_root = tmp_path / "schema"
    other_manifest, _ = _write_fixture(other_root, monkeypatch)
    payload = json.loads(other_manifest.read_text(encoding="utf-8"))
    payload["source"]["license"] = "unknown"
    other_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CompositionalTasksDataError, match="provenance"):
        load_compositional_tasks(
            other_root,
            other_manifest,
            expected_manifest_sha256=_sha256(other_manifest),
        )


def test_complete_group_split_helpers_never_split_the_statistical_unit() -> None:
    rows = []
    for animal in range(2):
        for session in range(2):
            for block in range(2):
                for trial in range(2):
                    rows.append(
                        {
                            "animal_id": f"animal-{animal}",
                            "session_id": f"animal-{animal}:session-{session}",
                            "trial_id": f"{animal}-{session}-{block}-{trial}",
                            "trial_order": block * 2 + trial,
                            "block_id": block,
                            "composition_id": (block + trial) % 2,
                            "cue": "left",
                            "behavior": "correct",
                            "stimulus_id": trial,
                            "action_id": trial,
                        }
                    )
    trials = pd.DataFrame(rows)
    definitions = (
        (leave_one_block_out_splits, ["session_id", "block_id"]),
        (leave_one_session_out_splits, ["session_id"]),
        (leave_one_animal_out_splits, ["animal_id"]),
        (leave_one_composition_out_splits, ["composition_id"]),
    )
    for function, columns in definitions:
        splits = function(trials)
        assert splits
        for split in splits:
            train_groups = {
                tuple(row)
                for row in trials.iloc[split.train_indices][columns].itertuples(
                    index=False, name=None
                )
            }
            test_groups = {
                tuple(row)
                for row in trials.iloc[split.test_indices][columns].itertuples(
                    index=False, name=None
                )
            }
            assert train_groups.isdisjoint(test_groups)


def test_past_only_belief_never_reads_current_or_test_truth() -> None:
    trials = pd.DataFrame(
        {
            "session_id": ["s0"] * 12,
            "trial_id": [f"trial-{index}" for index in range(12)],
            "trial_order": np.arange(12),
            "cue": np.tile(["cue-a", "cue-b"], 6),
            "choice": np.tile(["left", "right"], 6),
            "reaction_time": np.linspace(0.2, 0.8, 12),
            "composition_id": np.tile(["task-a", "task-b"], 6),
        }
    )
    estimator = PastOnlyBeliefEstimator(
        cue_columns=("cue",),
        behavior_columns=("choice", "reaction_time"),
        numeric_columns=("reaction_time",),
        ridge=0.1,
    ).fit(trials, range(8), label_column="composition_id")
    baseline = estimator.predict(trials, range(8, 12))
    changed_truth = trials.copy()
    changed_truth.loc[8:, "composition_id"] = "never-read-test-truth"
    changed = estimator.predict(changed_truth, range(8, 12))
    np.testing.assert_array_equal(baseline.probabilities, changed.probabilities)
    assert baseline.receipt.feature_lag_trials == 1
    assert not baseline.receipt.uses_current_trial_fields
    assert not baseline.receipt.uses_future_trials
    assert not baseline.receipt.accessed_test_truth
    assert baseline.receipt.fit_trial_keys == tuple(
        ("s0", f"trial-{index}") for index in range(8)
    )
    with pytest.raises(ValueError, match="truth/context/task"):
        PastOnlyBeliefEstimator(
            cue_columns=("true_context",),
            behavior_columns=("choice",),
        )


def test_past_only_belief_fit_checkpoint_is_independent_of_heldout_history() -> None:
    trials = pd.DataFrame(
        {
            "session_id": ["s0"] * 10,
            "trial_id": [f"trial-{index}" for index in range(10)],
            "trial_order": np.arange(10),
            "cue": np.tile(["cue-a", "cue-b"], 5),
            "choice": np.tile(["left", "right"], 5),
            "reaction_time": np.linspace(0.2, 0.9, 10),
            "composition_id": np.tile(["task-a", "task-b"], 5),
        }
    )
    train = np.asarray([0, 2, 4, 6, 8, 9])

    def fit(frame: pd.DataFrame) -> PastOnlyBeliefEstimator:
        return PastOnlyBeliefEstimator(
            cue_columns=("cue",),
            behavior_columns=("choice", "reaction_time"),
            numeric_columns=("reaction_time",),
            ridge=0.1,
        ).fit(frame, train, label_column="composition_id")

    baseline = fit(trials)
    changed = trials.copy()
    heldout = np.setdiff1d(np.arange(len(trials)), train)
    changed.loc[heldout, "session_id"] = "heldout-session"
    changed.loc[heldout, "trial_id"] = [
        f"changed-heldout-{index}" for index in range(len(heldout))
    ]
    changed.loc[heldout, "trial_order"] = np.arange(100, 100 + len(heldout))
    changed.loc[heldout, "cue"] = "unseen-heldout-cue"
    changed.loc[heldout, "choice"] = "unseen-heldout-choice"
    changed.loc[heldout, "reaction_time"] = 1_000.0
    changed.loc[heldout, "composition_id"] = "never-fit-heldout-truth"
    refit = fit(changed)

    assert refit.checkpoint_sha256_ == baseline.checkpoint_sha256_
    assert refit.fit_design_sha256_ == baseline.fit_design_sha256_
    assert refit.categories_ == baseline.categories_
    assert refit.numeric_mean_ == baseline.numeric_mean_
    assert refit.numeric_scale_ == baseline.numeric_scale_
    np.testing.assert_array_equal(refit.coefficients_, baseline.coefficients_)
    receipt = baseline.predict(trials, [1]).receipt
    assert receipt.fit_history_scope == "training_rows_only_within_group"
    assert receipt.fit_preprocessing_heldout_independent


def test_required_subject_session_block_and_composition_fields_fail_closed() -> None:
    invalid = pd.DataFrame(
        {
            "animal_id": ["animal"],
            "session_id": [""],
            "trial_id": ["trial"],
            "trial_order": [0],
            "block_id": [0],
            "composition_id": ["task"],
            "cue": ["cue"],
            "behavior": ["choice"],
            "stimulus_id": ["stimulus"],
            "action_id": ["action"],
        }
    )
    for function in (
        leave_one_block_out_splits,
        leave_one_session_out_splits,
        leave_one_animal_out_splits,
        leave_one_composition_out_splits,
    ):
        with pytest.raises(CompositionalTasksDataError, match="empty"):
            function(invalid)
