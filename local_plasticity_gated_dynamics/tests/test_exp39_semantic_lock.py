from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_exp39_semantic_lock import audit
from src.analysis.exp39_semantic_lock import (
    _verify_hash_receipt,
    portable_tape_fingerprint,
    validate_exp39_semantic_lock,
    verify_sha256_manifest,
)
from src.tasks.factorized_uncertainty import (
    FactorialStreamConfig,
    generate_uncertainty_tape,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exp39_semantic_lock_replays_frozen_package() -> None:
    receipt = audit(
        ROOT / "results/exp39_factorized_uncertainty_prospective_v1",
        ROOT / "provenance/exp39_semantic_lock_20260727.json",
    )
    assert receipt["audit_status"] == "passed"
    assert receipt["n_seeds"] == 30
    assert receipt["n_block_rows"] == 15360
    assert receipt["n_selection_rows"] == 2490
    assert receipt["aggregate_shard_identity"] == "passed"
    assert receipt["fit_test_tape_semantic_replay"] == "passed"
    assert receipt["fit_test_tape_exact_replay"] in {
        "passed",
        "platform_float_tail_variance_disclosed",
    }
    if receipt["fit_test_tape_exact_replay"] != "passed":
        assert receipt["exact_tape_digest_mismatches"]
    assert receipt["selection_argmin_replay"] == "passed"
    assert receipt["full_summary_replay"] == "passed"
    assert receipt["summary_derived_tables_replay"] == "passed"
    assert receipt["numeric_semantic_fingerprint"] == "passed"
    assert receipt["replay_environment_is_original_formal_environment"] is False
    assert receipt["replay_environment"]["scope"] == (
        "current_replay_only_not_original_formal_execution"
    )
    assert receipt["verified_hash_receipt_files"] == {
        "scientific_implementation": 11,
        "outcome_blind_parallel_execution": 4,
    }
    assert receipt["frozen_replay"]["registered_verdict"] == "support"
    assert receipt["claim_boundary"] == {
        "registered_average_unseen_composition_utility": "support",
        "uniform_cellwise_composition_utility": "oppose",
        "clean_three_factor_parameter_decomposition": "oppose",
        "faster_post_switch_release_than_seen_mode_imm": "oppose",
        "real_behavior_or_neural_utility": "inconclusive",
    }


def test_manifest_verifier_rejects_tampering_and_path_escape(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("original\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(f"{digest}  artifact.txt\n", encoding="utf-8")
    assert verify_sha256_manifest(manifest, project_root=tmp_path) == {
        "artifact.txt": digest
    }
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        verify_sha256_manifest(manifest, project_root=tmp_path)

    manifest.write_text(f"{digest}  ../artifact.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        verify_sha256_manifest(manifest, project_root=tmp_path)


def test_manifest_verifier_rejects_duplicates(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("content", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(
        f"{digest}  artifact.txt\n{digest}  artifact.txt\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        verify_sha256_manifest(manifest, project_root=tmp_path)


def test_hash_receipt_rejects_path_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "receipt.json"
    outside.write_text('{"files": {}}\n', encoding="utf-8")
    contract = {
        "path": "../receipt.json",
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    with pytest.raises(RuntimeError, match="unsafe"):
        _verify_hash_receipt(project.resolve(), contract)


def test_semantic_lock_rejects_an_unbound_result_directory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not the formal result"):
        validate_exp39_semantic_lock(
            ROOT,
            tmp_path,
            ROOT / "provenance/exp39_semantic_lock_20260727.json",
        )


def test_portable_tape_fingerprint_ignores_only_float_tail_noise() -> None:
    tape = generate_uncertainty_tape(
        seed=4,
        split="portable-test",
        cells=("000", "001"),
        config=FactorialStreamConfig(
            block_length=4, blocks_per_sequence=2, n_sequences=1
        ),
    )
    tail_changed = np.array(tape.observations, copy=True)
    tail_changed[0] = np.nextafter(tail_changed[0], np.inf)
    meaningfully_changed = np.array(tape.observations, copy=True)
    meaningfully_changed[0] += 1e-6
    original = portable_tape_fingerprint(tape, decimals=10)
    assert portable_tape_fingerprint(
        replace(tape, observations=tail_changed), decimals=10
    ) == original
    assert portable_tape_fingerprint(
        replace(tape, observations=meaningfully_changed), decimals=10
    ) != original
