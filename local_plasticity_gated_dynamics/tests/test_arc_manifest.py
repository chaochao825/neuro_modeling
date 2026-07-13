from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.data.arc_manifest import (
    ARCManifestError,
    validate_arc_acquisition_receipts,
    validate_arc_source_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "arc"
    for split in ("training", "evaluation"):
        directory = root / split
        directory.mkdir(parents=True)
        (directory / f"{split}.json").write_text(
            '{"test": [{"input": [[0]], "output": [[1]]}], '
            '"train": [{"input": [[1]], "output": [[0]]}]}\n',
            encoding="utf-8",
        )
    (root / "LICENSE").write_text("Apache License fixture\n", encoding="utf-8")
    manifest = tmp_path / "arc.sha256"
    entries = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.json"))
    manifest.write_text(
        "".join(f"{_sha(root / relative)}  {relative}\n" for relative in entries),
        encoding="utf-8",
        newline="\n",
    )
    config: dict[str, object] = {
        "expected_manifest_sha256": _sha(manifest),
        "expected_license_sha256": _sha(root / "LICENSE"),
        "expected_split_counts": {"training": 1, "evaluation": 1},
    }
    return root, manifest, config


def test_arc_manifest_verifies_every_source_file_and_license(tmp_path: Path) -> None:
    root, manifest, config = _fixture(tmp_path)
    receipt = validate_arc_source_manifest(root, manifest, **config)

    assert receipt.source_tree_verified is True
    assert receipt.n_files == 2
    assert receipt.split_counts == {"training": 1, "evaluation": 1}
    assert receipt.to_dict()["manifest_sha256"] == config["expected_manifest_sha256"]


def test_arc_manifest_rejects_tampering_and_unlisted_json(tmp_path: Path) -> None:
    root, manifest, config = _fixture(tmp_path)
    (root / "evaluation" / "evaluation.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ARCManifestError, match="file SHA-256 mismatch"):
        validate_arc_source_manifest(root, manifest, **config)

    root, manifest, config = _fixture(tmp_path / "extra")
    (root / "evaluation" / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ARCManifestError, match="tree coverage mismatch"):
        validate_arc_source_manifest(root, manifest, **config)


def test_arc_manifest_rejects_unsafe_or_reordered_entries(tmp_path: Path) -> None:
    root, manifest, config = _fixture(tmp_path)
    manifest.write_text(f"{'0' * 64}  training/../escape.json\n", encoding="utf-8")
    config["expected_manifest_sha256"] = _sha(manifest)
    with pytest.raises(ARCManifestError, match="invalid ARC source manifest line"):
        validate_arc_source_manifest(root, manifest, **config)


def test_arc_acquisition_receipts_bind_source_identity_and_manifest(
    tmp_path: Path,
) -> None:
    source = {
        "commit": "fixture-revision",
        "license": "Apache-2.0",
        "name": "ARC-fixture",
        "url": "https://example.test/ARC",
        "splits": {"training": {"tasks": 1}, "evaluation": {"tasks": 1}},
    }
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps({"datasets": [source]}, sort_keys=True), encoding="utf-8"
    )
    source_manifest_sha = "a" * 64
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "arc": [source],
                "arc_manifest_sha256": {
                    "ARC-fixture": source_manifest_sha,
                    "validation": _sha(validation),
                },
                "created_utc": "2026-01-01T00:00:00Z",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    receipt = validate_arc_acquisition_receipts(
        acquisition,
        validation,
        expected_acquisition_manifest_sha256=_sha(acquisition),
        expected_validation_sha256=_sha(validation),
        dataset_name="ARC-fixture",
        revision="fixture-revision",
        source_url="https://example.test/ARC",
        license_name="Apache-2.0",
        source_manifest_sha256=source_manifest_sha,
        expected_split_counts={"training": 1, "evaluation": 1},
    )
    assert receipt["source_acquisition_verified"] is True
    assert receipt["source_manifest_sha256"] == source_manifest_sha
