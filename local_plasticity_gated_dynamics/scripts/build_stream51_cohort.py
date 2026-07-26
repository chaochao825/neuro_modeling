"""Build the outcome-blind video split manifest from an official ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.stream51_streaming import (
    assign_stream51_video_splits,
    read_stream51_ordering,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cohort(
    ordering_path: str | Path,
    output_path: str | Path,
    *,
    split_salt: str,
    source_repo_commit: str,
) -> dict[str, object]:
    ordering = Path(ordering_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    payload = assign_stream51_video_splits(
        read_stream51_ordering(ordering), salt=split_salt
    )
    payload.update(
        {
            "source_ordering": ordering.name,
            "source_ordering_sha256": _sha256(ordering),
            "source_repo_commit": str(source_repo_commit),
            "outcome_fields_inspected": False,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordering", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--source-repo-commit", required=True)
    args = parser.parse_args()
    payload = build_cohort(
        args.ordering,
        args.output,
        split_salt=args.split_salt,
        source_repo_commit=args.source_repo_commit,
    )
    print(json.dumps({key: payload[key] for key in ("n_classes", "n_videos", "split_counts")}, sort_keys=True))


if __name__ == "__main__":
    main()
