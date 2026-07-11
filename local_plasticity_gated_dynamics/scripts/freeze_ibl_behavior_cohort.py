"""Freeze a balanced, behavior-only IBL BWM cohort with full provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibl_behavior_cohort import (  # noqa: E402
    IBLBehaviorCohortCriteria,
    balanced_session_order,
    behavior_session_qc,
    default_trials_table_provenance,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bwm-repo", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--target-sessions", type=int, default=30)
    parser.add_argument("--min-animals", type=int, default=10)
    parser.add_argument("--max-sessions-per-animal", type=int, default=3)
    parser.add_argument("--min-raw-trials", type=int, default=400)
    parser.add_argument("--min-analysis-trials", type=int, default=300)
    parser.add_argument("--min-valid-choices", type=int, default=300)
    parser.add_argument("--min-valid-feedback", type=int, default=300)
    parser.add_argument("--min-context-switches", type=int, default=8)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parser().parse_args()
    criteria = IBLBehaviorCohortCriteria(
        target_sessions=args.target_sessions,
        min_animals=args.min_animals,
        max_sessions_per_animal=args.max_sessions_per_animal,
        min_raw_trials=args.min_raw_trials,
        min_analysis_trials=args.min_analysis_trials,
        min_valid_choices=args.min_valid_choices,
        min_valid_feedback=args.min_valid_feedback,
        min_context_switches=args.min_context_switches,
    )
    bwm_repo = Path(args.bwm_repo).resolve()
    if not (bwm_repo / "brainwidemap").is_dir():
        raise FileNotFoundError("--bwm-repo must contain the brainwidemap package")
    sys.path.insert(0, str(bwm_repo))
    from brainwidemap import bwm_query, load_trials_and_mask  # type: ignore
    from one.api import ONE

    cohort_dir = Path(args.output_root).resolve() / args.cohort_id
    if cohort_dir.exists():
        raise FileExistsError(
            f"immutable cohort output already exists: {cohort_dir}; choose a new cohort-id"
        )
    cohort_dir.mkdir(parents=True)
    sessions_dir = cohort_dir / "sessions"
    sessions_dir.mkdir()
    source_commit = subprocess.run(
        ["git", "-C", str(bwm_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = {
        "schema_version": "1.0",
        "cohort_id": args.cohort_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_seed": 0,
        "selection_algorithm": "subject_round_robin_then_date_eid_v1",
        "bwm_query_source": "fixtures/2023_12_bwm_release.csv",
        "bwm_repository": "https://github.com/int-brain-lab/paper-brain-wide-map",
        "bwm_repository_commit": source_commit,
        "one_base_url": "https://openalyx.internationalbrainlab.org",
        "download_scope": "_ibl_trials.table.pqt_only",
        "criteria": asdict(criteria),
    }
    (cohort_dir / "cohort_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    candidates = balanced_session_order(bwm_query())
    candidates.to_csv(cohort_dir / "planned_sessions.csv", index=False)
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=Path(args.cache_dir).resolve(),
    )
    manifest_rows: list[dict[str, object]] = []
    eligible = 0
    animals: set[str] = set()
    eligible_by_animal: dict[str, int] = {}
    jsonl_path = cohort_dir / "cohort_manifest.jsonl"
    for candidate in candidates.to_dict("records"):
        if (
            eligible >= criteria.target_sessions
            and len(animals) >= criteria.min_animals
        ):
            break
        subject = str(candidate["subject"])
        if eligible_by_animal.get(subject, 0) >= criteria.max_sessions_per_animal:
            continue
        base = {
            **candidate,
            "cohort_id": args.cohort_id,
            "bwm_repository_commit": source_commit,
            "status": "failed",
            "eligible": False,
            "error_type": "",
            "error": "",
        }
        try:
            eid = str(candidate["eid"])
            details = one.list_datasets(eid, details=True)
            provenance = default_trials_table_provenance(details)
            revision = str(provenance["dataset_revision"]) or None
            source_table = one.load_dataset(
                eid,
                "_ibl_trials.table.pqt",
                collection="alf",
                revision=revision,
            )
            if not isinstance(source_table, pd.DataFrame):
                raise TypeError("ONE trials table did not load as a DataFrame")
            table, official_mask = load_trials_and_mask(
                one,
                eid,
                min_rt=0.08,
                max_rt=2.0,
                nan_exclude="default",
                exclude_unbiased=False,
                exclude_nochoice=False,
                revision=revision,
            )
            if not isinstance(table, pd.DataFrame):
                raise TypeError("official BWM trial loader did not return a DataFrame")
            qc = behavior_session_qc(
                table,
                criteria,
                official_bwm_mask=np.asarray(official_mask, dtype=bool),
            )
            qc["source_dataset_trial_count"] = int(len(source_table))
            qc["official_loader_trial_count"] = int(len(table))
            qc["official_trial_mask_protocol"] = (
                "load_trials_and_mask_min_rt_0p08_max_rt_2_nan_default"
            )
            analysis_frame = qc.pop("analysis_frame")
            dataset_qc_eligible = str(provenance["dataset_qc"]).upper() in {
                "PASS",
                "WARNING",
            }
            qc["dataset_qc_eligible"] = dataset_qc_eligible
            if not dataset_qc_eligible:
                qc["eligible"] = False
                qc["exclusion_reason"] = ";".join(
                    item for item in (str(qc["exclusion_reason"]), "dataset_qc") if item
                )
            base.update(provenance, **qc)
            if bool(qc["eligible"]):
                session_dir = sessions_dir / eid
                session_dir.mkdir()
                table_path = session_dir / "trials.csv"
                analysis_frame.to_csv(table_path, index=False)
                base.update(
                    status="eligible",
                    compact_table=str(table_path.relative_to(cohort_dir)),
                    compact_table_sha256=_sha256(table_path),
                )
                eligible += 1
                animals.add(subject)
                eligible_by_animal[subject] = eligible_by_animal.get(subject, 0) + 1
            else:
                base["status"] = "excluded"
        except Exception as error:
            base.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
            )
        manifest_rows.append(base)
        with jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(base, sort_keys=True, default=str) + "\n")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(cohort_dir / "cohort_manifest.csv", index=False)
    summary = {
        **config,
        "attempted_sessions": len(manifest),
        "eligible_sessions": eligible,
        "eligible_animals": len(animals),
        "target_met": eligible >= criteria.target_sessions
        and len(animals) >= criteria.min_animals,
    }
    (cohort_dir / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    if not summary["target_met"]:
        raise RuntimeError(
            "cohort target was not met; preserved manifest contains every attempted failure"
        )


if __name__ == "__main__":
    main()
