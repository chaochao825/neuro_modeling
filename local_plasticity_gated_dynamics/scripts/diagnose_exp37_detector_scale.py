#!/usr/bin/env python3
"""Post-hoc, development-only detector-scale diagnostic for Exp37.

This script cannot upgrade the preregistered Exp37 claim.  It reconstructs the
selected detector on the development session only and records whether the
frozen alarm grid was numerically reachable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config, seed_list
from experiments.exp37_core50_change_aware_prefix import (
    _prototypes,
    _store,
    validate_feature_cache,
    validate_preregistration,
)
from src.data.core50_streaming import prepare_core50_task
from src.models.bocpd_prefix import BOCPDConfig, bocpd_prefix_accumulator


QUANTILES = (0.5, 0.9, 0.99, 0.999)


def summarize_scores(values: ArrayLike, *, threshold: float) -> dict[str, Any]:
    """Return a finite, JSON-safe summary without selecting a new threshold."""

    scores = np.asarray(values, dtype=np.float64)
    limit = float(threshold)
    if scores.ndim != 1 or scores.size < 1:
        raise ValueError("scores must be a nonempty vector")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be finite probabilities")
    if not np.isfinite(limit) or not 0.0 < limit <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    return {
        "n": int(scores.size),
        "maximum": float(np.max(scores)),
        "quantiles": {
            str(value): float(result)
            for value, result in zip(
                QUANTILES, np.quantile(scores, QUANTILES), strict=True
            )
        },
        "n_at_or_above_frozen_minimum": int(np.sum(scores >= limit)),
    }


def _one_path(paths: Iterable[Path], *, label: str) -> Path:
    selected = tuple(paths)
    if len(selected) != 1:
        raise RuntimeError(f"expected one {label}, found {len(selected)}")
    return selected[0]


def diagnose(config_path: Path, results_root: Path) -> dict[str, Any]:
    """Reconstruct frozen selected detectors on s2 without external outcomes."""

    validate_preregistration(config_path)
    config = load_json_config(config_path)
    feature_cache = validate_feature_cache(config)
    store = _store(config)
    prototypes = _prototypes(store, config)
    minimum_threshold = float(min(config["bocpd_grid"]["alarm_threshold"]))
    all_scores: list[float] = []
    switch_scores: list[float] = []
    post_switch_scores: list[float] = []
    selections: list[dict[str, Any]] = []
    development_session = str(config["development_sessions"][0])
    post_window = int(config["stream"]["post_switch_window"])

    for seed in seed_list(config["seeds"]):
        selection_path = _one_path(
            results_root.glob(
                f"runs/**/seed_{seed}/*/selected_hyperparameters.json"
            ),
            label=f"selection for seed {seed}",
        )
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        if selected.get("used_external_labels") is not False:
            raise RuntimeError("diagnostic encountered external-label selection")
        detector = BOCPDConfig(**selected["detector"])
        selections.append(
            {
                "seed": seed,
                "temperature": float(selected["temperature"]),
                "detector": selected["detector"],
            }
        )
        for task_index in range(int(config["n_development_tasks"])):
            _, stream = prepare_core50_task(
                store,
                prototypes=prototypes,
                session_id=development_session,
                seed=seed,
                task_index=task_index,
                temperature=float(selected["temperature"]),
                stream_config=config["stream"],
            )
            trace = bocpd_prefix_accumulator(
                stream.evidence,
                stream_ids=stream.stream_ids,
                config=detector,
                mode="hard_reset",
            )
            all_scores.extend(trace.detector_scores.tolist())
            for index in np.flatnonzero(stream.switch_flags):
                switch_scores.append(float(trace.detector_scores[index]))
                post_switch_scores.extend(
                    trace.detector_scores[index : index + post_window].tolist()
                )

    return {
        "analysis_status": "post_hoc_failure_diagnostic_only",
        "claim_upgrade_allowed": False,
        "external_outcomes_used": False,
        "development_session": development_session,
        "frozen_minimum_alarm_threshold": minimum_threshold,
        "feature_cache": feature_cache,
        "selected_hyperparameters": selections,
        "all_development_hidden_frames": summarize_scores(
            all_scores, threshold=minimum_threshold
        ),
        "true_switch_frames": summarize_scores(
            switch_scores, threshold=minimum_threshold
        ),
        "post_switch_window_frames": summarize_scores(
            post_switch_scores, threshold=minimum_threshold
        ),
        "interpretation": (
            "The frozen detector score did not reach the minimum registered "
            "alarm threshold on development data; this diagnoses threshold/model "
            "scale mismatch and cannot rescue or replace the preregistered verdict."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prospective/exp37_core50_change_aware_prefix.json"),
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = diagnose(args.config.expanduser().resolve(), args.results_root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
