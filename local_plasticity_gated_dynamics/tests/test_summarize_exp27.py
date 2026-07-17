from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from experiments import exp27_low_dimensional_actuator_selector as exp27
from scripts.summarize_exp27 import (
    SelectorCollection,
    _canonical_sha256,
    _matching_attempt,
    collect_exp27_runs,
    write_exp27_summary,
)
from src.analysis.actuator_selector_metrics import SELECTOR_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _smoke_config() -> dict[str, object]:
    return json.loads(
        (
            PROJECT_ROOT
            / "configs"
            / "smoke"
            / "exp27_low_dimensional_actuator_selector.json"
        ).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def completed_smoke_runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    results_root = tmp_path_factory.mktemp("exp27_provenance_runs")
    config = _smoke_config()
    for seed in config["seeds"]:
        exp27.run_seed(
            config,
            int(seed),
            results_root,
            run_label="exp27-provenance-test",
        )
    return results_root


def _copy_completed_runs(source: Path, destination: Path) -> Path:
    copied = destination / "results"
    shutil.copytree(source, copied)
    return copied


def _attempt_path(results_root: Path, seed: int) -> Path:
    seed_root = (
        results_root
        / "runs"
        / exp27.EXPERIMENT
        / f"seed_{seed:04d}"
    )
    attempts = list(seed_root.iterdir())
    assert len(attempts) == 1
    return attempts[0]


def _read_metrics(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _refresh_receipt_hash(receipt: dict[str, object]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    digest = _canonical_sha256(payload)
    receipt["receipt_sha256"] = digest
    return digest


def _collect_smoke(results_root: Path) -> SelectorCollection:
    return collect_exp27_runs(
        results_root,
        config_path=(
            PROJECT_ROOT
            / "configs"
            / "smoke"
            / "exp27_low_dimensional_actuator_selector.json"
        ),
        run_label="exp27-provenance-test",
    )


def _smoke_records() -> pd.DataFrame:
    modes = ("routing", "gain", "low_rank")
    candidates = (
        (0.60, 0.55, 0.70),
        (0.50, 0.80, 0.70),
        (0.60, 0.55, 0.70),
        (0.50, 0.80, 0.70),
    )
    rows: list[dict[str, object]] = []
    for seed in (9000, 9001):
        for generator, utilities in enumerate(candidates):
            oracle_index = max(range(3), key=utilities.__getitem__)
            oracle_mode = modes[oracle_index]
            for selector in SELECTOR_MODES:
                if selector in {"oracle", "local_three_factor"}:
                    selected = oracle_mode
                elif selector == "fixed_best":
                    selected = "low_rank"
                else:
                    selected = "routing"
                rows.append(
                    {
                        "seed": seed,
                        "outer_seed": seed,
                        "source_seed": seed,
                        "generator_id": f"g{generator}",
                        "generator_split": "heldout",
                        "strict_unseen_composition": True,
                        "primary_endpoint_eligible": True,
                        "composition_overlap_secondary": False,
                        "selector": selector,
                        "mode_selected": selected,
                        "oracle_mode": oracle_mode,
                        "fixed_best_mode": "low_rank",
                        "utility": utilities[modes.index(selected)],
                        "candidate_routing_utility": utilities[0],
                        "candidate_gain_utility": utilities[1],
                        "candidate_low_rank_utility": utilities[2],
                        "train_mean_candidate_routing_utility": 0.55,
                        "train_mean_candidate_gain_utility": 0.60,
                        "train_mean_candidate_low_rank_utility": 0.70,
                        "training_source_seed_count": 1,
                        "outer_seed_excluded_from_training": True,
                        "train_split": "discovery",
                        "train_endpoint": "validation_balanced_accuracy",
                        "test_endpoint": "test_balanced_accuracy",
                        "plasticity_l1": (
                            1.0 if selector == "local_three_factor" else 0.0
                        ),
                        "plasticity_l2": (
                            0.5 if selector == "local_three_factor" else 0.0
                        ),
                        "status": "complete",
                    }
                )
    return pd.DataFrame(rows)


def test_write_exp27_summary_retains_smoke_scope_gate(tmp_path: Path) -> None:
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs"
            / "smoke"
            / "exp27_low_dimensional_actuator_selector.json"
        ).read_text(encoding="utf-8")
    )
    collection = SelectorCollection(
        raw=_smoke_records(),
        config=config,
        config_sha256="a" * 64,
        source_receipt={
            "raw_metrics_sha256": "b" * 64,
            "conclusion_sha256": "c" * 64,
        },
        source_receipt_sha256="d" * 64,
        attempts=(),
        run_git_commit="e" * 40,
        run_git_tree="f" * 40,
        runtime_identity={"python": "3.11.9", "packages": {"numpy": "1.26.4"}},
        run_label="unit-smoke",
    )

    conclusion = write_exp27_summary(collection, tmp_path, make_figure=False)

    assert conclusion.conclusion == "inconclusive"
    assert conclusion.complete_primary_coverage
    for name in (
        "raw_metrics.csv.gz",
        "seed_endpoints.csv",
        "summary.csv",
        "conclusion.json",
        "provenance.json",
        "report.md",
    ):
        assert (tmp_path / name).is_file()
    payload = json.loads((tmp_path / "conclusion.json").read_text(encoding="utf-8"))
    assert payload["statistics_unit"] == "outer_seed"
    assert payload["confirmatory_eligible"] is False
    assert "task-matched" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_matching_attempt_refuses_label_ambiguity(tmp_path: Path) -> None:
    for name in ("attempt-a", "attempt-b"):
        attempt = tmp_path / name
        attempt.mkdir()
        (attempt / "status.json").write_text(
            json.dumps({"status": "complete", "run_label": "same"}),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="2 attempts"):
        _matching_attempt(tmp_path, run_label="same")


def test_collector_replays_locked_source_and_selector_decisions(
    completed_smoke_runs: Path,
) -> None:
    collection = _collect_smoke(completed_smoke_runs)

    assert collection.raw.shape[0] == 2 * 12 * 4
    assert len(collection.attempts) == 2


def test_collector_rejects_coordinated_source_utility_tampering(
    completed_smoke_runs: Path,
    tmp_path: Path,
) -> None:
    results_root = _copy_completed_runs(completed_smoke_runs, tmp_path)
    metrics_path = _attempt_path(results_root, 9000) / "metrics.jsonl"
    rows = _read_metrics(metrics_path)
    target = str(rows[0]["generator_id"])
    for row in rows:
        if str(row["generator_id"]) != target:
            continue
        for column in (
            "candidate_routing_utility",
            "candidate_gain_utility",
            "candidate_low_rank_utility",
            "utility",
            "oracle_utility",
            "fixed_best_utility",
        ):
            row[column] = float(row[column]) - 0.01
    _write_metrics(metrics_path, rows)

    with pytest.raises(ValueError, match="reconstructed source test utilities"):
        _collect_smoke(results_root)


def test_collector_rejects_asserted_training_seed_tampering(
    completed_smoke_runs: Path,
    tmp_path: Path,
) -> None:
    results_root = _copy_completed_runs(completed_smoke_runs, tmp_path)
    metrics_path = _attempt_path(results_root, 9000) / "metrics.jsonl"
    rows = _read_metrics(metrics_path)
    for row in rows:
        row["training_source_seeds"] = [9000]
    _write_metrics(metrics_path, rows)

    with pytest.raises(ValueError, match="reconstructed training source seeds"):
        _collect_smoke(results_root)


@pytest.mark.parametrize(
    ("column", "expected_error"),
    (
        ("noise_std", "reconstructed fold mismatch: noise_std"),
        (
            "train_mean_candidate_gain_utility",
            "reconstructed training utility means",
        ),
    ),
)
def test_collector_rejects_source_metadata_or_train_mean_tampering(
    completed_smoke_runs: Path,
    tmp_path: Path,
    column: str,
    expected_error: str,
) -> None:
    results_root = _copy_completed_runs(completed_smoke_runs, tmp_path)
    metrics_path = _attempt_path(results_root, 9000) / "metrics.jsonl"
    rows = _read_metrics(metrics_path)
    for row in rows:
        row[column] = float(row[column]) + 0.01
    _write_metrics(metrics_path, rows)

    with pytest.raises(ValueError, match=expected_error):
        _collect_smoke(results_root)


def test_collector_rejects_self_consistent_normalizer_receipt_tampering(
    completed_smoke_runs: Path,
    tmp_path: Path,
) -> None:
    results_root = _copy_completed_runs(completed_smoke_runs, tmp_path)
    attempt = _attempt_path(results_root, 9000)
    receipt_path = attempt / "normalizer_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["center"][0] = float(receipt["center"][0]) + 0.25
    receipt["fingerprint"] = _canonical_sha256(
        {
            "feature_names": list(receipt["feature_names"][:-1]),
            "mean": receipt["center"],
            "scale": receipt["scale"],
            "n_fit_samples": receipt["train_n"],
        }
    )
    receipt_digest = _refresh_receipt_hash(receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = attempt / "metrics.jsonl"
    rows = _read_metrics(metrics_path)
    for row in rows:
        row["normalizer_receipt_sha256"] = receipt_digest
        row["normalizer_fit_fingerprint"] = receipt["fingerprint"]
    _write_metrics(metrics_path, rows)

    with pytest.raises(ValueError, match="normalizer center differs"):
        _collect_smoke(results_root)


def test_collector_rejects_self_consistent_decision_receipt_tampering(
    completed_smoke_runs: Path,
    tmp_path: Path,
) -> None:
    results_root = _copy_completed_runs(completed_smoke_runs, tmp_path)
    attempt = _attempt_path(results_root, 9000)
    receipt_path = attempt / "selector_training_receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    local = receipt["local_three_factor"]
    probabilities = np.asarray(local["test_probabilities"], dtype=np.float64)
    probabilities[0] = 0.99 * probabilities[0] + 0.01 / len(exp27.CANDIDATE_MODES)
    probabilities[0] /= np.sum(probabilities[0])
    assert np.argmax(probabilities[0]) == np.argmax(
        np.asarray(local["test_probabilities"], dtype=np.float64)[0]
    )
    local["test_probabilities"] = probabilities.tolist()
    local["test_decision_fingerprint"] = exp27._decision_fingerprint(
        local["test_generator_ids"], probabilities
    )
    local_digest = _refresh_receipt_hash(local)
    training_digest = _refresh_receipt_hash(receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics_path = attempt / "metrics.jsonl"
    rows = _read_metrics(metrics_path)
    target = str(local["test_generator_ids"][0])
    for row in rows:
        row["selector_training_receipts_sha256"] = training_digest
        if row["selector"] == "local_three_factor":
            row["selector_model_receipt_sha256"] = local_digest
            if str(row["generator_id"]) == target:
                row["selection_probability_routing"] = float(probabilities[0, 0])
                row["selection_probability_gain"] = float(probabilities[0, 1])
                row["selection_probability_low_rank"] = float(probabilities[0, 2])
    _write_metrics(metrics_path, rows)

    with pytest.raises(ValueError, match="training receipt differs"):
        _collect_smoke(results_root)
