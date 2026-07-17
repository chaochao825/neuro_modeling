"""Hash-locked independent Exp26 source panel for downstream Exp28 tests.

This module deliberately performs no selector fitting or statistical inference.
It reuses the frozen Exp26 generator and task-matched actuator implementation on
independent carrier/data seeds 30--59.  The old formal configuration, generator
manifest, train-only budget preflight, and every critical Exp26 implementation
file are hash bound before an attempt may start.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp26_actuator_phase_diagram as exp26
from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.actuator_manifest import GeneratorCell, manifest_hash
from src.utils.artifacts import ExperimentRun


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp28_exp26_independent_source_panel"
PROTOCOL_VERSION = "exp28_exp26_independent_source_v1"
EVIDENCE_SCHEMA_VERSION = "exp28_independent_source_evidence_v1"
PROFILE = "independent_test"
EXPECTED_SEEDS = tuple(range(30, 60))
EXPECTED_GENERATORS = 88
EXPECTED_MODES = tuple(exp26.MODES)
EXPECTED_ROWS_PER_SEED = EXPECTED_GENERATORS * len(EXPECTED_MODES)
EXPECTED_PANEL_ROWS = len(EXPECTED_SEEDS) * EXPECTED_ROWS_PER_SEED
REQUIRED_RUN_LABEL = "exp28-exp26-independent-source-v1"
SOURCE_CONFIG_RELATIVE_PATH = "configs/formal/exp26_actuator_phase_diagram.json"
SOURCE_PREFLIGHT_RELATIVE_PATH = (
    "results/exp26_actuator_matching_formal_v2_e08beaf/preflight"
)
SOURCE_PUBLISHED_MANIFEST_RELATIVE_PATH = (
    "results/exp26_actuator_matching_formal_v2_e08beaf/MANIFEST.sha256"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_RUNTIME_CONFIG_KEYS = {
    "config_path",
    "evidence_provenance",
    "experiment",
    "run_label",
    "seed",
}
_COPIED_SOURCE_KEYS = (
    "training_algorithm",
    "used_autograd",
    "used_bptt",
    "manifest",
    "carrier",
    "task",
    "actuator",
    "analysis",
)
_ROW_EVIDENCE_FIELDS = (
    "source_panel_evidence_schema",
    "independent_config_sha256",
    "independent_config_file_sha256",
    "source_contract_sha256",
    "source_exp26_config_sha256",
    "source_exp26_config_file_sha256",
    "source_exp26_manifest_sha256",
    "source_exp26_preflight_receipt_sha256",
    "source_exp26_critical_code_sha256",
    "run_git_commit",
    "run_git_tree",
    "run_git_dirty",
    "run_python_version",
    "run_numpy_version",
    "run_scipy_version",
    "run_scikit_learn_version",
    "run_pandas_version",
    "run_statsmodels_version",
    "run_label",
    "source_only",
    "standalone_inference_permitted",
)


@dataclass(frozen=True)
class SourceContract:
    """Validated immutable identities needed by one independent attempt."""

    independent_config_sha256: str
    independent_config_file_sha256: str
    source_contract_sha256: str
    source_config_sha256: str
    source_config_file_sha256: str
    source_manifest_sha256: str
    source_preflight_receipt_sha256: str
    source_critical_code_sha256: str
    current_git: Mapping[str, object]
    runtime_versions: Mapping[str, str]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only attempt-specific provenance from an Exp28 config."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return {
        str(key): value
        for key, value in config.items()
        if str(key) not in _RUNTIME_CONFIG_KEYS
    }


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    return _canonical_sha256(canonical_config_payload(config))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read hash-bound JSON object {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"hash-bound JSON must be an object: {path}")
    return dict(value)


def _config_source_file(config: Mapping[str, Any]) -> Path:
    value = config.get("config_path")
    if not isinstance(value, str) or not value:
        raise ValueError("independent source config requires config_path provenance")
    path = Path(value)
    if not path.is_file():
        raise ValueError("independent source config_path is not a regular file")
    source = _read_object(path)
    if canonical_config_payload(source) != canonical_config_payload(config):
        raise ValueError("runtime independent source config differs from its file")
    return path


def _exact_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_runtime_policy(
    config: Mapping[str, Any],
    runtime_versions: Mapping[str, object],
) -> dict[str, str]:
    policy = config.get("runtime_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("independent source config lacks runtime_policy")
    expected_python = policy.get("python_major_minor")
    python_version = runtime_versions.get("python")
    if (
        expected_python != "3.11"
        or not isinstance(python_version, str)
        or not python_version.startswith("3.11.")
    ):
        raise ValueError("independent source panel requires Python 3.11")
    expected_distributions = [
        "numpy",
        "scipy",
        "pandas",
        "scikit_learn",
        "statsmodels",
    ]
    if (
        policy.get("required_distributions")
        != ["numpy", "scipy", "pandas", "scikit_learn", "statsmodels"]
        or policy.get("exact_match_across_seeds") is not True
    ):
        raise ValueError("independent source runtime policy is not registered")
    resolved: dict[str, str] = {"python": python_version}
    for name in expected_distributions:
        value = runtime_versions.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"independent source runtime lacks {name}")
        resolved[name] = value
    return resolved


def _validate_current_git(current_git: Mapping[str, object]) -> dict[str, object]:
    commit = current_git.get("commit")
    tree = current_git.get("tree")
    dirty = current_git.get("dirty")
    if (
        not isinstance(commit, str)
        or not GIT_OBJECT_PATTERN.fullmatch(commit)
        or not isinstance(tree, str)
        or not GIT_OBJECT_PATTERN.fullmatch(tree)
        or dirty is not False
    ):
        raise ValueError(
            "independent source attempts require a clean identifiable git tree"
        )
    return {"commit": commit, "tree": tree, "dirty": False}


def validate_source_contract(
    config: Mapping[str, Any],
    *,
    current_git: Mapping[str, object] | None = None,
    runtime_versions: Mapping[str, object] | None = None,
) -> SourceContract:
    """Fail closed unless Exp28 is an exact independent replay of Exp26."""

    if (
        config.get("profile") != PROFILE
        or config.get("dev_only") is not False
        or config.get("protocol_version") != PROTOCOL_VERSION
        or config.get("required_run_label") != REQUIRED_RUN_LABEL
        or config.get("evidence_role") != "independent_test_source_only"
        or config.get("standalone_inference_permitted") is not False
    ):
        raise ValueError("independent source profile/protocol contract is invalid")
    seeds_value = config.get("seeds")
    if not isinstance(seeds_value, Sequence) or isinstance(
        seeds_value, (str, bytes)
    ):
        raise ValueError("independent source seeds must be a sequence")
    seeds = tuple(_exact_integer(seed, name="seed") for seed in seeds_value)
    if seeds != EXPECTED_SEEDS:
        raise ValueError("independent source panel requires exact seeds 30--59")
    if (
        config.get("training_algorithm")
        != "train_only_closed_form_task_matched_actuators"
        or config.get("used_autograd") is not False
        or config.get("used_bptt") is not False
    ):
        raise ValueError("independent source panel must retain non-BPTT Exp26 fitting")

    binding = config.get("source_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("independent source config lacks source_binding")
    if (
        binding.get("source_config_relative_path") != SOURCE_CONFIG_RELATIVE_PATH
        or binding.get("source_preflight_relative_path")
        != SOURCE_PREFLIGHT_RELATIVE_PATH
        or binding.get("source_published_manifest_relative_path")
        != SOURCE_PUBLISHED_MANIFEST_RELATIVE_PATH
    ):
        raise ValueError("independent source artifact paths are not registered")

    source_config_path = PROJECT_ROOT / SOURCE_CONFIG_RELATIVE_PATH
    source_config = _read_object(source_config_path)
    source_config_file_sha = _file_sha256(source_config_path)
    source_config_sha = exp26.canonical_config_sha256(source_config)
    if (
        source_config_file_sha != binding.get("source_config_file_sha256")
        or source_config_sha != binding.get("source_config_canonical_sha256")
    ):
        raise ValueError("old Exp26 formal config hash binding is invalid")
    for key in _COPIED_SOURCE_KEYS:
        if config.get(key) != source_config.get(key):
            raise ValueError(f"independent source {key} differs from frozen Exp26")
    actuator = config.get("actuator")
    if not isinstance(actuator, Mapping) or float(actuator.get("max_scale", -1.0)) != 128.0:
        raise ValueError("independent source max_scale must remain frozen at 128")

    cells = exp26._manifest(config)
    source_manifest_sha = manifest_hash(cells)
    if (
        len(cells) != EXPECTED_GENERATORS
        or len({cell.generator_id for cell in cells}) != EXPECTED_GENERATORS
        or source_manifest_sha != binding.get("source_manifest_sha256")
        or source_manifest_sha != source_config["manifest"]["expected_hash"]
    ):
        raise ValueError("old Exp26 generator manifest hash binding is invalid")

    preflight_path = PROJECT_ROOT / SOURCE_PREFLIGHT_RELATIVE_PATH
    receipt_sha = exp26._preflight_receipt_sha256(preflight_path)
    preflight_summary = _read_object(preflight_path / "preflight_summary.json")
    code_tree = _read_object(preflight_path / "code_tree_provenance.json")
    if (
        receipt_sha != binding.get("source_preflight_receipt_sha256")
        or preflight_summary.get("schema_version")
        != binding.get("source_preflight_schema")
        or preflight_summary.get("preflight_passed") is not True
        or preflight_summary.get("provenance_clean") is not True
        or preflight_summary.get("provenance_stable_during_run") is not True
        or preflight_summary.get("canonical_config_sha256") != source_config_sha
        or preflight_summary.get("manifest_hash") != source_manifest_sha
        or float(preflight_summary.get("derived_ceiling", -1.0)) != 128.0
        or int(preflight_summary.get("n_seeds", -1)) != 30
        or int(preflight_summary.get("n_generators", -1)) != EXPECTED_GENERATORS
    ):
        raise ValueError("old Exp26 train-only preflight binding is invalid")
    critical_file_hashes = binding.get("critical_file_sha256")
    if not isinstance(critical_file_hashes, Mapping):
        raise ValueError("independent source lacks critical Exp26 file hashes")
    observed_file_hashes = {
        str(relative): _file_sha256(PROJECT_ROOT / str(relative))
        for relative in critical_file_hashes
    }
    if (
        dict(critical_file_hashes) != observed_file_hashes
        or code_tree.get("critical_file_sha256") != observed_file_hashes
        or code_tree.get("critical_code_sha256")
        != binding.get("source_preflight_critical_code_sha256")
        or code_tree.get("git_commit") != binding.get("source_preflight_git_commit")
        or code_tree.get("git_tree") != binding.get("source_preflight_git_tree")
        or code_tree.get("stable_during_run") is not True
        or code_tree.get("git_dirty") is not False
    ):
        raise ValueError("critical Exp26 implementation hash binding is invalid")
    budget_policy = binding.get("budget_policy")
    if (
        not isinstance(budget_policy, Mapping)
        or budget_policy.get("selection_scope")
        != "exp26_seeds_0_29_training_blocks_only"
        or budget_policy.get("independent_panel_refit_permitted") is not False
        or float(budget_policy.get("frozen_max_scale", -1.0)) != 128.0
        or not np.isclose(
            float(budget_policy.get("source_required_scale_max", np.nan)),
            float(preflight_summary.get("registered_required_scale_max", np.nan)),
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise ValueError("independent source frozen budget policy is invalid")

    published_manifest = PROJECT_ROOT / SOURCE_PUBLISHED_MANIFEST_RELATIVE_PATH
    if (
        _file_sha256(published_manifest)
        != binding.get("source_published_manifest_file_sha256")
    ):
        raise ValueError("published Exp26 evidence manifest hash is invalid")

    config_path = _config_source_file(config)
    run_git = _validate_current_git(
        exp26.git_identity() if current_git is None else current_git
    )
    versions = _validate_runtime_policy(
        config,
        exp26.scientific_runtime_versions()
        if runtime_versions is None
        else runtime_versions,
    )
    return SourceContract(
        independent_config_sha256=canonical_config_sha256(config),
        independent_config_file_sha256=_file_sha256(config_path),
        source_contract_sha256=_canonical_sha256(binding),
        source_config_sha256=source_config_sha,
        source_config_file_sha256=source_config_file_sha,
        source_manifest_sha256=source_manifest_sha,
        source_preflight_receipt_sha256=receipt_sha,
        source_critical_code_sha256=str(
            binding["source_preflight_critical_code_sha256"]
        ),
        current_git=run_git,
        runtime_versions=versions,
    )


def build_evidence_provenance(
    contract: SourceContract,
    *,
    run_label: str,
) -> dict[str, object]:
    """Create one identity shared by the config, manifest, and every row."""

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_role": "independent_test_source_only",
        "standalone_inference_permitted": False,
        "independent_config_sha256": contract.independent_config_sha256,
        "independent_config_file_sha256": contract.independent_config_file_sha256,
        "source_contract_sha256": contract.source_contract_sha256,
        "source_exp26_config_sha256": contract.source_config_sha256,
        "source_exp26_config_file_sha256": contract.source_config_file_sha256,
        "source_exp26_manifest_sha256": contract.source_manifest_sha256,
        "source_exp26_preflight_receipt_sha256": (
            contract.source_preflight_receipt_sha256
        ),
        "source_exp26_critical_code_sha256": contract.source_critical_code_sha256,
        "run_git": dict(contract.current_git),
        "runtime_versions": dict(contract.runtime_versions),
        "run_label": run_label,
    }


def evidence_row_fields(provenance: Mapping[str, Any]) -> dict[str, object]:
    git = provenance["run_git"]
    versions = provenance["runtime_versions"]
    if not isinstance(git, Mapping) or not isinstance(versions, Mapping):
        raise ValueError("independent source provenance is malformed")
    return {
        "source_panel_evidence_schema": provenance["schema_version"],
        "independent_config_sha256": provenance["independent_config_sha256"],
        "independent_config_file_sha256": provenance[
            "independent_config_file_sha256"
        ],
        "source_contract_sha256": provenance["source_contract_sha256"],
        "source_exp26_config_sha256": provenance["source_exp26_config_sha256"],
        "source_exp26_config_file_sha256": provenance[
            "source_exp26_config_file_sha256"
        ],
        "source_exp26_manifest_sha256": provenance[
            "source_exp26_manifest_sha256"
        ],
        "source_exp26_preflight_receipt_sha256": provenance[
            "source_exp26_preflight_receipt_sha256"
        ],
        "source_exp26_critical_code_sha256": provenance[
            "source_exp26_critical_code_sha256"
        ],
        "run_git_commit": git["commit"],
        "run_git_tree": git["tree"],
        "run_git_dirty": git["dirty"],
        "run_python_version": versions["python"],
        "run_numpy_version": versions["numpy"],
        "run_scipy_version": versions["scipy"],
        "run_scikit_learn_version": versions["scikit_learn"],
        "run_pandas_version": versions["pandas"],
        "run_statsmodels_version": versions["statsmodels"],
        "run_label": provenance["run_label"],
        "source_only": True,
        "standalone_inference_permitted": False,
    }


def planned_conditions(config: Mapping[str, Any]) -> list[dict[str, object]]:
    return exp26._planned_conditions(config)


def _dimensions(
    cell: GeneratorCell,
    *,
    mode: str,
    manifest_receipt: str,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "generator_id": cell.generator_id,
        "generator_split": cell.generator_split,
        "alpha": cell.alpha,
        "transition_rank": cell.transition_rank,
        "input_rank": cell.input_rank,
        "delay": cell.delay,
        "noise_std": cell.noise_std,
        "rotation_seed": cell.rotation_seed,
        "actuator_mode": mode,
        "condition": mode,
        "manifest_hash": manifest_receipt,
        **evidence,
    }


def _record_all_failed(
    run: ExperimentRun,
    cells: Sequence[GeneratorCell],
    error: BaseException,
    *,
    receipt: str,
    evidence: Mapping[str, object],
) -> None:
    for cell in cells:
        for mode in EXPECTED_MODES:
            run.mark_condition_failure(
                error,
                **_dimensions(
                    cell,
                    mode=mode,
                    manifest_receipt=receipt,
                    evidence=evidence,
                ),
            )


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str | Path,
    *,
    run_label: str,
) -> Path:
    """Run one independent source seed without fitting any selector."""

    if run_label != REQUIRED_RUN_LABEL:
        raise ValueError(f"formal independent source requires {REQUIRED_RUN_LABEL!r}")
    contract = validate_source_contract(config)
    requested_seed = _exact_integer(seed, name="requested seed")
    if requested_seed not in EXPECTED_SEEDS:
        raise ValueError("requested independent source seed is not in 30--59")
    cells = exp26._manifest(config)
    receipt = manifest_hash(cells)
    initialize_seed(requested_seed)
    provenance = build_evidence_provenance(contract, run_label=run_label)
    evidence = evidence_row_fields(provenance)
    run_config = {**config, "evidence_provenance": provenance}
    with ExperimentRun(
        EXPERIMENT,
        requested_seed,
        run_config,
        results_root=results_root,
        run_label=run_label,
    ) as run:
        run.register_conditions(planned_conditions(run_config))
        try:
            carrier = exp26.make_carrier(
                exp26._carrier_config(run_config), requested_seed
            )
        except Exception as error:
            _record_all_failed(
                run,
                cells,
                error,
                receipt=receipt,
                evidence=evidence,
            )
        else:
            moment_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
            for cell in cells:
                try:
                    setup = exp26._setup_generator(
                        run_config,
                        carrier,
                        cell,
                        seed=requested_seed,
                        moment_cache=moment_cache,
                    )
                except Exception as error:
                    for mode in EXPECTED_MODES:
                        run.mark_condition_failure(
                            error,
                            **_dimensions(
                                cell,
                                mode=mode,
                                manifest_receipt=receipt,
                                evidence=evidence,
                            ),
                        )
                    continue
                for mode in EXPECTED_MODES:
                    dimensions = _dimensions(
                        cell,
                        mode=mode,
                        manifest_receipt=receipt,
                        evidence=evidence,
                    )
                    try:
                        metrics, valid = exp26._condition_metrics(
                            run_config,
                            setup,
                            mode=mode,
                        )
                        # ``source_only`` and
                        # ``standalone_inference_permitted`` are immutable
                        # evidence dimensions.  Do not duplicate them in the
                        # metric payload: ExperimentRun correctly rejects any
                        # metric/dimension overlap.
                        metrics["source_panel_protocol_version"] = PROTOCOL_VERSION
                        if valid:
                            run.record(metrics, **dimensions)
                        else:
                            failure_reasons = []
                            if not metrics["functional_budget_valid"]:
                                failure_reasons.append("functional_budget")
                            if not metrics["effective_dynamics_strictly_stable"]:
                                failure_reasons.append("effective_stability")
                            metrics["failure_reason"] = ",".join(failure_reasons)
                            run.record_failed_condition(metrics, **dimensions)
                    except Exception as error:
                        run.mark_condition_failure(error, **dimensions)

        end_git = exp26.git_identity()
        end_runtime = exp26.scientific_runtime_versions()
        if any(
            end_git.get(name) != contract.current_git.get(name)
            for name in ("commit", "tree", "dirty")
        ) or dict(end_runtime) != dict(contract.runtime_versions):
            raise RuntimeError(
                "independent source git/runtime identity changed during seed run"
            )
        return run.path


def _selected_seeds(config: Mapping[str, Any], override: str | None) -> Iterable[int]:
    selected = seed_list(override if override is not None else config["seeds"])
    if any(seed not in EXPECTED_SEEDS for seed in selected):
        raise ValueError("seed override must be a subset of 30--59")
    return selected


def main() -> None:
    parser = basic_parser(
        "Exp28 independent Exp26 source-only panel",
        "configs/formal/exp28_exp26_independent_source_panel.json",
    )
    parser.add_argument(
        "--run-label",
        required=True,
        help=f"must equal {REQUIRED_RUN_LABEL}",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    if args.run_label != REQUIRED_RUN_LABEL:
        parser.error(f"--run-label must equal {REQUIRED_RUN_LABEL}")
    for seed in _selected_seeds(config, args.seeds):
        print(
            run_seed(
                config,
                seed,
                args.results_root,
                run_label=args.run_label,
            )
        )


if __name__ == "__main__":
    main()
