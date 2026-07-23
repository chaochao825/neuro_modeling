"""Causal label-free actuator selection from within-video prediction consensus."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.orbit_streaming_metrics import (
    actuator_headroom,
    holm_adjust,
    paired_user_inference,
    reduce_to_user_accuracy,
    task_video_accuracy_rows,
)
from src.data.orbit_streaming import (
    OrbitEmbeddingEpisode,
    OrbitEpisodeSamplingConfig,
    OrbitFeatureStore,
    validate_user_disjoint_stores,
)
from src.models.causal_consensus_gate import (
    CausalConsensusConfig,
    CausalConsensusGate,
)
from src.models.streaming_fewshot_actuators import (
    ACTUATOR_NAMES,
    PersonalizedStreamingActuators,
    StreamingActuatorConfig,
    StreamingActuatorTrace,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp34_orbit_causal_consensus"
PROTOCOL_VERSION = "exp34_orbit_causal_consensus_v2_support_annotation_safe"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_SCHEMA = "exp34_formal_scale_authorization_v1"
AUTHORIZATION_EXCLUDED_CONFIG_KEYS = frozenset(
    {
        "formal_authorized",
        "authorization_reason",
        "authorization_receipt",
        "authorization_receipt_sha256",
        "config_path",
    }
)
IMPLEMENTATION_PATHS = (
    "experiments/exp34_orbit_causal_consensus.py",
    "src/models/causal_consensus_gate.py",
    "src/models/streaming_fewshot_actuators.py",
    "src/data/orbit_streaming.py",
    "src/analysis/orbit_streaming_metrics.py",
)
EVALUATION_CONDITIONS = (
    *ACTUATOR_NAMES,
    "selection_fixed_best",
    "causal_consensus",
    "memoryless_consensus",
    "delayed_consensus",
    "oracle_per_frame",
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formal_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash every registered formal setting except receipt plumbing."""

    payload = {
        key: value
        for key, value in dict(config).items()
        if key not in AUTHORIZATION_EXCLUDED_CONFIG_KEYS
    }
    return _canonical_sha256(payload)


def implementation_hashes() -> dict[str, str]:
    return {
        relative: _file_sha256(PROJECT_ROOT / relative)
        for relative in IMPLEMENTATION_PATHS
    }


def _validate_formal_authorization(config: Mapping[str, Any]) -> None:
    receipt_value = config.get("authorization_receipt")
    expected_digest = config.get("authorization_receipt_sha256")
    if not isinstance(receipt_value, str) or not receipt_value:
        raise ValueError("formal Exp34 requires an authorization receipt")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("formal Exp34 requires the registered receipt digest")
    receipt_path = Path(receipt_value)
    if not receipt_path.is_absolute():
        receipt_path = PROJECT_ROOT / receipt_path
    receipt_path = receipt_path.resolve()
    if not receipt_path.is_relative_to(PROJECT_ROOT) or not receipt_path.is_file():
        raise ValueError("formal Exp34 receipt must be a committed project artifact")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("formal Exp34 authorization receipt must be a JSON object")
    observed_digest = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if (
        observed_digest != _canonical_sha256(unsigned)
        or observed_digest != expected_digest
    ):
        raise ValueError("formal Exp34 authorization receipt hash mismatch")
    if (
        receipt.get("schema") != AUTHORIZATION_SCHEMA
        or receipt.get("protocol_version") != PROTOCOL_VERSION
        or receipt.get("authorized") is not True
        or receipt.get("scale_decision") != "scale-authorized"
    ):
        raise ValueError("formal Exp34 authorization receipt did not pass")
    if receipt.get("formal_config_fingerprint") != formal_config_fingerprint(config):
        raise ValueError("formal Exp34 config differs from its frozen receipt")
    if receipt.get("implementation_sha256") != implementation_hashes():
        raise ValueError("formal Exp34 implementation differs from its receipt")


def _sampling_config(config: Mapping[str, Any]) -> OrbitEpisodeSamplingConfig:
    return OrbitEpisodeSamplingConfig(**dict(config["sampling"]))


def _actuator_config(config: Mapping[str, Any]) -> StreamingActuatorConfig:
    return StreamingActuatorConfig(**dict(config["actuators"]))


def _gate_config(config: Mapping[str, Any]) -> CausalConsensusConfig:
    payload = dict(config["gate"])
    payload["tie_break_order"] = tuple(payload["tie_break_order"])
    return CausalConsensusConfig(**payload)


def _users(
    store: OrbitFeatureStore, requested: list[str] | tuple[str, ...] | None
) -> tuple[str, ...]:
    if not requested:
        return store.users
    selected = tuple(map(str, requested))
    if len(selected) != len(set(selected)):
        raise ValueError("requested users must be unique")
    missing = set(selected) - set(store.users)
    if missing:
        raise ValueError(f"requested users are absent: {sorted(missing)}")
    return selected


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp34 protocol version mismatch")
    profile = str(config.get("profile"))
    if profile not in {"smoke", "development", "formal"}:
        raise ValueError("profile must be smoke, development, or formal")
    if config.get("training_algorithm") != "causal_label_free_count_belief":
        raise ValueError("Exp34 must use the registered label-free belief update")
    for key in (
        "used_query_labels",
        "used_future_frames",
        "used_autograd",
        "used_bptt",
    ):
        if config.get(key) is not False:
            raise ValueError(f"Exp34 requires {key}=false")
    for name in ("n_selection_tasks_per_user", "n_eval_tasks_per_user"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(config.get("cache_features_in_memory", False), bool):
        raise ValueError("cache_features_in_memory must be boolean")
    if config.get("selection_split") not in {"train", "validation"}:
        raise ValueError("selection_split must be train or validation")
    if config.get("eval_split") not in {"validation", "test"}:
        raise ValueError("eval_split must be validation or test")
    selection_users = set(map(str, config.get("selection_user_ids", [])))
    eval_users = set(map(str, config.get("eval_user_ids", [])))
    if config["selection_split"] == config["eval_split"]:
        if not selection_users or not eval_users or selection_users & eval_users:
            raise ValueError(
                "same-split development requires explicit disjoint user panels"
            )
    if profile == "formal":
        if config.get("formal_authorized") is not True:
            raise ValueError("formal Exp34 is fail-closed without authorization")
        _validate_formal_authorization(config)
        if (config["selection_split"], config["eval_split"]) != (
            "validation",
            "test",
        ):
            raise ValueError("formal Exp34 selects on validation and evaluates test")
        if config["n_eval_tasks_per_user"] != 50:
            raise ValueError("formal ORBIT evaluation requires 50 tasks per test user")
        if selection_users or eval_users:
            raise ValueError("formal Exp34 must use all validation and test users")
    delay = config["analysis"].get("delay_intervention_frames")
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        raise ValueError("delay_intervention_frames must be a positive integer")
    _sampling_config(config)
    _actuator_config(config)
    gate = _gate_config(config)
    if gate.delay_frames != 0 or gate.reset_each_frame:
        raise ValueError("main Exp34 gate cannot be delayed or memoryless")


def _select_fixed_action(
    store: OrbitFeatureStore,
    users: tuple[str, ...],
    *,
    seed: int,
    config: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    correct = np.zeros(len(ACTUATOR_NAMES), dtype=np.int64)
    frames = 0
    failures: list[dict[str, object]] = []
    for user_id in users:
        for task_index in range(int(config["n_selection_tasks_per_user"])):
            try:
                episode = store.sample_episode(
                    user_id,
                    seed=derive_seed(seed, "selection", user_id, task_index),
                    task_index=task_index,
                    config=_sampling_config(config),
                )
                fitted = PersonalizedStreamingActuators.fit(
                    episode.support,
                    n_classes=episode.n_classes,
                    config=_actuator_config(config),
                )
                trace = fitted.trace(episode.query_observation)
                correct += np.sum(
                    trace.predictions == episode.query_labels[:, None], axis=0
                )
                frames += int(episode.query_labels.size)
            except Exception as error:
                failures.append(
                    {
                        "user_id": user_id,
                        "task_index": task_index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    if frames == 0:
        raise RuntimeError("fixed-action selection produced no frames")
    action = int(np.argmax(correct))
    return action, {
        "action_correct": correct.tolist(),
        "n_frames": frames,
        "selected_action": action,
        "selected_name": ACTUATOR_NAMES[action],
        "failures": failures,
    }


def _gate_output(
    episode: OrbitEmbeddingEpisode,
    trace: StreamingActuatorTrace,
    gate_config: CausalConsensusConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gate = CausalConsensusGate(
        len(ACTUATOR_NAMES), episode.n_classes, config=gate_config
    )
    selected = gate.trace(
        trace.predictions,
        video_ids=trace.video_ids,
        action_event_l1=trace.action_event_l1,
    )
    return (
        selected.predictions,
        selected.actions,
        selected.full_bank_event_l1,
        selected.count_state_l1,
    )


def _condition_output(
    condition: str,
    *,
    episode: OrbitEmbeddingEpisode,
    trace: StreamingActuatorTrace,
    fixed_action: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    n = episode.query_labels.size
    if condition in ACTUATOR_NAMES:
        action = ACTUATOR_NAMES.index(condition)
        actions = np.full(n, action, dtype=np.int64)
        scope = "selected_actuator"
    elif condition == "selection_fixed_best":
        actions = np.full(n, fixed_action, dtype=np.int64)
        scope = "selected_actuator"
    elif condition in {
        "causal_consensus",
        "memoryless_consensus",
        "delayed_consensus",
    }:
        gate_config = _gate_config(config)
        if condition == "memoryless_consensus":
            gate_config = replace(gate_config, reset_each_frame=True)
        elif condition == "delayed_consensus":
            gate_config = replace(
                gate_config,
                delay_frames=int(config["analysis"]["delay_intervention_frames"]),
            )
        predictions, actions, costs, state = _gate_output(episode, trace, gate_config)
        return predictions, actions, costs, state, "full_actuator_bank"
    elif condition == "oracle_per_frame":
        correct = trace.predictions == episode.query_labels[:, None]
        actions = np.where(
            np.any(correct, axis=1), np.argmax(correct, axis=1), fixed_action
        ).astype(np.int64)
        scope = "label_oracle_full_bank"
    else:
        raise ValueError(f"unknown Exp34 condition: {condition}")
    indices = np.arange(n)
    return (
        trace.predictions[indices, actions],
        actions,
        trace.action_event_l1[indices, actions],
        np.zeros(n, dtype=np.float64),
        scope,
    )


def _evaluate_episode(
    episode: OrbitEmbeddingEpisode,
    *,
    fixed_action: int,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fitted = PersonalizedStreamingActuators.fit(
        episode.support,
        n_classes=episode.n_classes,
        config=_actuator_config(config),
    )
    trace = fitted.trace(episode.query_observation)
    headroom = actuator_headroom(episode.query_labels, trace.predictions)
    rows = []
    for condition in EVALUATION_CONDITIONS:
        predictions, actions, costs, states, compute_scope = _condition_output(
            condition,
            episode=episode,
            trace=trace,
            fixed_action=fixed_action,
            config=config,
        )
        frame = task_video_accuracy_rows(
            user_id=episode.user_id,
            task_index=episode.task_index,
            condition=condition,
            labels=episode.query_labels,
            predictions=predictions,
            video_ids=episode.query_video_ids,
            selected_actions=actions,
        )
        for column, values in (
            ("mean_event_l1", costs),
            ("mean_belief_state_l1", states),
        ):
            mapping = {
                str(video_id): float(
                    np.mean(values[episode.query_video_ids == video_id])
                )
                for video_id in np.unique(episode.query_video_ids)
            }
            frame[column] = frame["video_id"].map(mapping)
        frame["compute_scope"] = compute_scope
        frame["status"] = "complete"
        frame["episode_fingerprint"] = episode.fingerprint
        frame["trace_fingerprint"] = trace.fingerprint
        rows.append(frame)
    diagnostic = {
        "user_id": episode.user_id,
        "task_index": episode.task_index,
        "n_classes": episode.n_classes,
        "best_fixed_accuracy": headroom.best_fixed_accuracy,
        "oracle_accuracy": headroom.oracle_accuracy,
        "oracle_gain": headroom.oracle_gain,
        "action_disagreement": headroom.action_disagreement,
        "support_write_l1": fitted.write_l1_cost,
        "support_write_l2": fitted.write_l2_cost,
    }
    return pd.concat(rows, ignore_index=True), diagnostic


def run_seed(config: Mapping[str, Any], *, seed: int, results_root: str | Path) -> Path:
    _validate_config(config)
    initialize_seed(seed)
    split_path = Path(str(config["official_splits_path"]))
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    feature_root = Path(str(config["feature_root"])).expanduser()
    selection_store = OrbitFeatureStore(
        feature_root,
        split=str(config["selection_split"]),
        official_splits_path=split_path,
        require_complete_split=bool(
            config.get("require_complete_selection_split", False)
        ),
        cache_videos=bool(config.get("cache_features_in_memory", False)),
    )
    if config["selection_split"] == config["eval_split"]:
        eval_store = selection_store
    else:
        eval_store = OrbitFeatureStore(
            feature_root,
            split=str(config["eval_split"]),
            official_splits_path=split_path,
            require_complete_split=bool(
                config.get("require_complete_eval_split", False)
            ),
            cache_videos=bool(config.get("cache_features_in_memory", False)),
        )
        validate_user_disjoint_stores((selection_store, eval_store))
    selection_users = _users(selection_store, config.get("selection_user_ids"))
    eval_users = _users(eval_store, config.get("eval_user_ids"))
    if set(selection_users) & set(eval_users):
        raise ValueError("selection and evaluation users must be disjoint")

    run_config = dict(config)
    run_config["selection_users"] = list(selection_users)
    run_config["eval_users"] = list(eval_users)
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
        run_label=str(config["profile"]),
    ) as run:
        run.register_conditions(
            [
                {
                    "user_id": user_id,
                    "task_index": task_index,
                    "condition": condition,
                }
                for user_id in eval_users
                for task_index in range(int(config["n_eval_tasks_per_user"]))
                for condition in EVALUATION_CONDITIONS
            ]
        )
        fixed_action, selection_audit = _select_fixed_action(
            selection_store,
            selection_users,
            seed=seed,
            config=config,
        )
        (run.path / "selection_audit.json").write_text(
            json.dumps(selection_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw_frames = []
        diagnostics = []
        for user_id in eval_users:
            for task_index in range(int(config["n_eval_tasks_per_user"])):
                try:
                    episode = eval_store.sample_episode(
                        user_id,
                        seed=derive_seed(seed, "eval", user_id, task_index),
                        task_index=task_index,
                        config=_sampling_config(config),
                    )
                    rows, diagnostic = _evaluate_episode(
                        episode, fixed_action=fixed_action, config=config
                    )
                    raw_frames.append(rows)
                    diagnostics.append(diagnostic)
                    for row in rows.to_dict("records"):
                        dimensions = {
                            key: row.pop(key)
                            for key in (
                                "user_id",
                                "task_index",
                                "video_id",
                                "condition",
                            )
                        }
                        run.record(row, **dimensions)
                except Exception as error:
                    for condition in EVALUATION_CONDITIONS:
                        run.mark_condition_failure(
                            error,
                            user_id=user_id,
                            task_index=task_index,
                            video_id="unavailable",
                            condition=condition,
                        )
        if not raw_frames:
            raise RuntimeError("all Exp34 evaluation tasks failed")
        raw = pd.concat(raw_frames, ignore_index=True)
        diagnostic_frame = pd.DataFrame(diagnostics)
        raw.to_csv(run.path / "raw_video_metrics.csv", index=False)
        diagnostic_frame.to_csv(run.path / "actuator_headroom.csv", index=False)
        user_rows = reduce_to_user_accuracy(raw)
        user_rows.to_csv(run.path / "user_metrics.csv", index=False)
        means = (
            user_rows.groupby("condition")["user_video_mean_accuracy"].mean().to_dict()
        )
        bootstrap = int(config["analysis"].get("bootstrap_samples", 20_000))
        inference = []
        for index, comparator in enumerate(
            ("selection_fixed_best", "memoryless_consensus", "delayed_consensus")
        ):
            inference.append(
                paired_user_inference(
                    user_rows,
                    method="causal_consensus",
                    comparator=comparator,
                    bootstrap_samples=bootstrap,
                    seed=derive_seed(seed, "inference", index),
                )
            )
        adjusted = holm_adjust(item.sign_flip_pvalue for item in inference)
        headroom = float(diagnostic_frame["oracle_gain"].mean())
        fixed_gain = float(means["causal_consensus"] - means["selection_fixed_best"])
        memory_gain = float(means["causal_consensus"] - means["memoryless_consensus"])
        retained = fixed_gain / headroom if headroom > 0.0 else 0.0
        profile = str(config["profile"])
        conclusion = "inconclusive"
        reason = "validation-only development cannot support a formal claim"
        if profile == "formal":
            minimum = float(config["analysis"]["minimum_accuracy_gain"])
            support = bool(
                fixed_gain >= minimum
                and memory_gain > 0.0
                and retained
                >= float(config["analysis"]["minimum_retained_oracle_headroom"])
                and inference[0].ci_low > 0.0
                and inference[1].ci_low > 0.0
                and adjusted[0] <= 0.05
                and adjusted[1] <= 0.05
            )
            if support:
                conclusion = "support"
                reason = "registered task and causal-memory gates passed"
            elif inference[0].ci_high < minimum or inference[1].ci_high <= 0.0:
                conclusion = "oppose"
                reason = "registered task or causal-memory effect was absent"
            else:
                reason = "formal user-level uncertainty did not resolve the claim"
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "profile": profile,
            "seed": seed,
            "selected_fixed_action": fixed_action,
            "selected_fixed_name": ACTUATOR_NAMES[fixed_action],
            "condition_user_mean_accuracy": means,
            "consensus_gain_over_selection_fixed": fixed_gain,
            "consensus_gain_over_memoryless": memory_gain,
            "mean_oracle_headroom": headroom,
            "retained_oracle_headroom_fraction": retained,
            "mean_action_disagreement": float(
                diagnostic_frame["action_disagreement"].mean()
            ),
            "feature_cache": {
                "selection": selection_store.cache_stats,
                "evaluation": eval_store.cache_stats,
            },
            "paired_user_inference": [asdict(item) for item in inference],
            "holm_adjusted_pvalues": adjusted.tolist(),
            "conclusion": conclusion,
            "conclusion_reason": reason,
        }
        (run.path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return run.path


def main() -> None:
    parser = basic_parser(
        __doc__ or EXPERIMENT,
        "configs/smoke/exp34_orbit_causal_consensus.json",
    )
    parser.add_argument("--feature-root", default=None)
    args = parser.parse_args()
    config = load_json_config(args.config)
    if args.feature_root is not None:
        config["feature_root"] = str(Path(args.feature_root).expanduser().resolve())
    seeds = seed_list(args.seeds if args.seeds is not None else config["seeds"])
    for seed in seeds:
        print(run_seed(config, seed=seed, results_root=args.results_root))


if __name__ == "__main__":
    main()
