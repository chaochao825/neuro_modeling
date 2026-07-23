"""ORBIT end-to-end test of reward-only actuator matching.

A frozen visual encoder supplies a common observation tape.  Four reusable
few-shot actuator motifs are fitted from labelled clean support videos.  A
small contextual bandit is trained on *executed* correctness rewards from
development users and deployed without query labels on disjoint users.
"""

from __future__ import annotations

from dataclasses import asdict
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
from src.models.streaming_fewshot_actuators import (
    ACTUATOR_NAMES,
    CONTEXT_FEATURE_NAMES,
    PersonalizedStreamingActuators,
    StreamingActuatorConfig,
    StreamingActuatorTrace,
)
from src.plasticity.contextual_bandit_selector import (
    ContextualBanditConfig,
    ContextualBanditReceipt,
    RewardOnlyContextualController,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp33_orbit_streaming_fewshot"
PROTOCOL_VERSION = "exp33_orbit_causal_cluve_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONDITIONS = (
    *ACTUATOR_NAMES,
    "train_fixed_best",
    "reward_only_local",
    "credit_shuffled_local",
    "oracle_per_frame",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sampling_config(config: Mapping[str, Any]) -> OrbitEpisodeSamplingConfig:
    return OrbitEpisodeSamplingConfig(**dict(config["sampling"]))


def _actuator_config(config: Mapping[str, Any]) -> StreamingActuatorConfig:
    return StreamingActuatorConfig(**dict(config["actuators"]))


def _controller_config(config: Mapping[str, Any]) -> ContextualBanditConfig:
    return ContextualBanditConfig(**dict(config["controller"]))


def _users(
    store: OrbitFeatureStore,
    requested: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if not requested:
        return store.users
    selected = tuple(map(str, requested))
    if len(selected) != len(set(selected)):
        raise ValueError("requested ORBIT users must be unique")
    missing = set(selected) - set(store.users)
    if missing:
        raise ValueError(
            f"requested users are absent from feature store: {sorted(missing)}"
        )
    return selected


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp33 protocol version mismatch")
    profile = str(config.get("profile"))
    if profile not in {"smoke", "development", "formal"}:
        raise ValueError("profile must be smoke, development, or formal")
    if config.get("training_algorithm") != "executed_reward_only_contextual_bandit":
        raise ValueError("Exp33 training algorithm must remain reward-only")
    if config.get("used_autograd") is not False or config.get("used_bptt") is not False:
        raise ValueError("the Exp33 local controller cannot use autograd or BPTT")
    for name in (
        "n_fit_tasks_per_user",
        "n_eval_tasks_per_user",
        "training_frame_stride",
    ):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    fit_split = str(config.get("fit_split"))
    eval_split = str(config.get("eval_split"))
    if fit_split not in {"train", "validation"}:
        raise ValueError("fit_split must be train or validation")
    if eval_split not in {"validation", "test"}:
        raise ValueError("eval_split must be validation or test")
    fit_users = set(map(str, config.get("fit_user_ids", [])))
    eval_users = set(map(str, config.get("eval_user_ids", [])))
    if fit_split == eval_split:
        if not fit_users or not eval_users or fit_users & eval_users:
            raise ValueError(
                "same-split development requires explicit disjoint fit/eval users"
            )
    if profile == "formal":
        if config.get("formal_authorized") is not True:
            raise ValueError("formal Exp33 is fail-closed without authorization")
        if (fit_split, eval_split) != ("train", "test"):
            raise ValueError(
                "formal Exp33 must fit on train users and evaluate test users"
            )
        if config["n_eval_tasks_per_user"] != 50:
            raise ValueError("formal ORBIT evaluation requires 50 tasks per test user")
        if fit_users or eval_users:
            raise ValueError("formal Exp33 must use every official train/test user")
    _sampling_config(config)
    _actuator_config(config)
    _controller_config(config)


def _training_indices(trace: StreamingActuatorTrace, stride: int) -> np.ndarray:
    selected: list[int] = []
    previous_video = ""
    within_video = 0
    for index, raw_video_id in enumerate(trace.video_ids):
        video_id = str(raw_video_id)
        if video_id != previous_video:
            within_video = 0
        if within_video % stride == 0:
            selected.append(index)
        within_video += 1
        previous_video = video_id
    return np.asarray(selected, dtype=np.int64)


def _fit_controllers(
    store: OrbitFeatureStore,
    users: tuple[str, ...],
    *,
    seed: int,
    config: Mapping[str, Any],
) -> tuple[ContextualBanditReceipt, ContextualBanditReceipt, int, dict[str, Any]]:
    controller_cfg = _controller_config(config)
    main = RewardOnlyContextualController(
        len(ACTUATOR_NAMES),
        len(CONTEXT_FEATURE_NAMES),
        config=controller_cfg,
        seed=derive_seed(seed, "exp33-controller-main"),
    )
    shuffled = RewardOnlyContextualController(
        len(ACTUATOR_NAMES),
        len(CONTEXT_FEATURE_NAMES),
        config=controller_cfg,
        seed=derive_seed(seed, "exp33-controller-shuffled"),
    )
    counterfactual_correct = np.zeros(len(ACTUATOR_NAMES), dtype=np.int64)
    counterfactual_total = 0
    task_failures: list[dict[str, object]] = []
    for user_id in users:
        for task_index in range(int(config["n_fit_tasks_per_user"])):
            try:
                episode = store.sample_episode(
                    user_id,
                    seed=derive_seed(seed, "fit-episode", user_id, task_index),
                    task_index=task_index,
                    config=_sampling_config(config),
                )
                actuators = PersonalizedStreamingActuators.fit(
                    episode.support,
                    n_classes=episode.n_classes,
                    config=_actuator_config(config),
                )
                trace = actuators.trace(episode.query_observation)
                indices = _training_indices(trace, int(config["training_frame_stride"]))
                counterfactual_correct += np.sum(
                    trace.predictions[indices] == episode.query_labels[indices, None],
                    axis=0,
                )
                counterfactual_total += int(indices.size)
                previous_main_video = ""
                previous_shuffled_video = ""
                for index in indices:
                    video_id = str(trace.video_ids[index])
                    if video_id != previous_main_video:
                        main.reset_sequence()
                    if video_id != previous_shuffled_video:
                        shuffled.reset_sequence()
                    label = int(episode.query_labels[index])
                    predictions = trace.predictions[index]

                    def main_reward(action: int) -> float:
                        return float(predictions[action] == label)

                    def shuffled_reward(action: int) -> float:
                        return float(predictions[action] == label)

                    main.train_step(trace.contexts[index], main_reward)
                    shuffled.train_step(
                        trace.contexts[index],
                        shuffled_reward,
                        credit_transform=lambda action: (
                            (action + 1) % len(ACTUATOR_NAMES)
                        ),
                    )
                    previous_main_video = video_id
                    previous_shuffled_video = video_id
            except Exception as error:
                task_failures.append(
                    {
                        "user_id": user_id,
                        "task_index": task_index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    if counterfactual_total == 0:
        raise RuntimeError("controller fitting produced no usable query frames")
    fixed_best = int(np.argmax(counterfactual_correct))
    audit = {
        "counterfactual_fit_correct": counterfactual_correct.tolist(),
        "counterfactual_fit_frames": counterfactual_total,
        "fixed_best_action": fixed_best,
        "fixed_best_name": ACTUATOR_NAMES[fixed_best],
        "task_failures": task_failures,
    }
    return main.receipt(), shuffled.receipt(), fixed_best, audit


def _condition_output(
    condition: str,
    *,
    episode: OrbitEmbeddingEpisode,
    trace: StreamingActuatorTrace,
    fixed_best: int,
    main_receipt: ContextualBanditReceipt,
    shuffled_receipt: ContextualBanditReceipt,
    controller_config: ContextualBanditConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = episode.query_labels.size
    if condition in ACTUATOR_NAMES:
        action = ACTUATOR_NAMES.index(condition)
        actions = np.full(n, action, dtype=np.int64)
    elif condition == "train_fixed_best":
        actions = np.full(n, fixed_best, dtype=np.int64)
    elif condition in {"reward_only_local", "credit_shuffled_local"}:
        receipt = main_receipt if condition == "reward_only_local" else shuffled_receipt
        controller = RewardOnlyContextualController.from_receipt(
            receipt,
            config=controller_config,
            seed=derive_seed(
                seed, "deploy", condition, episode.user_id, episode.task_index
            ),
        )
        selected = controller.predict_trace(trace)
        return selected.predictions, selected.actions, selected.selected_event_l1
    elif condition == "oracle_per_frame":
        correct = trace.predictions == episode.query_labels[:, None]
        actions = np.where(
            np.any(correct, axis=1), np.argmax(correct, axis=1), fixed_best
        ).astype(np.int64)
    else:
        raise ValueError(f"unknown Exp33 condition: {condition}")
    frame = np.arange(n)
    return (
        trace.predictions[frame, actions],
        actions,
        trace.action_event_l1[frame, actions],
    )


def _evaluate_episode(
    episode: OrbitEmbeddingEpisode,
    *,
    seed: int,
    config: Mapping[str, Any],
    fixed_best: int,
    main_receipt: ContextualBanditReceipt,
    shuffled_receipt: ContextualBanditReceipt,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actuators = PersonalizedStreamingActuators.fit(
        episode.support,
        n_classes=episode.n_classes,
        config=_actuator_config(config),
    )
    trace = actuators.trace(episode.query_observation)
    headroom = actuator_headroom(episode.query_labels, trace.predictions)
    rows: list[pd.DataFrame] = []
    for condition in EVALUATION_CONDITIONS:
        predictions, actions, costs = _condition_output(
            condition,
            episode=episode,
            trace=trace,
            fixed_best=fixed_best,
            main_receipt=main_receipt,
            shuffled_receipt=shuffled_receipt,
            controller_config=_controller_config(config),
            seed=seed,
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
        cost_by_video = {
            str(video_id): float(np.mean(costs[episode.query_video_ids == video_id]))
            for video_id in np.unique(episode.query_video_ids)
        }
        frame["mean_selected_event_l1"] = frame["video_id"].map(cost_by_video)
        frame["status"] = "complete"
        frame["episode_fingerprint"] = episode.fingerprint
        frame["trace_fingerprint"] = trace.fingerprint
        rows.append(frame)
    diagnostic = {
        "user_id": episode.user_id,
        "task_index": episode.task_index,
        "n_classes": episode.n_classes,
        "n_support_frames": int(episode.support.labels.size),
        "n_query_frames": int(episode.query_labels.size),
        "per_action_accuracy": headroom.per_action_accuracy.tolist(),
        "best_action": headroom.best_action,
        "best_fixed_accuracy": headroom.best_fixed_accuracy,
        "oracle_accuracy": headroom.oracle_accuracy,
        "oracle_gain": headroom.oracle_gain,
        "action_disagreement": headroom.action_disagreement,
        "support_write_l1": actuators.write_l1_cost,
        "support_write_l2": actuators.write_l2_cost,
    }
    return pd.concat(rows, ignore_index=True), diagnostic


def _receipt_payload(receipt: ContextualBanditReceipt) -> dict[str, Any]:
    payload = asdict(receipt)
    for key, value in tuple(payload.items()):
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
    return payload


def run_seed(config: Mapping[str, Any], *, seed: int, results_root: str | Path) -> Path:
    _validate_config(config)
    initialize_seed(seed)
    split_path = Path(str(config["official_splits_path"]))
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    feature_root = Path(str(config["feature_root"])).expanduser()
    fit_store = OrbitFeatureStore(
        feature_root,
        split=str(config["fit_split"]),
        official_splits_path=split_path,
        require_complete_split=bool(config.get("require_complete_fit_split", False)),
    )
    if config["eval_split"] == config["fit_split"]:
        eval_store = fit_store
    else:
        eval_store = OrbitFeatureStore(
            feature_root,
            split=str(config["eval_split"]),
            official_splits_path=split_path,
            require_complete_split=bool(
                config.get("require_complete_eval_split", False)
            ),
        )
        validate_user_disjoint_stores((fit_store, eval_store))
    fit_users = _users(fit_store, config.get("fit_user_ids"))
    eval_users = _users(eval_store, config.get("eval_user_ids"))
    if set(fit_users) & set(eval_users):
        raise ValueError("fit and evaluation users must be disjoint")

    run_config = dict(config)
    run_config["official_splits_sha256"] = _sha256(split_path)
    run_config["fit_users"] = list(fit_users)
    run_config["eval_users"] = list(eval_users)
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
        run_label=str(config["profile"]),
    ) as run:
        planned = [
            {
                "user_id": user_id,
                "task_index": task_index,
                "condition": condition,
            }
            for user_id in eval_users
            for task_index in range(int(config["n_eval_tasks_per_user"]))
            for condition in EVALUATION_CONDITIONS
        ]
        run.register_conditions(planned)
        main_receipt, shuffled_receipt, fixed_best, training_audit = _fit_controllers(
            fit_store, fit_users, seed=seed, config=config
        )
        (run.path / "controller_receipts.json").write_text(
            json.dumps(
                {
                    "main": _receipt_payload(main_receipt),
                    "credit_shuffled": _receipt_payload(shuffled_receipt),
                    "training_audit": training_audit,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raw_frames: list[pd.DataFrame] = []
        diagnostics: list[dict[str, Any]] = []
        for user_id in eval_users:
            for task_index in range(int(config["n_eval_tasks_per_user"])):
                try:
                    episode = eval_store.sample_episode(
                        user_id,
                        seed=derive_seed(seed, "eval-episode", user_id, task_index),
                        task_index=task_index,
                        config=_sampling_config(config),
                    )
                    rows, diagnostic = _evaluate_episode(
                        episode,
                        seed=seed,
                        config=config,
                        fixed_best=fixed_best,
                        main_receipt=main_receipt,
                        shuffled_receipt=shuffled_receipt,
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
            raise RuntimeError("all Exp33 evaluation tasks failed")
        raw = pd.concat(raw_frames, ignore_index=True)
        raw.to_csv(run.path / "raw_video_metrics.csv", index=False)
        diagnostic_frame = pd.DataFrame(diagnostics)
        diagnostic_frame.to_csv(run.path / "actuator_headroom.csv", index=False)
        user_rows = reduce_to_user_accuracy(raw)
        user_rows.to_csv(run.path / "user_metrics.csv", index=False)
        condition_summary = (
            user_rows.groupby("condition", as_index=False)["user_video_mean_accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        condition_summary.to_csv(run.path / "condition_summary.csv", index=False)
        means = dict(
            zip(
                condition_summary["condition"],
                condition_summary["mean"],
                strict=True,
            )
        )
        local_gain = float(means["reward_only_local"] - means["train_fixed_best"])
        bootstrap_samples = int(config["analysis"].get("bootstrap_samples", 20_000))
        statistics_seed = int(config["analysis"].get("statistics_seed", seed))
        fixed_inference = paired_user_inference(
            user_rows,
            method="reward_only_local",
            comparator="train_fixed_best",
            bootstrap_samples=bootstrap_samples,
            seed=derive_seed(statistics_seed, "local-vs-fixed"),
        )
        shuffled_inference = paired_user_inference(
            user_rows,
            method="reward_only_local",
            comparator="credit_shuffled_local",
            bootstrap_samples=bootstrap_samples,
            seed=derive_seed(statistics_seed, "local-vs-shuffled"),
        )
        adjusted_pvalues = holm_adjust(
            [fixed_inference.sign_flip_pvalue, shuffled_inference.sign_flip_pvalue]
        )
        mean_headroom = float(diagnostic_frame["oracle_gain"].mean())
        mean_disagreement = float(diagnostic_frame["action_disagreement"].mean())
        profile = str(config["profile"])
        conclusion = "inconclusive"
        reason = "development data cannot support a confirmatory claim"
        if profile == "formal":
            minimum = float(config["analysis"]["minimum_accuracy_gain"])
            headroom_gate = mean_headroom >= float(
                config["analysis"]["minimum_oracle_headroom"]
            ) and mean_disagreement >= float(
                config["analysis"]["minimum_action_disagreement"]
            )
            significance_gate = bool(
                np.all(adjusted_pvalues <= 0.05)
                and fixed_inference.ci_low > 0.0
                and shuffled_inference.ci_low > 0.0
            )
            if local_gain >= minimum and headroom_gate and significance_gate:
                conclusion = "support"
                reason = (
                    "registered task, headroom, credit-specificity, and user-level "
                    "inference gates passed"
                )
            elif fixed_inference.ci_high < minimum or not headroom_gate:
                conclusion = "oppose"
                reason = "registered task-effect or actuator-headroom gate was missed"
            else:
                conclusion = "inconclusive"
                reason = (
                    "effect direction was not resolved at the registered user level"
                )
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "profile": profile,
            "seed": seed,
            "fixed_best_action": fixed_best,
            "fixed_best_name": ACTUATOR_NAMES[fixed_best],
            "condition_user_mean_accuracy": means,
            "reward_only_gain_over_train_fixed": local_gain,
            "mean_oracle_headroom": mean_headroom,
            "mean_action_disagreement": mean_disagreement,
            "paired_user_inference": {
                "local_vs_train_fixed": asdict(fixed_inference),
                "local_vs_credit_shuffled": asdict(shuffled_inference),
                "holm_adjusted_pvalues": adjusted_pvalues.tolist(),
            },
            "main_update_l1": main_receipt.update_l1_cost,
            "main_update_l2": main_receipt.update_l2_cost,
            "main_reward_queries": main_receipt.n_reward_queries,
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
        "configs/smoke/exp33_orbit_streaming_fewshot.json",
    )
    parser.add_argument("--feature-root", default=None)
    args = parser.parse_args()
    config = load_json_config(args.config)
    if args.feature_root is not None:
        config["feature_root"] = str(Path(args.feature_root).expanduser().resolve())
    seeds = seed_list(args.seeds if args.seeds is not None else config["seeds"])
    for seed in seeds:
        path = run_seed(config, seed=seed, results_root=args.results_root)
        print(path)


if __name__ == "__main__":
    main()
