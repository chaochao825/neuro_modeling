"""Shared-basis switching dynamics on public macaque sequence-memory sessions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.manifold_metrics import fit_train_pca, principal_angles
from src.analysis.task_subspaces import fit_demixed_condition_subspace
from src.data.sequence_dataset import (
    SequenceDataError,
    bin_spikes,
    block_split_trials,
    discover_sequence_sessions,
    infer_column,
    load_sequence_session,
    smooth_spike_counts,
    unseen_combination_split,
)
from src.models.reduced_dynamics import LDSSequenceDataset, SwitchingLDS, retained_switching_gain
from src.utils.artifacts import ExperimentRun
from src.utils.splits import grouped_kfold


def _column(trials, configured: str | None, aliases: list[str], purpose: str) -> str:
    if configured:
        if configured not in trials:
            raise SequenceDataError(f"configured {purpose} column {configured!r} is absent")
        return configured
    return infer_column(trials, aliases, purpose=purpose)


def _condition_subspace_metrics(activity_train, trials_train, config, sample_ids):
    flattened = activity_train.mean(axis=1)
    aliases_by_factor = {
        "rank": ["rank", "position", "serial_position"],
        "rule": ["rule", "task_rule", "direction"],
        "operation": ["operation", "epoch", "sorting_operation"],
        "item": ["item", "stimulus", "target"],
    }
    columns = {}
    errors = {}
    for name, aliases in aliases_by_factor.items():
        try:
            columns[name] = _column(
                trials_train, config.get("columns", {}).get(name), aliases, name
            )
        except SequenceDataError as error:
            errors[name] = str(error)
    bases = {}
    for name, column in columns.items():
        try:
            labels = trials_train[column].to_numpy()
            n_classes = len(np.unique(labels))
            dim = min(int(config["subspace_dim"]), n_classes - 1, flattened.shape[1])
            if dim >= 1:
                bases[name] = fit_demixed_condition_subspace(
                    flattened,
                    labels,
                    nuisance_labels={
                        other: trials_train[other_column].to_numpy()
                        for other, other_column in columns.items()
                        if other != name
                    },
                    n_components=dim,
                    sample_ids=sample_ids,
                )
        except ValueError as error:
            errors[name] = str(error)
    required = set(config.get("required_subspaces", ["rank", "rule", "operation"]))
    missing = sorted(required - set(bases))
    if missing:
        raise SequenceDataError(
            "required demixed subspaces unavailable: "
            + "; ".join(f"{name}: {errors.get(name, 'not fitted')}" for name in missing)
        )
    metrics = {
        **{f"{name}_subspace_dim": basis.basis.shape[1] for name, basis in bases.items()},
        "subspace_method": "train_only_multivariate_demixed_condition_means",
        "subspace_errors": errors,
    }
    for first_name, first in bases.items():
        for second_name, second in bases.items():
            if first_name >= second_name:
                continue
            metrics[f"{first_name}_{second_name}_angles_deg"] = principal_angles(
                first.basis, second.basis, degrees=True
            ).tolist()
    return metrics


def _transform_trials(pca, activity):
    shape = activity.shape
    return pca.transform(activity.reshape(-1, shape[-1])).reshape(shape[0], shape[1], -1)


def _behavior_accuracy(train_activity, test_activity, train_labels, test_labels, seed):
    if len(np.unique(train_labels)) < 2:
        return float("nan")
    classifier = LogisticRegression(max_iter=1000, random_state=seed).fit(
        train_activity.mean(axis=1), train_labels
    )
    return float(
        accuracy_score(test_labels, classifier.predict(test_activity.mean(axis=1)))
    )


def _analyze_session(session_path: Path, config: dict, seed: int):
    session = load_sequence_session(session_path)
    counts, _ = bin_spikes(
        session,
        window_s=tuple(config["window_s"]),
        bin_size_ms=int(config["bin_size_ms"]),
    )
    causal = smooth_spike_counts(
        counts,
        bin_size_ms=int(config["bin_size_ms"]),
        sigma_ms=float(config["smoothing_ms"]),
        mode="causal",
    )
    display = smooth_spike_counts(
        counts,
        bin_size_ms=int(config["bin_size_ms"]),
        sigma_ms=float(config["smoothing_ms"]),
        mode="symmetric",
    )
    block_column = config.get("columns", {}).get("block")
    if block_column is None:
        for alias in ("block", "block_id", "trial_block"):
            if alias in session.trials:
                block_column = alias
                break
    _, _, blocks = block_split_trials(
        session.trials,
        block_column=block_column,
        contiguous_block_size=None if block_column else int(config["fallback_block_size"]),
        seed=seed,
    )
    rule_column = _column(
        session.trials,
        config.get("columns", {}).get("rule"),
        ["rule", "task_rule", "direction"],
        "rule",
    )
    behavior_column = config.get("columns", {}).get("behavior")
    if behavior_column is None:
        for alias in ("correct", "choice", "response"):
            if alias in session.trials:
                behavior_column = alias
                break
    folds = grouped_kfold(blocks, int(config["n_splits"]))
    for fold, (train_trials, test_trials) in enumerate(folds):
        train_flat = causal[train_trials].reshape(-1, causal.shape[-1])
        pca_dim = min(
            int(config["pca_dim"]),
            train_flat.shape[1],
            train_flat.shape[0] - 1,
        )
        pca = fit_train_pca(
            train_flat,
            pca_dim,
            normalize=True,
            sample_ids=np.repeat(session.trials.index.to_numpy()[train_trials], causal.shape[1]),
        )
        train_activity = _transform_trials(pca, causal[train_trials])
        test_activity = _transform_trials(pca, causal[test_trials])
        n_time = train_activity.shape[1]
        epoch = np.minimum(2, np.arange(n_time) * 3 // n_time)
        train_rule = session.trials.iloc[train_trials][rule_column].astype(str).to_numpy()
        test_rule = session.trials.iloc[test_trials][rule_column].astype(str).to_numpy()
        train_conditions = np.array(
            [[f"{rule}:epoch{value}" for value in epoch] for rule in train_rule], dtype=object
        )
        test_conditions = np.array(
            [[f"{rule}:epoch{value}" for value in epoch] for rule in test_rule], dtype=object
        )
        train_sequences = LDSSequenceDataset(
            tuple(train_activity), tuple(train_conditions), blocks[train_trials]
        )
        test_sequences = LDSSequenceDataset(
            tuple(test_activity), tuple(test_conditions), blocks[test_trials]
        )
        lds_dim = min(int(config["shared_dim"]), pca_dim)
        models = {
            "common": SwitchingLDS("common", lds_dim, ridge=float(config["ridge"])),
            "shared": SwitchingLDS("shared", lds_dim, ridge=float(config["ridge"])),
            "full": SwitchingLDS("full", lds_dim, ridge=float(config["ridge"])),
        }
        scores = {}
        fold_records = []
        for name, model in models.items():
            model.fit(train_sequences)
            score = model.score(test_sequences)
            scores[name] = score
            record = {
                "status": "complete",
                "session_id": session.session_id,
                "fold": fold,
                "model_family": name,
                "heldout_log_likelihood": score.log_likelihood,
                "heldout_nll_per_scalar": score.nll_per_scalar,
                "parameter_count": model.parameter_count(),
                "likelihood_type": "marginal_kalman_linear_gaussian",
                "aggregation_level": "session",
            }
            fold_records.append(record)
        common_nll = scores["common"].nll_per_scalar
        shared_nll = scores["shared"].nll_per_scalar
        full_nll = scores["full"].nll_per_scalar
        shared_record = next(
            record for record in fold_records if record["model_family"] == "shared"
        )
        shared_record["retained_switching_gain"] = retained_switching_gain(
            common_nll, shared_nll, full_nll
        )
        shared_record.update(
            _condition_subspace_metrics(
                display[train_trials],
                session.trials.iloc[train_trials],
                config,
                session.trials.index.to_numpy()[train_trials],
            )
        )
        if behavior_column is not None:
            shared_record["behavior_prediction_accuracy"] = _behavior_accuracy(
                train_activity,
                test_activity,
                session.trials.iloc[train_trials][behavior_column].to_numpy(),
                session.trials.iloc[test_trials][behavior_column].to_numpy(),
                seed,
            )
        # Stream each completed fold immediately so a later fold/session error
        # cannot erase already completed evidence from the immutable artifact.
        yield from fold_records

    factors = config.get("combination_factors", [])
    if factors and all(factor in session.trials for factor in factors):
        combo_train, combo_test, held = unseen_combination_split(
            session.trials,
            factor_columns=factors,
            seed=seed,
            holdout_fraction=float(config["combination_holdout_fraction"]),
        )
        train_flat = causal[combo_train].reshape(-1, causal.shape[-1])
        pca_dim = min(int(config["pca_dim"]), train_flat.shape[1], train_flat.shape[0] - 1)
        pca = fit_train_pca(train_flat, pca_dim, normalize=True)
        combo_train_activity = _transform_trials(pca, causal[combo_train])
        combo_test_activity = _transform_trials(pca, causal[combo_test])
        n_time = combo_train_activity.shape[1]
        epoch = np.minimum(2, np.arange(n_time) * 3 // n_time)
        combo_train_rule = session.trials.iloc[combo_train][rule_column].astype(str).to_numpy()
        combo_test_rule = session.trials.iloc[combo_test][rule_column].astype(str).to_numpy()
        combo_train_conditions = np.array(
            [[f"{rule}:epoch{value}" for value in epoch] for rule in combo_train_rule],
            dtype=object,
        )
        combo_test_conditions = np.array(
            [[f"{rule}:epoch{value}" for value in epoch] for rule in combo_test_rule],
            dtype=object,
        )
        train_sequences = LDSSequenceDataset(
            tuple(combo_train_activity), tuple(combo_train_conditions), blocks[combo_train]
        )
        test_sequences = LDSSequenceDataset(
            tuple(combo_test_activity), tuple(combo_test_conditions), blocks[combo_test]
        )
        lds_dim = min(int(config["shared_dim"]), pca_dim)
        for family in ("common", "shared", "full"):
            model = SwitchingLDS(family, lds_dim, ridge=float(config["ridge"])).fit(
                train_sequences
            )
            score = model.score(test_sequences)
            yield {
                "status": "complete",
                "session_id": session.session_id,
                "fold": "unseen_combination",
                "model_family": family,
                "heldout_log_likelihood": score.log_likelihood,
                "heldout_nll_per_scalar": score.nll_per_scalar,
                "parameter_count": model.parameter_count(),
                "heldout_combinations": [list(item) for item in held],
                "likelihood_type": "marginal_kalman_linear_gaussian",
                "aggregation_level": "session",
            }
        if behavior_column is not None:
            generalization = _behavior_accuracy(
                combo_train_activity,
                combo_test_activity,
                session.trials.iloc[combo_train][behavior_column].to_numpy(),
                session.trials.iloc[combo_test][behavior_column].to_numpy(),
                seed,
            )
            yield {
                "status": "complete",
                "session_id": session.session_id,
                "fold": "unseen_combination",
                "model_family": "behavior_decoder",
                "behavior_prediction_accuracy": generalization,
                "heldout_combinations": [list(item) for item in held],
                "aggregation_level": "session",
            }


def run_seed(config: dict, seed: int, results_root: str) -> Path:
    initialize_seed(seed)
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parents[1] / data_root
    run_config = {
        **config,
        "resolved_data_root": str(data_root),
        "training_algorithm": "train_only_switching_lds_kalman_pca_ridge",
        "used_autograd": False,
        "parent_checkpoint": None,
    }
    with ExperimentRun("exp05_sequence_real_data", seed, run_config, results_root=results_root) as run:
        try:
            sessions = discover_sequence_sessions(data_root)
        except FileNotFoundError:
            sessions = []
        planned = [{"session_id": path.name} for path in sessions] or [{"session_id": "unavailable"}]
        run.register_conditions(planned)
        if not sessions:
            run.mark_condition_failure(
                FileNotFoundError(
                    "no accessible trials.csv/units.csv/spikes.mat session; the referenced "
                    "Zenodo record is restricted and requires user-granted access"
                ),
                session_id="unavailable",
                aggregation_level="session",
            )
            return run.path
        for session_path in sessions:
            try:
                for record in _analyze_session(session_path, config, seed):
                    run.record(record, session_id=record.pop("session_id"))
            except Exception as error:
                run.mark_condition_failure(
                    error, session_id=session_path.name, aggregation_level="session"
                )
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "sequence data", "configs/formal/exp05_sequence_real_data.json"
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
