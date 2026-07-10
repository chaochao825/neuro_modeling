"""Trial-level shared-basis IBL context-switch analysis with session statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.lead_lag import causal_within_block_bias, switch_latency_summary
from src.analysis.manifold_metrics import fit_train_pca
from src.data.ibl_loader import (
    CachedIBLSessionSource,
    IBLDataError,
    IBLSessionSource,
    OneAPISource,
    TrialNuisanceResidualizer,
    load_ibl_trial_data,
)
from src.models.reduced_dynamics import LDSSequenceDataset, SwitchingLDS, retained_switching_gain
from src.utils.artifacts import ExperimentRun
from src.utils.splits import grouped_kfold


FULL_NUISANCE = ("stimulus", "choice", "wheel", "reward", "reaction_time", "pose")
BEHAVIOR_NUISANCE = ("stimulus", "wheel", "reward", "reaction_time", "pose")


def _contiguous_positions(original_indices):
    """Return observed-index contiguous segments without revealing context switches."""

    indices = np.asarray(original_indices)
    if indices.ndim != 1:
        raise ValueError("original_indices must be a vector")
    if indices.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    return [segment for segment in np.split(np.arange(indices.size), boundaries) if segment.size]


def _block_sequences(latent, conditions, blocks, original_indices):
    sequences = []
    condition_sequences = []
    position_sequences = []
    sequence_groups = []
    # Sequence boundaries encode missing/non-adjacent trials only. They must
    # not encode true context-block boundaries: hidden-context scoring must
    # infer those switches rather than receive them through filter resets.
    for segment in _contiguous_positions(original_indices):
        if segment.size < 2:
            continue
        sequences.append(latent[segment])
        condition_sequences.append(np.asarray(conditions)[segment])
        position_sequences.append(segment)
        sequence_groups.append(blocks[segment[0]])
    if not sequences:
        raise IBLDataError("no adjacent within-block trial sequences remain")
    return (
        LDSSequenceDataset(
            tuple(sequences), tuple(condition_sequences), np.asarray(sequence_groups)
        ),
        position_sequences,
    )


def _broad_region_masks(regions: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.char.upper(np.asarray(regions).astype(str))
    patterns = {
        "thalamus": ("MD", "LP", "PO", "VAL", "VM", "TH"),
        "striatum": ("CP", "ACB", "STR", "CAUD", "PUT"),
        "frontal": ("MOS", "MOP", "ACA", "PL", "ILA", "ORB", "FRP"),
    }
    return {
        name: np.array([any(label.startswith(pattern) for pattern in prefixes) for label in labels])
        for name, prefixes in patterns.items()
    }


def _view_records(data, view: str, config: dict, seed: int):
    activity = data.activity[view]
    covariates = data.view_covariates[view].reset_index(drop=True)
    valid = data.valid_masks[view].copy()
    required = list(FULL_NUISANCE)
    valid &= ~covariates[required].isna().any(axis=1).to_numpy()
    selected = np.flatnonzero(valid)
    if selected.size < int(config["minimum_trials"]):
        raise IBLDataError(f"{view} has too few complete trials after nuisance filtering")
    flat = activity[selected].reshape(selected.size, -1)
    selected_covariates = covariates.iloc[selected].reset_index(drop=True)
    blocks = selected_covariates["block_id"].to_numpy()
    unique_blocks = np.unique(blocks)
    n_splits = min(int(config["n_splits"]), len(unique_blocks))
    if n_splits < 2:
        raise IBLDataError("at least two complete context blocks are required")
    folds = grouped_kfold(blocks, n_splits)
    oof_context_score = np.full(selected.size, np.nan)
    region_masks = _broad_region_masks(data.regions)
    oof_region_scores = {
        name: np.full(selected.size, np.nan) for name, mask in region_masks.items() if mask.any()
    }
    for fold, (train, test) in enumerate(folds):
        residualizer = TrialNuisanceResidualizer(FULL_NUISANCE).fit(
            selected_covariates.iloc[train], flat[train], sample_ids=selected[train]
        )
        train_residual = residualizer.transform(selected_covariates.iloc[train], flat[train])
        test_residual = residualizer.transform(selected_covariates.iloc[test], flat[test])
        pca_dim = min(int(config["pca_dim"]), train_residual.shape[1], train_residual.shape[0] - 1)
        pca = fit_train_pca(train_residual, pca_dim, normalize=True, sample_ids=selected[train])
        train_latent = pca.transform(train_residual)
        test_latent = pca.transform(test_residual)
        all_residual = residualizer.transform(selected_covariates, flat)
        all_latent = pca.transform(all_residual)
        context_values = selected_covariates["probability_left"].to_numpy(dtype=float)
        train_dataset, train_positions = _block_sequences(
            train_latent,
            context_values[train],
            blocks[train],
            selected[train],
        )
        test_dataset, test_positions = _block_sequences(
            test_latent,
            context_values[test],
            blocks[test],
            selected[test],
        )
        lds_dim = min(int(config["shared_dim"]), pca_dim)
        models = {
            "common": SwitchingLDS("common", lds_dim, ridge=float(config["ridge"])),
            "shared": SwitchingLDS("shared", lds_dim, ridge=float(config["ridge"])),
            "full": SwitchingLDS("full", lds_dim, ridge=float(config["ridge"])),
        }
        scores = {}
        fold_records = []
        for family, model in models.items():
            model.fit(train_dataset)
            score = model.score_hidden_context(
                test_dataset,
                stay_probability=float(config.get("context_stay_probability", 0.95)),
            )
            scores[family] = score
            fold_records.append(
                {
                    "status": "complete",
                    "view": view,
                    "fold": fold,
                    "model_family": family,
                    "heldout_log_likelihood": score.log_likelihood,
                    "heldout_nll_per_scalar": score.nll_per_scalar,
                    "parameter_count": model.parameter_count(),
                    "likelihood_type": "marginal_hidden_context_kalman",
                    "aggregation_level": "session",
                    "n_trials_total": int(len(covariates)),
                    "n_trials_complete": int(selected.size),
                    "n_trials_invalid_timing": int(
                        np.sum(~covariates["timing_valid"].to_numpy(dtype=bool))
                    ),
                }
            )
        shared_record = fold_records[1]
        shared_record["retained_switching_gain"] = retained_switching_gain(
            scores["common"].nll_per_scalar,
            scores["shared"].nll_per_scalar,
            scores["full"].nll_per_scalar,
        )

        # Context scores come from a continuous, hidden-context IMM filter.
        # No held-out probabilityLeft labels are supplied to this filter.
        shared_model = models["shared"]
        inferred_context = np.full(selected.size, np.nan)
        for positions in _contiguous_positions(selected):
            hidden = shared_model.filter_hidden_context_sequence(
                all_latent[positions],
                stay_probability=float(
                    config.get("context_stay_probability", 0.95)
                ),
            )
            numeric_conditions = np.asarray(hidden.condition_labels, dtype=float)
            inferred_context[positions] = (
                hidden.context_probability @ numeric_conditions
            )
        if not np.isfinite(inferred_context).all():
            raise RuntimeError("segmented hidden-context posterior is incomplete")
        oof_context_score[test] = inferred_context[test]
        context_labels = context_values.astype(str)

        train_reshaped = train_residual.reshape(len(train), activity.shape[1], activity.shape[2])
        test_reshaped = test_residual.reshape(len(test), activity.shape[1], activity.shape[2])
        for region_name, mask in region_masks.items():
            if region_name not in oof_region_scores:
                continue
            region_train = train_reshaped[:, :, mask].mean(axis=1)
            region_test = test_reshaped[:, :, mask].mean(axis=1)
            region_classifier = LogisticRegression(max_iter=1000, random_state=seed).fit(
                region_train, context_labels[train]
            )
            probabilities = region_classifier.predict_proba(region_test)
            oof_region_scores[region_name][test] = (
                probabilities @ region_classifier.classes_.astype(float)
            )

        behavior_residualizer = TrialNuisanceResidualizer(BEHAVIOR_NUISANCE).fit(
            selected_covariates.iloc[train], flat[train], sample_ids=selected[train]
        )
        behavior_train = behavior_residualizer.transform(
            selected_covariates.iloc[train], flat[train]
        )
        behavior_test = behavior_residualizer.transform(
            selected_covariates.iloc[test], flat[test]
        )
        behavior_pca = fit_train_pca(
            behavior_train,
            pca_dim,
            normalize=True,
            sample_ids=selected[train],
        )
        choice_classifier = LogisticRegression(max_iter=1000, random_state=seed).fit(
            behavior_pca.transform(behavior_train),
            selected_covariates.iloc[train]["choice"].to_numpy(),
        )
        shared_record["behavior_prediction_accuracy"] = float(
            accuracy_score(
                selected_covariates.iloc[test]["choice"].to_numpy(),
                choice_classifier.predict(behavior_pca.transform(behavior_test)),
            )
        )
        for record in fold_records:
            yield record
    if not np.isfinite(oof_context_score).all():
        raise RuntimeError("OOF context score is incomplete")
    choice = selected_covariates["choice"].to_numpy(dtype=float)
    behavior_bias = causal_within_block_bias(choice, blocks)
    latency = switch_latency_summary(oof_context_score, behavior_bias, blocks)
    region_latencies = {}
    for name, score in oof_region_scores.items():
        if np.isfinite(score).all():
            region_latencies[name] = switch_latency_summary(
                score, behavior_bias, blocks
            ).median_latent_lead_trials
    yield {
        "status": "complete",
        "view": view,
        "fold": "session_switch",
        "model_family": "lead_lag",
        "latent_lead_trials": latency.median_latent_lead_trials,
        "n_switches": latency.n_switches,
        "latent_source": "shared_switching_lds_hidden_context_imm_posterior",
        "condition_schedule_observed": False,
        "region_latent_lead_trials": region_latencies,
        "lead_lag_is_causal_claim": False,
        "aggregation_level": "session",
        "n_trials_total": int(len(covariates)),
        "n_trials_complete": int(selected.size),
        "n_trials_invalid_timing": int(
            np.sum(~covariates["timing_valid"].to_numpy(dtype=bool))
        ),
    }


def run_seed(
    config: dict,
    seed: int,
    results_root: str,
    *,
    source: IBLSessionSource | None = None,
) -> Path:
    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "train_only_trial_shared_basis",
        "used_autograd": False,
        "parent_checkpoint": None,
    }
    with ExperimentRun("exp06_ibl_context_switch", seed, run_config, results_root=results_root) as run:
        try:
            if source is None:
                cache_dir = Path(config["cache_dir"])
                if not cache_dir.is_absolute():
                    cache_dir = Path(__file__).resolve().parents[1] / cache_dir
                cached_paths = config.get("cached_session_paths")
                source = (
                    CachedIBLSessionSource(
                        cache_dir=cache_dir,
                        session_paths=cached_paths,
                    )
                    if cached_paths
                    else OneAPISource(cache_dir=cache_dir)
                )
            eids = [str(eid) for eid in config.get("eids", [])]
            if not eids:
                eids = source.search_sessions(limit=int(config["n_sessions"]))
            if not 1 <= len(eids) <= 5:
                raise ValueError("IBL initial analysis requires 1-5 sessions")
        except Exception as error:
            run.register_conditions([{"session_id": "discovery", "view": "discovery"}])
            run.mark_condition_failure(
                error,
                session_id="discovery",
                view="discovery",
                aggregation_level="session",
            )
            return run.path
        planned = [
            {"session_id": eid, "view": view}
            for eid in eids
            for view in ("stimulus_pre", "movement_pre")
        ]
        run.register_conditions(planned)
        for eid in eids:
            try:
                data = load_ibl_trial_data(
                    source,
                    eid,
                    bin_size_ms=int(config["bin_size_ms"]),
                    pre_window_s=tuple(config["pre_window_s"]),
                )
                for view in ("stimulus_pre", "movement_pre"):
                    try:
                        for record in _view_records(data, view, config, seed):
                            run.record(
                                record,
                                session_id=eid,
                                animal_id=data.animal_id,
                            )
                    except Exception as error:
                        run.mark_condition_failure(
                            error,
                            session_id=eid,
                            animal_id=data.animal_id,
                            view=view,
                            aggregation_level="session",
                        )
            except Exception as error:
                for view in ("stimulus_pre", "movement_pre"):
                    run.mark_condition_failure(
                        error,
                        session_id=eid,
                        view=view,
                        aggregation_level="session",
                    )
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "IBL", "configs/formal/exp06_ibl_context_switch.json"
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
