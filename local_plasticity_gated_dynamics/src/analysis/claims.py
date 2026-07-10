"""Pre-registered, evidence-gated classification of the core propositions.

Only ``formal`` artifacts enter this module.  Seed-level propositions require
all twenty planned seeds.  Real-data folds are first averaged within a session
and, when ``animal_id`` is available, sessions are then averaged within animal.
The reported ``p_value`` is the Holm-adjusted Wilcoxon p value across the full
family of registered propositions; conclusions use the pre-registered paired
bootstrap interval and never promote incomplete evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

from src.analysis.model_comparison import paired_bootstrap, paired_wilcoxon


SEED_PLAN = 20


@dataclass(frozen=True)
class ClaimResult:
    claim_id: str
    experiment: str
    metric: str
    comparison: str
    stats_unit: str
    n_planned: int
    n_complete: int
    n_failed: int
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    effect_size: float | None
    p_value: float | None
    multiplicity_method: str
    conclusion: str
    criterion: str
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _inconclusive(
    claim_id: str,
    experiment: str,
    metric: str,
    comparison: str,
    stats_unit: str,
    criterion: str,
    note: str,
    *,
    n_planned: int = 0,
    n_complete: int = 0,
    n_failed: int = 0,
) -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        experiment=experiment,
        metric=metric,
        comparison=comparison,
        stats_unit=stats_unit,
        n_planned=n_planned,
        n_complete=n_complete,
        n_failed=n_failed,
        estimate=None,
        ci_low=None,
        ci_high=None,
        effect_size=None,
        p_value=None,
        multiplicity_method="holm(full_registered_family)",
        conclusion="inconclusive",
        criterion=criterion,
        note=note,
    )


def _paired_claim(
    *,
    claim_id: str,
    experiment: str,
    metric: str,
    comparison: str,
    stats_unit: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    unit_ids: np.ndarray,
    n_planned: int,
    n_failed: int,
    minimum_units: int,
    support_low: float | None = None,
    support_high: float | None = None,
    oppose_below: float | None = None,
    oppose_above: float | None = None,
    criterion: str,
    seed: int = 0,
) -> ClaimResult:
    """Evaluate one paired contrast after enforcing the evidence plan."""

    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    unit_ids = np.asarray(unit_ids, dtype=object)
    valid = (
        candidate.ndim == reference.ndim == unit_ids.ndim == 1
        and len(candidate) == len(reference) == len(unit_ids)
    )
    if not valid:
        raise ValueError("paired claim arrays must be aligned one-dimensional vectors")
    finite = np.isfinite(candidate) & np.isfinite(reference)
    candidate, reference, unit_ids = candidate[finite], reference[finite], unit_ids[finite]
    n_complete = len(set(unit_ids.tolist()))
    if n_failed or n_complete < minimum_units or n_complete < n_planned:
        reasons: list[str] = []
        if n_failed:
            reasons.append(f"{n_failed} planned independent unit(s) failed")
        if n_complete < minimum_units:
            reasons.append(f"requires at least {minimum_units} complete independent units")
        if n_complete < n_planned:
            reasons.append(f"only {n_complete}/{n_planned} planned units are complete")
        return _inconclusive(
            claim_id,
            experiment,
            metric,
            comparison,
            stats_unit,
            criterion,
            "; ".join(reasons),
            n_planned=n_planned,
            n_complete=n_complete,
            n_failed=n_failed,
        )

    bootstrap = paired_bootstrap(
        candidate,
        reference,
        unit_ids=unit_ids,
        replicate_unit=stats_unit,
        n_resamples=5000,
        confidence=0.95,
        seed=seed,
    )
    wilcoxon = paired_wilcoxon(
        candidate,
        reference,
        unit_ids=unit_ids,
        replicate_unit=stats_unit,
    )
    support = True
    if support_low is not None:
        support &= bootstrap.ci_low >= support_low
    if support_high is not None:
        support &= bootstrap.ci_high <= support_high
    oppose = False
    if oppose_below is not None:
        oppose |= bootstrap.ci_high < oppose_below
    if oppose_above is not None:
        oppose |= bootstrap.ci_low > oppose_above
    conclusion = "support" if support else ("oppose" if oppose else "inconclusive")
    return ClaimResult(
        claim_id=claim_id,
        experiment=experiment,
        metric=metric,
        comparison=comparison,
        stats_unit=stats_unit,
        n_planned=n_planned,
        n_complete=bootstrap.n_units,
        n_failed=n_failed,
        estimate=bootstrap.estimate,
        ci_low=bootstrap.ci_low,
        ci_high=bootstrap.ci_high,
        effect_size=bootstrap.estimate,
        p_value=wilcoxon.p_value,
        multiplicity_method="holm(full_registered_family)",
        conclusion=conclusion,
        criterion=criterion,
        note=(
            "paired 95% bootstrap CI at the declared independent-unit level; "
            "p_value awaits full-family Holm adjustment"
        ),
    )


def _series(frame: pd.DataFrame, column: str, default: object = None) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index, dtype=object)


def _eq(frame: pd.DataFrame, column: str, value: object) -> pd.Series:
    return _series(frame, column).eq(value)


def _complete(frame: pd.DataFrame) -> pd.Series:
    return _eq(frame, "status", "complete")


def _failed_units(
    frame: pd.DataFrame,
    unit: str,
    mask: pd.Series | None = None,
    *,
    completed_units: Iterable[object] = (),
) -> int:
    selected = _eq(frame, "status", "failed")
    if mask is not None:
        scoped = mask.reindex(frame.index, fill_value=False)
        # A materialized top-level failure has no condition dimensions, but it
        # invalidates every preregistered panel in that run.  Keep it in each
        # relevant failure count instead of silently dropping an empty retry.
        run_level = (
            _series(frame, "run_level_failure", False)
            .fillna(False)
            .astype(bool)
        )
        selected &= scoped | run_level
    failed = frame.loc[selected]
    if failed.empty:
        return 0
    complete = {str(value) for value in completed_units}
    if unit in failed:
        identifiers = failed[unit].dropna().astype(str)
        if not identifiers.empty:
            return int(len(set(identifiers.tolist()) - complete))
    return int(len(failed))


def select_latest_attempts(formal: pd.DataFrame) -> pd.DataFrame:
    """Select the newest immutable run attempt for each experiment and seed.

    Raw artifacts remain untouched.  This prevents a corrected formal retry
    from being permanently contaminated by an older failed attempt.  Synthetic
    or legacy tables without run provenance are intentionally left unchanged.
    """

    required = {"experiment", "seed", "run_id"}
    if formal.empty or not required <= set(formal):
        return formal
    pieces: list[pd.DataFrame] = []
    for _, group in formal.groupby(["experiment", "seed"], sort=False, dropna=False):
        valid_run = group["run_id"].notna()
        timing = pd.DataFrame(
            {
                "run_id": group.loc[valid_run, "run_id"],
                "started": pd.to_datetime(
                    _series(group, "run_started_at").loc[valid_run],
                    errors="coerce",
                    utc=True,
                    format="mixed",
                ),
                "recorded": pd.to_datetime(
                    _series(group, "recorded_at").loc[valid_run],
                    errors="coerce",
                    utc=True,
                    format="mixed",
                ),
            }
        )
        run_times = timing.groupby("run_id", sort=False, dropna=False).agg(
            started=("started", "max"),
            recorded=("recorded", "max"),
        )
        # Attempt identity is defined by its immutable start time.  The final
        # metric time is only a legacy fallback; an older long-running attempt
        # must never supersede a later retry merely because it finished later.
        run_times["attempt_time"] = run_times["started"].fillna(
            run_times["recorded"]
        )
        valid = run_times["attempt_time"].notna()
        if not valid.any():
            pieces.append(group)
            continue
        ordered = run_times.loc[valid].sort_values(
            ["attempt_time", "recorded"], kind="stable", na_position="first"
        )
        latest_run = ordered.index[-1]
        latest = group.loc[group["run_id"].eq(latest_run)].copy()
        run_status = _series(latest, "run_status")
        has_nonterminal_status = (
            run_status.notna()
            & ~run_status.isin({"complete", "complete_with_failures"})
        ).any()
        run_level_failure = _series(latest, "run_level_failure", False).fillna(False).astype(bool).any()
        if has_nonterminal_status or run_level_failure:
            # A streamed condition is not independent evidence until its
            # enclosing run reaches a terminal state.  Preserve its dimensions
            # but make every cell fail the evidence gate.
            latest.loc[:, "status"] = "failed"
        pieces.append(latest)
    return pd.concat(pieces, axis=0).sort_index() if pieces else formal.iloc[:0].copy()


def _paired_rows(
    frame: pd.DataFrame,
    first_filter: pd.Series,
    second_filter: pd.Series,
    metric: str,
    unit: str,
    *,
    pair_columns: Iterable[str] = (),
    required_pairs: set[tuple[object, ...]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair rows within strata, then give every independent unit equal weight."""

    if metric not in frame or unit not in frame:
        return np.array([]), np.array([]), np.array([], dtype=object)
    pair_columns = tuple(column for column in pair_columns if column in frame)
    keys = [unit, *pair_columns]

    def select(mask: pd.Series, name: str) -> pd.DataFrame:
        selected = frame.loc[mask & _complete(frame), [*keys, metric]].copy()
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        selected = selected.dropna(subset=[unit, metric])
        return (
            selected.groupby(keys, as_index=False, dropna=False)[metric]
            .mean()
            .rename(columns={metric: name})
        )

    first = select(first_filter, "first")
    second = select(second_filter, "second")
    paired = first.merge(second, on=keys, validate="one_to_one")
    if paired.empty:
        return np.array([]), np.array([]), np.array([], dtype=object)
    if required_pairs is not None and pair_columns:
        observed = paired[list(pair_columns)].apply(tuple, axis=1)
        paired = paired.assign(_pair=observed)
        good_units = [
            identifier
            for identifier, rows in paired.groupby(unit, sort=False)
            if set(rows["_pair"].tolist()) == required_pairs
        ]
        paired = paired.loc[paired[unit].isin(good_units)].drop(columns="_pair")
    paired = paired.groupby(unit, as_index=False)[["first", "second"]].mean()
    return (
        paired["first"].to_numpy(float),
        paired["second"].to_numpy(float),
        paired[unit].to_numpy(object),
    )


def _primary_phase2(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the preregistered 80/20 E/I architecture when dimensions exist."""

    if frame.empty:
        return frame
    run_level_failure = (
        _eq(frame, "status", "failed")
        & _series(frame, "run_level_failure", False).fillna(False).astype(bool)
    )
    if "architecture" in frame and frame["architecture"].eq("ei_n512_fi20_gain1").any():
        return frame.loc[
            frame["architecture"].eq("ei_n512_fi20_gain1") | run_level_failure
        ].copy()
    if {"model_kind", "inhibitory_fraction"} <= set(frame):
        fraction = pd.to_numeric(frame["inhibitory_fraction"], errors="coerce")
        mask = frame["model_kind"].eq("ei") & np.isclose(fraction, 0.2, atol=1e-8)
        if mask.any():
            return frame.loc[mask | run_level_failure].copy()
    if "architecture" in frame or "model_kind" in frame:
        return frame.loc[run_level_failure].copy()
    return frame.copy()


def _real_unit_column(frame: pd.DataFrame) -> tuple[str, str]:
    """Return the valid highest-level unit column and its model-comparison label."""

    if "animal_id" in frame and frame["animal_id"].notna().any():
        return "animal_id", "animal"
    if "session_id" in frame:
        return "session_id", "session"
    if "session" in frame:
        return "session", "session"
    return "session_id", "session"


def _session_column(frame: pd.DataFrame) -> str | None:
    for name in ("session_id", "session"):
        if name in frame:
            return name
    return None


def _model_unit_table(
    frame: pd.DataFrame,
    metric: str,
    *,
    fold_filter: pd.Series | None = None,
    failure_filter: pd.Series | None = None,
) -> tuple[pd.DataFrame, str, str, int]:
    """Aggregate fold/view -> session -> animal and pivot model families."""

    session = _session_column(frame)
    unit_column, stats_unit = _real_unit_column(frame)
    if failure_filter is None:
        failure_filter = pd.Series(True, index=frame.index, dtype=bool)
    if session is None or metric not in frame or "model_family" not in frame:
        return (
            pd.DataFrame(),
            unit_column,
            stats_unit,
            _failed_units(frame, unit_column, failure_filter),
        )
    mask = _complete(frame)
    if fold_filter is not None:
        mask &= fold_filter
    columns = [session, "model_family", metric]
    repeated_columns = [name for name in ("fold", "view") if name in frame]
    columns.extend(repeated_columns)
    has_animal = stats_unit == "animal" and "animal_id" in frame
    if has_animal:
        columns.append("animal_id")
    selected = frame.loc[mask, columns].copy()
    required_models = {"common", "shared", "full"}
    selected = selected.loc[selected["model_family"].isin(required_models)]
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected.dropna(subset=[session, "model_family", metric])
    if selected.empty:
        return (
            pd.DataFrame(),
            unit_column,
            stats_unit,
            _failed_units(frame, unit_column, failure_filter),
        )
    # A missing model in any planned fold/view is not allowed to disappear in
    # a session mean.  Invalidate the whole independent unit before inference.
    cell_keys = [session, *repeated_columns]
    cell_models = selected.groupby(cell_keys, sort=False, dropna=False)[
        "model_family"
    ].agg(lambda values: frozenset(values))
    bad_cells = cell_models.loc[cell_models.map(set) != required_models]
    bad_sessions = (
        set(bad_cells.index.get_level_values(session).tolist())
        if repeated_columns
        else set(bad_cells.index.tolist())
    )
    # Any relevant session failure invalidates that session. This is crucial
    # for streaming experiments: earlier folds may already be on disk when a
    # later fold raises, but a partial fold panel is never evidence.
    failed_rows = frame.loc[
        _eq(frame, "status", "failed")
        & failure_filter.reindex(frame.index, fill_value=False)
    ]
    for _, failed_row in failed_rows.iterrows():
        if session not in failed_row or pd.isna(failed_row[session]):
            continue
        bad_sessions.add(failed_row[session])
    implicit_failed = 0
    if bad_sessions:
        if has_animal:
            bad_animals = selected.loc[
                selected[session].isin(bad_sessions), "animal_id"
            ].dropna()
            implicit_failed = int(bad_animals.nunique()) or len(bad_sessions)
        else:
            implicit_failed = len(bad_sessions)
        selected = selected.loc[~selected[session].isin(bad_sessions)]
    if selected.empty:
        return (
            pd.DataFrame(),
            unit_column,
            stats_unit,
            max(_failed_units(frame, unit_column, failure_filter), implicit_failed),
        )

    session_keys = [session, "model_family"]
    if has_animal:
        session_keys.insert(1, "animal_id")
    # All folds and both IBL views are repeated measurements within session.
    aggregated = selected.groupby(session_keys, as_index=False, dropna=False)[metric].mean()
    if has_animal:
        variation = aggregated.groupby(session, dropna=False)["animal_id"].nunique()
        if (variation != 1).any() or aggregated["animal_id"].isna().any():
            return (
                pd.DataFrame(),
                unit_column,
                stats_unit,
                max(1, _failed_units(frame, unit_column, failure_filter)),
            )
        aggregated = aggregated.groupby(
            ["animal_id", "model_family"], as_index=False, dropna=False
        )[metric].mean()
        unit_column = "animal_id"
    table = aggregated.pivot(index=unit_column, columns="model_family", values=metric)
    complete_units = table.dropna(
        subset=[name for name in ("common", "shared", "full") if name in table]
    ).index.tolist()
    return (
        table,
        unit_column,
        stats_unit,
        max(
            _failed_units(
                frame,
                unit_column,
                failure_filter,
                completed_units=complete_units,
            ),
            implicit_failed,
        ),
    )


def _real_plan(frame: pd.DataFrame, unit_column: str, n_failed: int) -> int:
    explicit = pd.to_numeric(_series(frame, "n_planned"), errors="coerce").dropna()
    if not explicit.empty:
        return max(2, int(explicit.max()))
    observed = int(frame[unit_column].dropna().nunique()) if unit_column in frame else 0
    return max(2, observed + n_failed)


def _real_scalar_values(
    frame: pd.DataFrame, metric: str, mask: pd.Series
) -> tuple[np.ndarray, np.ndarray, str, str, int]:
    """Aggregate only complete session/view panels through the real-data hierarchy."""

    session = _session_column(frame)
    unit_column, stats_unit = _real_unit_column(frame)
    failure_scope = mask | (
        _eq(frame, "status", "failed") & _series(frame, "model_family").isna()
    )
    n_failed = _failed_units(frame, unit_column, failure_scope)
    if session is None or metric not in frame:
        return np.array([]), np.array([], dtype=object), unit_column, stats_unit, n_failed
    columns = [session, metric]
    if "view" in frame:
        columns.append("view")
    has_animal = stats_unit == "animal" and "animal_id" in frame
    if has_animal:
        columns.append("animal_id")
    selected = frame.loc[mask & _complete(frame), columns].copy()
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected.dropna(subset=[session, metric])
    if selected.empty:
        return np.array([]), np.array([], dtype=object), unit_column, stats_unit, n_failed

    bad_sessions = set(
        frame.loc[
            _eq(frame, "status", "failed")
            & failure_scope.reindex(frame.index, fill_value=False),
            session,
        ]
        .dropna()
        .tolist()
    )
    if "view" in frame:
        experiments = set(_series(frame, "experiment").dropna().astype(str).tolist())
        expected_views = (
            {"stimulus_pre", "movement_pre"}
            if "exp06_ibl_context_switch" in experiments
            else set(frame.loc[failure_scope, "view"].dropna().astype(str).tolist())
        )
        if expected_views:
            observed_views = selected.groupby(session, dropna=False)["view"].agg(
                lambda values: set(values.dropna().astype(str).tolist())
            )
            bad_sessions.update(
                observed_views.loc[observed_views.map(set) != expected_views]
                .index.tolist()
            )

    if bad_sessions:
        if has_animal:
            bad_animals = set(
                frame.loc[frame[session].isin(bad_sessions), "animal_id"]
                .dropna()
                .astype(str)
                .tolist()
            )
            n_failed = max(n_failed, len(bad_animals) or len(bad_sessions))
            if bad_animals:
                selected = selected.loc[
                    ~selected["animal_id"].astype(str).isin(bad_animals)
                ]
            else:
                selected = selected.loc[~selected[session].isin(bad_sessions)]
        else:
            n_failed = max(n_failed, len(bad_sessions))
            selected = selected.loc[~selected[session].isin(bad_sessions)]
    if selected.empty:
        return np.array([]), np.array([], dtype=object), unit_column, stats_unit, n_failed

    session_keys = [session]
    if has_animal:
        session_keys.append("animal_id")
    aggregated = selected.groupby(session_keys, as_index=False, dropna=False)[metric].mean()
    if has_animal:
        if aggregated["animal_id"].isna().any():
            return np.array([]), np.array([], dtype=object), unit_column, stats_unit, max(1, n_failed)
        aggregated = aggregated.groupby("animal_id", as_index=False)[metric].mean()
        unit_column = "animal_id"
    return (
        aggregated[metric].to_numpy(float),
        aggregated[unit_column].to_numpy(object),
        unit_column,
        stats_unit,
        n_failed,
    )


def _phase_match_is_exact(frame: pd.DataFrame) -> tuple[bool, str]:
    required_flags = (
        "mean_rate_match_exact",
        "per_trial_spike_count_match_exact",
        "mean_coupling_match_exact",
    )
    if not set(required_flags) <= set(frame) or "shared_source_fingerprint" not in frame:
        return False, "exact rate/spike/coupling flags or source fingerprint are absent"
    relevant = frame.loc[
        _complete(frame)
        & _series(frame, "phase_condition").isin(["in_phase", "no_oscillation"])
    ]
    if relevant.empty or not relevant[list(required_flags)].fillna(False).astype(bool).all(axis=None):
        return False, "one or more exact matching flags are false"
    counts = relevant.groupby("seed")["phase_condition"].nunique()
    fingerprints = relevant.groupby("seed")["shared_source_fingerprint"].nunique(dropna=False)
    if not (counts.eq(2).all() and fingerprints.eq(1).all()):
        return False, "in-phase/no-oscillation sources are not exactly paired within seed"
    return True, "all exact matching flags true and source fingerprints identical within seed"


def _holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        sorted_values * (len(values) - np.arange(len(values)))
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _apply_full_family_holm(results: list[ClaimResult]) -> list[ClaimResult]:
    """Include unavailable hypotheses as p=1 so the registered family stays fixed."""

    raw = np.asarray(
        [result.p_value if result.p_value is not None and np.isfinite(result.p_value) else 1.0
         for result in results],
        dtype=float,
    )
    adjusted = _holm_adjust(raw)
    output: list[ClaimResult] = []
    family_size = len(results)
    for result, adjusted_p in zip(results, adjusted, strict=True):
        if result.p_value is None or not np.isfinite(result.p_value):
            output.append(result)
            continue
        raw_p = result.p_value
        output.append(
            replace(
                result,
                p_value=float(adjusted_p),
                note=(
                    f"{result.note}; p_value is Holm-adjusted across all {family_size} "
                    f"registered claims (raw Wilcoxon p={raw_p:.12g}); bootstrap criterion "
                    "determines the three-way conclusion"
                ),
            )
        )
    return output


def evaluate_core_claims(raw_metrics: pd.DataFrame) -> list[ClaimResult]:
    """Evaluate A1--E2; absent, failed, or incomplete formal evidence stays inconclusive."""

    if not isinstance(raw_metrics, pd.DataFrame):
        raise TypeError("raw_metrics must be a pandas DataFrame")
    formal = (
        raw_metrics.loc[_eq(raw_metrics, "profile", "formal")].copy()
        if "profile" in raw_metrics
        else raw_metrics.iloc[:0].copy()
    )
    formal = select_latest_attempts(formal)
    results: list[ClaimResult] = []

    # Phase 1: rank and feedback alignment.
    exp01 = formal.loc[_eq(formal, "experiment", "exp01_feedback_dimension_sweep")].copy()
    core = _eq(exp01, "grid", "core") & _eq(exp01, "feedback_mode", "aligned")
    d4 = core & pd.to_numeric(_series(exp01, "feedback_dim"), errors="coerce").eq(4)
    rank_values, _, rank_units = _paired_rows(
        exp01, d4, d4, "effective_rank", "seed"
    )
    results.append(
        _paired_claim(
            claim_id="A1_rank_matches_feedback",
            experiment="exp01",
            metric="effective_rank",
            comparison="aligned d=4 minus target rank 4",
            stats_unit="seed",
            candidate=rank_values,
            reference=np.full(len(rank_values), 4.0),
            unit_ids=rank_units,
            n_planned=SEED_PLAN,
            n_failed=_failed_units(exp01, "seed", d4),
            minimum_units=SEED_PLAN,
            support_low=-0.5,
            support_high=0.5,
            oppose_below=-1.0,
            oppose_above=1.0,
            criterion="95% CI of effective-rank minus 4 lies within [-0.5, 0.5]",
        )
    )
    # The preregistered full-feedback comparator is N=128. Never substitute the
    # largest surviving dimension from a partial/aborted sweep.
    full_dim = 128
    full = core & pd.to_numeric(_series(exp01, "feedback_dim"), errors="coerce").eq(full_dim)
    first, second, units = _paired_rows(exp01, d4, full, "latent_r2", "seed")
    results.append(
        _paired_claim(
            claim_id="A2_d4_r2_noninferior_full",
            experiment="exp01",
            metric="latent_r2",
            comparison=f"aligned d=4 minus aligned d={full_dim}",
            stats_unit="seed",
            candidate=first,
            reference=second,
            unit_ids=units,
            n_planned=SEED_PLAN,
            n_failed=_failed_units(exp01, "seed", d4 | full),
            minimum_units=SEED_PLAN,
            support_low=-0.01,
            oppose_below=-0.01,
            criterion="one-sided non-inferiority margin is -0.01 latent R2",
        )
    )
    orthogonal = (
        _eq(exp01, "grid", "core")
        & _eq(exp01, "feedback_mode", "orthogonal")
        & pd.to_numeric(_series(exp01, "feedback_dim"), errors="coerce").eq(4)
    )
    first, second, units = _paired_rows(exp01, d4, orthogonal, "latent_r2", "seed")
    results.append(
        _paired_claim(
            claim_id="A3_alignment_is_necessary",
            experiment="exp01",
            metric="latent_r2",
            comparison="aligned d=4 minus orthogonal d=4",
            stats_unit="seed",
            candidate=first,
            reference=second,
            unit_ids=units,
            n_planned=SEED_PLAN,
            n_failed=_failed_units(exp01, "seed", d4 | orthogonal),
            minimum_units=SEED_PLAN,
            support_low=0.10,
            oppose_below=0.02,
            criterion="aligned latent-R2 advantage CI lower bound is at least 0.10",
        )
    )

    # Phase 2: primary 80/20 E/I architecture, both oracle and learned gates.
    phase2_names = {"exp02_context_ei_oracle_gate", "exp03_context_ei_learned_gate"}
    phase2 = _primary_phase2(formal.loc[_series(formal, "experiment").isin(phase2_names)])
    required_experiments = {(name,) for name in phase2_names}
    local_mask = _eq(phase2, "condition", "local")
    bptt_mask = _eq(phase2, "condition", "bptt")
    local, bptt, units = _paired_rows(
        phase2,
        local_mask,
        bptt_mask,
        "accuracy",
        "seed",
        pair_columns=("experiment",),
        required_pairs=required_experiments,
    )
    b1 = _paired_claim(
        claim_id="B1_local_reaches_task_threshold",
        experiment="exp02/03",
        metric="accuracy",
        comparison="local minus 90% of paired BPTT accuracy",
        stats_unit="seed",
        candidate=local,
        reference=0.9 * bptt,
        unit_ids=units,
        n_planned=SEED_PLAN,
        n_failed=_failed_units(phase2, "seed", local_mask | bptt_mask),
        minimum_units=SEED_PLAN,
        support_low=0.0,
        oppose_below=0.0,
        criterion="local accuracy >= 0.85 OR >= 90% of paired BPTT (95% CI)",
    )
    if b1.estimate is not None:
        absolute = paired_bootstrap(
            local,
            np.full(len(local), 0.85),
            unit_ids=units,
            replicate_unit="seed",
            n_resamples=5000,
            confidence=0.95,
            seed=1,
        )
        support = b1.ci_low is not None and (b1.ci_low >= 0.0 or absolute.ci_low >= 0.0)
        oppose = b1.ci_high is not None and b1.ci_high < 0.0 and absolute.ci_high < 0.0
        b1 = replace(
            b1,
            conclusion="support" if support else ("oppose" if oppose else "inconclusive"),
            note=f"{b1.note}; absolute accuracy-minus-0.85 CI [{absolute.ci_low:.6g}, {absolute.ci_high:.6g}]",
        )
    results.append(b1)

    no_gate_mask = _eq(phase2, "condition", "no-gate")
    no_gate, local_switch, units = _paired_rows(
        phase2,
        no_gate_mask,
        local_mask,
        "switch_cost",
        "seed",
        pair_columns=("experiment",),
        required_pairs=required_experiments,
    )
    results.append(
        _paired_claim(
            claim_id="B2_gate_reduces_switch_cost",
            experiment="exp02/03",
            metric="switch_cost",
            comparison="no-gate switch cost minus gated-local switch cost",
            stats_unit="seed",
            candidate=no_gate,
            reference=local_switch,
            unit_ids=units,
            n_planned=SEED_PLAN,
            n_failed=_failed_units(phase2, "seed", no_gate_mask | local_mask),
            minimum_units=SEED_PLAN,
            support_low=0.0,
            oppose_below=0.0,
            criterion="no-gate switch cost exceeds gated-local switch cost (paired CI)",
        )
    )

    exp02 = phase2.loc[_eq(phase2, "experiment", "exp02_context_ei_oracle_gate")].copy()
    stability_metric = next(
        (
            metric
            for metric in ("jacobian_max_real_part", "jacobian_unstable_fraction")
            if metric in exp02 and pd.to_numeric(exp02[metric], errors="coerce").notna().any()
        ),
        "jacobian_max_real_part",
    )
    no_home_mask = _eq(exp02, "condition", "no-homeostasis")
    exp02_local_mask = _eq(exp02, "condition", "local")
    no_home, home, units = _paired_rows(
        exp02,
        no_home_mask,
        exp02_local_mask,
        stability_metric,
        "seed",
    )
    results.append(
        _paired_claim(
            claim_id="B3_homeostasis_stabilizes",
            experiment="exp02",
            metric=stability_metric,
            comparison="no-homeostasis minus local E/I instability",
            stats_unit="seed",
            candidate=no_home,
            reference=home,
            unit_ids=units,
            n_planned=SEED_PLAN,
            n_failed=_failed_units(
                exp02, "seed", no_home_mask | exp02_local_mask
            ),
            minimum_units=SEED_PLAN,
            support_low=0.0,
            oppose_below=0.0,
            criterion="removing inhibitory homeostasis increases Jacobian instability",
        )
    )

    rank_metric = next(
        (
            metric
            for metric in ("raw_update_effective_rank", "total_update_effective_rank")
            if metric in phase2 and pd.to_numeric(phase2[metric], errors="coerce").notna().any()
        ),
        "raw_update_effective_rank",
    )
    full_feedback_mask = _eq(phase2, "condition", "full-feedback")
    full_rank, local_rank, units = _paired_rows(
        phase2,
        full_feedback_mask,
        local_mask,
        rank_metric,
        "seed",
        pair_columns=("experiment",),
        required_pairs=required_experiments,
    )
    results.append(
        _paired_claim(
            claim_id="B4_local_rank_below_full_feedback",
            experiment="exp02/03",
            metric=rank_metric,
            comparison="full-feedback update rank minus low-dimensional local update rank",
            stats_unit="seed",
            candidate=full_rank,
            reference=local_rank,
            unit_ids=units,
            n_planned=SEED_PLAN,
            n_failed=_failed_units(
                phase2, "seed", full_feedback_mask | local_mask
            ),
            minimum_units=SEED_PLAN,
            support_low=0.0,
            oppose_below=0.0,
            criterion="local three-factor update rank is lower than full-feedback rank",
        )
    )

    # Phase 2.5: phase gating only after exact source/rate/spike/coupling matching.
    exp04 = formal.loc[_eq(formal, "experiment", "exp04_phase_gating")].copy()
    in_phase_mask = _eq(exp04, "phase_condition", "in_phase")
    no_osc_mask = _eq(exp04, "phase_condition", "no_oscillation")
    in_phase, no_osc, units = _paired_rows(
        exp04,
        in_phase_mask,
        no_osc_mask,
        "decoding_accuracy",
        "seed",
    )
    exact, exact_note = _phase_match_is_exact(exp04)
    if exact:
        c1 = _paired_claim(
            claim_id="C1_phase_effect_survives_rate_match",
            experiment="exp04",
            metric="decoding_accuracy",
            comparison="in-phase minus no-oscillation decoding accuracy",
            stats_unit="seed",
            candidate=in_phase,
            reference=no_osc,
            unit_ids=units,
            n_planned=SEED_PLAN,
            n_failed=_failed_units(exp04, "seed", in_phase_mask | no_osc_mask),
            minimum_units=SEED_PLAN,
            support_low=0.02,
            oppose_below=0.02,
            criterion="exactly matched in-phase accuracy advantage CI exceeds 0.02",
        )
        c1 = replace(c1, note=f"{c1.note}; {exact_note}")
    else:
        c1 = _inconclusive(
            "C1_phase_effect_survives_rate_match",
            "exp04",
            "decoding_accuracy",
            "in-phase minus no-oscillation decoding accuracy",
            "seed",
            "exactly matched in-phase accuracy advantage CI exceeds 0.02",
            exact_note,
            n_planned=SEED_PLAN,
            n_complete=len(set(units.tolist())),
            n_failed=_failed_units(exp04, "seed", in_phase_mask | no_osc_mask),
        )
    results.append(c1)

    # Phase 3: folds are averaged within session and then within animal.
    exp05 = formal.loc[_eq(formal, "experiment", "exp05_sequence_real_data")].copy()
    ordinary = ~_eq(exp05, "fold", "unseen_combination")
    ordinary_failures = ordinary | (
        _eq(exp05, "status", "failed") & _series(exp05, "fold").isna()
    )
    nll05, unit05, stats05, failed05 = _model_unit_table(
        exp05,
        "heldout_nll_per_scalar",
        fold_filter=ordinary,
        failure_filter=ordinary_failures,
    )
    params05, _, _, _ = _model_unit_table(
        exp05,
        "parameter_count",
        fold_filter=ordinary,
        failure_filter=ordinary_failures,
    )
    required_models = {"common", "shared", "full"}
    if required_models <= set(nll05) and {"shared", "full"} <= set(params05):
        joined = nll05[list(required_models)].join(
            params05[["shared", "full"]], how="inner", rsuffix="_params"
        )
        denominator = joined["common"] - joined["full"]
        retained = (joined["common"] - joined["shared"]) / denominator
        valid = np.isfinite(retained) & denominator.gt(0)
        joined, retained = joined.loc[valid], retained.loc[valid]
        plan05 = _real_plan(exp05, unit05, failed05)
        d1 = _paired_claim(
            claim_id="D1_shared_basis_near_full",
            experiment="exp05",
            metric="retained_switching_gain",
            comparison="retained full-vs-common gain minus 0.95",
            stats_unit=stats05,
            candidate=retained.to_numpy(float),
            reference=np.full(len(retained), 0.95),
            unit_ids=retained.index.to_numpy(object),
            n_planned=plan05,
            n_failed=failed05,
            minimum_units=2,
            support_low=0.0,
            oppose_below=0.0,
            criterion="retained switching gain >= 0.95 and shared parameters < full",
        )
        if d1.estimate is not None:
            parameter = paired_bootstrap(
                joined["full_params"].to_numpy(float),
                joined["shared_params"].to_numpy(float),
                unit_ids=joined.index.to_numpy(object),
                replicate_unit=stats05,
                n_resamples=5000,
                confidence=0.95,
                seed=4,
            )
            support = d1.ci_low is not None and d1.ci_low >= 0 and parameter.ci_low > 0
            oppose = d1.ci_high is not None and (d1.ci_high < 0 or parameter.ci_high <= 0)
            d1 = replace(
                d1,
                conclusion="support" if support else ("oppose" if oppose else "inconclusive"),
                note=f"{d1.note}; full-minus-shared parameter CI [{parameter.ci_low:.6g}, {parameter.ci_high:.6g}]",
            )
    else:
        d1 = _inconclusive(
            "D1_shared_basis_near_full", "exp05", "retained_switching_gain",
            "shared versus common/full", stats05,
            "retained switching gain >= 0.95 and shared parameters < full",
            "complete common/shared/full formal panel unavailable",
            n_planned=_real_plan(exp05, unit05, failed05), n_failed=failed05,
        )
    results.append(d1)

    unseen = _eq(exp05, "fold", "unseen_combination")
    unseen_failures = unseen | (
        _eq(exp05, "status", "failed") & _series(exp05, "fold").isna()
    )
    unseen05, unseen_unit, unseen_stats, unseen_failed = _model_unit_table(
        exp05,
        "heldout_nll_per_scalar",
        fold_filter=unseen,
        failure_filter=unseen_failures,
    )
    if {"shared", "full"} <= set(unseen05):
        unseen05 = unseen05[["shared", "full"]].dropna()
        d2 = _paired_claim(
            claim_id="D2_unseen_sequence_generalization",
            experiment="exp05",
            metric="heldout_nll_per_scalar",
            comparison="full minus shared NLL on unseen combinations",
            stats_unit=unseen_stats,
            candidate=unseen05["full"].to_numpy(float),
            reference=unseen05["shared"].to_numpy(float),
            unit_ids=unseen05.index.to_numpy(object),
            n_planned=_real_plan(exp05, unseen_unit, unseen_failed),
            n_failed=unseen_failed,
            minimum_units=2,
            support_low=0.0,
            oppose_below=0.0,
            criterion="shared held-out NLL is below full LDS on unseen combinations",
        )
    else:
        d2 = _inconclusive(
            "D2_unseen_sequence_generalization", "exp05", "heldout_nll_per_scalar",
            "full minus shared NLL on unseen combinations", unseen_stats,
            "shared held-out NLL is below full LDS on unseen combinations",
            "complete unseen-combination shared/full panel unavailable",
            n_planned=_real_plan(exp05, unseen_unit, unseen_failed), n_failed=unseen_failed,
        )
    results.append(d2)

    # Phase 4: hidden-context IBL switching LDS and descriptive lead/lag.
    exp06 = formal.loc[_eq(formal, "experiment", "exp06_ibl_context_switch")].copy()
    model06 = _series(exp06, "model_family").isin(["common", "shared", "full"])
    model06_failures = model06 | (
        _eq(exp06, "status", "failed") & _series(exp06, "model_family").isna()
    )
    nll06, unit06, stats06, failed06 = _model_unit_table(
        exp06,
        "heldout_nll_per_scalar",
        fold_filter=model06,
        failure_filter=model06_failures,
    )
    if required_models <= set(nll06):
        panel = nll06[list(required_models)].dropna()
        denominator = panel["common"] - panel["full"]
        retained = (panel["common"] - panel["shared"]) / denominator
        valid = np.isfinite(retained) & denominator.gt(0)
        panel, retained = panel.loc[valid], retained.loc[valid]
        e1 = _paired_claim(
            claim_id="E1_ibl_shared_switching",
            experiment="exp06",
            metric="heldout_nll_per_scalar",
            comparison="common minus shared hidden-context NLL",
            stats_unit=stats06,
            candidate=panel["common"].to_numpy(float),
            reference=panel["shared"].to_numpy(float),
            unit_ids=panel.index.to_numpy(object),
            n_planned=_real_plan(exp06, unit06, failed06),
            n_failed=failed06,
            minimum_units=2,
            support_low=0.0,
            oppose_below=0.0,
            criterion="shared improves on common and retains >= 0.95 of full-model gain",
        )
        if e1.estimate is not None:
            gain = paired_bootstrap(
                retained.to_numpy(float),
                np.full(len(retained), 0.95),
                unit_ids=retained.index.to_numpy(object),
                replicate_unit=stats06,
                n_resamples=5000,
                confidence=0.95,
                seed=5,
            )
            support = e1.ci_low is not None and e1.ci_low > 0 and gain.ci_low >= 0
            oppose = e1.ci_high is not None and (e1.ci_high <= 0 or gain.ci_high < 0)
            e1 = replace(
                e1,
                conclusion="support" if support else ("oppose" if oppose else "inconclusive"),
                note=f"{e1.note}; retained-gain-minus-0.95 CI [{gain.ci_low:.6g}, {gain.ci_high:.6g}]",
            )
    else:
        e1 = _inconclusive(
            "E1_ibl_shared_switching", "exp06", "heldout_nll_per_scalar",
            "shared versus common/full hidden-context LDS", stats06,
            "shared improves on common and retains >= 0.95 of full-model gain",
            "complete common/shared/full formal panel unavailable",
            n_planned=_real_plan(exp06, unit06, failed06), n_failed=failed06,
        )
    results.append(e1)

    lead_mask = _eq(exp06, "model_family", "lead_lag")
    schedule_valid = (
        not exp06.loc[
            lead_mask & _complete(exp06), "condition_schedule_observed"
        ].fillna(True).astype(bool).any()
        if "condition_schedule_observed" in exp06
        else False
    )
    if "lead_lag_is_causal_claim" in exp06:
        schedule_valid &= not exp06.loc[
            lead_mask & _complete(exp06), "lead_lag_is_causal_claim"
        ].fillna(True).astype(bool).any()
    lead, lead_ids, lead_unit, lead_stats, lead_failed = _real_scalar_values(
        exp06, "latent_lead_trials", lead_mask
    )
    if schedule_valid:
        e2 = _paired_claim(
            claim_id="E2_latent_precedes_behavior_bias",
            experiment="exp06",
            metric="latent_lead_trials",
            comparison="latent lead minus zero trials",
            stats_unit=lead_stats,
            candidate=lead,
            reference=np.zeros(len(lead)),
            unit_ids=lead_ids,
            n_planned=_real_plan(exp06, lead_unit, lead_failed),
            n_failed=lead_failed,
            minimum_units=2,
            support_low=0.0,
            oppose_below=0.0,
            criterion="independent-unit bootstrap CI of latent lead is above zero",
        )
        e2 = replace(e2, note=f"{e2.note}; descriptive temporal association, not causal")
    else:
        e2 = _inconclusive(
            "E2_latent_precedes_behavior_bias", "exp06", "latent_lead_trials",
            "latent lead minus zero trials", lead_stats,
            "independent-unit bootstrap CI of latent lead is above zero",
            "hidden-context schedule provenance is absent or held-out truth was observed",
            n_planned=_real_plan(exp06, lead_unit, lead_failed),
            n_complete=len(set(lead_ids.tolist())), n_failed=lead_failed,
        )
    results.append(e2)

    return _apply_full_family_holm(results)
