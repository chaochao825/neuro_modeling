"""Pre-registered, evidence-gated classification of the core propositions.

Only ``formal`` artifacts enter this module.  Seed-level propositions require
their complete experiment-specific seed plan.  Real-data folds are first
averaged within a session and, when ``animal_id`` is available, sessions are
then averaged within animal.
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
from src.analysis.p2_protocol import FORMAL_P2_PROTOCOL_ID


SEED_PLAN = 20
P0_SEED_PLAN = 30
P0_PLANNED_SEEDS = frozenset(range(P0_SEED_PLAN))
P0_BUDGET_PANELS = ("l1", "l2")
P0_REQUIRED_CLAIMS = (
    "P0a_aligned_task_improves_prediction_vs_frozen",
    "P0b_aligned_task_beats_shuffled",
    "P0c_aligned_adds_value_over_matched_homeostasis",
    "P0d_local_absolute_accuracy",
    "P0e_local_noninferior_tuned_bptt",
    "P0f_local_noninferior_tuned_gru",
)
P2_SEED_PLAN = 30
P2_PLANNED_SEEDS = frozenset(range(P2_SEED_PLAN))
P2_Q = (0.55, 0.70, 0.85, 1.0)
P2_H = (0.01, 0.05, 0.10, 0.20)
P2_GATES = (
    "oracle_bayes",
    "supervised_upper_bound",
    "learned_hmm",
    "md_recurrent_belief",
    "no_gate",
)
P2_INTERVENTIONS = ("clamp", "delay", "shuffle")
P2_PRIMARY_CLAIMS = (
    "P2a_hmm_context_nll",
    "P2b_md_context_nll",
    "P2c_md_context_brier",
    "P2d_md_calibration",
    "P2e_md_switch_latency",
    "P2f_md_false_switch",
    "P2g_md_behavior",
    "P2h_md_retains_oracle_gain",
    "P2i_md_energy",
    "P2j_clamp_causal",
    "P2k_delay_causal",
    "P2l_shuffle_causal",
)
P2_REQUIRED_CLAIMS = (
    "P2b_md_context_nll",
    "P2c_md_context_brier",
    "P2d_md_calibration",
    "P2e_md_switch_latency",
    "P2f_md_false_switch",
    "P2g_md_behavior",
    "P2h_md_retains_oracle_gain",
    "P2j_clamp_causal",
    "P2k_delay_causal",
    "P2l_shuffle_causal",
)


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
    valid = candidate.ndim == reference.ndim == unit_ids.ndim == 1 and len(
        candidate
    ) == len(reference) == len(unit_ids)
    if not valid:
        raise ValueError("paired claim arrays must be aligned one-dimensional vectors")
    finite = np.isfinite(candidate) & np.isfinite(reference)
    candidate, reference, unit_ids = (
        candidate[finite],
        reference[finite],
        unit_ids[finite],
    )
    n_complete = len(set(unit_ids.tolist()))
    if n_failed or n_complete < minimum_units or n_complete < n_planned:
        reasons: list[str] = []
        if n_failed:
            reasons.append(f"{n_failed} planned independent unit(s) failed")
        if n_complete < minimum_units:
            reasons.append(
                f"requires at least {minimum_units} complete independent units"
            )
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
    if conclusion == "support" and support_low is not None and support_high is not None:
        lower_test = paired_wilcoxon(
            candidate,
            reference + support_low,
            unit_ids=unit_ids,
            replicate_unit=stats_unit,
            alternative="greater",
        )
        upper_test = paired_wilcoxon(
            candidate,
            reference + support_high,
            unit_ids=unit_ids,
            replicate_unit=stats_unit,
            alternative="less",
        )
        raw_p_value = max(lower_test.p_value, upper_test.p_value)
        p_value_definition = "two-one-sided margin tests"
    elif conclusion == "support" and support_low is not None:
        directional = paired_wilcoxon(
            candidate,
            reference + support_low,
            unit_ids=unit_ids,
            replicate_unit=stats_unit,
            alternative="greater",
        )
        raw_p_value = directional.p_value
        p_value_definition = "one-sided support-margin test"
    elif conclusion == "support" and support_high is not None:
        directional = paired_wilcoxon(
            candidate,
            reference + support_high,
            unit_ids=unit_ids,
            replicate_unit=stats_unit,
            alternative="less",
        )
        raw_p_value = directional.p_value
        p_value_definition = "one-sided support-margin test"
    elif (
        conclusion == "oppose"
        and oppose_below is not None
        and bootstrap.ci_high < oppose_below
    ):
        directional = paired_wilcoxon(
            candidate,
            reference + oppose_below,
            unit_ids=unit_ids,
            replicate_unit=stats_unit,
            alternative="less",
        )
        raw_p_value = directional.p_value
        p_value_definition = "one-sided oppose-margin test"
    elif conclusion == "oppose" and oppose_above is not None:
        directional = paired_wilcoxon(
            candidate,
            reference + oppose_above,
            unit_ids=unit_ids,
            replicate_unit=stats_unit,
            alternative="greater",
        )
        raw_p_value = directional.p_value
        p_value_definition = "one-sided oppose-margin test"
    else:
        directional = paired_wilcoxon(
            candidate,
            reference,
            unit_ids=unit_ids,
            replicate_unit=stats_unit,
        )
        raw_p_value = directional.p_value
        p_value_definition = "two-sided zero-difference diagnostic"
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
        p_value=raw_p_value,
        multiplicity_method="holm(full_registered_family)",
        conclusion=conclusion,
        criterion=criterion,
        note=(
            "paired 95% bootstrap CI at the declared independent-unit level; "
            f"{p_value_definition} awaits full-family Holm adjustment"
        ),
    )


def _series(frame: pd.DataFrame, column: str, default: object = None) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index, dtype=object)


def _eq(frame: pd.DataFrame, column: str, value: object) -> pd.Series:
    return _series(frame, column).eq(value)


def _strict_boolean_value(value: object) -> bool | None:
    """Parse only unambiguous booleans from JSON or compact CSV artifacts."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "true":
            return True
        if token == "false":
            return False
    return None


def _strict_boolean_requirement(
    frame: pd.DataFrame, column: str, expected: bool
) -> bool:
    """Require an explicit boolean value, including safe compact-CSV tokens."""

    if frame.empty or column not in frame or not isinstance(expected, bool):
        return False

    parsed = frame[column].map(_strict_boolean_value)
    return bool(parsed.notna().all() and parsed.eq(expected).all())


def _strict_boolean_mask(frame: pd.DataFrame, column: str, expected: bool) -> pd.Series:
    """Return a serialization-safe boolean selector for artifact rows."""

    if column not in frame or not isinstance(expected, bool):
        return pd.Series(False, index=frame.index, dtype=bool)
    parsed = frame[column].map(_strict_boolean_value)
    return parsed.notna() & parsed.eq(expected)


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
            _series(frame, "run_level_failure", False).fillna(False).astype(bool)
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
        run_times["attempt_time"] = run_times["started"].fillna(run_times["recorded"])
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
        run_level_failure = (
            _series(latest, "run_level_failure", False).fillna(False).astype(bool).any()
        )
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


def _prepare_p0_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict P0 evidence to preregistered seeds and restore sparse dimensions.

    Failed condition artifacts historically carried only their stable condition
    name.  Recovering the boolean mechanism axes from that name lets failures
    remain in the same evidential scope as complete rows.  Existing explicit
    dimensions always take precedence.
    """

    if frame.empty or "seed" not in frame:
        return frame.iloc[:0].copy()
    seeds = pd.to_numeric(frame["seed"], errors="coerce")
    planned = seeds.isin(P0_PLANNED_SEEDS)
    prepared = frame.loc[planned].copy()
    prepared["seed"] = seeds.loc[planned].astype(int)

    condition = _series(prepared, "condition", "").fillna("").astype(str)
    parts = condition.str.split("__", n=2, expand=True).reindex(columns=range(3))
    mechanism_token = parts[0].fillna("").str.lower()
    parsed_flags = {
        "task_plasticity_enabled": mechanism_token.str.contains("task", regex=False),
        "homeostasis_enabled": mechanism_token.str.contains("homeostasis", regex=False),
        "normalization_enabled": mechanism_token.str.contains(
            "normalization", regex=False
        ),
    }
    for column, parsed in parsed_flags.items():
        existing = _series(prepared, column)
        prepared[column] = existing.where(existing.notna(), parsed)

    parsed_feedback = parts[1].where(parts[1].notna(), None)
    parsed_budget = parts[2].where(parts[2].notna(), None)
    for column, parsed in (
        ("feedback_mode", parsed_feedback),
        ("budget_norm", parsed_budget),
    ):
        existing = _series(prepared, column)
        prepared[column] = existing.where(existing.notna(), parsed)
    return prepared


def _p0_failed_seed_ids(frame: pd.DataFrame, mask: pd.Series) -> set[int]:
    """Return failed/invalid P0 seeds, including completed budget shortfalls."""

    if frame.empty or "seed" not in frame:
        return set()
    scope = mask.reindex(frame.index, fill_value=False)
    status = _series(frame, "status")
    run_level = _series(frame, "run_level_failure", False).fillna(False).astype(bool)
    condition = _series(frame, "condition", "").fillna("").astype(str)
    local_cell = condition.str.contains("__", regex=False)
    setup_failure = condition.eq("setup") & status.isin({"failed", "invalid"})
    invalid_budget = (
        scope
        & status.eq("complete")
        & local_cell
        & ~_strict_boolean_mask(frame, "budget_match_valid", True)
    )
    failed = (
        (scope & status.isin({"failed", "invalid"}))
        | (run_level & status.eq("failed"))
        | setup_failure
    )
    selected = pd.to_numeric(
        frame.loc[failed | invalid_budget, "seed"], errors="coerce"
    ).dropna()
    return {int(value) for value in selected.tolist() if int(value) in P0_PLANNED_SEEDS}


def _joint_budget_panel_claim(
    *,
    claim_id: str,
    experiment: str,
    metric: str,
    comparison: str,
    stats_unit: str,
    criterion: str,
    panels: dict[str, ClaimResult],
    panel_complete_seed_ids: dict[str, set[int]],
    failed_seed_ids: set[int],
) -> ClaimResult:
    """Combine L1/L2 panel claims with an intersection-union decision.

    No panel values are averaged.  A directional conclusion requires the same
    conclusion in both panels; its raw joint p value is the maximum of the two
    panel p values and is subsequently included in the ordinary Holm family.
    """

    if set(panels) != set(P0_BUDGET_PANELS):
        raise ValueError("P0 joint claims require exactly l1 and l2 panels")
    if set(panel_complete_seed_ids) != set(P0_BUDGET_PANELS):
        raise ValueError("P0 joint claims require complete seed IDs for both panels")
    ordered = [panels[name] for name in P0_BUDGET_PANELS]
    jointly_complete = set.intersection(
        *(panel_complete_seed_ids[name] for name in P0_BUDGET_PANELS)
    )
    n_complete = len(jointly_complete)
    panel_notes = "; ".join(
        (
            f"{name}: conclusion={item.conclusion}, n={item.n_complete}, "
            f"estimate={item.estimate}, CI=[{item.ci_low}, {item.ci_high}], "
            f"raw_p={item.p_value}"
        )
        for name, item in zip(P0_BUDGET_PANELS, ordered, strict=True)
    )
    if failed_seed_ids or n_complete < P0_SEED_PLAN:
        reasons: list[str] = []
        if failed_seed_ids:
            reasons.append(
                "failed/invalid planned seeds="
                + ",".join(str(value) for value in sorted(failed_seed_ids))
            )
        if n_complete < P0_SEED_PLAN:
            reasons.append(
                f"only {n_complete}/{P0_SEED_PLAN} planned seeds complete in both panels"
            )
        return _inconclusive(
            claim_id,
            experiment,
            metric,
            comparison,
            stats_unit,
            criterion,
            "; ".join(reasons) + f"; panel audit: {panel_notes}",
            n_planned=P0_SEED_PLAN,
            n_complete=n_complete,
            n_failed=len(failed_seed_ids),
        )

    conclusions = {item.conclusion for item in ordered}
    unanimous = conclusions in ({"support"}, {"oppose"})
    p_values = [item.p_value for item in ordered]
    if not unanimous or any(
        value is None or not np.isfinite(value) for value in p_values
    ):
        return ClaimResult(
            claim_id=claim_id,
            experiment=experiment,
            metric=metric,
            comparison=comparison,
            stats_unit=stats_unit,
            n_planned=P0_SEED_PLAN,
            n_complete=n_complete,
            n_failed=0,
            estimate=None,
            ci_low=None,
            ci_high=None,
            effect_size=None,
            p_value=None,
            multiplicity_method="holm(full_registered_family)",
            conclusion="inconclusive",
            criterion=criterion,
            note=(
                "L1/L2 panel conclusions are not unanimous; no cross-panel "
                f"averaging is permitted; panel audit: {panel_notes}"
            ),
        )

    conclusion = ordered[0].conclusion
    estimates = [float(item.estimate) for item in ordered if item.estimate is not None]
    ci_lows = [float(item.ci_low) for item in ordered if item.ci_low is not None]
    ci_highs = [float(item.ci_high) for item in ordered if item.ci_high is not None]
    if len(estimates) != 2 or len(ci_lows) != 2 or len(ci_highs) != 2:
        raise RuntimeError(
            "complete P0 panels must expose estimates and confidence bounds"
        )
    # Report the conservative panel estimate in the declared direction and an
    # envelope over both intervals.  Individual values remain in ``note``.
    estimate = min(estimates) if conclusion == "support" else max(estimates)
    return ClaimResult(
        claim_id=claim_id,
        experiment=experiment,
        metric=metric,
        comparison=comparison,
        stats_unit=stats_unit,
        n_planned=P0_SEED_PLAN,
        n_complete=n_complete,
        n_failed=0,
        estimate=estimate,
        ci_low=min(ci_lows),
        ci_high=max(ci_highs),
        effect_size=estimate,
        p_value=float(max(float(value) for value in p_values if value is not None)),
        multiplicity_method="holm(full_registered_family)",
        conclusion=conclusion,
        criterion=criterion,
        note=(
            "intersection-union across separately matched L1/L2 panels; raw joint "
            f"p=max(panel p) awaits full-family Holm adjustment; panel audit: {panel_notes}"
        ),
    )


def _p0_overall_gate(
    adjusted_claims: list[ClaimResult],
    *,
    complete_seed_ids_by_claim: dict[str, set[int]],
    failed_seed_ids_by_claim: dict[str, set[int]],
) -> ClaimResult:
    """Derive the non-inferential P0 stage gate after constituent Holm tests."""

    lookup = {item.claim_id: item for item in adjusted_claims}
    constituents = [lookup.get(claim_id) for claim_id in P0_REQUIRED_CLAIMS]
    conclusions = [
        item.conclusion if item is not None else "inconclusive" for item in constituents
    ]
    if all(value == "support" for value in conclusions):
        conclusion = "support"
    elif any(value == "oppose" for value in conclusions):
        conclusion = "oppose"
    else:
        conclusion = "inconclusive"
    audit = "; ".join(
        f"{claim_id}={value}"
        for claim_id, value in zip(P0_REQUIRED_CLAIMS, conclusions, strict=True)
    )
    complete_sets = [
        complete_seed_ids_by_claim.get(claim_id, set())
        for claim_id in P0_REQUIRED_CLAIMS
    ]
    jointly_complete = set.intersection(*complete_sets) if complete_sets else set()
    any_failed = set().union(
        *(
            failed_seed_ids_by_claim.get(claim_id, set())
            for claim_id in P0_REQUIRED_CLAIMS
        )
    )
    return ClaimResult(
        claim_id="P0_overall",
        experiment="exp07",
        metric="noninferential_stage_gate",
        comparison="conjunction of Holm-adjusted P0a--P0f conclusions",
        stats_unit="seed",
        n_planned=P0_SEED_PLAN,
        n_complete=len(jointly_complete),
        n_failed=len(any_failed),
        estimate=None,
        ci_low=None,
        ci_high=None,
        effect_size=None,
        p_value=None,
        multiplicity_method="derived_after_holm(no_additional_test)",
        conclusion=conclusion,
        criterion=(
            "support iff every Holm-adjusted P0a--P0f claim supports; oppose iff "
            "at least one opposes; otherwise inconclusive"
        ),
        note=f"non-inferential stage gate; {audit}",
    )


@dataclass(frozen=True)
class _StrictP2Panel:
    """Leakage- and pairing-validated formal hidden-context seed panel."""

    frame: pd.DataFrame
    complete_seed_ids: frozenset[int]
    failed_seed_ids: frozenset[int]
    issues: tuple[str, ...]


_P2_METRICS = (
    "context_nll",
    "context_brier",
    "context_ece",
    "switch_latency_trials",
    "false_switch_rate",
    "behavior_balanced_accuracy",
    "energy_proxy_per_trial",
)
_P2_COMMON_BOOLEAN_PROVENANCE = {
    "hidden_context_task": True,
    "cue_encodes_observation_not_state": True,
    "gate_test_accessed_true_context": False,
    "third_factor_accessed_true_context": False,
    "oracle_warm_start_used": False,
    "md_fit_used_context_bias": False,
    "gate_fit_accessed_task_target": False,
    "gate_test_accessed_task_target": False,
    "gate_test_future_observations_accessed": False,
    "state_label_alignment_accessed_true_context": False,
    "test_switch_boundaries_accessed_by_model": False,
    "preprocessing_fit_train_only": True,
    "hyperparameters_preregistered": True,
    "dev_used_for_selection": False,
    "train_dev_test_episode_disjoint": True,
    "belief_online_causal": True,
    "predictions_frozen_before_truth_scoring": True,
}
_P2_PAIRING_IDENTIFIERS = (
    "random_tape_id",
    "hidden_state_tape_id",
    "observation_tape_id",
    "task_tape_id",
    "noise_tape_id",
    "network_initialization_id",
    "split_id",
    "readout_fit_data_id",
    "readout_protocol_id",
)
_P2_INTERVENTION_PROVENANCE = {
    "intervention_postfit": True,
    "intervention_reuses_intact_checkpoint": True,
    "intervention_reuses_intact_readout": True,
    "intervention_permutation_accessed_true_context": False,
}


def _p2_expected_cells() -> frozenset[tuple[float, float, str, str]]:
    base = {
        (reliability, hazard, gate, "none")
        for reliability in P2_Q
        for hazard in P2_H
        for gate in P2_GATES
    }
    interventions = {
        (reliability, hazard, "md_recurrent_belief", intervention)
        for reliability in P2_Q
        for hazard in P2_H
        for intervention in P2_INTERVENTIONS
    }
    return frozenset(base | interventions)


def _p2_grid_value(value: object, allowed: tuple[float, ...]) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    matches = [item for item in allowed if np.isclose(numeric, item, atol=1e-12)]
    return matches[0] if len(matches) == 1 else None


def _p2_identifier_is_present(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, (list, dict)):
        try:
            if bool(pd.isna(value)):
                return False
        except (TypeError, ValueError):
            pass
    return bool(str(value).strip())


def _strict_p2_panel(frame: pd.DataFrame) -> _StrictP2Panel:
    """Require the exact 128-cell grid before any seed-level averaging."""

    empty = _StrictP2Panel(
        frame=frame.iloc[:0].copy(),
        complete_seed_ids=frozenset(),
        failed_seed_ids=frozenset(),
        issues=(),
    )
    if frame.empty:
        return replace(empty, issues=("formal exp09 evidence is unavailable",))
    if "seed" not in frame:
        return replace(empty, issues=("exp09 rows are missing the seed field",))

    numeric_seed = pd.to_numeric(frame["seed"], errors="coerce")
    planned_mask = numeric_seed.isin(P2_PLANNED_SEEDS)
    selected = frame.loc[planned_mask].copy()
    selected["seed"] = numeric_seed.loc[planned_mask].astype(int)
    observed_seeds = frozenset(int(value) for value in selected["seed"].tolist())
    if selected.empty:
        return replace(
            empty,
            issues=("no preregistered exp09 seed in the exact 0..29 plan",),
        )

    required_columns = {
        "seed",
        "status",
        "cue_reliability",
        "context_hazard",
        "gate_model",
        "intervention",
        "eligible_switch_count",
        "gate_fit_accessed_true_context",
        "gate_fit_used_batch_smoothing",
        "eligible_for_p2_support",
        "gate_received_true_q_h",
        "gate_fit_supervision",
        "true_context_access_scope",
        "checkpoint_id",
        "readout_id",
        "belief_trajectory_id",
        "intact_belief_trajectory_id",
        "p2_protocol_id",
        "train_trial_count",
        "dev_trial_count",
        "test_trial_count",
        "latency_limit_trials",
        "latency_sustain_trials",
        "posterior_threshold",
        "minimum_state_duration",
        "switch_tolerance_trials",
        "minimum_eligible_switches",
        "delay_trials",
        *_P2_METRICS,
        *_P2_COMMON_BOOLEAN_PROVENANCE,
        *_P2_PAIRING_IDENTIFIERS,
        *_P2_INTERVENTION_PROVENANCE,
    }
    missing_columns = sorted(required_columns - set(selected))
    if missing_columns:
        return replace(
            empty,
            failed_seed_ids=observed_seeds,
            issues=("missing strict exp09 columns: " + ", ".join(missing_columns),),
        )

    selected["cue_reliability"] = selected["cue_reliability"].map(
        lambda value: _p2_grid_value(value, P2_Q)
    )
    selected["context_hazard"] = selected["context_hazard"].map(
        lambda value: _p2_grid_value(value, P2_H)
    )
    selected["gate_model"] = selected["gate_model"].astype("string")
    selected["intervention"] = selected["intervention"].astype("string")

    invalid_seeds: set[int] = set()
    incomplete_seeds: set[int] = set()
    issues: list[str] = []
    expected_cells = _p2_expected_cells()
    for seed in sorted(P2_PLANNED_SEEDS):
        rows = selected.loc[selected["seed"].eq(seed)]
        if rows.empty:
            incomplete_seeds.add(seed)
            continue
        invalid_dimensions = (
            rows["cue_reliability"].isna()
            | rows["context_hazard"].isna()
            | ~rows["gate_model"].isin(P2_GATES)
            | ~rows["intervention"].isin(("none", *P2_INTERVENTIONS))
        )
        if invalid_dimensions.any():
            invalid_seeds.add(seed)
            issues.append(f"seed {seed}: invalid or unregistered grid dimension")
        cells = [
            (
                float(row.cue_reliability),
                float(row.context_hazard),
                str(row.gate_model),
                str(row.intervention),
            )
            for row in rows.loc[~invalid_dimensions].itertuples(index=False)
        ]
        observed_cells = set(cells)
        if len(cells) != len(observed_cells):
            invalid_seeds.add(seed)
            issues.append(f"seed {seed}: duplicate exp09 condition cell")
        if observed_cells != expected_cells:
            incomplete_seeds.add(seed)
            missing_count = len(expected_cells - observed_cells)
            extra_count = len(observed_cells - expected_cells)
            issues.append(
                f"seed {seed}: incomplete exact 128-cell grid "
                f"(missing={missing_count}, extra={extra_count})"
            )
            if extra_count:
                invalid_seeds.add(seed)
        if not rows["status"].eq("complete").all():
            invalid_seeds.add(seed)
            issues.append(f"seed {seed}: failed or invalid planned exp09 cell")

    def invalidate_rows(mask: pd.Series, message: str) -> None:
        affected = set(selected.loc[mask, "seed"].astype(int).tolist())
        if affected:
            invalid_seeds.update(affected)
            issues.append(f"{message}; seeds={','.join(map(str, sorted(affected)))}")

    for field, expected in _P2_COMMON_BOOLEAN_PROVENANCE.items():
        parsed = selected[field].map(_strict_boolean_value)
        invalidate_rows(
            parsed.isna() | parsed.ne(expected),
            f"{field} is not uniformly {expected}",
        )

    supervised = selected["gate_model"].eq("supervised_upper_bound")
    fit_access = selected["gate_fit_accessed_true_context"].map(_strict_boolean_value)
    eligible = selected["eligible_for_p2_support"].map(_strict_boolean_value)
    receives_q_h = selected["gate_received_true_q_h"].map(_strict_boolean_value)
    batch_smoothing = selected["gate_fit_used_batch_smoothing"].map(
        _strict_boolean_value
    )
    oracle = selected["gate_model"].eq("oracle_bayes")
    invalidate_rows(
        fit_access.isna() | fit_access.ne(supervised),
        "gate_fit_accessed_true_context violates the supervised-only exception",
    )
    invalidate_rows(
        eligible.isna() | eligible.ne(~supervised),
        "eligible_for_p2_support does not exclude exactly the supervised upper bound",
    )
    invalidate_rows(
        receives_q_h.isna() | receives_q_h.ne(oracle),
        "gate_received_true_q_h does not isolate oracle Bayes",
    )
    learned_hmm = selected["gate_model"].eq("learned_hmm")
    invalidate_rows(
        batch_smoothing.isna() | batch_smoothing.ne(learned_hmm),
        "gate_fit_used_batch_smoothing does not isolate learned-HMM train EM",
    )

    expected_supervision = selected["gate_model"].map(
        {
            "oracle_bayes": "known_generative_params",
            "supervised_upper_bound": "train_context_labels",
            "learned_hmm": "none",
            "md_recurrent_belief": "none",
            "no_gate": "none",
        }
    )
    invalidate_rows(
        selected["gate_fit_supervision"].astype("string").ne(expected_supervision),
        "gate_fit_supervision is inconsistent with the registered gate",
    )
    expected_scope = pd.Series(
        np.where(
            supervised,
            "train_gate_fit_and_evaluation",
            "evaluation_only",
        ),
        index=selected.index,
        dtype="string",
    )
    invalidate_rows(
        selected["true_context_access_scope"].astype("string").ne(expected_scope),
        "true_context_access_scope is inconsistent with gate capability",
    )

    intervention_rows = selected["intervention"].isin(P2_INTERVENTIONS)
    for field, expected in _P2_INTERVENTION_PROVENANCE.items():
        parsed = selected[field].map(_strict_boolean_value)
        invalidate_rows(
            intervention_rows & (parsed.isna() | parsed.ne(expected)),
            f"{field} is invalid for a post-fit intervention",
        )

    protocol_valid = (
        selected["p2_protocol_id"].astype("string").eq(FORMAL_P2_PROTOCOL_ID)
    )
    invalidate_rows(~protocol_valid, "p2_protocol_id is not the formal preregistration")
    exact_protocol_values = {
        "train_trial_count": 6000.0,
        "dev_trial_count": 2000.0,
        "test_trial_count": 4000.0,
        "latency_limit_trials": 5.0,
        "latency_sustain_trials": 2.0,
        "posterior_threshold": 0.8,
        "minimum_state_duration": 5.0,
        "switch_tolerance_trials": 1.0,
        "minimum_eligible_switches": 20.0,
        "delay_trials": 1.0,
    }
    for field, expected in exact_protocol_values.items():
        observed = pd.to_numeric(selected[field], errors="coerce")
        invalidate_rows(
            ~np.isclose(observed.to_numpy(float), expected, atol=1e-12),
            f"{field} differs from the formal P2 protocol",
        )

    numeric_metrics: dict[str, pd.Series] = {}
    for metric in (*_P2_METRICS, "eligible_switch_count"):
        numeric_metrics[metric] = pd.to_numeric(selected[metric], errors="coerce")
        selected[metric] = numeric_metrics[metric]
    finite = pd.Series(True, index=selected.index, dtype=bool)
    for metric in _P2_METRICS:
        finite &= np.isfinite(numeric_metrics[metric].to_numpy(float))
    invalidate_rows(~finite, "one or more registered P2 metrics are non-finite")
    invalidate_rows(
        numeric_metrics["context_nll"].lt(0.0)
        | numeric_metrics["switch_latency_trials"].lt(0.0)
        | numeric_metrics["energy_proxy_per_trial"].le(0.0),
        "NLL/latency/energy violates its numeric domain",
    )
    for metric in (
        "context_brier",
        "context_ece",
        "false_switch_rate",
        "behavior_balanced_accuracy",
    ):
        invalidate_rows(
            ~numeric_metrics[metric].between(0.0, 1.0, inclusive="both"),
            f"{metric} lies outside [0, 1]",
        )
    switch_counts = numeric_metrics["eligible_switch_count"]
    invalidate_rows(
        ~np.isfinite(switch_counts.to_numpy(float))
        | switch_counts.lt(20.0)
        | ~np.isclose(switch_counts, np.rint(switch_counts), atol=1e-9),
        "eligible_switch_count is not an integer >=20",
    )

    for (seed, reliability, hazard), rows in selected.groupby(
        ["seed", "cue_reliability", "context_hazard"],
        dropna=False,
        sort=False,
    ):
        seed = int(seed)
        for field in _P2_PAIRING_IDENTIFIERS:
            values = rows[field]
            if (
                not values.map(_p2_identifier_is_present).all()
                or values.nunique(dropna=False) != 1
            ):
                invalid_seeds.add(seed)
                issues.append(
                    f"seed {seed}, q={reliability}, h={hazard}: "
                    f"pairing identifier {field} differs or is missing"
                )
        if rows["eligible_switch_count"].nunique(dropna=False) != 1:
            invalid_seeds.add(seed)
            issues.append(
                f"seed {seed}, q={reliability}, h={hazard}: "
                "eligible switch count differs across paired gates"
            )
        md_rows = rows.loc[rows["gate_model"].eq("md_recurrent_belief")]
        for field in ("checkpoint_id", "readout_id"):
            values = md_rows[field]
            if (
                len(values) != 4
                or not values.map(_p2_identifier_is_present).all()
                or values.nunique(dropna=False) != 1
            ):
                invalid_seeds.add(seed)
                issues.append(
                    f"seed {seed}, q={reliability}, h={hazard}: MD {field} "
                    "is not reused by all three interventions"
                )
        intact_ids = md_rows["intact_belief_trajectory_id"]
        belief_ids = md_rows["belief_trajectory_id"]
        base_md = md_rows.loc[md_rows["intervention"].eq("none")]
        intervened_md = md_rows.loc[md_rows["intervention"].isin(P2_INTERVENTIONS)]
        trajectory_valid = (
            len(md_rows) == 4
            and intact_ids.map(_p2_identifier_is_present).all()
            and intact_ids.nunique(dropna=False) == 1
            and belief_ids.map(_p2_identifier_is_present).all()
            and belief_ids.nunique(dropna=False) == 4
            and len(base_md) == 1
            and base_md["belief_trajectory_id"].iloc[0]
            == base_md["intact_belief_trajectory_id"].iloc[0]
            and len(intervened_md) == 3
            and intervened_md["belief_trajectory_id"]
            .ne(intervened_md["intact_belief_trajectory_id"])
            .all()
        )
        if not trajectory_valid:
            invalid_seeds.add(seed)
            issues.append(
                f"seed {seed}, q={reliability}, h={hazard}: MD intervention "
                "belief trajectories do not branch uniquely from one intact trajectory"
            )

    bad_seeds = invalid_seeds | incomplete_seeds
    complete_seeds = P2_PLANNED_SEEDS - bad_seeds
    eligible_frame = selected.loc[selected["seed"].isin(complete_seeds)].copy()
    if incomplete_seeds:
        issues.append(
            "incomplete planned seeds="
            + ",".join(str(value) for value in sorted(incomplete_seeds))
        )
    return _StrictP2Panel(
        frame=eligible_frame,
        complete_seed_ids=frozenset(complete_seeds),
        failed_seed_ids=frozenset(invalid_seeds),
        issues=tuple(dict.fromkeys(issues)),
    )


def _p2_gate_values(
    panel: _StrictP2Panel,
    metric: str,
    *,
    gate: str,
    intervention: str = "none",
    hazards: tuple[float, ...] = P2_H,
) -> tuple[np.ndarray, np.ndarray]:
    frame = panel.frame
    required = {"seed", "gate_model", "intervention", "context_hazard", metric}
    if frame.empty or not required <= set(frame):
        return np.array([]), np.array([], dtype=object)
    mask = (
        _eq(frame, "gate_model", gate)
        & _eq(frame, "intervention", intervention)
        & _series(frame, "context_hazard").isin(hazards)
    )
    selected = frame.loc[mask, ["seed", metric]].copy()
    if selected.empty:
        return np.array([]), np.array([], dtype=object)
    aggregated = selected.groupby("seed", as_index=False, sort=False)[metric].mean()
    return aggregated[metric].to_numpy(float), aggregated["seed"].to_numpy(object)


def _p2_paired_values(
    panel: _StrictP2Panel,
    metric: str,
    *,
    first_gate: str,
    second_gate: str,
    first_intervention: str = "none",
    second_intervention: str = "none",
    hazards: tuple[float, ...] = P2_H,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, first_ids = _p2_gate_values(
        panel,
        metric,
        gate=first_gate,
        intervention=first_intervention,
        hazards=hazards,
    )
    second, second_ids = _p2_gate_values(
        panel,
        metric,
        gate=second_gate,
        intervention=second_intervention,
        hazards=hazards,
    )
    first_frame = pd.DataFrame({"seed": first_ids, "first": first})
    second_frame = pd.DataFrame({"seed": second_ids, "second": second})
    paired = first_frame.merge(second_frame, on="seed", validate="one_to_one")
    return (
        paired["first"].to_numpy(float),
        paired["second"].to_numpy(float),
        paired["seed"].to_numpy(object),
    )


def _p2_retained_oracle_gain(
    panel: _StrictP2Panel,
) -> tuple[np.ndarray, np.ndarray]:
    frame = panel.frame
    required = {
        "seed",
        "cue_reliability",
        "context_hazard",
        "gate_model",
        "intervention",
        "behavior_balanced_accuracy",
    }
    if frame.empty or not required <= set(frame):
        return np.array([]), np.array([], dtype=object)
    selected = frame.loc[
        _eq(frame, "intervention", "none")
        & _series(frame, "gate_model").isin(
            ["md_recurrent_belief", "oracle_bayes", "no_gate"]
        ),
        [
            "seed",
            "cue_reliability",
            "context_hazard",
            "gate_model",
            "behavior_balanced_accuracy",
        ],
    ]
    if selected.empty:
        return np.array([]), np.array([], dtype=object)
    table = selected.pivot(
        index=["seed", "cue_reliability", "context_hazard"],
        columns="gate_model",
        values="behavior_balanced_accuracy",
    )
    retained = (
        table["md_recurrent_belief"]
        - table["no_gate"]
        - 0.9 * (table["oracle_bayes"] - table["no_gate"])
    )
    seed_values = retained.groupby(level="seed").mean()
    return seed_values.to_numpy(float), seed_values.index.to_numpy(object)


def _evaluate_p2_claims(
    frame: pd.DataFrame,
) -> tuple[
    list[ClaimResult],
    dict[str, set[int]],
    dict[str, set[int]],
    tuple[str, ...],
]:
    panel = _strict_p2_panel(frame)
    results: list[ClaimResult] = []
    complete_by_claim: dict[str, set[int]] = {}
    failed_by_claim: dict[str, set[int]] = {}

    def append_claim(
        *,
        claim_id: str,
        metric: str,
        comparison: str,
        criterion: str,
        candidate: np.ndarray,
        reference: np.ndarray,
        unit_ids: np.ndarray,
        support_low: float | None = None,
        support_high: float | None = None,
        oppose_below: float | None = None,
        oppose_above: float | None = None,
    ) -> None:
        complete_by_claim[claim_id] = {
            int(value) for value in unit_ids.tolist() if int(value) in P2_PLANNED_SEEDS
        }
        failed_by_claim[claim_id] = set(panel.failed_seed_ids)
        result = _paired_claim(
            claim_id=claim_id,
            experiment="exp09",
            metric=metric,
            comparison=comparison,
            stats_unit="seed",
            candidate=candidate,
            reference=reference,
            unit_ids=unit_ids,
            n_planned=P2_SEED_PLAN,
            n_failed=len(panel.failed_seed_ids),
            minimum_units=P2_SEED_PLAN,
            support_low=support_low,
            support_high=support_high,
            oppose_below=oppose_below,
            oppose_above=oppose_above,
            criterion=criterion,
            seed=len(results) + 101,
        )
        if panel.issues:
            result = replace(
                result,
                note=result.note + "; strict P2 panel: " + " | ".join(panel.issues),
            )
        results.append(result)

    no_nll, hmm_nll, units = _p2_paired_values(
        panel,
        "context_nll",
        first_gate="no_gate",
        second_gate="learned_hmm",
    )
    append_claim(
        claim_id="P2a_hmm_context_nll",
        metric="context_nll",
        comparison="no-gate minus learned-HMM context NLL",
        criterion="learned HMM improves context NLL by at least 0.02 nats/trial",
        candidate=no_nll,
        reference=hmm_nll,
        unit_ids=units,
        support_low=0.02,
        oppose_below=0.0,
    )
    no_nll, md_nll, units = _p2_paired_values(
        panel,
        "context_nll",
        first_gate="no_gate",
        second_gate="md_recurrent_belief",
    )
    append_claim(
        claim_id="P2b_md_context_nll",
        metric="context_nll",
        comparison="no-gate minus MD-belief context NLL",
        criterion="MD belief improves context NLL by at least 0.02 nats/trial",
        candidate=no_nll,
        reference=md_nll,
        unit_ids=units,
        support_low=0.02,
        oppose_below=0.0,
    )
    no_brier, md_brier, units = _p2_paired_values(
        panel,
        "context_brier",
        first_gate="no_gate",
        second_gate="md_recurrent_belief",
    )
    append_claim(
        claim_id="P2c_md_context_brier",
        metric="context_brier",
        comparison="no-gate minus MD-belief Brier score",
        criterion="MD belief improves Brier score by at least 0.01",
        candidate=no_brier,
        reference=md_brier,
        unit_ids=units,
        support_low=0.01,
        oppose_below=0.0,
    )
    md_ece, units = _p2_gate_values(panel, "context_ece", gate="md_recurrent_belief")
    append_claim(
        claim_id="P2d_md_calibration",
        metric="context_ece",
        comparison="MD-belief ECE minus absolute calibration threshold 0.05",
        criterion="MD ECE upper CI <=0.05; ECE lower CI >=0.10 opposes",
        candidate=md_ece,
        reference=np.full(len(md_ece), 0.05),
        unit_ids=units,
        support_high=0.0,
        oppose_above=0.05,
    )
    md_latency, oracle_latency, units = _p2_paired_values(
        panel,
        "switch_latency_trials",
        first_gate="md_recurrent_belief",
        second_gate="oracle_bayes",
    )
    append_claim(
        claim_id="P2e_md_switch_latency",
        metric="switch_latency_trials",
        comparison="MD-belief minus oracle-Bayes switch latency",
        criterion="MD excess switch latency upper CI <=1 trial",
        candidate=md_latency,
        reference=oracle_latency,
        unit_ids=units,
        support_high=1.0,
        oppose_above=1.0,
    )
    md_false, oracle_false, units = _p2_paired_values(
        panel,
        "false_switch_rate",
        first_gate="md_recurrent_belief",
        second_gate="oracle_bayes",
    )
    append_claim(
        claim_id="P2f_md_false_switch",
        metric="false_switch_rate",
        comparison="MD-belief minus oracle-Bayes false-switch rate",
        criterion="MD excess false-switch-rate upper CI <=0.01",
        candidate=md_false,
        reference=oracle_false,
        unit_ids=units,
        support_high=0.01,
        oppose_above=0.01,
    )
    md_accuracy, no_accuracy, units = _p2_paired_values(
        panel,
        "behavior_balanced_accuracy",
        first_gate="md_recurrent_belief",
        second_gate="no_gate",
    )
    append_claim(
        claim_id="P2g_md_behavior",
        metric="behavior_balanced_accuracy",
        comparison="MD-belief minus no-gate held-out balanced accuracy",
        criterion="MD gate improves held-out balanced accuracy by at least 0.02",
        candidate=md_accuracy,
        reference=no_accuracy,
        unit_ids=units,
        support_low=0.02,
        oppose_below=0.0,
    )
    retained, units = _p2_retained_oracle_gain(panel)
    append_claim(
        claim_id="P2h_md_retains_oracle_gain",
        metric="behavior_balanced_accuracy",
        comparison="MD gain minus 90% of oracle-Bayes gain over no gate",
        criterion="MD retains at least 90% of the paired oracle behavioral gain",
        candidate=retained,
        reference=np.zeros(len(retained)),
        unit_ids=units,
        support_low=0.0,
        oppose_below=0.0,
    )
    md_energy, no_energy, units = _p2_paired_values(
        panel,
        "energy_proxy_per_trial",
        first_gate="md_recurrent_belief",
        second_gate="no_gate",
    )
    append_claim(
        claim_id="P2i_md_energy",
        metric="log_energy_ratio",
        comparison="log(MD-belief energy / no-gate energy)",
        criterion="MD energy upper ratio CI <=1.10",
        candidate=np.log(md_energy),
        reference=np.log(no_energy),
        unit_ids=units,
        support_high=float(np.log(1.10)),
        oppose_above=float(np.log(1.10)),
    )
    for claim_id, intervention, label, hazards in (
        ("P2j_clamp_causal", "clamp", "clamp", P2_H),
        ("P2k_delay_causal", "delay", "one-trial delay", (0.10, 0.20)),
        ("P2l_shuffle_causal", "shuffle", "trajectory shuffle", P2_H),
    ):
        intact, intervened, units = _p2_paired_values(
            panel,
            "behavior_balanced_accuracy",
            first_gate="md_recurrent_belief",
            second_gate="md_recurrent_belief",
            second_intervention=intervention,
            hazards=hazards,
        )
        append_claim(
            claim_id=claim_id,
            metric="behavior_balanced_accuracy",
            comparison=f"intact MD-belief minus {label} accuracy",
            criterion=f"post-fit {label} reduces balanced accuracy by at least 0.01",
            candidate=intact,
            reference=intervened,
            unit_ids=units,
            support_low=0.01,
            oppose_below=0.0,
        )
    if tuple(item.claim_id for item in results) != P2_PRIMARY_CLAIMS:
        raise RuntimeError("P2 claim construction does not match the fixed registry")
    return results, complete_by_claim, failed_by_claim, panel.issues


def _p2_overall_gate(
    adjusted_claims: list[ClaimResult],
    *,
    complete_seed_ids_by_claim: dict[str, set[int]],
    failed_seed_ids_by_claim: dict[str, set[int]],
    panel_issues: tuple[str, ...],
) -> ClaimResult:
    """Derive the leakage-safe P2 mechanism gate after constituent Holm tests."""

    lookup = {item.claim_id: item for item in adjusted_claims}
    constituents = [lookup.get(claim_id) for claim_id in P2_REQUIRED_CLAIMS]
    conclusions = [
        item.conclusion if item is not None else "inconclusive" for item in constituents
    ]
    if all(value == "support" for value in conclusions):
        conclusion = "support"
    elif any(value == "oppose" for value in conclusions):
        conclusion = "oppose"
    else:
        conclusion = "inconclusive"
    complete_sets = [
        complete_seed_ids_by_claim.get(claim_id, set())
        for claim_id in P2_REQUIRED_CLAIMS
    ]
    jointly_complete = set.intersection(*complete_sets) if complete_sets else set()
    failed = set().union(
        *(
            failed_seed_ids_by_claim.get(claim_id, set())
            for claim_id in P2_REQUIRED_CLAIMS
        )
    )
    audit = "; ".join(
        f"{claim_id}={value}"
        for claim_id, value in zip(P2_REQUIRED_CLAIMS, conclusions, strict=True)
    )
    issue_note = " | ".join(panel_issues) if panel_issues else "none"
    return ClaimResult(
        claim_id="P2_overall",
        experiment="exp09",
        metric="noninferential_stage_gate",
        comparison="conjunction of leakage-safe Holm-adjusted P2 claims",
        stats_unit="seed",
        n_planned=P2_SEED_PLAN,
        n_complete=len(jointly_complete),
        n_failed=len(failed),
        estimate=None,
        ci_low=None,
        ci_high=None,
        effect_size=None,
        p_value=None,
        multiplicity_method="derived_after_holm(no_additional_test)",
        conclusion=conclusion,
        criterion=(
            "support iff every critical Holm-adjusted P2 claim supports; oppose iff "
            "at least one opposes; otherwise inconclusive"
        ),
        note=f"non-inferential P2 stage gate; {audit}; strict panel issues: {issue_note}",
    )


def _primary_phase2(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the preregistered 80/20 E/I architecture when dimensions exist."""

    if frame.empty:
        return frame
    run_level_failure = _eq(frame, "status", "failed") & _series(
        frame, "run_level_failure", False
    ).fillna(False).astype(bool)
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
    aggregated = selected.groupby(session_keys, as_index=False, dropna=False)[
        metric
    ].mean()
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


@dataclass(frozen=True)
class _StrictIblPanel:
    """One strictly validated stimulus-pre model panel for P6 inference."""

    nll: pd.DataFrame
    parameters: pd.DataFrame
    sessions: frozenset[object]
    animals: frozenset[object]
    session_animal_pairs: frozenset[tuple[object, object]]
    n_failed: int
    issues: tuple[str, ...]


def _strict_ibl_model_panel(
    frame: pd.DataFrame,
    *,
    failure_filter: pd.Series,
    provenance: dict[str, bool],
) -> _StrictIblPanel:
    """Validate P6 cells before any session or animal aggregation.

    A session is usable only when every stimulus-pre fold has exactly one row
    for each registered model, all provenance is explicit, and counted
    parameters are finite non-negative integers that do not change across
    folds.  If one session of an animal is invalid, the whole animal is
    removed so a partial independent unit can never enter inference.
    """

    required_models = frozenset({"common", "shared", "full"})
    session = _session_column(frame)
    empty = _StrictIblPanel(
        nll=pd.DataFrame(),
        parameters=pd.DataFrame(),
        sessions=frozenset(),
        animals=frozenset(),
        session_animal_pairs=frozenset(),
        n_failed=0,
        issues=(),
    )
    if session is None:
        return replace(empty, issues=("missing session identifier",))
    if "animal_id" not in frame:
        return replace(empty, issues=("missing animal_id",))

    model_scope = _series(frame, "model_family").isin(required_models)
    selected = frame.loc[model_scope & _complete(frame)].copy()
    n_failed = _failed_units(
        frame,
        "animal_id",
        failure_filter,
    )
    if selected.empty:
        return replace(
            empty,
            n_failed=n_failed,
            issues=("complete common/shared/full stimulus-pre panel unavailable",),
        )

    issues: list[str] = []
    bad_sessions: set[object] = set()
    required_columns = {
        session,
        "animal_id",
        "view",
        "fold",
        "model_family",
        "heldout_nll_per_scalar",
        "parameter_count",
    }
    missing = sorted(required_columns - set(selected))
    if missing:
        return replace(
            empty,
            n_failed=n_failed,
            issues=(f"missing strict panel columns: {', '.join(missing)}",),
        )

    missing_identity = selected[session].isna() | selected["animal_id"].isna()
    if missing_identity.any():
        bad_sessions.update(selected.loc[missing_identity, session].dropna().tolist())
        issues.append("session/animal identity is missing")

    wrong_view = selected["view"].isna() | selected["view"].ne("stimulus_pre")
    if wrong_view.any():
        bad_sessions.update(selected.loc[wrong_view, session].dropna().tolist())
        issues.append("model panel contains a non-stimulus-pre or missing view")

    for field, expected in provenance.items():
        if field not in selected:
            issues.append(f"missing provenance field {field}")
            bad_sessions.update(selected[session].dropna().tolist())
            continue
        parsed = selected[field].map(_strict_boolean_value)
        invalid = parsed.isna() | parsed.ne(expected)
        if invalid.any():
            issues.append(f"{field} is not uniformly {expected}")
            bad_sessions.update(selected.loc[invalid, session].dropna().tolist())

    selected["_nll"] = pd.to_numeric(
        selected["heldout_nll_per_scalar"], errors="coerce"
    )
    selected["_parameters"] = pd.to_numeric(
        selected["parameter_count"], errors="coerce"
    )
    nll_valid = np.isfinite(selected["_nll"].to_numpy(float))
    parameter_values = selected["_parameters"].to_numpy(float)
    parameter_valid = (
        np.isfinite(parameter_values)
        & (parameter_values >= 0)
        & np.isclose(parameter_values, np.rint(parameter_values), rtol=0.0, atol=1e-9)
    )
    invalid_numeric = ~(nll_valid & parameter_valid)
    if invalid_numeric.any():
        bad_sessions.update(selected.loc[invalid_numeric, session].dropna().tolist())
        issues.append(
            "NLL must be finite and parameter_count must be a finite non-negative integer"
        )

    cell_keys = [session, "view", "fold"]
    row_counts = selected.groupby(
        [*cell_keys, "model_family"], dropna=False, sort=False
    ).size()
    duplicate_cells = row_counts.loc[row_counts.ne(1)]
    if not duplicate_cells.empty:
        bad_sessions.update(
            duplicate_cells.index.get_level_values(session).dropna().tolist()
        )
        issues.append("duplicate (session, view, fold, model) cell")

    cell_models = selected.groupby(cell_keys, dropna=False, sort=False)[
        "model_family"
    ].agg(lambda values: frozenset(values.tolist()))
    incomplete_cells = cell_models.loc[cell_models.ne(required_models)]
    if not incomplete_cells.empty:
        bad_sessions.update(
            incomplete_cells.index.get_level_values(session).dropna().tolist()
        )
        issues.append("a stimulus-pre fold lacks exactly common/shared/full")

    if selected["fold"].isna().any():
        bad_sessions.update(
            selected.loc[selected["fold"].isna(), session].dropna().tolist()
        )
        issues.append("fold identifier is missing")
    planned_folds = frozenset(selected["fold"].dropna().tolist())
    session_folds = selected.groupby(session, dropna=False, sort=False)["fold"].agg(
        lambda values: frozenset(values.dropna().tolist())
    )
    incomplete_fold_sets = session_folds.loc[session_folds.ne(planned_folds)]
    if not incomplete_fold_sets.empty:
        bad_sessions.update(incomplete_fold_sets.index.dropna().tolist())
        issues.append("sessions do not share the complete registered fold set")

    animal_variation = selected.groupby(session, dropna=False, sort=False)[
        "animal_id"
    ].nunique(dropna=False)
    invalid_mapping = animal_variation.loc[animal_variation.ne(1)]
    if not invalid_mapping.empty:
        bad_sessions.update(invalid_mapping.index.dropna().tolist())
        issues.append("a session does not map to exactly one animal")

    parameter_variation = selected.groupby(
        [session, "model_family"], dropna=False, sort=False
    )["_parameters"].nunique(dropna=False)
    varying_parameters = parameter_variation.loc[parameter_variation.ne(1)]
    if not varying_parameters.empty:
        bad_sessions.update(
            varying_parameters.index.get_level_values(session).dropna().tolist()
        )
        issues.append("parameter_count changes across folds or is missing")

    failed_sessions = set(
        frame.loc[
            _eq(frame, "status", "failed")
            & failure_filter.reindex(frame.index, fill_value=False),
            session,
        ]
        .dropna()
        .tolist()
    )
    if failed_sessions:
        bad_sessions.update(failed_sessions)

    bad_animals = set(
        selected.loc[selected[session].isin(bad_sessions), "animal_id"]
        .dropna()
        .tolist()
    )
    n_failed = max(n_failed, len(bad_animals) or len(bad_sessions))
    eligible = selected.loc[
        ~selected[session].isin(bad_sessions) & ~selected["animal_id"].isin(bad_animals)
    ].copy()
    if eligible.empty:
        return replace(
            empty,
            n_failed=n_failed,
            issues=tuple(dict.fromkeys(issues))
            or ("no strictly eligible stimulus-pre session",),
        )

    session_model = eligible.groupby(
        [session, "animal_id", "model_family"],
        as_index=False,
        dropna=False,
        sort=False,
    ).agg(_nll=("_nll", "mean"), _parameters=("_parameters", "first"))
    animal_model = session_model.groupby(
        ["animal_id", "model_family"],
        as_index=False,
        dropna=False,
        sort=False,
    ).agg(_nll=("_nll", "mean"), _parameters=("_parameters", "mean"))
    nll = animal_model.pivot(index="animal_id", columns="model_family", values="_nll")
    parameters = animal_model.pivot(
        index="animal_id", columns="model_family", values="_parameters"
    )
    return _StrictIblPanel(
        nll=nll,
        parameters=parameters,
        sessions=frozenset(eligible[session].dropna().tolist()),
        animals=frozenset(eligible["animal_id"].dropna().tolist()),
        session_animal_pairs=frozenset(
            eligible[[session, "animal_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ),
        n_failed=n_failed,
        issues=tuple(dict.fromkeys(issues)),
    )


def _planned_ibl_animals(frame: pd.DataFrame) -> int:
    """Return the preregistered animal count, never the minimum threshold."""

    explicit = pd.to_numeric(
        _series(frame, "n_planned_animals"), errors="coerce"
    ).dropna()
    if not explicit.empty:
        return max(1, int(explicit.max()))
    if "animal_id" in frame:
        return max(1, int(frame["animal_id"].dropna().nunique()))
    return 1


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
        return (
            np.array([]),
            np.array([], dtype=object),
            unit_column,
            stats_unit,
            n_failed,
        )
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
        return (
            np.array([]),
            np.array([], dtype=object),
            unit_column,
            stats_unit,
            n_failed,
        )

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
                observed_views.loc[
                    observed_views.map(set) != expected_views
                ].index.tolist()
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
        return (
            np.array([]),
            np.array([], dtype=object),
            unit_column,
            stats_unit,
            n_failed,
        )

    session_keys = [session]
    if has_animal:
        session_keys.append("animal_id")
    aggregated = selected.groupby(session_keys, as_index=False, dropna=False)[
        metric
    ].mean()
    if has_animal:
        if aggregated["animal_id"].isna().any():
            return (
                np.array([]),
                np.array([], dtype=object),
                unit_column,
                stats_unit,
                max(1, n_failed),
            )
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
    if (
        not set(required_flags) <= set(frame)
        or "shared_source_fingerprint" not in frame
    ):
        return False, "exact rate/spike/coupling flags or source fingerprint are absent"
    relevant = frame.loc[
        _complete(frame)
        & _series(frame, "phase_condition").isin(["in_phase", "no_oscillation"])
    ]
    if relevant.empty or not relevant[list(required_flags)].fillna(False).astype(
        bool
    ).all(axis=None):
        return False, "one or more exact matching flags are false"
    counts = relevant.groupby("seed")["phase_condition"].nunique()
    fingerprints = relevant.groupby("seed")["shared_source_fingerprint"].nunique(
        dropna=False
    )
    if not (counts.eq(2).all() and fingerprints.eq(1).all()):
        return (
            False,
            "in-phase/no-oscillation sources are not exactly paired within seed",
        )
    return (
        True,
        "all exact matching flags true and source fingerprints identical within seed",
    )


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
        [
            result.p_value
            if result.p_value is not None and np.isfinite(result.p_value)
            else 1.0
            for result in results
        ],
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
        corrected_conclusion = result.conclusion
        if result.conclusion in {"support", "oppose"} and adjusted_p > 0.05:
            corrected_conclusion = "inconclusive"
        output.append(
            replace(
                result,
                conclusion=corrected_conclusion,
                p_value=float(adjusted_p),
                note=(
                    f"{result.note}; p_value is Holm-adjusted across all {family_size} "
                    f"registered claims (raw Wilcoxon p={raw_p:.12g}); a directional "
                    "bootstrap criterion can support/oppose only when Holm p<=0.05"
                ),
            )
        )
    return output


def evaluate_core_claims(raw_metrics: pd.DataFrame) -> list[ClaimResult]:
    """Evaluate registered claims; incomplete formal evidence stays inconclusive."""

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
    exp01 = formal.loc[
        _eq(formal, "experiment", "exp01_feedback_dimension_sweep")
    ].copy()
    core = _eq(exp01, "grid", "core") & _eq(exp01, "feedback_mode", "aligned")
    d4 = core & pd.to_numeric(_series(exp01, "feedback_dim"), errors="coerce").eq(4)
    rank_values, _, rank_units = _paired_rows(exp01, d4, d4, "effective_rank", "seed")
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
    full = core & pd.to_numeric(_series(exp01, "feedback_dim"), errors="coerce").eq(
        full_dim
    )
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
    phase2 = _primary_phase2(
        formal.loc[_series(formal, "experiment").isin(phase2_names)]
    )
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
    results.append(
        _paired_claim(
            claim_id="B1a_local_absolute_accuracy",
            experiment="exp02/03",
            metric="accuracy",
            comparison="local accuracy minus absolute threshold 0.85",
            stats_unit="seed",
            candidate=local,
            reference=np.full(len(local), 0.85),
            unit_ids=units,
            n_planned=SEED_PLAN,
            n_failed=_failed_units(phase2, "seed", local_mask | bptt_mask),
            minimum_units=SEED_PLAN,
            support_low=0.0,
            oppose_below=0.0,
            criterion="local absolute accuracy-minus-0.85 CI is reported independently",
        )
    )
    results.append(
        _paired_claim(
            claim_id="B1b_local_relative_noninferiority",
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
            criterion=(
                "local relative non-inferiority to 90% of paired BPTT is reported "
                "independently of absolute performance"
            ),
        )
    )

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
    hidden_gate_rows = phase2.loc[(no_gate_mask | local_mask) & _complete(phase2)]
    hidden_gate_requirements = {
        "hidden_context_task": True,
        "cue_encodes_observation_not_state": True,
        "gate_test_accessed_true_context": False,
        "gate_fit_accessed_true_context": False,
        "third_factor_accessed_true_context": False,
        "oracle_warm_start_used": False,
        "md_fit_used_context_bias": False,
    }
    hidden_gate_provenance_valid = not hidden_gate_rows.empty
    for field, expected in hidden_gate_requirements.items():
        if not _strict_boolean_requirement(hidden_gate_rows, field, expected):
            hidden_gate_provenance_valid = False
            break
    if hidden_gate_provenance_valid:
        b2 = _paired_claim(
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
    else:
        b2 = _inconclusive(
            "B2_gate_reduces_switch_cost",
            "exp02/03",
            "switch_cost",
            "no-gate switch cost minus gated-local switch cost",
            "seed",
            (
                "hidden-context gate improves switch cost without true-context "
                "access in the cue, gate fit/test, oracle warm start, or recurrent "
                "third factor"
            ),
            (
                "legacy exp02/03 lacks leakage-free hidden-context provenance; "
                "exp03 is a supervised/oracle-warm-start upper bound and the "
                "legacy no-gate third factor receives true context"
            ),
            n_planned=SEED_PLAN,
            n_complete=0,
            n_failed=_failed_units(phase2, "seed", no_gate_mask | local_mask),
        )
    results.append(b2)

    # P2 hidden-context audit. The five gates share an exact 4x4 HMM grid;
    # supervised context labels are retained only as an explicitly ineligible
    # upper bound. Every registered P2 claim is appended even when exp09 has
    # not run so the project-wide Holm family never depends on data arrival.
    (
        p2_results,
        p2_complete_seed_ids_by_claim,
        p2_failed_seed_ids_by_claim,
        (p2_panel_issues),
    ) = _evaluate_p2_claims(
        formal.loc[_eq(formal, "experiment", "exp09_hidden_context_gate")].copy()
    )
    results.extend(p2_results)

    exp02 = phase2.loc[_eq(phase2, "experiment", "exp02_context_ei_oracle_gate")].copy()
    stability_metric = next(
        (
            metric
            for metric in ("jacobian_max_real_part", "jacobian_unstable_fraction")
            if metric in exp02
            and pd.to_numeric(exp02[metric], errors="coerce").notna().any()
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
            n_failed=_failed_units(exp02, "seed", no_home_mask | exp02_local_mask),
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
            if metric in phase2
            and pd.to_numeric(phase2[metric], errors="coerce").notna().any()
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
            n_failed=_failed_units(phase2, "seed", full_feedback_mask | local_mask),
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
            comparison="retained full-vs-common gain minus 0.90",
            stats_unit=stats05,
            candidate=retained.to_numpy(float),
            reference=np.full(len(retained), 0.90),
            unit_ids=retained.index.to_numpy(object),
            n_planned=plan05,
            n_failed=failed05,
            minimum_units=2,
            support_low=0.0,
            oppose_below=0.0,
            criterion="retained switching gain >= 0.90 and shared parameters < full",
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
            oppose = d1.ci_high is not None and (
                d1.ci_high < 0 or parameter.ci_high <= 0
            )
            d1 = replace(
                d1,
                conclusion="support"
                if support
                else ("oppose" if oppose else "inconclusive"),
                note=f"{d1.note}; full-minus-shared parameter CI [{parameter.ci_low:.6g}, {parameter.ci_high:.6g}]",
            )
    else:
        d1 = _inconclusive(
            "D1_shared_basis_near_full",
            "exp05",
            "retained_switching_gain",
            "shared versus common/full",
            stats05,
            "retained switching gain >= 0.90 and shared parameters < full",
            "complete common/shared/full formal panel unavailable",
            n_planned=_real_plan(exp05, unit05, failed05),
            n_failed=failed05,
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
            "D2_unseen_sequence_generalization",
            "exp05",
            "heldout_nll_per_scalar",
            "full minus shared NLL on unseen combinations",
            unseen_stats,
            "shared held-out NLL is below full LDS on unseen combinations",
            "complete unseen-combination shared/full panel unavailable",
            n_planned=_real_plan(exp05, unseen_unit, unseen_failed),
            n_failed=unseen_failed,
        )
    results.append(d2)

    # Phase 4: hidden-context IBL switching LDS and descriptive lead/lag.
    # The preregistered primary view is stimulus-pre. Movement-pre and future
    # full-trial sensitivity analyses remain separate repeated outcomes and are
    # never averaged into the primary claim.
    exp06 = formal.loc[_eq(formal, "experiment", "exp06_ibl_context_switch")].copy()
    stimulus_pre = _eq(exp06, "view", "stimulus_pre")
    viewless_failure = _eq(exp06, "status", "failed") & _series(exp06, "view").isna()
    exp06_primary = exp06.loc[stimulus_pre | viewless_failure].copy()
    model06 = _series(exp06_primary, "model_family").isin(["common", "shared", "full"])
    model06_failures = model06 | (
        _eq(exp06_primary, "status", "failed")
        & _series(exp06_primary, "model_family").isna()
    )
    method_requirements06 = {
        "hierarchical_observation_model": True,
        "nested_cv_latent_dimension": True,
        "unit_qc_applied": True,
        "context_coverage_valid": True,
        "parameter_count_includes_preprocessing": True,
        "hidden_context_inference": True,
        "test_context_observed": False,
        "belief_filter_used_true_block_boundaries": False,
        "condition_schedule_observed": False,
    }
    strict06 = _strict_ibl_model_panel(
        exp06_primary,
        failure_filter=model06_failures,
        provenance=method_requirements06,
    )
    required_model_order = ("common", "shared", "full")
    required_models = set(required_model_order)
    nll06 = strict06.nll
    params06 = strict06.parameters
    failed06 = strict06.n_failed
    stats06 = "animal"
    session_count06 = len(strict06.sessions)
    animal_count06 = len(strict06.animals)
    planned_animals06 = _planned_ibl_animals(exp06_primary)
    method_issues06 = list(strict06.issues)
    cohort_valid06 = session_count06 >= 20 and animal_count06 >= 5
    panel_columns_valid = required_models <= set(nll06) and required_models <= set(
        params06
    )
    panel_index_valid = nll06.index.equals(params06.index)
    e1_ready = (
        panel_columns_valid
        and panel_index_valid
        and cohort_valid06
        and not method_issues06
    )
    panel = pd.DataFrame()
    retained = pd.Series(dtype=float)
    if e1_ready:
        panel = nll06[list(required_model_order)]
        denominator = panel["common"] - panel["full"]
        retained = (panel["common"] - panel["shared"]) / denominator
        retained_valid = bool(
            np.isfinite(retained.to_numpy(float)).all()
            and np.isfinite(denominator.to_numpy(float)).all()
            and denominator.gt(0).all()
        )
        if not retained_valid:
            method_issues06.append(
                "every animal must have a finite positive full-vs-common gain denominator"
            )
            e1_ready = False

    if e1_ready:
        params06 = params06[list(required_model_order)]
        e1 = _paired_claim(
            claim_id="E1_ibl_shared_switching",
            experiment="exp06",
            metric="heldout_nll_per_scalar",
            comparison="common minus shared hidden-context NLL",
            stats_unit=stats06,
            candidate=panel["common"].to_numpy(float),
            reference=panel["shared"].to_numpy(float),
            unit_ids=panel.index.to_numpy(object),
            n_planned=planned_animals06,
            n_failed=failed06,
            minimum_units=5,
            support_low=0.0,
            oppose_below=0.0,
            criterion=(
                "stimulus-pre hierarchical shared model improves on common, retains "
                ">=0.90 of full-model gain, and uses fewer counted parameters"
            ),
        )
        if e1.estimate is not None and required_models <= set(params06):
            gain = paired_bootstrap(
                retained.to_numpy(float),
                np.full(len(retained), 0.90),
                unit_ids=retained.index.to_numpy(object),
                replicate_unit=stats06,
                n_resamples=5000,
                confidence=0.95,
                seed=5,
            )
            parameter = paired_bootstrap(
                params06["full"].to_numpy(float),
                params06["shared"].to_numpy(float),
                unit_ids=params06.index.to_numpy(object),
                replicate_unit=stats06,
                n_resamples=5000,
                confidence=0.95,
                seed=6,
            )
            support = (
                e1.ci_low is not None
                and e1.ci_low > 0
                and gain.ci_low >= 0
                and parameter.ci_low > 0
            )
            oppose = e1.ci_high is not None and (
                e1.ci_high <= 0 or gain.ci_high < 0 or parameter.ci_high <= 0
            )
            e1 = replace(
                e1,
                conclusion="support"
                if support
                else ("oppose" if oppose else "inconclusive"),
                note=(
                    f"{e1.note}; retained-gain-minus-0.90 CI "
                    f"[{gain.ci_low:.6g}, {gain.ci_high:.6g}]; full-minus-shared "
                    f"parameter CI [{parameter.ci_low:.6g}, {parameter.ci_high:.6g}]; "
                    f"eligible cohort {animal_count06} animals/{session_count06} sessions"
                ),
            )
    else:
        reasons = [
            f"strict eligible cohort has {animal_count06} animals/{session_count06} "
            "sessions (minimum 5/20)"
        ]
        reasons.extend(method_issues06)
        if not panel_columns_valid or not panel_index_valid:
            reasons.append("complete common/shared/full stimulus-pre panel unavailable")
        e1 = _inconclusive(
            "E1_ibl_shared_switching",
            "exp06",
            "heldout_nll_per_scalar",
            "shared versus common/full hidden-context LDS",
            stats06,
            (
                "stimulus-pre hierarchical shared model improves on common, "
                "retains >=0.90 of full gain, and uses fewer counted parameters"
            ),
            "; ".join(reasons),
            n_planned=planned_animals06,
            n_complete=animal_count06,
            n_failed=failed06,
        )
    results.append(e1)

    # Lead/lag is likewise stimulus-pre only. A movement-pre failure cannot be
    # averaged with or cancel the primary view. The behavioral estimator must
    # explicitly attest that it did not reset on true block boundaries.
    lead_frame = exp06_primary.copy()
    lead_mask = _eq(lead_frame, "model_family", "lead_lag")
    complete_lead = lead_frame.loc[lead_mask & _complete(lead_frame)]
    lead_requirements = {
        **method_requirements06,
        "lead_lag_is_causal_claim": False,
        "behavior_bias_used_true_block_boundaries": False,
    }
    lead_issues: list[str] = []
    for field, expected in lead_requirements.items():
        if not _strict_boolean_requirement(complete_lead, field, expected):
            lead_issues.append(f"{field} is not uniformly {expected}")

    session_column06 = _session_column(lead_frame)
    lead_failure_scope = lead_mask | (
        _eq(lead_frame, "status", "failed") & _series(lead_frame, "model_family").isna()
    )
    lead_failed = _failed_units(lead_frame, "animal_id", lead_failure_scope)
    lead_stats = "animal"
    lead = np.array([], dtype=float)
    lead_ids = np.array([], dtype=object)
    lead_sessions: frozenset[object] = frozenset()
    lead_animals: frozenset[object] = frozenset()
    lead_session_animal_pairs: frozenset[tuple[object, object]] = frozenset()
    required_lead_columns = {
        "animal_id",
        "view",
        "latent_lead_trials",
    }
    lead_structure_valid = True
    if session_column06 is None:
        lead_issues.append("missing lead session identifier")
        lead_structure_valid = False
    else:
        required_lead_columns.add(session_column06)
    missing_lead_columns = sorted(required_lead_columns - set(complete_lead))
    if missing_lead_columns:
        lead_issues.append(f"missing lead columns: {', '.join(missing_lead_columns)}")
        lead_structure_valid = False
    if not complete_lead.empty and not missing_lead_columns:
        prepared_lead = complete_lead.copy()
        prepared_lead["_lead"] = pd.to_numeric(
            prepared_lead["latent_lead_trials"], errors="coerce"
        )
        invalid_lead = (
            prepared_lead[session_column06].isna()
            | prepared_lead["animal_id"].isna()
            | prepared_lead["view"].ne("stimulus_pre")
            | ~np.isfinite(prepared_lead["_lead"].to_numpy(float))
        )
        if invalid_lead.any():
            lead_issues.append("lead rows have invalid identity, view, or value")
            lead_structure_valid = False
        lead_cell_keys = [session_column06, "view"]
        if "fold" in prepared_lead and prepared_lead["fold"].notna().any():
            lead_cell_keys.append("fold")
        duplicate_lead = prepared_lead.groupby(
            lead_cell_keys, dropna=False, sort=False
        ).size()
        if duplicate_lead.ne(1).any():
            lead_issues.append("duplicate lead session/view/fold cell")
            lead_structure_valid = False
        lead_mapping = prepared_lead.groupby(
            session_column06, dropna=False, sort=False
        )["animal_id"].nunique(dropna=False)
        if lead_mapping.ne(1).any():
            lead_issues.append("a lead session does not map to exactly one animal")
            lead_structure_valid = False
        if lead_structure_valid:
            lead_sessions = frozenset(prepared_lead[session_column06].dropna().tolist())
            lead_animals = frozenset(prepared_lead["animal_id"].dropna().tolist())
            lead_session_animal_pairs = frozenset(
                prepared_lead[[session_column06, "animal_id"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            session_lead = prepared_lead.groupby(
                [session_column06, "animal_id"],
                as_index=False,
                dropna=False,
                sort=False,
            )["_lead"].mean()
            animal_lead = session_lead.groupby(
                "animal_id", as_index=False, dropna=False, sort=False
            )["_lead"].mean()
            lead = animal_lead["_lead"].to_numpy(float)
            lead_ids = animal_lead["animal_id"].to_numpy(object)

    same_cohort = (
        lead_sessions == strict06.sessions
        and lead_animals == strict06.animals
        and lead_session_animal_pairs == strict06.session_animal_pairs
        and len(lead_sessions) >= 20
        and len(lead_animals) >= 5
    )
    model_method_valid = (
        cohort_valid06 and not strict06.issues and strict06.n_failed == 0
    )
    if not same_cohort:
        lead_issues.append(
            "lead records do not exactly match the strict E1 session/animal cohort"
        )
    if not model_method_valid:
        lead_issues.append("the strict E1 model cohort or method provenance is invalid")

    if not lead_issues and lead_failed == 0:
        e2 = _paired_claim(
            claim_id="E2_latent_precedes_behavior_bias",
            experiment="exp06",
            metric="latent_lead_trials",
            comparison="latent lead minus zero trials",
            stats_unit=lead_stats,
            candidate=lead,
            reference=np.zeros(len(lead)),
            unit_ids=lead_ids,
            n_planned=planned_animals06,
            n_failed=lead_failed,
            minimum_units=5,
            support_low=0.0,
            oppose_below=0.0,
            criterion="independent-unit bootstrap CI of latent lead is above zero",
        )
        e2 = replace(
            e2,
            note=(
                f"{e2.note}; descriptive temporal association, not causal; "
                f"eligible cohort {len(lead_animals)} animals/{len(lead_sessions)} sessions"
            ),
        )
    else:
        e2 = _inconclusive(
            "E2_latent_precedes_behavior_bias",
            "exp06",
            "latent_lead_trials",
            "latent lead minus zero trials",
            lead_stats,
            "independent-unit bootstrap CI of latent lead is above zero",
            "; ".join(dict.fromkeys(lead_issues))
            or "strict stimulus-pre lead panel unavailable",
            n_planned=planned_animals06,
            n_complete=len(set(lead_ids.tolist())),
            n_failed=lead_failed,
        )
    results.append(e2)

    # P0 mechanism-identifiability: restrict evidence to exactly the registered
    # seed IDs, evaluate L1 and L2 independently, and combine them only through
    # an intersection-union decision.  Averages across budget panels are never
    # admissible evidence.
    p0 = _prepare_p0_frame(
        formal.loc[_eq(formal, "experiment", "exp07_mechanism_identifiability")].copy()
    )
    task_enabled = (
        _series(p0, "task_plasticity_enabled", False).fillna(False).astype(bool)
    )
    homeostasis_enabled = (
        _series(p0, "homeostasis_enabled", False).fillna(False).astype(bool)
    )
    normalization_enabled = (
        _series(p0, "normalization_enabled", False).fillna(False).astype(bool)
    )
    aligned = _eq(p0, "feedback_mode", "aligned")
    shuffled = _eq(p0, "feedback_mode", "shuffled")
    task_only = task_enabled & ~homeostasis_enabled & ~normalization_enabled
    task_homeostasis = (
        task_enabled & homeostasis_enabled & ~normalization_enabled & aligned
    )
    primary_local = task_enabled & homeostasis_enabled & normalization_enabled & aligned
    homeostasis_only = (
        ~task_enabled & homeostasis_enabled & ~normalization_enabled & aligned
    )
    frozen = ~task_enabled & ~homeostasis_enabled & ~normalization_enabled & aligned
    valid_budget = _strict_boolean_mask(p0, "budget_match_valid", True)
    p0_complete_seed_ids_by_claim: dict[str, set[int]] = {}
    p0_failed_seed_ids_by_claim: dict[str, set[int]] = {}

    def p0_joint_claim(
        *,
        claim_id: str,
        metric: str,
        comparison: str,
        criterion: str,
        first_mask: pd.Series,
        second_mask: pd.Series,
        fixed_reference: float | None = None,
        reference_multiplier: float = 1.0,
        second_is_panel_invariant: bool = False,
    ) -> ClaimResult:
        panel_results: dict[str, ClaimResult] = {}
        panel_complete_seed_ids: dict[str, set[int]] = {}
        failed_seed_ids: set[int] = set()
        for panel_index, norm in enumerate(P0_BUDGET_PANELS):
            norm_mask = _eq(p0, "budget_norm", norm)
            selected_first = first_mask & norm_mask & valid_budget
            if second_is_panel_invariant:
                selected_second = second_mask
                failure_scope = (first_mask & norm_mask) | second_mask
            else:
                selected_second = second_mask & norm_mask & valid_budget
                failure_scope = (first_mask | second_mask) & norm_mask
            first, second, units = _paired_rows(
                p0,
                selected_first,
                selected_second,
                metric,
                "seed",
            )
            if fixed_reference is not None:
                second = np.full(len(first), fixed_reference, dtype=float)
            else:
                second = reference_multiplier * second
            panel_failed = _p0_failed_seed_ids(p0, failure_scope)
            failed_seed_ids.update(panel_failed)
            panel_complete_seed_ids[norm] = {
                int(value) for value in units.tolist() if int(value) in P0_PLANNED_SEEDS
            }
            panel_results[norm] = _paired_claim(
                claim_id=f"{claim_id}__{norm}_panel",
                experiment="exp07",
                metric=metric,
                comparison=f"{comparison} [{norm} panel]",
                stats_unit="seed",
                candidate=first,
                reference=second,
                unit_ids=units,
                n_planned=P0_SEED_PLAN,
                n_failed=len(panel_failed),
                minimum_units=P0_SEED_PLAN,
                support_low=0.0,
                oppose_below=0.0,
                criterion=f"{criterion} [{norm} panel]",
                seed=panel_index,
            )
        p0_complete_seed_ids_by_claim[claim_id] = set.intersection(
            *(panel_complete_seed_ids[name] for name in P0_BUDGET_PANELS)
        )
        p0_failed_seed_ids_by_claim[claim_id] = failed_seed_ids.copy()
        return _joint_budget_panel_claim(
            claim_id=claim_id,
            experiment="exp07",
            metric=metric,
            comparison=comparison,
            stats_unit="seed",
            criterion=criterion,
            panels=panel_results,
            panel_complete_seed_ids=panel_complete_seed_ids,
            failed_seed_ids=failed_seed_ids,
        )

    results.append(
        p0_joint_claim(
            claim_id="P0a_aligned_task_improves_prediction_vs_frozen",
            metric="heldout_masked_mse",
            comparison="frozen MSE minus aligned task-plastic MSE",
            criterion=(
                "aligned task plasticity lowers held-out prediction MSE versus a "
                "bitwise-frozen recurrent network in both L1/L2 panels"
            ),
            first_mask=frozen,
            second_mask=task_only & aligned,
        )
    )
    results.append(
        p0_joint_claim(
            claim_id="P0b_aligned_task_beats_shuffled",
            metric="heldout_masked_mse",
            comparison="shuffled MSE minus aligned MSE at matched task budget",
            criterion=(
                "aligned feedback lowers held-out prediction MSE versus shuffled "
                "feedback under separately exact L1 and L2 task budgets"
            ),
            first_mask=task_only & shuffled,
            second_mask=task_only & aligned,
        )
    )
    results.append(
        p0_joint_claim(
            claim_id="P0c_aligned_adds_value_over_matched_homeostasis",
            metric="heldout_masked_mse",
            comparison=("homeostasis-only MSE minus aligned task+homeostasis MSE"),
            criterion=(
                "adding aligned task plasticity improves held-out prediction over "
                "the same-budget homeostasis-only control in both panels"
            ),
            first_mask=homeostasis_only,
            second_mask=task_homeostasis,
        )
    )
    results.append(
        p0_joint_claim(
            claim_id="P0d_local_absolute_accuracy",
            metric="accuracy",
            comparison="aligned local accuracy minus 0.85",
            criterion=("absolute accuracy >=0.85 independently in both L1/L2 panels"),
            first_mask=primary_local,
            second_mask=primary_local,
            fixed_reference=0.85,
        )
    )
    for claim_id, baseline_condition, label in (
        ("P0e_local_noninferior_tuned_bptt", "tuned-bptt", "tuned BPTT"),
        ("P0f_local_noninferior_tuned_gru", "tuned-gru", "tuned GRU"),
    ):
        baseline = _eq(p0, "condition", baseline_condition)
        results.append(
            p0_joint_claim(
                claim_id=claim_id,
                metric="accuracy",
                comparison=f"aligned local minus 90% of {label} accuracy",
                criterion=(
                    f"relative non-inferiority to 90% of {label} independently "
                    "in both L1/L2 panels and independently of absolute accuracy"
                ),
                first_mask=primary_local,
                second_mask=baseline,
                reference_multiplier=0.9,
                second_is_panel_invariant=True,
            )
        )

    # P1 rank-stage theorem audit.  These claims establish the revised
    # mathematical mechanism only; they do not substitute for P0 held-out
    # behavior/prediction evidence.
    p1 = formal.loc[_eq(formal, "experiment", "exp08_rank_stage_validation")].copy()
    p1_primary = (
        _eq(p1, "parameterization", "direct")
        & pd.to_numeric(_series(p1, "requested_feedback_dim"), errors="coerce").eq(4.0)
        & pd.to_numeric(_series(p1, "feedback_angle_degrees"), errors="coerce").eq(0.0)
        & _strict_boolean_mask(p1, "geometry_valid", True)
    )
    identity_residual, _, units = _paired_rows(
        p1,
        p1_primary,
        p1_primary,
        "masked_identity_max_abs_residual",
        "seed",
    )
    results.append(
        _paired_claim(
            claim_id="P1a_masked_outer_product_identity",
            experiment="exp08",
            metric="masked_identity_max_abs_residual",
            comparison="masked identity residual minus zero",
            stats_unit="seed",
            candidate=identity_residual,
            reference=np.zeros(len(identity_residual)),
            unit_ids=units,
            n_planned=P0_SEED_PLAN,
            n_failed=_failed_units(p1, "seed", p1_primary),
            minimum_units=P0_SEED_PLAN,
            support_high=1e-12,
            oppose_above=1e-8,
            criterion=("M⊙uv^T equals diag(u)Mdiag(v) to <=1e-12 max residual"),
        )
    )

    tangent, _, units = _paired_rows(
        p1,
        p1_primary,
        p1_primary,
        "lowdim_credit_tangent_dimension",
        "seed",
    )
    results.append(
        _paired_claim(
            claim_id="P1b_credit_tangent_respects_feedback_bound",
            experiment="exp08",
            metric="lowdim_credit_tangent_dimension",
            comparison="credit tangent dimension minus feedback dimension 4",
            stats_unit="seed",
            candidate=tangent,
            reference=np.full(len(tangent), 4.0),
            unit_ids=units,
            n_planned=P0_SEED_PLAN,
            n_failed=_failed_units(p1, "seed", p1_primary),
            minimum_units=P0_SEED_PLAN,
            support_high=0.5,
            oppose_above=0.5,
            criterion=(
                "instantaneous credit tangent does not exceed feedback dimension "
                "within 0.5 numerical-dimension tolerance"
            ),
        )
    )

    if {
        "masked_numerical_rank",
        "lowdim_credit_tangent_dimension",
    } <= set(p1.columns):
        p1["physical_minus_tangent_dimension"] = pd.to_numeric(
            p1["masked_numerical_rank"], errors="coerce"
        ) - pd.to_numeric(p1["lowdim_credit_tangent_dimension"], errors="coerce")
    physical_gap, _, units = _paired_rows(
        p1,
        p1_primary,
        p1_primary,
        "physical_minus_tangent_dimension",
        "seed",
    )
    results.append(
        _paired_claim(
            claim_id="P1c_highrank_physical_update_coexists_with_lowdim_credit",
            experiment="exp08",
            metric="physical_minus_tangent_dimension",
            comparison="masked physical rank minus credit tangent dimension",
            stats_unit="seed",
            candidate=physical_gap,
            reference=np.zeros(len(physical_gap)),
            unit_ids=units,
            n_planned=P0_SEED_PLAN,
            n_failed=_failed_units(p1, "seed", p1_primary),
            minimum_units=P0_SEED_PLAN,
            support_low=1.0,
            oppose_below=1.0,
            criterion=(
                "masked physical numerical rank exceeds credit tangent dimension; "
                "this theoretical claim does not imply held-out task support"
            ),
        )
    )

    adjusted = _apply_full_family_holm(results)
    return [
        *adjusted,
        _p0_overall_gate(
            adjusted,
            complete_seed_ids_by_claim=p0_complete_seed_ids_by_claim,
            failed_seed_ids_by_claim=p0_failed_seed_ids_by_claim,
        ),
        _p2_overall_gate(
            adjusted,
            complete_seed_ids_by_claim=p2_complete_seed_ids_by_claim,
            failed_seed_ids_by_claim=p2_failed_seed_ids_by_claim,
            panel_issues=p2_panel_issues,
        ),
    ]
