"""Animal-level inference for the post-hoc Exp40 IBL utility audit."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


CONDITIONS = (
    "history_only",
    "learned_hmm_mean",
    "semimarkov_mean",
    "semimarkov_release",
    "semimarkov_concentration",
    "factorized_state",
    "oracle_context_mean",
)


@dataclass(frozen=True)
class StateUtilityClaim:
    claim: str
    endpoint: str
    effect_definition: str
    minimum_effect: float
    estimate: float
    ci_low: float
    ci_high: float
    n_animals: int
    positive_animals: int
    wilcoxon_two_sided_p_at_margin: float
    holm_p: float
    conclusion: str
    evidence_tier: str = "posthoc_development"
    confirmatory_eligible: bool = False


def _validate_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = {
        "eid",
        "animal_id",
        "condition",
        "status",
        "profile",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} frame lacks columns: {missing}")
    scoped = frame.loc[frame["condition"].astype(str).isin(CONDITIONS)].copy()
    if scoped.empty:
        raise ValueError(f"{label} frame contains no Exp40 conditions")
    duplicates = scoped.duplicated(["eid", "condition"], keep=False)
    if duplicates.any():
        raise ValueError(f"{label} contains duplicated session-condition cells")
    counts = scoped.groupby("eid")["condition"].nunique()
    if not (counts == len(CONDITIONS)).all():
        raise ValueError(f"{label} does not contain a complete registered grid")
    if set(scoped["condition"].astype(str)) != set(CONDITIONS):
        raise ValueError(f"{label} condition family is incomplete")
    animal_counts = scoped[["eid", "animal_id"]].drop_duplicates()
    if animal_counts["animal_id"].astype(str).duplicated().any():
        raise ValueError(f"{label} requires one session per independent animal")
    for _, rows in scoped.groupby("eid", sort=False):
        statuses = set(rows["status"].astype(str))
        if statuses not in ({"complete"}, {"failed"}, {"invalid"}):
            raise ValueError(
                f"{label} session conditions have asymmetric completion status"
            )
        if statuses != {"complete"} and "error" in rows:
            errors = set(rows["error"].fillna("").astype(str))
            if len(errors) != 1:
                raise ValueError(f"{label} session failures are condition-selective")
    return scoped


def _complete_wide(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"Exp40 frame lacks endpoint {column}")
    complete = frame.loc[frame["status"].astype(str).eq("complete")].copy()
    complete[column] = pd.to_numeric(complete[column], errors="coerce")
    if complete[column].isna().any():
        raise ValueError(f"complete Exp40 rows have non-finite {column}")
    wide = complete.pivot(
        index=["eid", "animal_id"], columns="condition", values=column
    )
    if wide.isna().any().any() or set(wide.columns) != set(CONDITIONS):
        raise ValueError(f"{column} does not form a complete paired panel")
    return wide


def exp40_animal_effects(
    primary_frame: pd.DataFrame,
    probe_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per planned animal, including endpoint-ineligible rows."""

    primary = _validate_frame(primary_frame, label="primary")
    probe = _validate_frame(probe_frame, label="probe")
    primary_keys = primary[["eid", "animal_id"]].drop_duplicates()
    probe_keys = probe[["eid", "animal_id"]].drop_duplicates()
    if set(map(tuple, primary_keys.to_numpy())) != set(
        map(tuple, probe_keys.to_numpy())
    ):
        raise ValueError("primary and probe cohorts differ")

    low_column = "test_low_contrast_choice_nll"
    primary_low = _complete_wide(primary, low_column)
    probe_low = _complete_wide(probe, low_column)
    if not primary_low.index.equals(probe_low.index):
        raise ValueError("primary and probe complete panels differ")
    context = _complete_wide(primary, "context_nll")
    factorized = primary.loc[
        primary["condition"].astype(str).eq("factorized_state")
        & primary["status"].astype(str).eq("complete")
    ].set_index(["eid", "animal_id"])
    factorized_probe = probe.loc[
        probe["condition"].astype(str).eq("factorized_state")
        & probe["status"].astype(str).eq("complete")
    ].set_index(["eid", "animal_id"])
    required_factorized = {
        "factorized_nll_gain_vs_primary_baseline",
        "clamp_release_low_contrast_nll_harm",
        "clamp_concentration_low_contrast_nll_harm",
        "selected_primary_baseline",
    }
    missing = sorted(required_factorized - set(factorized.columns))
    if missing:
        raise ValueError(f"factorized records lack columns: {missing}")
    if not factorized.index.equals(factorized_probe.index):
        raise ValueError("primary and probe factorized panels differ")

    effects = primary_keys.copy().sort_values(["animal_id", "eid"])
    effects["endpoint_status"] = "failed_endpoint_eligibility"
    status_lookup = (
        primary.groupby(["eid", "animal_id"], as_index=False)["status"]
        .first()
        .set_index(["eid", "animal_id"])["status"]
    )
    complete_index = set(primary_low.index)
    rows = []
    for item in effects.itertuples(index=False):
        key = (str(item.eid), str(item.animal_id))
        row: dict[str, object] = {
            "eid": key[0],
            "animal_id": key[1],
            "endpoint_status": str(status_lookup.loc[key]),
        }
        if key in complete_index:
            selected = str(factorized.loc[key, "selected_primary_baseline"])
            row.update(
                context_nll_gain_hmm_minus_semimarkov=float(
                    context.loc[key, "learned_hmm_mean"]
                    - context.loc[key, "semimarkov_mean"]
                ),
                primary_gain_selected_baseline_minus_factorized=float(
                    factorized.loc[key, "factorized_nll_gain_vs_primary_baseline"]
                ),
                probe_gain_selected_baseline_minus_factorized=float(
                    factorized_probe.loc[key, "factorized_nll_gain_vs_primary_baseline"]
                ),
                release_clamp_nll_harm=float(
                    factorized.loc[key, "clamp_release_low_contrast_nll_harm"]
                ),
                concentration_clamp_nll_harm=float(
                    factorized.loc[key, "clamp_concentration_low_contrast_nll_harm"]
                ),
                primary_selected_baseline=selected,
                probe_selected_baseline=str(
                    factorized_probe.loc[key, "selected_primary_baseline"]
                ),
                primary_factorized_nll=float(primary_low.loc[key, "factorized_state"]),
                probe_factorized_nll=float(probe_low.loc[key, "factorized_state"]),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must contain at least two animals")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(n_resamples), values.size))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _margin_p(values: np.ndarray, margin: float) -> float:
    shifted = values - float(margin)
    if np.allclose(shifted, 0.0):
        return 1.0
    return float(
        wilcoxon(shifted, alternative="two-sided", zero_method="wilcox").pvalue
    )


def exp40_claims(
    effects: pd.DataFrame,
    *,
    seed: int = 20260726,
    n_resamples: int = 20_000,
) -> pd.DataFrame:
    """Evaluate the bounded development claims with one animal per replicate."""

    specifications = (
        (
            "context_decoding_gain",
            "context_nll_gain_hmm_minus_semimarkov",
            "learned-HMM context NLL minus semi-Markov context NLL",
            0.02,
        ),
        (
            "any_behavioral_utility",
            "primary_gain_selected_baseline_minus_factorized",
            "dev-selected baseline NLL minus factorized-state NLL",
            0.0,
        ),
        (
            "meaningful_behavioral_utility",
            "primary_gain_selected_baseline_minus_factorized",
            "dev-selected baseline NLL minus factorized-state NLL",
            0.005,
        ),
        (
            "release_actuator_contribution",
            "release_clamp_nll_harm",
            "release-clamp NLL minus intact factorized NLL",
            0.002,
        ),
        (
            "precision_actuator_contribution",
            "concentration_clamp_nll_harm",
            "precision-clamp NLL minus intact factorized NLL",
            0.002,
        ),
    )
    provisional = []
    for index, (claim, endpoint, definition, margin) in enumerate(specifications):
        values = pd.to_numeric(effects[endpoint], errors="coerce").dropna().to_numpy()
        low, high = _bootstrap_interval(
            values, seed=seed + index, n_resamples=n_resamples
        )
        provisional.append(
            {
                "claim": claim,
                "endpoint": endpoint,
                "effect_definition": definition,
                "minimum_effect": margin,
                "estimate": float(np.mean(values)),
                "ci_low": low,
                "ci_high": high,
                "n_animals": int(values.size),
                "positive_animals": int(np.sum(values > 0.0)),
                "wilcoxon_two_sided_p_at_margin": _margin_p(values, margin),
            }
        )
    adjusted = multipletests(
        [row["wilcoxon_two_sided_p_at_margin"] for row in provisional],
        method="holm",
    )[1]
    claims = []
    for row, holm_p in zip(provisional, adjusted, strict=True):
        margin = float(row["minimum_effect"])
        if holm_p <= 0.05 and float(row["ci_low"]) > margin:
            conclusion = "support"
        elif holm_p <= 0.05 and float(row["ci_high"]) < margin:
            conclusion = "oppose"
        else:
            conclusion = "inconclusive"
        claims.append(
            StateUtilityClaim(
                **row,
                holm_p=float(holm_p),
                conclusion=conclusion,
            )
        )
    return pd.DataFrame([asdict(claim) for claim in claims])


def exp40_condition_summary(
    primary_frame: pd.DataFrame,
    probe_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label, frame in (
        ("registered_readout", primary_frame),
        ("assay_probe", probe_frame),
    ):
        scoped = _validate_frame(frame, label=label)
        for condition, group in scoped.groupby("condition", sort=False):
            complete = group.loc[group["status"].astype(str).eq("complete")]
            rows.append(
                {
                    "analysis": label,
                    "condition": condition,
                    "planned_animals": int(group["animal_id"].nunique()),
                    "complete_animals": int(complete["animal_id"].nunique()),
                    "failed_animals": int(
                        group.loc[
                            ~group["status"].astype(str).eq("complete"), "animal_id"
                        ].nunique()
                    ),
                    "mean_test_low_contrast_choice_nll": float(
                        pd.to_numeric(
                            complete["test_low_contrast_choice_nll"], errors="coerce"
                        ).mean()
                    ),
                    "mean_context_nll": float(
                        pd.to_numeric(complete["context_nll"], errors="coerce").mean()
                    ),
                    "behavior_parameter_count": int(
                        pd.to_numeric(
                            complete["behavior_parameter_count"], errors="raise"
                        ).iloc[0]
                    ),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "CONDITIONS",
    "StateUtilityClaim",
    "exp40_animal_effects",
    "exp40_claims",
    "exp40_condition_summary",
]
