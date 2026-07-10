"""Paired model inference at seed, session, or animal level only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd


FloatArray = NDArray[np.float64]
ALLOWED_REPLICATE_UNITS = frozenset({"seed", "session", "animal"})


def validate_replicate_unit(replicate_unit: str) -> str:
    """Validate the independent replicate level and explicitly reject neurons."""

    if not isinstance(replicate_unit, str):
        raise TypeError("replicate_unit must be a string")
    normalized = replicate_unit.strip().lower()
    if normalized not in ALLOWED_REPLICATE_UNITS:
        allowed = ", ".join(sorted(ALLOWED_REPLICATE_UNITS))
        raise ValueError(
            f"replicate_unit must be one of {{{allowed}}}; neurons are not independent replicates"
        )
    return normalized


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_unit_ids(unit_ids: ArrayLike, *, length: int) -> NDArray[np.object_]:
    identifiers = np.asarray(unit_ids, dtype=object)
    if identifiers.ndim != 1 or identifiers.size != length:
        raise ValueError("unit_ids must be one-dimensional and match value rows")
    for identifier in identifiers.tolist():
        if identifier is None or isinstance(identifier, (bool, np.bool_)):
            raise ValueError("unit_ids must be non-missing hashable identifiers")
        try:
            hash(identifier)
        except TypeError:
            raise ValueError("unit_ids must be hashable scalar identifiers") from None
        try:
            missing = identifier != identifier
        except (TypeError, ValueError):
            raise ValueError("unit_ids must be non-missing scalar identifiers") from None
        if not isinstance(missing, (bool, np.bool_)) or bool(missing):
            raise ValueError("unit_ids must be non-missing scalar identifiers")
    return identifiers


def _prepare_paired_units(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    unit_ids: ArrayLike,
    replicate_unit: str,
) -> tuple[FloatArray, FloatArray, NDArray[np.object_], str]:
    unit_kind = validate_replicate_unit(replicate_unit)
    candidate_values = _finite_vector(candidate, name="candidate")
    reference_values = _finite_vector(reference, name="reference")
    if candidate_values.shape != reference_values.shape:
        raise ValueError("candidate and reference must have identical shapes")
    identifiers = _validate_unit_ids(unit_ids, length=candidate_values.size)

    # Aggregate folds/trials within each independent unit before inference.
    # Dictionary insertion order preserves the caller's deterministic order.
    grouped: dict[object, list[float]] = {}
    for identifier, candidate_value, reference_value in zip(
        identifiers.tolist(), candidate_values.tolist(), reference_values.tolist(), strict=True
    ):
        if identifier not in grouped:
            grouped[identifier] = [0.0, 0.0, 0.0]
        grouped[identifier][0] += candidate_value
        grouped[identifier][1] += reference_value
        grouped[identifier][2] += 1.0
    if len(grouped) < 2:
        raise ValueError("paired inference requires at least two independent units")

    aggregated_candidate = np.asarray(
        [totals[0] / totals[2] for totals in grouped.values()], dtype=np.float64
    )
    aggregated_reference = np.asarray(
        [totals[1] / totals[2] for totals in grouped.values()], dtype=np.float64
    )
    aggregated_ids = np.asarray(list(grouped.keys()), dtype=object)
    return aggregated_candidate, aggregated_reference, aggregated_ids, unit_kind


def _validate_bootstrap_options(
    *, n_resamples: int, confidence: float, seed: int
) -> tuple[int, float, int]:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)):
        raise TypeError("n_resamples must be an integer")
    if int(n_resamples) < 100:
        raise ValueError("n_resamples must be at least 100")
    if isinstance(confidence, bool) or not np.isscalar(confidence):
        raise TypeError("confidence must be a scalar")
    confidence_value = float(confidence)
    if not np.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    return int(n_resamples), confidence_value, int(seed)


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Percentile bootstrap of candidate-minus-reference unit differences."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_units: int
    n_resamples: int
    replicate_unit: str
    unit_differences: FloatArray


def paired_bootstrap(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    unit_ids: ArrayLike,
    replicate_unit: str,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int,
    statistic: Literal["mean", "median"] = "mean",
) -> PairedBootstrapResult:
    """Bootstrap whole paired seed/session/animal units with replacement."""

    candidate_values, reference_values, _, unit_kind = _prepare_paired_units(
        candidate,
        reference,
        unit_ids=unit_ids,
        replicate_unit=replicate_unit,
    )
    n_resamples_value, confidence_value, seed_value = _validate_bootstrap_options(
        n_resamples=n_resamples, confidence=confidence, seed=seed
    )
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    differences = candidate_values - reference_values
    reducer = np.mean if statistic == "mean" else np.median
    estimate = float(reducer(differences))
    rng = np.random.default_rng(seed_value)
    indices = rng.integers(
        0, differences.size, size=(n_resamples_value, differences.size), endpoint=False
    )
    samples = reducer(differences[indices], axis=1)
    alpha = (1.0 - confidence_value) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return PairedBootstrapResult(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence=confidence_value,
        n_units=int(differences.size),
        n_resamples=n_resamples_value,
        replicate_unit=unit_kind,
        unit_differences=differences.copy(),
    )


@dataclass(frozen=True)
class PairedWilcoxonResult:
    """Wilcoxon signed-rank result on aggregated independent units."""

    statistic: float
    p_value: float
    n_units: int
    n_nonzero: int
    alternative: str
    replicate_unit: str


def paired_wilcoxon(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    unit_ids: ArrayLike,
    replicate_unit: str,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> PairedWilcoxonResult:
    """Run Wilcoxon after aggregating repeated rows within each legal unit."""

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    candidate_values, reference_values, _, unit_kind = _prepare_paired_units(
        candidate,
        reference,
        unit_ids=unit_ids,
        replicate_unit=replicate_unit,
    )
    differences = candidate_values - reference_values
    nonzero = int(np.count_nonzero(differences))
    if nonzero == 0:
        statistic_value, p_value = 0.0, 1.0
    else:
        from scipy.stats import wilcoxon

        result = wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative=alternative,
            method="auto",
        )
        statistic_value, p_value = float(result.statistic), float(result.pvalue)
    return PairedWilcoxonResult(
        statistic=statistic_value,
        p_value=p_value,
        n_units=int(differences.size),
        n_nonzero=nonzero,
        alternative=alternative,
        replicate_unit=unit_kind,
    )


@dataclass(frozen=True)
class PairedInferenceResult:
    """Paired bootstrap interval and signed-rank test from identical units."""

    bootstrap: PairedBootstrapResult
    wilcoxon: PairedWilcoxonResult


def paired_inference(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    unit_ids: ArrayLike,
    replicate_unit: str,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int,
    statistic: Literal["mean", "median"] = "mean",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> PairedInferenceResult:
    """Compute both paired procedures using the same explicit unit level."""

    bootstrap = paired_bootstrap(
        candidate,
        reference,
        unit_ids=unit_ids,
        replicate_unit=replicate_unit,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        statistic=statistic,
    )
    signed_rank = paired_wilcoxon(
        candidate,
        reference,
        unit_ids=unit_ids,
        replicate_unit=replicate_unit,
        alternative=alternative,
    )
    return PairedInferenceResult(bootstrap=bootstrap, wilcoxon=signed_rank)


def _holm_adjust(p_values: Sequence[float]) -> FloatArray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("p_values must be a non-empty finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values)
    sorted_values = values[order]
    multipliers = values.size - np.arange(values.size)
    adjusted_sorted = np.maximum.accumulate(sorted_values * multipliers)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _normalize_metrics(metric_columns: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(metric_columns, str):
        metrics = (metric_columns,)
    else:
        metrics = tuple(metric_columns)
    if not metrics or any(not isinstance(metric, str) or not metric for metric in metrics):
        raise ValueError("metric_columns must contain at least one valid column name")
    if len(set(metrics)) != len(metrics):
        raise ValueError("metric_columns cannot contain duplicates")
    return metrics


def _higher_is_better_map(
    metrics: Sequence[str], higher_is_better: bool | Mapping[str, bool]
) -> dict[str, bool]:
    if isinstance(higher_is_better, (bool, np.bool_)):
        return {metric: bool(higher_is_better) for metric in metrics}
    if not isinstance(higher_is_better, Mapping):
        raise TypeError("higher_is_better must be boolean or a metric-to-boolean mapping")
    missing = set(metrics) - set(higher_is_better)
    if missing:
        raise ValueError(f"higher_is_better is missing metrics: {sorted(missing)}")
    mapping: dict[str, bool] = {}
    for metric in metrics:
        value = higher_is_better[metric]
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("higher_is_better mapping values must be boolean")
        mapping[metric] = bool(value)
    return mapping


def _optional_column_tuple(
    columns: str | Sequence[str] | None, *, name: str
) -> tuple[str, ...]:
    if columns is None:
        return ()
    if isinstance(columns, str):
        normalized = (columns,)
    else:
        normalized = tuple(columns)
    if any(not isinstance(column, str) or not column for column in normalized):
        raise ValueError(f"{name} must contain valid column names")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} cannot contain duplicate columns")
    return normalized


def build_model_comparison_table(
    results: pd.DataFrame,
    *,
    metric_columns: str | Sequence[str],
    unit_column: str,
    replicate_unit: str,
    model_column: str = "model",
    parameter_column: str | None = "parameter_count",
    pair_columns: str | Sequence[str] | None = None,
    session_column: str | None = None,
    baseline_model: str = "common",
    models: Sequence[str] = ("common", "shared", "full"),
    higher_is_better: bool | Mapping[str, bool] = True,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int,
) -> pd.DataFrame:
    """Build a paired common/shared/full LDS comparison table.

    ``pair_columns`` identifies folds/blocks that must be present for every
    selected model within a unit (and session, when supplied).  With no pair
    columns, exactly one row per unit/model is required.  For animal-level
    inference, ``session_column`` enables the required fold -> session ->
    animal aggregation so animals with more sessions or folds do not receive
    extra weight.

    The selected models must form a complete paired panel; missing,
    duplicated, or non-finite cells raise instead of being silently discarded.
    Holm correction is performed separately within each metric across the
    selected non-baseline model comparisons; ``holm_family`` records that
    family explicitly.  Extra models are excluded unless named in ``models``.
    """

    import pandas as pd

    if not isinstance(results, pd.DataFrame) or results.empty:
        raise ValueError("results must be a non-empty pandas DataFrame")
    if not results.columns.is_unique:
        raise ValueError("results cannot contain duplicate column names")
    unit_kind = validate_replicate_unit(replicate_unit)
    for column_name, label in ((unit_column, "unit_column"), (model_column, "model_column")):
        if not isinstance(column_name, str) or not column_name:
            raise ValueError(f"{label} must be a valid column name")
    lowered_unit_column = unit_column.lower()
    if "neuron" in lowered_unit_column or "cell" in lowered_unit_column:
        raise ValueError("neuron/cell columns cannot be used as replicate units")
    for named_level in ALLOWED_REPLICATE_UNITS:
        if named_level in lowered_unit_column and named_level != unit_kind:
            raise ValueError(
                f"unit_column appears to identify {named_level!r}, not {unit_kind!r} replicates"
            )
    metrics = _normalize_metrics(metric_columns)
    paired_columns = _optional_column_tuple(pair_columns, name="pair_columns")
    if session_column is not None:
        if not isinstance(session_column, str) or not session_column:
            raise ValueError("session_column must be None or a valid column name")
        if unit_kind != "animal":
            raise ValueError("session_column is only valid for animal-level inference")
    structural_columns = [unit_column, model_column]
    if session_column is not None:
        structural_columns.append(session_column)
    structural_columns.extend(paired_columns)
    if len(set(structural_columns)) != len(structural_columns):
        raise ValueError("unit/model/session/pair columns must be distinct")
    if set(metrics) & set(structural_columns):
        raise ValueError("metric columns cannot also be grouping columns")
    direction = _higher_is_better_map(metrics, higher_is_better)
    n_resamples, confidence, seed = _validate_bootstrap_options(
        n_resamples=n_resamples, confidence=confidence, seed=seed
    )
    if isinstance(models, str):
        raise TypeError("models must be a sequence of model names, not a string")
    selected_models = tuple(models)
    if (
        len(selected_models) < 2
        or any(not isinstance(model, str) or not model for model in selected_models)
        or len(set(selected_models)) != len(selected_models)
    ):
        raise ValueError("models must contain at least two distinct non-empty names")
    if not isinstance(baseline_model, str) or not baseline_model:
        raise ValueError("baseline_model must be a non-empty string")
    if baseline_model not in selected_models:
        raise ValueError("baseline_model must be present in models")

    required_columns = {*structural_columns, *metrics}
    if parameter_column is not None:
        if not isinstance(parameter_column, str) or not parameter_column:
            raise ValueError("parameter_column must be None or a valid column name")
        if parameter_column in structural_columns or parameter_column in metrics:
            raise ValueError("parameter_column must be distinct from grouping and metrics")
        required_columns.add(parameter_column)
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(f"results is missing columns: {sorted(missing_columns)}")

    data = results.loc[results[model_column].isin(selected_models), list(required_columns)].copy()
    observed_models = set(data[model_column].dropna().tolist())
    missing_models = set(selected_models) - observed_models
    if missing_models:
        raise ValueError(f"results is missing selected models: {sorted(missing_models)}")
    if data[structural_columns].isna().any(axis=None):
        raise ValueError("unit/model/session/pair identifiers cannot be missing")
    if data.empty:
        raise ValueError("results has no rows for selected models")

    for metric in metrics:
        if pd.api.types.is_bool_dtype(data[metric]):
            raise TypeError(f"metric {metric!r} must be numeric, not boolean")
        try:
            data[metric] = pd.to_numeric(data[metric], errors="raise")
        except (TypeError, ValueError):
            raise TypeError(f"metric {metric!r} must be numeric") from None
        if not np.all(np.isfinite(data[metric].to_numpy(dtype=float))):
            raise ValueError(f"metric {metric!r} must be finite; failed runs cannot be hidden")

    parameter_summary: dict[str, tuple[float, float, float]] = {}
    if parameter_column is not None:
        if pd.api.types.is_bool_dtype(data[parameter_column]):
            raise TypeError("parameter counts must be numeric, not boolean")
        try:
            data[parameter_column] = pd.to_numeric(data[parameter_column], errors="raise")
        except (TypeError, ValueError):
            raise TypeError("parameter counts must be numeric") from None
        parameter_values = data[parameter_column].to_numpy(dtype=float)
        if (
            not np.all(np.isfinite(parameter_values))
            or np.any(parameter_values < 0.0)
            or np.any(parameter_values != np.floor(parameter_values))
        ):
            raise ValueError("parameter counts must be finite non-negative integers")
        parameter_level = [unit_column]
        if session_column is not None:
            parameter_level.append(session_column)
        parameter_level.append(model_column)
        parameter_variation = data.groupby(
            parameter_level, sort=False, dropna=False
        )[parameter_column].nunique(dropna=False)
        if (parameter_variation > 1).any():
            raise ValueError(
                "parameter count must be constant across paired folds within a fitted model"
            )

    cell_columns = [unit_column]
    if session_column is not None:
        cell_columns.append(session_column)
    cell_columns.extend(paired_columns)
    cell_model_columns = [*cell_columns, model_column]
    cell_counts = data.groupby(
        cell_model_columns, sort=False, dropna=False
    ).size()
    if (cell_counts != 1).any():
        if paired_columns:
            raise ValueError(
                "each unit/session/pair/model cell must contain exactly one row"
            )
        raise ValueError(
            "multiple rows per unit/session/model require pair_columns or prior aggregation"
        )
    presence = cell_counts.unstack(model_column, fill_value=0)
    presence = presence.reindex(columns=selected_models, fill_value=0)
    if (presence == 0).any(axis=None):
        raise ValueError(
            "selected models do not form a complete paired fold/block panel"
        )
    if data[unit_column].nunique(dropna=False) < 2:
        raise ValueError("model comparison requires at least two independent units")

    value_columns = list(metrics)
    if parameter_column is not None:
        value_columns.append(parameter_column)
    session_level_columns = [unit_column]
    if session_column is not None:
        session_level_columns.append(session_column)
    session_model_columns = [*session_level_columns, model_column]
    if paired_columns:
        session_level = data.groupby(
            session_model_columns, sort=False, as_index=False, dropna=False
        )[value_columns].mean()
    else:
        session_level = data[[*session_model_columns, *value_columns]].copy()

    if session_column is not None:
        grouped = session_level.groupby(
            [unit_column, model_column], sort=False, as_index=False, dropna=False
        )[value_columns].mean()
    else:
        grouped = session_level
    if grouped.duplicated([unit_column, model_column]).any():
        raise RuntimeError("hierarchical aggregation did not produce one row per unit/model")

    parameter_summary = {
        model: (np.nan, np.nan, np.nan) for model in selected_models
    }
    if parameter_column is not None:
        for model in selected_models:
            values = grouped.loc[
                grouped[model_column] == model, parameter_column
            ].to_numpy(dtype=float)
            parameter_summary[model] = (
                float(np.mean(values)), float(np.min(values)), float(np.max(values))
            )
    records: list[dict[str, object]] = []
    pvalue_locations: dict[str, list[int]] = {metric: [] for metric in metrics}
    raw_pvalues: dict[str, list[float]] = {metric: [] for metric in metrics}

    for metric_index, metric in enumerate(metrics):
        panel = grouped.pivot(index=unit_column, columns=model_column, values=metric)
        panel = panel.loc[:, list(selected_models)]
        if panel.isna().any(axis=None):
            raise ValueError("paired panel contains missing aggregated metric values")
        identifiers = panel.index.to_numpy(dtype=object)
        baseline_values = panel[baseline_model].to_numpy(dtype=float)
        for model_index, model in enumerate(selected_models):
            model_values = panel[model].to_numpy(dtype=float)
            mean_value = float(np.mean(model_values))
            median_value = float(np.median(model_values))
            std_value = float(np.std(model_values, ddof=1))
            parameter_mean, parameter_min, parameter_max = parameter_summary[model]
            if model == baseline_model:
                difference = improvement = 0.0
                difference_low = difference_high = 0.0
                improvement_low = improvement_high = 0.0
                p_value = p_holm = 1.0
            else:
                comparison_seed = int(
                    np.random.SeedSequence([int(seed), metric_index, model_index])
                    .generate_state(1)[0]
                )
                inference = paired_inference(
                    model_values,
                    baseline_values,
                    unit_ids=identifiers,
                    replicate_unit=unit_kind,
                    n_resamples=n_resamples,
                    confidence=confidence,
                    seed=comparison_seed,
                )
                difference = inference.bootstrap.estimate
                difference_low = inference.bootstrap.ci_low
                difference_high = inference.bootstrap.ci_high
                if direction[metric]:
                    improvement = difference
                    improvement_low, improvement_high = difference_low, difference_high
                else:
                    improvement = -difference
                    improvement_low, improvement_high = -difference_high, -difference_low
                p_value = inference.wilcoxon.p_value
                p_holm = np.nan
            record = {
                "metric": metric,
                "model": model,
                "baseline_model": baseline_model,
                "replicate_unit": unit_kind,
                "n_units": int(panel.shape[0]),
                "mean": mean_value,
                "std": std_value,
                "median": median_value,
                "difference_vs_baseline": difference,
                "difference_ci_low": difference_low,
                "difference_ci_high": difference_high,
                "improvement_vs_baseline": improvement,
                "improvement_ci_low": improvement_low,
                "improvement_ci_high": improvement_high,
                "wilcoxon_p": p_value,
                "wilcoxon_p_holm": p_holm,
                "holm_family": f"metric={metric};models_vs={baseline_model}",
                "parameter_count_mean": parameter_mean,
                "parameter_count_min": parameter_min,
                "parameter_count_max": parameter_max,
            }
            records.append(record)
            if model != baseline_model:
                pvalue_locations[metric].append(len(records) - 1)
                raw_pvalues[metric].append(p_value)

    for metric in metrics:
        adjusted = _holm_adjust(raw_pvalues[metric])
        for record_index, p_value in zip(
            pvalue_locations[metric], adjusted.tolist(), strict=True
        ):
            records[record_index]["wilcoxon_p_holm"] = p_value
    return pd.DataFrame.from_records(records)


def common_shared_full_comparison_table(
    results: pd.DataFrame, **kwargs: object
) -> pd.DataFrame:
    """Alias enforcing the requested common/shared/full model order."""

    if "models" in kwargs:
        raise TypeError("common_shared_full_comparison_table fixes the models argument")
    return build_model_comparison_table(
        results, models=("common", "shared", "full"), **kwargs
    )
