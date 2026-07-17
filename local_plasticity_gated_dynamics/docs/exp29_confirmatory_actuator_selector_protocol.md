# Exp29 one-shot confirmatory actuator-selector protocol

## Registration boundary

This protocol must be committed before any seed 60--89 source output is read.
It fits one selector set on the already frozen Exp26 seeds 0--29 meta panel and
evaluates that set exactly once on an immutable Exp29 source package. Exp28
seeds 30--59 and all post-hoc ceiling-amendment results are excluded.

The registration is
`configs/formal/exp29_confirmatory_actuator_selector.json`. Its model,
normalization, endpoints, statistics, heldout generator registry, seed sets,
and failure rule are frozen now. Only the future immutable package path, its
receipt/conclusion/raw/payload hashes, and `confirmatory_readiness=ready` may be
filled after packaging. The source config canonical hash, source config file
hash, and source-contract hash are already bound.

The two-stage registration is protected by the hard-coded analysis-contract
SHA-256 `c680985c2d23c0b230be185b7f28ceb8a41e0377429f6fbd085c24a87e379e69`.
Its canonical payload replaces only those five package-materialization fields
and readiness with a sentinel; every other config field remains hashed. Thus
mechanical materialization leaves the contract unchanged, while changing a
seed, generator, feature, model hyperparameter, endpoint, statistic, decision
rule, or source identity invalidates it. The config also binds normalized
runner and summarizer source hashes. Normalization ignores exactly one
analysis-contract digest literal and no executable code.

## One frozen fit

There is one root fit seed, `2801`.

1. Load the hash-bound Exp26 formal result for seeds 0--29.
2. Fit the feature normalizer on discovery features only.
3. Fit fixed-best using discovery validation utility only.
4. Fit the local three-factor selector and deterministic CPU GRU-BPTT baseline
   once on those same meta rows.
5. Freeze every fit and decision rule.
6. Load the hash-bound Exp29 package with `require_complete=true`.
7. Evaluate seeds 60--89 heldout cells without any refit or preprocessing fit.

The GRU is an isolated BPTT baseline. The local main model discloses and
enforces `used_autograd=false` and `used_bptt=false`. Confirmatory utilities,
feasibility flags, and discovery rows are never training inputs.

## Unconditional deployment endpoints

All 44 registered heldout cells in each of the 30 evaluation seeds enter both
primary endpoints. Composition overlap and feasibility do not remove cells.
The seed, not a cell, generator, neuron, or time bin, is the statistical unit.

The selector dictionary is `routing`, `gain`, and `low_rank`. If a requested
active actuator is infeasible, its registered deployment utility is the exact
same-cell frozen utility. The oracle choice set is frozen plus feasible active
families; it cannot choose an infeasible active row. RGL remains a source-only
combined control and is not a selector candidate.

For each seed, average over all 44 cells and compute:

1. `local_minus_fixed_best`;
2. `local_noninferiority_contrast = (local - fixed) - 0.8 * (oracle - fixed)`.

Support requires both 95% seed-bootstrap lower bounds above zero and both
positive one-sided seed sign-flip tests to pass Holm correction at 0.05.
Oppose requires at least one non-positive upper bound and its negative test to
pass Holm correction. Otherwise the result is inconclusive. Unlike Exp28,
there is no `force_inconclusive` switch.

These endpoints test unconditional deployed selector utility. An infeasible
fallback cell remains evidence for deployed performance but cannot support a
matched-budget mechanism claim. Family infeasibility and local fallback rates
are reported separately by seed.

## Immutable package and invalidity

The runner binds the package receipt file, conclusion file, canonical raw
JSONL, receipt payload, registered source config, and source contract hashes.
It also requires the exact package/protocol/evidence schemas, source-only role,
standalone-inference flags, complete 30x88x5 source coverage, one source
attempt per seed, and `source_panel_valid=true`.

The selector registers all 30x44x4 conditions before package loading or model
fitting. Missing source cells, a failed/invalid source row, package tampering,
model/runtime failure, incomplete semantic output, or a second selector
attempt makes the confirmatory conclusion **INVALID**. Such a failure is not
converted to scientific infeasibility or an inconclusive result, and no
replacement attempt is allowed.

## Execution boundary

Do not run the selector until the seven source/adapter files and this selector
registration are committed and pushed, the source panel has run exactly once,
and its immutable package hashes have been mechanically inserted. The runner
and summarizer both require a clean, stable Git commit/tree. Tests use only
synthetic seeds and must never instantiate the real 60--89 task runner.
