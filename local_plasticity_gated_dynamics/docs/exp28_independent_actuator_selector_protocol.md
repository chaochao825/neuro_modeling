# Exp28 post-hoc amended actuator-selector sensitivity protocol

## Status and question

This protocol is permanently labelled
`post_hoc_amended_sensitivity_non_confirmatory`. It asks whether one
low-dimensional local selector, fitted once on the frozen Exp26 seeds `0--29`
discovery/validation panel, shows the same directional pattern on the amended
seeds `30--59` heldout/test panel. The selector chooses among the frozen
task-matched families `routing`, `gain`, and `low_rank`.

The original ceiling-128 source protocol retained 13,199 of 13,200 complete
cells; seed 52's routing cell failed the frozen reachability ceiling. That
source protocol is therefore **OPPOSE**, and the corresponding selector formal
analysis is **INCONCLUSIVE**. Raising the ceiling from 128 to 256 after panel
inspection cannot restore confirmatory independence, even though the amendment
is explicit and the full 30-seed panel is rerun.

This is a post-hoc sensitivity analysis of Exp27. It does not train
the recurrent carrier, readout, actuator bases, or family policies. It also
does not infer hidden context: the three cue steps are prospective
generator-provided task-demand descriptors.

## Frozen fit and amended evaluation

There is exactly one registered root fit seed, `selector_fit_seed=2801`.

1. Load the hash-bound Exp26 formal source for seeds `0--29`.
2. Fit one feature normalizer on discovery rows only.
3. Fit one fixed-best family from discovery validation utilities only.
4. Fit one local three-factor selector and one deterministic CPU GRU-BPTT
   baseline on the same normalized discovery/validation rows.
5. Freeze all four policies, including the oracle ceiling and fixed-best
   comparator definitions.
6. Load the separately packaged seeds `30--59` panel whose sole registered
   protocol amendment raises the reachability ceiling from 128 to 256.
7. Predict every independent heldout row with the identical frozen learned
   models. No independent discovery row, validation utility, test utility,
   normalization statistic, or evaluation seed may alter the fit.

The inference scope is
`fixed_meta_train_30_post_hoc_amended_test_seeds`. The evaluation seed,
not a generator cell, neuron, or time bin, is the sensitivity-analysis unit.
Inference is conditional on this one frozen meta-training panel and the single
registered optimizer root seed `2801`; it does not marginalize uncertainty
over alternative meta panels or selector initialization seeds.

## Directional endpoints and forced overall conclusion

Only heldout generators whose `(alpha, transition_rank, input_rank)` triple
is absent from meta-training discovery rows enter the directional endpoint.
For every independent seed, first average over its strict-unseen generators.

The two registered seed-level contrasts are:

1. `local_minus_fixed_best`.
2. `local_noninferiority_contrast = (local - fixed) - 0.8 * (oracle - fixed)`.

`directional_sensitivity` is labelled support if both 95% seed-bootstrap lower
bounds exceed zero and both positive one-sided sign-flip tests pass Holm
correction at `p < 0.05`. It is labelled oppose if at least one contrast's
upper bound is non-positive and its negative test passes Holm correction; all
other outcomes are inconclusive. This directional label is descriptive.

The overall three-class conclusion is always **INCONCLUSIVE**. The config must
set `analysis.force_inconclusive=true`, and neither configuration nor observed
performance can disable this rule. Oracle is a test-aware ceiling, never a
trainable comparator.

## Fail-closed package binding

[`configs/formal/exp28_independent_actuator_selector.json`](../configs/formal/exp28_independent_actuator_selector.json)
is the completed, hash-bound amended-sensitivity registration. It records
`sensitivity_readiness=ready` and the project-relative package path
`results/exp28_source_amend1_v1_28b6c76/package`. Git tracks the immutable
archive and package metadata; the package directory must first be materialized
from that archive for a replay. The runner still fails closed if readiness is
changed, a path/hash is replaced by a placeholder, or any bound identity
differs. The completed config must be committed before the sensitivity run.

The receipt-file and conclusion-file values are SHA-256 hashes of
`source_panel_receipt.json` and `conclusion.json`. Additional bindings include
`receipt_payload_sha256`, `raw_metrics_sha256`, registered config file
and canonical hashes, and `protocol_amendment_sha256`. The runner also requires
the amended package schema/protocol, amendment id, 128-to-256 ceiling values,
performance-inspection disclosure, inference status, and
`confirmatory_independence_restored=false` to match exactly.

The run writes self-hashed and file-hashed source/package, single-fit, and
all-test decision receipts. The collector reloads both sources, reconstructs
the frozen meta fold, refits once using seed `2801`, replays all decisions,
and requires exact semantic-row equality. Every planned condition is
registered before source-package validation or selector fitting, and failures
are retained.

## Fresh confirmatory reservation

Seeds `60--89` are reserved under the separate namespace
`exp28_fresh_independent_confirmatory_v2_seeds60_89`. They are not authorized
for or mixed into this amended sensitivity analysis. A future confirmatory
config must use a newly frozen source protocol and those wholly unseen seeds;
it must not inherit the amended package's confirmatory label.

## Interpretation boundary

A directional-support result would show a consistent sensitivity pattern for
supervised task-demand-to-family selection over a frozen dictionary. It would
not show confirmatory independent-seed generalization, nor would it show
hidden belief inference, de-novo actuator formation, online scalar-reward
credit assignment, matched update budgets between local and GRU models, or a
single global biophysical motif dictionary. The local third factor remains
the full three-candidate validation-utility vector. The result likewise does
not estimate meta-training-panel or optimizer-seed uncertainty and cannot
supersede the original frozen-protocol **OPPOSE** result.
