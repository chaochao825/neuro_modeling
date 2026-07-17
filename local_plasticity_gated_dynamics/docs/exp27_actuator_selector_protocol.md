# Exp27: low-dimensional actuator selector

## Registered question

Exp27 asks whether a learned low-dimensional selector can choose among the
three *already fitted and frozen* Exp26 actuator families: input routing,
population gain, and a low-rank recurrent operator.  It does not refit the
carrier, the actuator bases, the readout, or any Exp26 trajectory.  The Exp26
raw panel and conclusion are immutable candidate libraries guarded by exact
SHA-256 receipts.

The confirmatory comparison is the local three-factor selector against the
best single actuator family chosen without the outer seed.  The oracle is a
test-aware ceiling only.  A deterministic CPU GRU trained by BPTT is an
isolated engineering baseline and is never described as local learning.

## Nested seed and composition split

For outer seed `s`, all selector fitting and fixed-family selection use only:

1. source seeds other than `s`;
2. Exp26 `discovery` generator compositions; and
3. Exp26 validation balanced accuracy.

The frozen selector is then evaluated once on seed `s`, Exp26 `heldout`
generator compositions, and test balanced accuracy.  Discovery and held-out
generator IDs are disjoint, but the formal manifests contain one overlapping
`(alpha, transition rank, input rank)` triple.  All 44 held-out cells per seed
are retained; the one overlap is explicitly secondary and the confirmatory
primary endpoint uses only the 43 strict-unseen triples.  Thus neither time bins, neurons, the outer
seed, held-out compositions, nor outer test utilities enter selector fitting
or hyperparameter selection.  Hyperparameters are fixed in the JSON config.

Each outer seed contributes one row per held-out generator and selector:
`oracle`, `gru_bptt`, `local_three_factor`, and `fixed_best`.  Candidate
utilities for all three frozen actuator
families are retained on every row so downstream analysis need not reconstruct
or selectively discard failed comparisons.

## Frozen dictionary and selector inputs

The dictionary order is registered as `routing`, `gain`, `low_rank`.  It is
also the deterministic tie-break order.  Selector features are only the
task-side Gramian demand fraction `chi`, state/input demands, transition/input
ranks, delay, and noise.  `alpha` and the audited demand cross term are retained
for audit/stratification but are forbidden selector inputs.  The transformation
uses log demands, log2 ranks, scaled delay, log noise, and an explicit bias;
normalization is fitted on the nested training fold only.  The three temporal
cues are demand, ranks, and timing/noise.

The local model uses 200 epochs, learning rate 0.05, temperature 1.0,
teacher temperature 0.05, and L2 coefficient 1e-4.  Its update is explicitly
three-factor and sets both `used_autograd` and `used_bptt` to false.  The GRU
has hidden dimension 8, 200 epochs, learning rate 0.02, weight decay 1e-4,
runs deterministically on CPU, and is labelled with both flags true.

Here the controller input sometimes denoted `b_t` is a prospective sequence
of task-demand cues supplied by the registered generator descriptors.  It is
not an online belief inferred from hidden observations.  Exp27 therefore tests
actuator selection under unseen task compositions, not hidden-state inference,
and must not be cited as evidence that a belief gate was learned online.

Each Exp26 candidate family was itself task-matched on that generator's
training blocks and then frozen.  Consequently the three entries are frozen
*family policies* for the present audit, not one global set of fixed
biophysical motif matrices shared unchanged across all tasks.  Exp27 learns
only which family policy to select; success cannot establish a universal
biophysical motif dictionary.

The local training third factor is also privileged: it is the complete
three-candidate validation-utility vector transformed into a soft teacher,
not a scalar reward observed from only the chosen action.  With the registered
equal belief/eligibility decays, the implementation is best interpreted as a
hand-written local soft-label multinomial update, not as independent evidence
for temporal e-prop or forward-sensitivity credit assignment.  Local and GRU
update L1/L2 values remain descriptive because their parameterizations and
update schedules differ; they are not a matched plasticity budget.

Outer-seed LOSO models share most of their meta-training seeds.  The
registered seed bootstrap and sign-flip tests are therefore cross-fitted,
conditional inference rather than 30 fully independent model fits.  A fixed
meta-train/independent-test seed split is reserved as a sensitivity analysis.
The term `strict-unseen` refers only to the registered `(alpha, transition
rank, input rank)` triple; delay, noise, and rotation coordinates need not all
be novel simultaneously.

## Outcomes and interpretation

For each held-out composition, utility is the Exp26 test balanced accuracy of
the selected frozen actuator.  Regret is oracle utility minus selected
utility.  Recovered oracle gain is

`(selector utility - fixed utility) / (oracle utility - fixed utility)`

when the denominator is positive; zero-denominator cells are retained and
marked inapplicable for that ratio.  The registered system-level target is
that the local selector recovers at least 80% of the oracle-over-fixed gain
and outperforms the fixed actuator across independent outer seeds.  A later
summary must classify the result as support, oppose, or inconclusive; matrix
rank alone is not supporting evidence.

The smoke profile uses seeds 9000 and 9001 and is always development-only and
inconclusive.  It retains 12 held-out cells per seed, of which four are strict
unseen triples; its 20 training epochs are a runtime check and not the frozen
200-epoch formal estimate.  Formal evidence requires seeds 0--29, the fixed shared run label
`exp27-formal-v1`, a clean and stable Git commit/tree, complete source/hash
validation, and full planned-condition coverage.  Any mismatch fails closed
while preserving the complete plan and failure records through
`ExperimentRun`.
