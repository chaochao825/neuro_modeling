# Exp26 task--actuator matching protocol

## Scope

Exp26 asks a narrower question than Exp23: given a fixed high-rank
Dale-compatible recurrent carrier and a one-dimensional centered context
signal, which *single actuator family* best realizes a previously unseen task
change? It does not claim that a biological controller has learned the
actuator, and it does not train the frozen carrier.

The registered task difference is

\[
A_{+}-A_{-}=\alpha\,\Delta A,\qquad
B_{+}-B_{-}=(1-\alpha)\,\Delta B.
\]

`rank(Delta A)` is scanned over 1, 2, 4, and 8; `rank(Delta B)` is scanned over
1, 2, and 4. Their orientations are independent QR/Haar draws. Their
amplitudes are independent log-uniform draws and are never normalized to equal
weighted demand, so the theoretical coordinate below is not an alias for
`alpha`.

## Prospective demand coordinate

For the stable reference carrier, the infinite-horizon sensitivity is

\[
D_A^2=\operatorname{tr}(W_o\Delta A W_c\Delta A^\top),\qquad
D_B^2=\operatorname{tr}(W_o\Delta B\Sigma_u\Delta B^\top),
\]

with \(W_c\) and \(W_o\) the controllability and observability Gramians.
Exp26's primary finite-trial implementation uses the stricter event-local form:
each correction current is weighted by the remaining-horizon observability
Gramian and by the registered state/input second moment at that step. The
reported coordinate is

\[
\chi=\frac{D_A}{D_A+D_B}.
\]

The generator law makes the contemporaneous state--input cross moment zero.
The finite training sample cross term is nevertheless reported. A nonzero
empirical cross audit is not silently folded into either marginal demand.

The necessary recurrent rank is audited as

\[
r\geq \operatorname{rank}(P_o\Delta A P_c),
\]

along with 99%/99.9% weighted-energy ranks and Eckart--Young tail fractions.
This is a necessary bound, not a sufficiency claim for a fixed factor
dictionary.

## Paired actuator comparison

The paired modes are `frozen`, `routing`, `gain`, `low_rank`, and `rgl`.

- `routing` fits a rank-limited input operator.
- `gain` fits a population-wise gain on the frozen recurrent plus input drive.
- `low_rank` fits a rank-limited recurrent operator.
- `rgl` is a sequential composite ceiling and never enters the primary winner
  label.

All modes share the carrier, task, block order, initial state, context, process
noise, and readout. Fits use training blocks only. Each active correction is
scaled so that its teacher-forced correction-current L2 RMS equals the true
training task-residual RMS. L1 current is reported descriptively and is not
called matched. The frozen condition is the intentional zero-current control.

The primary balanced-accuracy endpoint is the sign of the *control-induced
observable*, i.e. target/candidate output relative to their paired frozen
rollout. A scalar calibration readout is fitted only on training blocks and
then shared. Absolute final-output behavior, state NRMSE, zero-input NRMSE, and
output R2 are secondary. This causal-contrast endpoint makes actuator fidelity
measurable without allowing carrier/noise-dominated absolute outputs to mask
the controlled computation.

The frozen base obeys Dale column signs and is full rank. Effective fitted
corrections are not Dale constrained; Exp26 must therefore be described as a
high-rank Dale-compatible carrier with unconstrained effective control, not as
a fully Dale-constrained plastic network.

## Train-only budget reachability

Before any formal validation or test endpoint is evaluated, a dedicated
preflight enumerates all 30 seeds, 88 registered generators, and four active
actuator modes using a train-split-only generator. It does not construct a
validation/test split and does not call a readout, behavior scorer, or rollout.
The resulting seven-file receipt must be staged outside the Git worktree or
under the ignored `results/runs` tree so that the formal runner can require the
same clean commit/tree at preflight and execution time.
The first frozen revision used `max_scale=25`; the preflight retained this as a
failed design audit because 144 of 10,560 active fits were blocked even though
all required scales were finite. The maximum required scale was 90.0921.

The revised ceiling follows one outcome-independent rule:

\[
s_{\max}=2^{\lceil\log_2(1.25\,s_{\mathrm{train,max}})\rceil}=128.
\]

Changing this ceiling does not change the registered residual budget, raw
actuator direction, or unique matching scale. It only removes an arbitrary
numerical admission gate. Poorly aligned but finite actuators remain in the
panel: for example, the largest-scale routing fit has negative training
explained fraction and is therefore a useful negative control, not a successful
fit. Effective-dynamics stability is still evaluated separately and fails
closed.

## Manifest and inference

The analytic grid has 2,112 cells:

- `alpha = 0.0, 0.1, ..., 1.0`;
- transition rank `1, 2, 4, 8`;
- input rank `1, 2, 4`;
- delay `0, 4, 12, 24`;
- noise standard deviation `0.1, 0.3, 0.6, 1.0`.

The formal E/I tier uses a performance-independent, hash-locked balanced
subset of 44 discovery and 44 held-out generators. Each generator is split by
whole block into train/validation/test. Formal inference uses seeds 0--29;
smoke uses disjoint seeds 9000 and 9001 and is permanently development-only.

For seed \(s\), the decision threshold is fitted from discovery-validation
rows belonging to the other 29 seeds. The held-out-test family advantage is

\[
\Delta_{s,g}=BA_{\text{low-rank}}
 -\max(BA_{\text{routing}},BA_{\text{gain}}).
\]

Values within 0.01 are preregistered ties. The three co-primary seed-level
endpoints are held-out Spearman correlation, threshold-classifier balanced
accuracy, and AUROC. Whole-seed bootstrap intervals and one-sided sign-flip
tests are used; the three tests receive Holm correction and form an
intersection--union gate. Support additionally requires positive held-out
AUROC gain of `chi` over raw `alpha`. Generator, trial, neuron, and time point
are never treated as independent replicates.

Every planned row is retained. Missing seeds, duplicate attempts, failed
cells, invalid budgets, unstable effective dynamics, or inconsistent
manifests make the formal conclusion inconclusive rather than being dropped.
Every formal row also binds the canonical config hash, manifest hash, analysis
settings, Git commit/tree, Python version, and scientific-package versions.
Cross-seed disagreement in any of these receipts makes the summary fail closed.

## Current status

The two-seed smoke completed all 240 planned rows with exact paired tape
receipts and valid functional budgets and remains forcibly `inconclusive`. The
initial formal launch was stopped before any validation/test evaluation when
the train-only reachability audit rejected the old numerical ceiling. The clean
v3 preflight subsequently passed all 10,560 train-only fits with zero blockers.

The frozen 30-seed formal-v2 panel then completed all 13,200 rows with no
failed, invalid, duplicate, budget-invalid, or missing cell. The registered
conclusion is **support**: held-out seed-level Spearman rho was 0.7605 (95% CI
0.7443--0.7757), classifier balanced accuracy was 0.8572 (0.8375--0.8758), and
classifier AUROC was 0.9467 (0.9365--0.9560). All three Holm-adjusted one-sided
tests had `p=2.99997e-05`. Gramian `chi` exceeded raw `alpha` by 0.1148 AUROC
(0.1009--0.1286; `p=9.9999e-06`), satisfying the incremental gate.

This unlocks a separately registered low-dimensional selector experiment. It
does not turn Exp26 into evidence that a local biological controller learned
the selector, and RGL remains only a descriptive ceiling. The immutable
receipt, compressed raw metrics, exact run archive, logs, report, and figures
are stored under
`results/exp26_actuator_matching_formal_v2_e08beaf/`.
