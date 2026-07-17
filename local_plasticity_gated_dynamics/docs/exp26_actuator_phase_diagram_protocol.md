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

## Current status

The two-seed smoke completed all 240 planned rows with exact paired tape
receipts and valid functional budgets. Its seed-level held-out Spearman values
were 0.856 and 0.725. These numbers validate execution only: the smoke summary
is forcibly `inconclusive`, and no threshold, grid cell, or formal criterion is
changed from this result. A low-dimensional actuator selector is permitted
only after the frozen formal Exp26 analysis passes its registered gates.
