# Exp41 matched-identifiability development protocol

Status: **development only; no confirmatory claim is authorized**.

Exp41 is a successor experiment, not a modification or rescue of Exp39. Its
purpose is to determine whether the apparent three-coordinate benefit survives
when process and observation noise have matched one-step marginal variance and
when reduced-factor controls receive the same fit and selection boundary.
Functional update budgets are measured here but are not yet matched.

## Motivation and novelty boundary

Exp39 already contains a fast continuation/reset posterior through its jump
mixture. Exp41 must not claim to introduce the first fast release mechanism.
Its new question is narrower: can lag-0/lag-1 increment covariance provide
useful Q/R discrimination, and does that discrimination improve held-out
prediction beyond total-variance or tied-Q/R controls?

For a no-jump random walk observed with independent noise,

\[
x_t=x_{t-1}+w_t,\qquad y_t=x_t+v_t,
\]

with \(\operatorname{Var}(w_t)=Q\) and
\(\operatorname{Var}(v_t)=R\), the observed increments obey

\[
\gamma_0=\operatorname{Var}(\Delta y_t)=Q+2R,\qquad
\gamma_1=\operatorname{Cov}(\Delta y_t,\Delta y_{t-1})=-R.
\]

This use of lagged covariance is aligned with the behavioral signature in
[Piray and Daw (2024)](https://www.nature.com/articles/s41467-024-53459-z):
volatility and observation stochasticity have opposite effects on learning,
and the sign of recent temporal correlation helps distinguish them. That work
is motivation and a future external-data target, not evidence for Exp41.

Exp41 therefore pairs regimes with the same \(Q+2R\) but different Q/R
allocation:

| Pair | Regime A \((Q,R)\) | Regime B \((Q,R)\) | Matched \(Q+2R\) |
|---|---:|---:|---:|
| low | (0.0400, 0.0100) | (0.0025, 0.02875) | 0.0600 |
| high | (0.0800, 0.0200) | (0.0100, 0.05500) | 0.1200 |

The main matched panel fixes hazard to zero. Hazard recovery and Q/R recovery
must not be conflated in this diagnostic.

## Causal information boundary

Deployable methods may consume only current and past observations plus their
own previous state. They may not receive true regime, block identity, Q, R,
future observations, test labels, or oracle change flags. Generator labels are
available only to oracle baselines and post-run recovery metrics.

All methods within a seed share the same fit tape, test tape, block order,
observations, initialization, and numerical precision. Hyperparameters and any
fixed Q/R allocation are selected on the fit tape only. Test blocks are never
used for selection, normalization, clipping-bound estimation, or early stopping.

## Development method panel

The minimum panel is:

1. `selected_fixed_jump`: fit-selected static Q/R filter;
2. `current_online_em`: the frozen current Exp39 adaptive implementation;
3. `h_plus_total_variance`: one adaptive total-increment-variance coordinate,
   with its Q/R allocation fixed using fit data only. At H=0 with fixed hazard,
   this is mathematically identical to a `tied_qr` controller with one shared
   scalar and fixed Q/R ratio, so `tied_qr` is not executed as a fake second
   baseline;
4. `autocov_factorized`: causal lag-0/lag-1 sufficient-statistic updates with
   explicit clipping and no truth access;
5. `generator_supported_seen_regime_imm`: a privileged reference whose mode
   definitions use the registered generator Q/R regimes; only its switching
   hyperparameter is selected on fit data;
6. `dynamic_qr_oracle`: true per-block Q/R, reported only as unattainable
   headroom.

Generator-supported IMM and dynamic oracle are never targets that a deployable
method must beat. The reduced-factor baseline receives the same fit tape and
hyperparameter-selection boundary as the full controller.

## Endpoints and diagnostics

The independent unit is seed. Metrics are first averaged within seed and
regime/cell before any paired inference.

Primary development endpoints:

- held-out predictive NLL on each matched member and seed-balanced aggregate;
- latent-state MSE on the same rows;
- full-controller gain over `h_plus_total_variance` (the single executed form
  of the mathematically equivalent fixed-allocation `tied_qr` baseline);
- correct ordering/discrimination of the two members of each matched pair.

Mandatory diagnostics:

- early windows of 1, 4, 8, and 16 steps, plus a late window;
- Q/R log error and signed bias;
- lag-0/lag-1 covariance recovery;
- clipping/saturation fraction and invalid-row count;
- per-cell seed coverage and paired-tape digest equality;
- update count and cumulative L1/L2 movement for adaptive coordinates.

Cross-loading is evaluated on a separate orthogonal continuous diagnostic
panel. The deliberately anticorrelated members of a matched pair cannot be
used to claim a diagonal loading matrix.

The current development implementation records parameter-update count and
cumulative L1/L2 movement but does **not** yet match those budgets across
adaptive methods. It must therefore emit `budget_matched=false`, keep the
verdict inconclusive, and cannot satisfy the development go gate regardless of
descriptive performance.

## Development seeds and formal lock

- Disclosed development seeds: `41000--41007`.
- Reserved formal seeds: `41100--41129`.

The reserved seeds must not be generated, previewed, or used for tuning until
all of the following exist in a **protocol-only commit/tag**:

1. final generator parameters, block lengths, method grids, bounds, and RNG
   domain labels;
2. exact primary estimands, MCIDs, multiple-comparison family, and conjunctive
   decision rule;
3. config and implementation hashes plus artifact schema;
4. a validator that can replay tape digests, selection, paired coverage, and
   the summary;
5. explicit failure behavior retaining every failed or invalid seed.

No formal run is part of the 2026-07-27 development stage. Exact formal MCIDs
remain intentionally unset until the development panel is complete; choosing
them after observing reserved seeds is prohibited.

## Development go/no-go decision

Exp41 may advance to a formal freeze only if the full/autocovariance controller
shows a consistent held-out direction beyond the non-duplicated reduced-factor
control, discriminates both matched pairs without truth access, and does not
obtain its aggregate gain solely from one cell or the late window. Otherwise:

- if `h_plus_total_variance` (equivalently, tied Q/R at fixed allocation)
  matches the full controller, separate Q/R control is unsupported and the
  active claim contracts to total uncertainty;
- if oracle Q/R helps but deployable methods do not, inference is the limiting
  component;
- if the oracle itself has negligible advantage, this matched task cannot
  identify useful actuator specialization and Exp42 stops.

These development outcomes guide protocol design but are not paper evidence.
