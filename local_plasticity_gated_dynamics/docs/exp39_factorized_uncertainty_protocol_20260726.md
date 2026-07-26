# Exp39 Prospective Protocol: Factorized Uncertainty and Compositional Memory

Status: **frozen before formal-seed generation**  
Frozen at: 2026-07-26T13:40:10Z  
Protocol: exp39_factorized_uncertainty_v1

## Question and rationality

The scalar-retention lineage conflated three causes of prediction error that
have different normative consequences:

- abrupt environmental changes increase the posterior probability of replacing
  the current latent state;
- gradual process volatility increases state-transition uncertainty;
- observation noise decreases the weight assigned to the current observation.

This separation is mathematically identifiable only when the factors are
independently manipulated. Exp38 did not contain those interventions and is
therefore ineligible evidence for the new claim.

The method is a three-scalar causal controller \((h_t,Q_t,R_t)\) coupled to a
one-dimensional jump/Kalman belief. At each observation it computes a
continuation-versus-reset responsibility and local conditional second moments,
then exponentially updates \(h_t,Q_t,R_t\). It uses no future observation,
autograd, BPTT, or test-time parameter optimization.

## Novelty and non-claim boundary

Hazard learning, Kalman filtering, interacting multiple models, joint
volatility/stochasticity estimation, input-gated delta memory, and directional
forgetting are established ingredients. Relevant primary references include:

- Piray and Daw's volatility/stochasticity model:
  <https://www.nature.com/articles/s41467-024-53459-z>
- hierarchical learning under volatility:
  <https://www.nature.com/articles/s41467-021-26731-9>
- the interacting multiple-model algorithm:
  <https://ieeexplore.ieee.org/abstract/document/1299>
- Gated DeltaNet:
  <https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf>
- SIFt-RLS directional forgetting:
  <https://arxiv.org/abs/2404.10844>
- STAD adaptive state-space filtering:
  <https://arxiv.org/abs/2407.12492>

The eligible contribution is narrower: whether a three-state local controller
trained only on isolated uncertainty factors composes them on unseen joint
conditions, while preserving distinct causal clamp effects. Low dimension or
Bayesian filtering alone is not a contribution.

## Generator and split

The latent scalar follows a jump-diffusion observation model:

\[
x_t =
\begin{cases}
\tilde x_t,\quad \tilde x_t\sim\mathcal N(0,4), & J_t=1,\\
x_{t-1}+w_t,\quad w_t\sim\mathcal N(0,Q_t), & J_t=0,
\end{cases}
\qquad
J_t\sim\operatorname{Bernoulli}(h_t),
\qquad
y_t=x_t+\epsilon_t,\quad \epsilon_t\sim\mathcal N(0,R_t).
\]

Registered low/high levels are:

| Factor | Low | High |
|---|---:|---:|
| \(h\) | 0.0025 | 0.06 |
| \(Q\) | 0.0025 | 0.04 |
| \(R\) | 0.01 | 0.16 |

Each sequence has 16 balanced, randomly ordered 96-step blocks. Time points are
never randomly split. Four complete sequences form each fit or test tape.
Sequences 0--1 calibrate the Gaussian fixed baselines; sequences 2--3 select
all hyperparameters.

Fit conditions contain only the baseline and one-factor elevations:
000, 100, 010, 001.

The primary test panel contains the unseen pairwise and triple compositions:
110, 101, 011, 111. Test tapes also contain the four seen conditions for clamp
selectivity and in-distribution diagnostics.

All methods receive exactly the same observations, order, latent realization,
jump events, sequence boundaries, and random seed. They never receive block
boundaries or true factor values. Tape SHA-256 digests are stored per seed.

## Methods and train-only selection

1. **Best fixed**: fit-only selection over causal EMA, rolling windows, and a
   stationary jump/Kalman grid. EMA/window predictive variance is estimated on
   calibration sequences and selected on disjoint selection sequences.
2. **Seen-mode IMM**: a four-mode interacting multiple-model filter containing
   exactly the four fit conditions. Its symmetric mode-transition probability
   is selected only on fit sequences. This is the strong compositional
   baseline the factorized method must beat.
3. **Factorized controller**: three online-EM adaptation rates are selected only
   on fit sequences. Initial \(h,Q,R\) values are fixed geometric midpoints of
   the registered ranges.
4. **Eight-mode factorial IMM**: receives all eight generator modes, including
   held-out combinations. It is a privileged upper bound, not a fair baseline.
5. **Dynamic oracle**: receives true time-varying \(h_t,Q_t,R_t\). It is an
   upper bound.
6. **Clamp h/Q/R**: replaces only the named effective controller component with
   its initial value; all other state updates and the observation tape remain
   unchanged.

The registered adaptation grids are:

- \(h\): 0.002, 0.01, 0.05;
- \(Q\): 0.08, 0.2, 0.5;
- \(R\): 0.08, 0.2, 0.5.

These ranges were chosen using eight disclosed development seeds
39000--39007. Development v1 and v2 are retained under results/development/;
neither is claim-eligible. Formal seeds 39100--39129 had not been generated or
inspected at freeze.

## Endpoints and independent unit

The independent unit is the seed. Block and time-point rows are descriptive
within-seed measurements and are never treated as independent replicates.

Primary endpoint: source-sequence/block-equal one-step predictive NLL on unseen
compositions.

Secondary endpoints:

- filtered latent-state MSE;
- early versus late post-block-change NLL;
- seed-level log-parameter correlation and absolute log error;
- clamp penalty and factor selectivity.

Five one-sided seed-level sign tests form one Holm family:

1. factorized over selected fixed;
2. factorized over seen-mode IMM;
3. \(h\)-clamp selectivity;
4. \(Q\)-clamp selectivity;
5. \(R\)-clamp selectivity.

Seed bootstrap confidence intervals use 100,000 resamples and fixed statistics
seed 39039.

## Frozen acceptance and stop rule

Utility requires both:

- factorized over best fixed: mean NLL gain at least 0.01, positive in at least
  24/30 seeds, Holm-adjusted \(p\le 0.05\);
- factorized over seen-mode IMM: mean NLL gain at least 0.005, positive in at
  least 21/30 seeds, Holm-adjusted \(p\le 0.05\).

For each of \(h,Q,R\), clamping must cause:

- mean penalty at the high level at least 0.002 nats;
- high-minus-low penalty at least 0.002 nats;
- positive selectivity in at least 21/30 seeds;
- Holm-adjusted \(p\le 0.05\).

All five gates are conjunctive. Failure of any gate yields **oppose**, freezes
IBL behavioral/neural stages, and cannot be rescued by post-hoc threshold,
generator, network, or baseline changes. Passing yields only bounded synthetic
support and does not by itself establish a real-data or biological claim.

