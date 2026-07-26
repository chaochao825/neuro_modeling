# Exp42 actuator-factorization audit plan

Status: **conditionally planned and locked**. Exp42 must not run unless Exp41
meets its development go criteria and a separate prospective protocol is
frozen before any Exp42 outcome is observed.

Outcome note (2026-07-27): Exp41 separated matched Q/R regimes but failed the
required predictive-utility and early-transition gates against total variance
and current online-EM. Exp42 is therefore **not authorized and was not run**.
This document is retained as a conditional historical design, not an active
experiment protocol.

## Question

Exp39 shows average synthetic prediction benefit but leaves inference and
actuation confounded. Exp42 asks which component limits performance:

1. can the controller infer distinct change, process-noise, and
   observation-noise signals from causal observations; and
2. when those signals are available, do separate write, retain, and release
   actions improve held-out utility beyond a matched lower-dimensional action?

This is an actuator-matching audit, not a claim that three physical latent
variables or biological circuits have already been identified.

The separation between event inference and parameter adaptation follows the
run-length posterior view of
[Bayesian online changepoint detection](https://arxiv.org/abs/0710.3742).
The actuator question is also distinct from merely adding another gate:
[Gated DeltaNet](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html)
already demonstrates that rapid erasure and targeted delta updates are
complementary. Any Exp42 contribution must therefore come from a falsifiable
mapping between uncertainty source and action, not from gating alone.

## Entry gate

Exp42 is eligible only if Exp41 demonstrates all of the following without
truth access:

- stable held-out gain beyond both `h_plus_total_variance` and tied-Q/R;
- correct discrimination of both matched-Q/R pairs;
- no single-cell or late-window-only explanation of the aggregate effect;
- complete paired coverage, frozen replay, and no leakage finding.

If any condition fails, Exp42 remains unexecuted and the active model is
contracted according to the Exp41 stop rule.

## Factorized inference panel

Subject to the prospective freeze, the inference panel should include:

- R-only and Q-only adaptive filters;
- `h + total variance` and tied-Q/R controls;
- frozen Exp39 online-EM coordinates;
- Exp41 autocovariance coordinates;
- seen-mode IMM;
- a causal BOCPD or particle approximation with fit-only hyperparameters;
- a GRU ceiling trained only on the training split and clearly separated from
  the local/no-online-gradient method;
- factorial and dynamic oracles reported only as headroom.

All trainable comparators must receive the same training blocks and selection
opportunities. Oracle information, larger mode dictionaries, and gradient
training cost must be disclosed rather than hidden inside a method label.

## Factorized actuator panel

The actuator audit separates actions that a single retention scalar previously
conflated:

\[
m_t=(1-r_t)\left[(1-\alpha_t)m_{t-1}+\alpha_t q_t\right],
\]

where \(\alpha_t\) controls write strength and \(r_t\) controls release. A
precision/uncertainty state controls predictive gain separately. Candidate
actions are:

- fixed write/retain/release;
- direct adaptive write only;
- direct release/reset only;
- joint write and release;
- Q-directed process adaptation;
- R-directed sensory precision adaptation;
- the complete factorized action.

The Exp39 continuation/reset mixture already supplies a fast-reset mechanism;
Exp42's contribution, if any, is the matched decomposition and causal audit of
which inferred coordinate should drive which action.

## Causal interventions

Each intervention shares the same tape and base inference trace:

- clamp one inferred coordinate to its fit-split reference;
- delay a coordinate or action by registered lags;
- shuffle within an allowed causal block partition using a frozen permutation;
- force, suppress, or time-shift release around registered transitions;
- swap Q and R coordinates after normalizing each within its fit-derived log
  range;
- supply oracle inference to the deployable actuator;
- supply deployable inference to an oracle-matched actuator.

The last two interventions distinguish an inference ceiling from an actuator
ceiling. Raw Q/R swapping without fit-derived scale normalization is not a
valid selectivity test.

## Functional budget matching

Comparisons report, and where feasible match, all of the following:

- number of adaptive scalar updates;
- cumulative L1 and L2 coordinate movement;
- reset/release mass;
- effective write mass or target effective sample size;
- number of maintained modes/states;
- online multiply-add and persistent-state counts.

Budget matching cannot make privileged truth or BPTT equivalent to local
execution, so those differences remain explicit ceiling annotations.

## Required endpoints

Primary endpoints use seed as the independent unit:

- held-out predictive NLL and latent-state MSE;
- early transition windows at 1, 4, 8, and 16 steps;
- late-regime NLL;
- matched-pair and per-cell non-inferiority;
- gain over the strongest reduced-factor baseline under the functional budget.

Secondary diagnostics include parameter/cross-loading recovery on an
orthogonal panel, switch latency, false release rate, target ESS, saturation,
interference, and oracle headroom closed. Aggregate support is invalid if it is
driven only by high-R cells, only by late adaptation, or only by calibration
without latent-state improvement.

## Evidence and execution boundary

Exp42 has no allocated formal seeds, no frozen MCIDs, and no authorized formal
run. Before execution it requires its own protocol-only commit/tag, data and
code hashes, complete artifact schema, replay validator, multiple-comparison
family, failure-retention rule, and support/oppose/inconclusive decision table.

Only a successful Exp42 formal result may motivate a new outcome-blind IBL
behavioral cohort. Neural activity, participating E/I execution, and biological
mapping remain separate later contracts and cannot be inferred from a
synthetic actuator result.

Before IBL neural analysis, the preferred external behavioral bridge is the
public volatility/stochasticity task of Piray and Daw, whose data and code are
available from [Zenodo](https://doi.org/10.5281/zenodo.13840905). Participants,
not trials, are the statistical unit; preprocessing and model selection must
remain inside participant-level training folds.
