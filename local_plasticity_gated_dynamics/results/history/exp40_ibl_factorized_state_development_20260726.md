# Exp40 IBL factorized-state development audit

Status: **historical post-hoc development evidence**

Archived on: 2026-07-26

Confirmatory cohort: **not frozen and not opened**

Neural stage: **locked**

## Why this audit was run

Exp38 showed stable evidence accumulation but failed to learn useful release at
visible Stream-51 boundaries. Exp39 then supported a narrower synthetic claim:
separate causal states for jump hazard, process variance, and observation
variance generalized across unseen uncertainty compositions. It did not recover
all generating parameters and was worse than the seen-mode IMM immediately
after switches.

IBL supplies a genuinely hidden block prior, but it does not independently
manipulate a time-varying sensory-noise parameter. Porting the synthetic
\((h,Q,R)\) labels literally would therefore be non-identifiable. Exp40 instead
tested a task-identifiable state:

\[
z_t=(\text{prior log odds},\;P(\text{recent change}\mid y_{<t}),\;
\text{run-length concentration}).
\]

The semi-Markov observer consumes only past binary stimulus sides. It assumes
the public task structure (90-trial unbiased burn-in, 0.2/0.8 emissions, and
bounded block durations) but never accepts per-trial `probabilityLeft`. The
release coordinate is a lagged BOCPD posterior over a five-observation recent
change window, not a true switch flag or a fixed hazard.

This decomposition is motivated but not made novel by prior work. The IBL
task uses uncued 0.2/0.8 blocks and recent large-scale analysis finds subjective
priors in both behavior and distributed neural activity
([International Brain Laboratory, *Nature*, 2025](https://www.nature.com/articles/s41586-025-09226-1)).
Volatility and observation stochasticity have distinct learning-rate effects
([Piray and Daw, *Nature Communications*, 2024](https://www.nature.com/articles/s41467-024-53459-z)),
and gated decay plus directional delta updates are established sequence-model
components
([Gated DeltaNet, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf)).
Exp40's eligible question is therefore empirical downstream utility, not the
novelty of a low-dimensional gate or a run-length filter.

## Data and leakage boundary

- Cohort: the existing Exp11 30-session/30-animal BWM behavior cohort.
- Evidence tier: outcome-exposed development only; no confirmatory claim is
  allowed.
- Split: chronological whole blocks into train/dev/test.
- Gate fit: training-prefix stimulus sides only.
- Readout: scaler and logistic model fit on train+dev only; regularization
  selected on dev; test is score-only.
- Primary endpoint: low-contrast (absolute contrast at most 0.0625) held-out
  choice NLL.
- Independent unit: animal. No neuron, trial, or time-bin is treated as a
  replicate.
- Baselines: history-only, learned-HMM mean, and semi-Markov mean. The primary
  reference is selected on dev without test access.
- Diagnostics: release-only, precision-only, both states, and an explicit
  truth-context readout. Truth context is never eligible as a gate.

All 210 planned animal-condition cells were retained. Twenty-seven animals
formed the complete endpoint panel. Three animals failed all seven conditions
symmetrically because their chronological test folds contained fewer than
eight low-contrast choices. This is an endpoint-eligibility failure, not a
condition-selective deletion.

## Registered development result

| Condition | Readout parameters | Mean low-contrast choice NLL | Mean context NLL |
|---|---:|---:|---:|
| History only | 7 | 0.483938 | 0.693147 |
| Learned HMM mean | 8 | 0.482630 | 0.307239 |
| Semi-Markov mean | 8 | 0.488656 | 0.235848 |
| Semi-Markov + release | 10 | 0.490924 | 0.235848 |
| Semi-Markov + precision | 10 | 0.495793 | 0.235848 |
| Factorized three-state | 12 | 0.496734 | 0.235848 |
| Truth-context diagnostic | 8 | 0.506526 | 0.000000 |

The task-structured observer improved context NLL over the learned HMM by
0.071391 nats/trial (95% animal bootstrap [0.049444, 0.095949], Holm
\(p=0.000364\)). This is **support** for structured block decoding only.

Dev-selected baseline minus factorized-state NLL was -0.010723
[-0.022459, 0.001010], positive in 9/27 animals. Any positive utility is
**inconclusive**. The 0.005 nats/trial meaningful-utility claim is **oppose**
(Holm \(p=0.0452\)).

Clamping release caused 0.001786 [-0.004956, 0.009137] NLL harm and is
**inconclusive**. Clamping run-length precision changed NLL by -0.005114
[-0.009175, -0.001242], so the registered positive contribution is
**oppose** (Holm \(p=0.00554\)). Better block-label decoding did not translate
into better prediction of the animal's choices.

## Bounded assay probe

After the registered development result was inspected, one post-outcome probe
selected regularization on all dev trials rather than the smaller low-contrast
subset and extended the strong-regularization grid. It did not alter the
observer, states, test fold, endpoint, or baseline family.

The factorized mean NLL improved from 0.496734 to 0.475206, showing that the
first readout selection had material variance. Nevertheless, its gain over the
dev-selected baseline remained -0.003617 nats/trial and was positive in only
12/27 animals. This probe cannot be selected in place of the registered result
and does not unlock confirmation.

## What failed and what remains valid

Valid retained insight:

- public duration structure improves causal recovery of the experimenter's
  hidden block;
- decoding accuracy and downstream utility are distinct endpoints;
- the implementation now exposes a causal recent-change posterior rather than
  relabeling a fixed hazard as release;
- every normalization and model-selection step is fold-restricted.

Rejected or unsupported interpretation:

- context NLL improvement is not evidence for an effective actuator;
- release and precision did not add held-out behavioral utility;
- the experiment does not identify a time-varying sensory-noise state;
- true block context is not a behavioral upper bound because animals act on a
  subjective belief that can lag the experimenter's state;
- neural decoding cannot rescue a failed behavioral-utility gate.

## Stop decision and artifacts

The disjoint 95-animal candidate pool identified outcome-blind from the current
IBL release was not frozen or opened. The development gate failed before a new
cohort was justified. A future confirmation must use a task that independently
manipulates environmental volatility and observation noise and must first pass
held-out utility before neural analysis.

- Main report: `results/exp40_ibl_state_utility_report.md`
- Claim table: `results/exp40_ibl_state_utility_claims.csv`
- Animal effects including all three failures:
  `results/exp40_ibl_state_utility_animal_effects.csv`
- Figure: `results/exp40_ibl_state_utility.pdf`
- Immutable run hashes: `results/exp40_ibl_state_utility_receipt.json`
- Raw runs on the 210 workspace are named in the receipt; both 210-row metric
  files, configs, planned grids, logs, raw test predictions, and failures remain
  intact.
