# Exp32: persistent belief control with sparse delayed reward

## Frozen smoke outcome (retained failure)

The independent smoke panel `9300--9304` completed every registered cell and
passed its access/pairing audit, but the preregistered primary did **not** pass
the scale gate. At hazard `0.05`, feedback `1/8`, delay `4`, mean
`persistent_rpe_local - train_fixed_best` was `+0.00352` with only 3/5 positive
seeds, below the `0.02` MCID and 4/5 sign requirement. The original formal
config is therefore explicitly fail-closed. All raw rows, logs, receipts,
summary tables, and figures are retained under
`results/exp32_persistent_sparse_feedback_smoke_v1_failed/`.

The full development grid motivated a new, independently seeded two-timescale
claim in `exp32_evidence_per_dwell_boundary_protocol.md`; it does not convert
this failed primary into support.

## Question and claim boundary

Exp32 asks whether a one-dimensional reward-prediction-error state can select
between the two **fixed** Exp31 actuators in a continuous hidden-state stream.
The selector is never reset at context changes and receives only the reward of
the executed actuator when that reward is scheduled to be observable.

This is a controller-identifiability experiment.  It does not contain a
participating high-rank E/I carrier, does not establish a thalamic biological
identity, and does not test real neural data.

## Frozen task and access contract

- The hidden demand is a two-state symmetric HMM with per-trial hazards
  `0.01, 0.05, 0.10, 0.20`.
- State 0 uses the high-reliability/high-memory-pressure Exp31 cell; state 1
  uses the low-reliability/low-memory-pressure cell.  The routing and dense
  associative motifs are immutable across every condition.
- Separate train and test potential-outcome tapes are generated.  The train
  tape alone estimates the fixed comparator and Bayesian emission table.
- Test feedback fractions are exactly `1/2, 1/4, 1/8, 1/16`, nested on a
  common randomized tape.  Delays are `0, 4, 16` trials.  All delay cells use
  the same eligible feedback positions, so observed-label count is matched.
- Local and Bayesian selectors receive `(executed_action, scalar_reward)`
  only.  They never receive hidden state, unexecuted reward, candidate utility,
  reliability, memory pressure, or switch times.  The true-state oracle is
  explicitly labelled and excluded from deployable comparisons.
- Full-stream behavioral accuracy includes exploration, adaptation and switch
  costs.  No time point is randomly assigned to a train/test fold.

## Registered methods

1. `train_fixed_best`: one train-selected actuator for the full test stream.
2. `matched_random`: random action with the same test potential outcomes.
3. `cumulative_sample_average`: executed-reward sample average without
   forgetting.
4. `persistent_rpe_local`: exponentially retained action values updated by
   local action eligibility times scalar reward prediction error.
5. `credit_shuffled_local`: identical local rule with reward credited to the
   opposite action.
6. `no_feedback_local`: same controller and action RNG with reward withheld.
7. `bayes_reward_filter`: train-estimated emissions and the registered hazard,
   but no true test state; delayed observations are incorporated by fixed-lag
   forward recomputation.
8. `oracle_hidden_state`: train-estimated winning actuator selected using the
   true test state; upper bound only.

The local rule uses fixed `alpha=0.30`, `retention=0.98`,
`temperature=0.08`, and prior action value `0.5`.  These values are frozen
before the five-seed smoke panel and are not retuned per hazard, feedback
fraction, delay, seed or test tape.

## Primary and negative tests

The primary cell is hazard `0.05`, feedback fraction `0.125`, delay `4`.
The independent unit is the seed and the endpoint is paired full-stream
accuracy.

Smoke scaling requires all five seeds and all access/pairing audits, plus:

- mean `persistent_rpe_local - train_fixed_best >= 0.02` in the primary cell;
- at least 4/5 positive seed effects;
- mean `persistent_rpe_local - credit_shuffled_local >= 0.02`;
- at least 20% of the train-state-oracle opportunity retained;
- both actuators win in their registered train state for every seed.

The one-shot formal panel uses 30 new seeds.  Support requires the 95% paired
whole-seed bootstrap lower bound to exceed `0.02` for the primary contrast and
`0.02` for credit specificity, and both one-sided paired sign-flip tests to
pass Holm correction at alpha `0.05`.  It also requires every access and
pairing audit.  If the primary upper bound is at most zero, the claim is
`oppose`; otherwise a failed conjunction is `inconclusive`.

The feedback-1/16 and hazard-0.20 cells are registered stress tests, not extra
ways to pass the primary claim.  Their failures must be retained and used to
locate the controller's operating boundary.

## Exp23 failure probe registered before reanalysis

The Exp23 raw formal-v2 archive is reanalysed without rerunning or selecting
seeds.  The probe reports, for both task variants:

- natural and matched-scale behavioral gain over frozen;
- natural-to-matched control-axis amplification;
- BPTT and exact-forward reachability relative to the registered `0.03` MCID;
- task-loss gain versus balanced-accuracy gain;
- seed-level Spearman association between update cosine and behavioral gain.

The probe is diagnostic only.  It may narrow the interpretation of the
existing `oppose` result but cannot convert Exp23 into support.
