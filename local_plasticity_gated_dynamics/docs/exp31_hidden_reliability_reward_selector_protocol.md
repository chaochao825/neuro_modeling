# Exp31 hidden-reliability reward-only selector protocol

## Why this experiment replaces an immediate Exp30 scale

Exp30 is retained as a positive-control wiring test. Its target explicitly
mixes direct and associative components, its one-hot memory retrieves exactly,
its matched rule reads the demand coordinate, and each demand/mode receives a
separate train-only output scale. More Exp30 seeds would narrow uncertainty
around a constructed crossover without removing those structural shortcuts.

Exp31 asks a narrower and more falsifiable question: can a controller choose
between two fixed motifs when the useful motif is hidden and the controller
sees only rewards from actions it actually executed?

## Task and actuator dictionary

Each trial stores dense bipolar unit keys and random scalar values. The target
is the value associated with the query key. It is never an actuator mixture.

- Input routing receives a noisy copy of the target. Its correctness
  probability is a hidden block state, fixed within a block.
- Associative memory performs fixed local outer-product writes and dense
  content-addressed retrieval. Cross-talk grows with binding load and
  distractor writes.
- Query-shuffled memory uses exactly the same writes and final memory state but
  cyclically shifts the query tape across trials. Thus write L1/L2 and state
  norm are exactly identical while trial-specific correspondence is broken.

There is no demand-specific gain fit. Every prediction has magnitude one. A
high-rank E/I carrier is deliberately absent from this controller
identifiability panel; consequently Exp31 cannot support a carrier-dynamics
claim.

## Reward-only local rule

The probe prefix contains an exactly balanced randomized action schedule. On
trial \(t\), the selector receives only the executed action \(a_t\) and its
binary scalar reward \(r_t\):

\[
Q_{a_t}\leftarrow Q_{a_t}+\frac{1}{n_{a_t}}(r_t-Q_{a_t}).
\]

This is action eligibility times a scalar reward-prediction error. The
selector API has no true-context, prospective-descriptor, unexecuted-reward,
or candidate-utility argument. After the probe prefix it freezes
\(\arg\max_a Q_a\) for the rest of the block. No autograd or BPTT is used.

## Controls and splits

- fixed routing;
- fixed associative memory;
- exact-write-budget query-shuffled memory;
- one global fixed actuator selected on training blocks only;
- a random block selector with the identical forced-probe schedule;
- reward-only local selection;
- a true-hidden-state oracle whose cell map is estimated on training blocks
  only and is labelled strictly as an upper bound.

Train and test contain disjoint whole blocks. All controls within a seed share
the target, cue flips, keys, values, queries, distractors, trial order, and
probe schedule. Trial/time points are never randomly reassigned.

## Preregistered scale gate

Smoke seeds are 9200--9204 and are disjoint from formal seeds 0--29. The
primary endpoint is seed-level full-block accuracy, including the exploration
prefix:

\[
\Delta=U_{\mathrm{reward\text{-}only}}-U_{\mathrm{train\text{-}fixed}}.
\]

Scaling is authorized only if all feedback-access and pairing audits pass,
four of five seeds have \(\Delta>0\), mean \(\Delta\ge 0.03\), both routing-
and memory-favorable cells occur, the reliability crossover is at least 0.10,
associative/query-shuffled specificity is at least 0.05, fixed performance is
neither floor nor ceiling, memory accuracy falls with preregistered
interference pressure, and the selector retains at least 25% of the oracle
opportunity. Failed conditions and failed seeds remain in the panel.

Formal inference uses exactly 30 seeds, seed-level paired bootstrap intervals,
paired random-sign tests, and Holm correction for the four registered claims.
No interim seed peeking, replacement, or early stopping is allowed.
