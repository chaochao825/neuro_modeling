# Exp30 associative-actuator trend protocol

## Purpose and status

Exp30 is an additive exploratory **positive-control sanity test** of the
associative-memory component of the Actuator Matching Principle. Its target is
constructed from the two mechanism-accessible components, so the crossover is
expected if the pipeline is wired correctly; it is not an independent learned
task decomposition. It does not modify the frozen Exp26/29 source,
selector, or evidence bundles. The five-seed smoke panel may be labelled only
`trend-positive` or `trend-not-established`; its formal claim classification is
always `inconclusive`.

## Paired task

Each trial writes several trial-specific one-hot key/scalar-value bindings,
passes through a distractor delay, and then presents a query key plus an
independent direct cue. For registered memory demand \(\mu\),

\[
y=\sqrt{1-\mu}\,d+\sqrt{\mu}\,M q+\epsilon,
\qquad
M=\sum_i v_i k_i^\top.
\]

The query contains no target value. The same key can retrieve either sign on
different trials. Train and test are disjoint whole blocks; time points are
never reassigned independently.

## Conditions

- `frozen`: no control current;
- `routing`: query-time direct cue only;
- `low_rank`: one-dimensional compressed history state;
- `associative`: local outer-product write and content-addressed read;
- `associative_shuffled`: values are cyclically permuted across keys, preserving
  write-update L1/L2 exactly while breaking correspondence;
- `fixed_best`: one single actuator selected from training macro utility;
- `matched`: routing for \(\mu<0.5\), associative memory for \(\mu>0.5\);
- `combined`: direct and associative components together, used as an upper
  capability reference.

All active single-actuator outputs receive a separate training-only gain for
each demand cell so that query-output RMS matches the target. This is not a
matched physical write, state, or energy budget across actuator families.
Associative and shuffled memory alone have exactly matched write-update L1/L2.
Every condition shares the high-rank Dale-compatible carrier, injection axis,
readout, trials, target noise, and actuator dictionary. The bridge is
identity-calibrated: carrier parameters are never updated and carrier dynamics
do not contribute task computation. No autograd or BPTT is used.

## Development trend gate

The smoke panel uses seeds 9100--9104 and \(\mu\in\{0,.25,.75,1\}\). A seed is
positive only if all of the following hold:

1. routing beats associative memory at the low-demand endpoint;
2. associative memory beats routing at the high-demand endpoint;
3. associative advantage has Spearman correlation greater than 0.8 with
   \(\mu\);
4. demand-matched macro score exceeds fixed-best macro score;
5. associative memory beats its update-budget-matched shuffled control at high
   demand.

At least four of five seeds are required for `trend-positive`. Passing this
gate permits scaling but remains formally `inconclusive`.

## Formal scale (registered but not yet run)

The formal config fixes seeds 0--29, a larger N=128 carrier, eight bindings,
longer delay, and six demand values. Before a formal run, the analysis package
must add seed bootstrap intervals, the registered Holm family, explicit
all-attempt inventory binding, and immutable source/result hashes. The narrow
positive-control statement can pass only if the crossover interaction,
matched-minus-fixed utility, and high-demand associative-minus-shuffled
contrast all have positive lower confidence bounds. Even then, the supported
statement is limited to: **given the explicit direct/retrieval decomposition
and oracle demand coordinate, matching a single fixed motif beats one fixed
family on this constructed panel**. It cannot support carrier dynamics, learned
task decomposition, or the full Principle. Strong neural sequence baselines
are not a gate for this mechanism-first stage.
