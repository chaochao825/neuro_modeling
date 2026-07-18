# Exp32 v2: confirmatory feedback-memory-timescale phase diagram

## Why this is a new analysis protocol

The preregistered Exp32 v1 smoke primary (hazard `0.05`, feedback `1/8`,
delay `4`) failed its scale gate: the five-seed mean local-minus-fixed gain
was only about `0.0035`. That formal configuration remains fail-closed and the
failed result is retained.

The complete registered v1 stress grid showed a specific development result.
At hazard `0.01`, the unchanged local controller improved over fixed in every
feedback/delay cell, whereas at hazard `0.05` the local and known-hazard
Bayesian controllers both had little or no advantage in the original primary
cell. This suggested a two-timescale boundary rather than a scalar threshold:

- `lambda = feedback_fraction / hazard` is the expected number of observable
  rewards per hidden-state dwell;
- `chi = hazard * tau_q`, where `tau_q = -1/log(0.98)`, compares environment
  switching with the controller's fixed memory timescale;
- `kappa = (1 - 2*hazard)^(delay + 1)` records evidence staleness.

This v2 protocol was frozen after that development result and before the new
seeds `32000--32029`. It keeps the actuator dictionary, controller,
`alpha=0.30`, `retention=0.98`, `temperature=0.08`, and reward-only access
contract fixed. It expands the task to 4096-trial streams, hazards
`0.01, 0.02, 0.04`, feedback fractions `1/32, 1/16, 1/8, 1/4, 1/2`, and
delays `0, 4, 16` to identify the two timescales. No v1 seed is used
inferentially. The transition receipt cryptographically binds the failed v1
panel and this independent formal panel.

## Confirmatory cells and contrasts

The slow-switch replication cell is hazard `0.01`, feedback `1/8`, delay `4`,
giving `12.5` expected observable rewards per dwell. The seed is the
independent unit. Three one-sided contrasts form one Holm family:

1. `persistent_rpe_local - train_fixed_best`, MCID `0.02`;
2. local minus the **opposite-action eligibility intervention** (legacy ID
   `credit_shuffled_local`), MCID `0.02`;
3. within-seed coefficient of `log2(lambda)` in the frozen response model,
   MCID `0.005` accuracy per evidence doubling.

Support requires all three 95% whole-seed bootstrap lower bounds to exceed
their MCIDs, all three paired sign-flip tests to pass Holm correction at
`0.05`, at least 24/30 positive primary effects, at least 25% of the
slow-switch oracle opportunity retained, all 30 seeds, and every
access/pairing/provenance audit.

The response model is fitted independently within each seed over the entire
grid:

`gain = beta0 + beta_lambda log2(lambda) + beta_chi log2(chi) +`
`beta_delay[-log2(kappa)] + beta_interaction log2(lambda)log2(chi)`.

A separate structural claim tests whether `lambda` alone is insufficient. At
delay 4, slow-minus-fast local gain is averaged on two exact iso-lambda lines:

- `lambda=6.25`: `(h=.01,f=.0625)` versus `(h=.04,f=.25)`;
- `lambda=12.5`: `(h=.01,f=.125)` versus `(h=.04,f=.5)`.

Its whole-seed CI lower bound must exceed `0.02`, with a one-sided paired
sign-flip `p<0.05`. A secondary delay probe averages delay-0 minus delay-16
effects at hazard `.02` and feedback `.25,.5`; it is not an extra pass route.

All hazards, feedback fractions, delays, seeds, and failures remain in the
output. The hazard-`0.05` v1 failure remains a development result and is not
relabelled as confirmatory evidence.

Formal provenance hashes the experiment/config helpers, task and actuator
implementations, local controller, artifact/seed utilities, every metric and
statistical helper, summarizer, and figure script. A live formal summary must
match the execution commit and Git tree. Historical reanalysis requires an
explicit `--archived-reanalysis` opt-in: it still verifies the frozen config,
authorization receipt, and all critical-file hashes, but is labelled as an
archive operation and cannot claim that the analyst's current checkout is the
execution checkout. The configured Holm family must exactly equal the three
contrasts above, including order and names.

## Claim boundary and audit semantics

The local selector has two internal action-value states and a one-dimensional
action-control coordinate; its reported NLL/Brier values are policy proxies,
not calibrated context posteriors. Delayed RPE is evaluated against the
current value at delivery time. The opposite-eligibility intervention is not
a random shuffle and its reward-update norm is reported rather than claimed
to be matched. The Bayesian comparator uses train state labels and both train
potential outcomes, so it is a supervised upper comparator.

Passing this protocol would support a bounded result: a reward-only local
controller can select between fixed routing and associative actuators in an
observable slow-switch regime, with performance jointly governed by evidence
availability and controller/environment timescale mismatch. It would not
establish a participating E/I carrier, low-rank physical connectivity, a
thalamic identity, or a real-neural-data advantage.
