# Actuator Matching Principle: critical audit and scale decision

## Executive conclusion

The most defensible current hypothesis is not that low-dimensional feedback
creates a low-rank physical recurrent matrix. It is that a fixed high-rank
carrier can expose a small dictionary of functionally distinct actuators, while
a low-dimensional belief/reward controller selects the actuator appropriate to
the current demand.

The repository now contains a coherent synthetic chain for this narrower
claim, plus explicit negative and unresolved results. It still does **not**
establish a participating high-rank E/I carrier, a thalamic biological identity,
or an advantage on multi-session neural activity.

| proposition | evidence | conclusion |
| --- | --- | --- |
| Task demand determines which actuator family is useful | Exp24/26, 30 seeds | support |
| Executed scalar reward can learn a reusable two-actuator selector in reset blocks | Exp31, 30 seeds | support |
| The same controller persists through a continuous hidden HMM without reset in a slow-switch regime | Exp32 main layer, 30 independent seeds | support |
| Feedback-per-dwell plus controller/environment timescale mismatch explains the registered phase structure | Exp32 iso-lambda layer | inconclusive |
| The Exp23 multiplicative total-drive axis and state-displacement-matched local rule improve behavior | Exp23 formal-v2 and fixed probe | oppose |
| Shared gated dynamics beat common dynamics on canonical multi-animal neural data | Exp25 | inconclusive; canonical neural bundle absent |
| The controller acts through a participating stable Dale E/I carrier | not tested by Exp31/32 | inconclusive |

## What changed after probing failures

### Exp23: a bounded negative result, not a universal rejection

The immutable 30-seed archive was reanalysed without rerunning or selecting
seeds. In the delayed task, local balanced-accuracy gain over frozen was
`-0.01133` under the matched state-displacement budget but only `+0.00056` at
the learned axis's natural scale. Matching amplified that local direction by a
median `83.4x`. Natural-scale delayed BPTT achieved `+0.01796` with a 95% seed
bootstrap CI `[+0.01162,+0.02444]`; exact forward sensitivity achieved
`+0.00194` with CI `[+0.00056,+0.00338]`.

This supports two criticisms. First, the selected axis has limited behavioral
headroom under the tested optimizer and protocol. Second, matching one scalar
state-displacement target can greatly amplify a weak direction and is not a
neutral proxy for functional plasticity. The registered `oppose` conclusion
therefore applies only to this drive-gain axis, rule and budget construction.

### Exp32 v1: retain the failure, then ask a different question

The original primary cell (`hazard=.05`, `feedback=.125`, `delay=4`) produced
only `+0.00352` local-minus-fixed accuracy and 3/5 positive seeds. It failed the
scale gate; its formal config remains unauthorized. The complete development
grid suggested that observability and memory timescale, rather than a scalar
feedback threshold, were the informative axes. That observation was used only
to freeze a new protocol and disjoint seeds `32000--32029`; the controller was
not retuned.

### Exp32 v2: controller support, phase structure unresolved

All 30 seeds and 10,800 rows completed from clean commit `49aaaf3`.

- Local minus train-fixed: `+0.04349`, CI `[+0.03446,+0.05289]`, 28/30 positive.
- Local minus opposite-action eligibility: `+0.08241`, CI `[+0.07667,+0.08779]`.
- Evidence response: `+0.01011` accuracy per doubling, CI `[+0.00962,+0.01062]`.
- Oracle opportunity retained: `0.297`.

Those preregistered components pass their MCIDs and Holm family, so the main
controller layer is **support**. However, the iso-lambda slow-minus-fast
contrast is only `+0.01195`, CI `[+0.00320,+0.02089]`, below its `0.02` MCID;
its one-sided threshold test gives `p=.955`. The timescale-structure layer and
the joint phase claim remain **inconclusive**.

## Relation to recent work

The revised architecture is consistent with evidence that distributed neural
dynamics can reuse shared computational motifs across tasks rather than learn
a new full circuit each time ([Driscoll et al., Nature Neuroscience 2024](https://www.nature.com/articles/s41593-024-01668-6)).
It also respects the observation that a low-dimensional latent circuit need not
imply a low-rank full connectivity matrix ([Langdon and Engel, Nature
Neuroscience 2025](https://www.nature.com/articles/s41593-025-01869-7)).

The controller interpretation is biologically suggestive, not identified:
PFC--MD models support context inference and gain-like control
([Zheng et al., Nature Communications 2024](https://www.nature.com/articles/s41467-024-52289-3));
ACC prediction-error signals provide a plausible scalar teaching variable
([Cole et al., Nature Communications 2024](https://www.nature.com/articles/s41467-024-51368-9));
and mediodorsal thalamus has been linked to uncertainty-dependent PFC control
([Zhang et al., Nature Communications 2025](https://www.nature.com/articles/s41467-025-58011-1)).
None of these papers makes the implemented two-value selector a literal MD or
ACC model.

Small recurrent networks can expose interpretable low-dimensional algorithms
([Nature 2025 tiny-RNN study](https://www.nature.com/articles/s41586-025-09142-4)),
while recent primate compositional-task data provide a direct future test of
shared task structure ([Tafazoli et al., Nature 2025](https://www.nature.com/articles/s41586-025-09805-2)).
The present synthetic result is a mechanism audit before that real-data test,
not a substitute for it.

## Remaining methodological weaknesses

1. Exp32 selects between only two hand-fixed actuators; it does not learn a
   reusable actuator dictionary.
2. The opposite-eligibility intervention is not update-budget matched; its
   mean reward-update L1 ratio is `1.215`. It supports credit-location
   specificity, not a matched-plasticity-efficiency claim.
3. The local internal state contains two action values. Only the action-control
   coordinate is one-dimensional, and reported context scores are policy
   proxies rather than calibrated hidden-state posteriors.
4. The Bayesian comparator uses training state labels and both training
   potential outcomes; it is a supervised upper comparator, not a deployable
   local learner.
5. Exp31/32 do not contain a participating E/I carrier. Fixed actuator outputs
   are evaluated behaviorally, so stability, closure and normal contraction
   remain untested in this chain.
6. Exp25 fails closed because the reviewed canonical multi-session neural
   bundle is absent. Existing IBL behavior results support hidden prior
   inference, not shared neural dynamics.

## Highest-value next scale gate

The next experiment should be one controlled bridge, not a broad benchmark
expansion:

1. Insert the frozen routing/associative/internal-dynamics motifs into one
   stable, high-rank Dale E/I carrier.
2. Let the unchanged reward-only controller modulate only the registered
   low-dimensional actuator coordinates; include frozen, no-gate and
   opposite-eligibility controls on paired tapes.
3. Match credit interventions by cumulative L1 and L2 in a separate registered
   panel, ideally with three actuators so a binary complement cannot mimic
   correct credit.
4. Require held-out behavior, low reduced-dynamics closure error, decay of
   normal perturbations, stable Jacobian margin and bounded saturation before
   increasing network size.
5. Only after that gate passes, evaluate the shared-basis belief-gated model on
   the canonical primate compositional data and multi-animal IBL neural
   sessions. Session/animal remains the replicate; all preprocessing and latent
   dimension selection stay inside training folds.

Scaling neuron count alone is not informative. The next claim must be earned by
the carrier participating in computation and by improving held-out behavior or
prediction, not merely by producing a low matrix rank.
