# Exp43 fast/slow causal decomposition development protocol

Status: **prospectively specified, development only**. This protocol does not
authorize a formal claim, Exp39/41 retuning, access to reserved formal seeds,
or promotion of the locked Exp42 plan.

## Provenance and relation to Exp42

Exp42 was a conditional actuator-factorization plan whose entry gate required
Exp41 to beat total variance and online EM. Exp41 failed that gate, so Exp42 is
retained as an unexecuted historical plan. Exp43 asks a materially narrower
question on new tapes: which inference-to-action path limits the already
implemented Exp39 actuator?

## Primary question

With the state-update algebra held fixed, does replacing either of the two
inference paths by truth expose usable held-out headroom?

The two axes are:

- event path: learned same-observation jump posterior versus the true jump
  event supplied only to the post-observation release action;
- uncertainty path: learned online Q/R state versus generating Q/R supplied to
  the predictive gain as a privileged ceiling.

The four paired arms are:

| Arm | Release action | Q/R gain |
|---|---|---|
| `learned_event_learned_qr` | learned jump posterior | local online EM |
| `oracle_event_learned_qr` | true event after observing the current sample | local online EM |
| `learned_event_oracle_qr` | learned jump posterior | generating Q/R |
| `oracle_event_oracle_qr` | true event after observing the current sample | generating Q/R |

Oracle arms are headroom diagnostics and can never be called deployable.
The event oracle cannot improve the predictive score of the event sample
itself; it can only change the filtered state and future predictions.

## New data contract

- Development seeds: `43000--43007`.
- Reserved formal seeds: `43100--43129`; they must remain untouched in this
  study unless a later protocol-only commit explicitly authorizes them.
- Fit and test tapes use different RNG namespaces and digests.
- Fit cells are isolated factors: `000`, `100`, `010`, `001`.
- Test cells are unseen joint factors: `110`, `101`, `011`, `111`.
- Block lengths vary causally within each sequence and are balanced over the
  configured length set; no time point is randomly split.
- All methods receive the identical observations, sequence boundaries, and
  selected fit hyperparameters. Only explicitly labelled oracle arms receive
  generating variables.

The task remains Gaussian and well specified for this first localization
audit. Outliers, autocorrelated sensory noise, missing observations, and
continuous parameter levels are later stress tests, not hidden degrees of
freedom in this probe.

## Comparators

- current factorized online EM (`learned_event_learned_qr`);
- `h + total variance` / tied-Q/R reduced controller, selected on the same fit
  tape;
- seen-mode IMM using only the four fit cells;
- fixed jump filter selected on the same fit tape;
- the four 2x2 exchange arms;
- dynamic truth only as a labelled upper bound.

BOCPD/robust BOCPD and an online-gradient RTU/GRU ceiling are required for a
later formal study, not for this bounded mechanism-localization probe.

## Selection and causality

- Initial states and adaptation rates are selected using fit-tape predictive
  NLL only.
- Test observations, latent states, cells, jumps, and generating parameters
  are unavailable to selection.
- The learned controller has no true-context, block-boundary, future-sample,
  BPTT, autograd, or online-gradient input.
- A future edit to an observation, oracle trace, or regime label must not alter
  any earlier output.
- The locked Exp39 model and result files are not modified.

## Primary development endpoints

All summaries use seed as the independent unit and retain every failed seed.

1. overall held-out predictive NLL and latent-state MSE;
2. post-jump future-prediction windows of 1, 4, 8, and 16 samples;
3. post-Q/R-regime-switch windows of 1, 4, 8, and 16 samples;
4. late-regime NLL;
5. per-cell NLL, especially cell `110`;
6. jump posterior AUC, false-release rate, release mass, write gain, and Q/R
   clipping/saturation.

The event window starts at the next prediction after an event. This prevents
the post-observation oracle release from receiving credit for predicting the
event that revealed it.

## Development decision table

Thresholds localize opportunity; they do not create a confirmatory claim.

- **Actuator headroom present:** oracle-both improves learned-by-learned NLL by
  at least 0.03 nats and latent MSE in at least 7/8 seeds.
- **Event path promising:** oracle-event/learned-Q/R improves the cumulative
  1--8 post-jump NLL by at least 0.02 nats and latent MSE in at least 7/8 seeds.
- **Q/R path promising:** learned-event/oracle-Q/R improves overall or late
  NLL by at least 0.02 nats, improves latent MSE in at least 7/8 seeds, and is
  non-negative on cell `110` in at least 7/8 seeds.
- **Deployable advance gate:** learned-by-learned must beat both total variance
  and seen-mode IMM overall, improve latent MSE, avoid a negative test cell,
  and not lose in every registered early window.

If the actuator-headroom gate fails, stop this synthetic actuator lineage. If
only one inference path has headroom, contract the model and develop only that
path. If oracle paths help but the deployable gate fails, report failure
localization and do not access the reserved seeds.

## Required artifacts

The runner must save the resolved config, code and tape digests, environment,
per-seed metadata/status, fit selections, raw block/event metrics, paired seed
metrics, failures, summary, report, and a manifest. Validation must replay the
method panel and reject missing arms, digest mismatches, future leakage, or a
formal-claim flag.
