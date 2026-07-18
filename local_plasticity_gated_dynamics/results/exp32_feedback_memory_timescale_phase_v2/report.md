# Exp32 persistent sparse-feedback reward belief

- Profile: `formal`; seeds: 30.
- Checkout policy: `live`; current-checkout reproducibility claimed: `True`.
- Scale decision: **formal-complete**.
- Claim classification: **inconclusive**.
- Primary local minus train-fixed: +0.0435.
- Local minus opposite eligibility: +0.0824.
- Local minus no-forgetting: +0.0435.
- Oracle opportunity retained: 0.297.
- Main controller claim: **support**.
- Timescale-structure claim: **inconclusive**.
- Evidence-response slope: +0.0101 accuracy / doubling.
- Iso-lambda slow-minus-fast effect: +0.0119.
- Short-minus-long delay effect: +0.0332.
- Opposite/executed reward-update L1 ratio: 1.215.

This independent confirmation was frozen after the original
primary smoke cell failed. It tests a two-timescale phase claim
without retuning the local controller.

The primary cell is a continuous HMM stream with no reset,
hazard 0.01, reward fraction 0.125, and delay 4 trials.  The local method receives
only executed scalar reward. It has two internal action values and
one action-control coordinate; its context scores are policy proxies.
The Bayesian comparator knows the registered hazard and uses supervised
train-state emissions, but not the test state. The opposite-credit
condition is an eligibility-location intervention, not a random shuffle.
The true-state oracle is
not deployable.  This experiment contains no participating E/I
carrier and makes no real-data or biological-identity claim.
