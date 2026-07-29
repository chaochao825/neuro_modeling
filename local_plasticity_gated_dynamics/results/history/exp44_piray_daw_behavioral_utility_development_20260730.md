# Exp44 Piray--Daw behavioral-utility development audit

## Registered conclusion

The prospective Experiment-1 development gate is **opposed**. All 223 eligible
participants completed the five-fold participant-held analysis, but the
factorized local controller did not reliably improve human bucket-update
prediction over fixed gain or total uncertainty and did not express the
registered volatility direction. Experiment 2 and POPGym remain unexecuted.

The frozen execution used commit
`42af52bb2d6769f41fe9d6f9a4825a0d60ecd8a0` and tag
`exp44-dev-v1-preoutcome-r2-20260730`. Its manifest SHA-256 is
`b30e193c9cfc0beb54e040a99781e5eb9dd723dce514af66b73ea9a5bc12a440`.
The clean regression contained 1,416 passing tests. The original pre-outcome
tag and its import-stage failure are separately retained in
`exp44_piray_daw_entrypoint_failure_20260730.md`; no data or metric was read in
that failed launch. Its raw `launch.log` and `exit_status.txt` are preserved in
`results/history/exp44_piray_daw_entrypoint_failure_20260730/` with SHA-256
`e27baa0e395543fa28d008fff4dd50779b95368748aa3f2458c167ac2e4b103d` and
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`.

## Primary held-out results

All gains below are comparator minus factorized, so positive values favor the
factorized controller.

| Endpoint | Comparator | Mean gain | 95% participant bootstrap CI | Holm p | Registered verdict |
|---|---|---:|---:|---:|---|
| update NLL | fixed gain | +0.001044 | [-0.006093, +0.010351] | 1.000000 | oppose |
| update NLL | total uncertainty | +0.006105 | [-0.002445, +0.017697] | 0.974001 | oppose |
| update MSE | fixed gain | +0.019918 | [-0.385452, +0.522493] | 1.000000 | oppose |
| update MSE | total uncertainty | +0.324224 | [-0.153438, +0.951971] | 0.974001 | oppose |

Cellwise non-inferiority versus total uncertainty passed, but this cannot
rescue the conjunction. The executed gain's high-minus-low Q effect was
`-0.000702`, opposite to the required direction; its low-minus-high R effect
was only `+0.001130`. The seven registered clauses therefore passed only
cellwise non-inferiority and the non-applicable particle-retention clause.

The selected local-EM candidate used identical initialization and adaptation
for both coordinates: `Q0=R0=8`, `beta_Q=beta_R=0.02`. Because its two update
equations are symmetric under those values, its Q and R traces remained
identical. Its mean gain was nearly constant across the four cells:
`0.619700, 0.618731, 0.618303, 0.617867`. Thus model selection collapsed the
nominally factorized controller to an effectively one-coordinate solution.

## Strong-comparator interpretation

The negative result is not caused by a lack of Q/R structure in the human
task. The autocovariance and hierarchical-particle traces expressed much
larger condition-dependent gain changes, but both predicted held-out human
updates worse than the selected local controller. Factorized minus
autocovariance improvement was `+0.023895` NLL and `+1.531729` MSE; factorized
minus particle improvement was `+0.007164` NLL and `+0.463559` MSE.

The true-Q/R Kalman ceiling tracked the hidden bird better in aggregate, yet
predicted human updating substantially worse: factorized improved NLL by
`+0.093632` and MSE by `+6.381001` relative to that privileged model. This is a
useful boundary: statistically normative environmental tracking is not the
same endpoint as predicting human control behavior.

## Outcome-exposed localization

The following calculation was performed only after the registered gate was
read and is descriptive. Replaying the source paper's per-participant,
per-block no-intercept regression of bucket update on prediction error gives
mean empirical gains

`[0.683002, 0.730009, 0.632599, 0.670371]`

for cells `(Q,R)=(4,16),(49,16),(4,64),(49,64)`. With participant bootstrap,
the empirical high-minus-low Q effect is `+0.042389` with interval
`[+0.026124,+0.058880]`; the empirical low-minus-high R effect is `+0.055021`
with interval `[+0.036552,+0.073335]`. These reproduce the known behavioral
direction and localize the failure to the deployed inference/calibration path,
not to absence of a behavioral signal. They are not a rescue analysis and do
not unlock Experiment 2.

## Evidence decision

Exp44 closes the first real-behavior gate for the current module:

\[
\text{statistical Q/R information: supported}
\quad\not\Rightarrow\quad
\text{current causal controller has held-out behavioral utility}.
\]

Consequently:

1. do not tune the exposed Experiment-1 result or reinterpret a positive point
   estimate with a crossing interval as support;
2. do not run Experiment 2 under the Exp44 contract;
3. do not start the conditional POPGym scale study, neural analysis, or E/I
   carrier stage;
4. if the question is revisited, use a separately named prospective contract
   that distinguishes environment inference from causal participant-specific
   response calibration using only past actions. It must earn utility on an
   untouched cohort before any control-scale claim.

Canonical raw evidence is in
`results/exp44_piray_daw_qr_behavior_development_v1/`. The independent
post-outcome validator replays artifact hashes, participant-level contrasts,
bootstrap intervals, Holm adjustment, every gate clause, and both downstream
locks without importing the experiment decision function. Its explicitly
post-outcome `posthoc_validation.json` receipt has SHA-256
`c19ddb1e89a60be3aa8c9c2c9716b4b79040dc08b8242ca760caefb8e4c93896`;
it is intentionally not retrofitted into the frozen run manifest.
