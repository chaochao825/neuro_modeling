# Exp44 Piray--Daw Q/R behavioral-utility protocol

Status: **prospective development protocol; Experiment 1 only**. The public
paper's human effects are known, and the source schema and hashes were inspected
on 2026-07-30. No Local Plasticity project model comparison was run before this
protocol and its configuration were written. Experiment 2 remains execution
locked until the development decision and implementation hashes are archived.

## Question and claim boundary

Exp44 tests the missing second half of the current evidence chain:

\[
\text{Q/R information exists in temporal statistics}
\longrightarrow
\text{a causal low-dimensional controller predicts real trial updates}.
\]

The Piray--Daw task is a random-walk observation model,

\[
x_t=x_{t-1}+w_t,\quad w_t\sim\mathcal N(0,Q),\qquad
y_t=x_t+v_t,\quad v_t\sim\mathcal N(0,R).
\]

Volatility \(Q\) should increase the effective Kalman gain, while observation
noise \(R\) should decrease it:

\[
P_t^-=P_{t-1}+Q_t,\qquad
K_t=\frac{P_t^-}{P_t^-+R_t}.
\]

The participant places bucket \(b_t\) before observing bag \(y_t\). The causal
behavioral prediction is therefore

\[
\widehat{b_{t+1}-b_t}=K_t(y_t-b_t),
\]

where \(K_t\) may use bag history through trial \(t-1\), but no future bag,
future bucket, true factor label, hidden bird position, or test-participant
response. This previsible-gain convention is shared by every deployable method.

The task announces every new bird/weather block. Every method receives the
same block reset. Exp44 therefore **does not test hidden change-point
detection, switch latency, or a release gate**. BOCPD is not a primary
comparator here because its target event is observed rather than latent.

## Source and exposure audit

The only eligible source is Zenodo record
`10.5281/zenodo.13840905`, version v1.0, CC-BY-4.0. File sizes and MD5 values
are fixed in `provenance/piray_daw_zenodo_v1.json`; the loader must reject an
archive or extracted schema that does not match.

Schema inspection established:

- Experiment 1: 223 participants;
- Experiment 2: 420 participants;
- four blocks and 50 trials per block;
- participant arrays: bucket, response time, and randomized block order;
- shared task arrays: bag, hidden bird, true volatility, true stochasticity;
- the Experiment 1 and Experiment 2 bag and bird arrays are byte-identical.

The last fact changes the evidential scope. Experiment 2 is an independent
participant replication on the **same stimulus tape**. It may confirm
population behavioral generalization, but it cannot establish unseen-stream,
unseen-task, or control generalization. Those claims are reserved for a later
POPGym stage with independently generated episodes.

The upstream archive contains fitted MATLAB artifacts. Exp44 may inspect the
authors' equations and code but must not use `model_*.mat`, hidden bird
positions, or true Q/R labels to fit a deployable controller. The upstream
`hpl_fit.m` snapshot also refers to a participant `bird` field absent from the
released participant schema; Exp44 therefore reimplements the stated causal
particle equations against bags instead of replaying fitted upstream outputs.

## Methods

All methods see the identical centered bag prefix, block reset, participant
bucket at the conditioning trial, and trial mask.

1. `fixed_gain`: one constant \(K\), selected inside the training participants.
2. `total_uncertainty`: one locally updated variance coordinate with a fixed
   Q/R allocation, selected inside training participants.
3. `factorized_local_em`: separate bounded Q and R coordinates updated by
   local conditional second-moment targets. This is the registered primary
   method and uses no autograd, BPTT, future data, or true condition.
4. `autocovariance_qr`: the Exp41-style causal increment covariance estimator,
   retained as a diagnostic rather than silently replacing the primary method.
5. `hierarchical_particle`: a causal Rao--Blackwellized particle implementation
   of the published hierarchical model, using only bags. It is the strong
   probabilistic comparator; Monte Carlo seeds are averaged before participant
   statistics and are not treated as independent samples.
6. `oracle_qr`: true block Q/R supplied to the same Kalman actuator. This is a
   privileged ceiling only and can never support a deployable claim.

Grid membership is frozen in the JSON configuration. Each family receives its
own training-only selection; all candidates and failed evaluations remain in
the raw artifacts. No method is selected because it wins on Experiment 2.

## Splits and likelihood

Experiment 1 uses deterministic five-fold participant cross-validation.
Candidate selection and Gaussian response-scale estimation use only the four
training folds; the held-out fold supplies participant-level predictions.
Trials are never randomly divided. Because the bag tape is shared, this split
tests held-out people, not held-out environmental trajectories.

The primary endpoint is mean conditional-update Gaussian NLL per participant
over trials 1--49 of all four blocks. The Gaussian scale is fitted on training
residuals separately for each method and then frozen for the held-out fold.
Conditional-update MSE is co-primary for numerical interpretation; free-running
bucket MSE, hidden-bird MSE, bag predictive NLL, early/late NLL, cellwise NLL,
Q/R estimates, and gain traces are secondary or diagnostic.

True Q/R and hidden bird arrays enter only evaluation code. Directional gain
effects are computed after predictions are frozen:

- mean gain under high Q must exceed mean gain under low Q;
- mean gain under high R must be below mean gain under low R.

The participant is the only inferential unit. Trial, block, response, and
Monte Carlo particle are never independent replicates. Paired bootstrap
intervals and multiplicity-adjusted participant tests are reported.

## Development gate and stopping rule

The Experiment 1 gate is a conjunction. `factorized_local_em` must:

1. improve conditional-update NLL over both selected `fixed_gain` and
   `total_uncertainty` by at least 0.005 nats/trial on average;
2. have a participant-bootstrap lower confidence bound above zero for both
   comparisons;
3. show both registered Q/R gain directions;
4. have no one factorial cell below `total_uncertainty` by more than
   0.005 nats/trial;
5. retain at least 90% of the improvement supplied by the hierarchical
   particle comparator over fixed gain, when that comparator improves.

If any clause fails, Exp44 is archived as development evidence and Experiment
2 remains unexecuted. A successful gate authorizes only a new immutable
confirmation configuration and implementation receipt. It does not itself
upgrade a paper claim.

Experiment 2, if unlocked, tests the same conjunction on 420 new participants
with all parameters, response scales, code hashes, and decision thresholds
frozen from Experiment 1. It supports only same-tape cross-participant
behavioral utility.

## Conditional POPGym stage

POPGym is not part of Exp44 and must not run merely because a Q/R estimate is
decodable. It is unlocked only by a successful Experiment 2 behavioral gate.
Its role is different: new episodes and task instances test whether the module
improves partially observable control, online adaptation, sample efficiency,
and scaling beyond a fixed memory timescale. Comparators must include a strong
recurrent baseline and an online recurrent method such as RTU, with matched
parameter and interaction budgets. Neural, tracking/BCI/IBL, and participating
E/I carrier stages remain downstream of that result.

## Permitted conclusions

- `support`: causal factorized Q/R state adds held-out participant behavior
  utility beyond fixed and total uncertainty under the complete gate.
- `oppose`: a registered superiority or non-inferiority clause fails with
  adequate eligible participants.
- `inconclusive`: integrity, missing-data, numerical, or eligibility failure
  prevents the registered contrast.

Even a supported Exp44 cannot establish unseen-sequence control, neural
representation, local synaptic implementation, E/I participation, or low
physical rank.
