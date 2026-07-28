# Exp43 fast/slow causal decomposition decision

Status: **historical mixed development result; no formal advancement**.

Exp43 was specified before outcomes in
[`docs/exp43_fast_slow_causal_decomposition_protocol_20260728.md`](../../docs/exp43_fast_slow_causal_decomposition_protocol_20260728.md).
Its implementation was frozen at commit `24464da2bdd3785dd442832d18e4858b10084842`
with implementation digest
`68fc6cf4bd67444c5a2c3dc19d98e08191b051b71615fe8c7944cc48460132d1`.
All eight disclosed development seeds completed and no reserved formal seed
was accessed. The independent
[`exp43_development_replay_receipt_210_20260728.json`](../../provenance/exp43_development_replay_receipt_210_20260728.json)
passes source/artifact hashes, aggregate replay, and summary replay. The raw
result package and its own manifest are retained at
[`results/exp43_fast_slow_causal_decomposition_development_v1`](../exp43_fast_slow_causal_decomposition_development_v1/report.md).

## Decision

The 2x2 signal-exchange audit localizes useful headroom in both inference
paths while rejecting advancement of the deployable controller:

\[
\boxed{\text{the actuator can use correct signals, but learned event and Q/R
inference do not yet supply them reliably enough.}}
\]

Q/R inference is the larger aggregate bottleneck; event inference matters
specifically after jumps. This is a development-level causal localization,
not evidence that an oracle is deployable, not a formal upgrade of Exp39, and
not a reason to execute Exp42 or access seeds `43100--43129`.

## Registered gates

Positive values mean replacing the named learned path with privileged truth
improves NLL. Event truth is injected only after scoring the event sample, so
the registered event window begins at the next prediction.

| Gate | Registered effect | Positive seeds | MCID / criterion | Result |
|---|---:|---:|---:|---|
| actuator headroom, oracle both | +0.090067 NLL; +0.020102 latent MSE | 8/8; 8/8 | NLL >= 0.03 and MSE 7/8 | **pass** |
| event path, next 8 predictions | +0.031488 NLL; +0.005895 latent MSE | 8/8; 8/8 | NLL >= 0.02 and MSE 7/8 | **pass** |
| Q/R path, overall | +0.076979 NLL; +0.012827 latent MSE | 8/8; 8/8 | NLL >= 0.02 and MSE 7/8 | **pass** |
| deployable learned controller | +0.013021 NLL vs total variance | 6/8 | both baselines, MSE, every cell | **fail** |

The learned controller does outperform the generator-supported seen-mode IMM
on aggregate NLL by +0.066212 in 8/8 seeds, but its latent-MSE contrast is
positive in 7/8. Against the simpler total-variance controller, NLL is positive
in only 6/8 and latent MSE in only 5/8. The registered no-negative-cell
condition also fails. Aggregate improvement against one comparator therefore
cannot override the conjunction gate.

## Cell heterogeneity (post-hoc descriptive audit)

Positive values favor the learned event + learned Q/R controller. These cell
rows were not separate confirmatory claims.

| Cell | total variance minus learned NLL | seen IMM minus learned NLL | Interpretation |
|---|---:|---:|---|
| `011` | +0.026217 (6/8) | +0.058324 (6/8) | descriptive positive |
| `101` | +0.015482 (7/8) | +0.041943 (7/8) | descriptive positive |
| `110` | **-0.002957 (4/8)** | **-0.014193 (3/8)** | repeated failure |
| `111` | +0.021430 (7/8) | +0.164836 (8/8) | strongest descriptive positive |

Cell `110` is also worse in latent MSE: total variance minus learned is
-0.004366 in 0/8 positive seeds. Thus the aggregate result again cannot be
called uniform composition generalization.

The parameter audit explains the failure without rescuing it. In cell `110`,
true Q/R are 0.04/0.01, while length-weighted learned means are approximately
0.0196/0.0512. The controller therefore assigns too much variance to sensory
noise and too little to state change, reducing write gain in exactly the regime
where rapid tracking is required. Supplying oracle Q/R improves cell-`110` NLL
by +0.125209 in 8/8 seeds. This shows that the fixed actuator has usable
headroom; it does not validate the learned estimator.

## Event quality is not event utility

The learned detector has mean AUC 0.9106, mean recall 0.6398 at threshold 0.5,
and false-release rate 0.00483. Nevertheless, oracle event actions still
improve the next eight predictions by +0.031488 NLL in every seed. AUC is
therefore not a sufficient endpoint: probability timing and the downstream
action must be evaluated on held-out prediction.

The two oracle improvements are mostly additive. The post-hoc NLL interaction
is only +0.00316 (6/8 positive), so the experiment does not reveal a hidden
large synergy that would justify scaling both modules indiscriminately.

## Claim classification

| Claim | Conclusion |
|---|---|
| Existing actuator has a non-trivial oracle ceiling | **Support**, development scope |
| Learned event inference leaves post-jump utility on the table | **Support**, development scope |
| Learned Q/R inference is the dominant aggregate bottleneck | **Support**, development scope |
| Deployable factorized controller beats both reduced and IMM comparators uniformly | **Oppose** |
| Clean Q/R recovery in unseen joint regimes | **Oppose**, especially cell `110` |
| Formal confirmation, real behavior, neural dynamics, or E/I mechanism | **Inconclusive / untested** |

## Consequence

Do not tune the current actuator, add carrier scale, or use reserved seeds.
The next evidence-bearing bridge should test inference calibration on a public
task that independently manipulates volatility and observation stochasticity,
with held-out participant update prediction as the endpoint. Robust BOCPD or a
hierarchical particle filter belongs in the fast-event comparator set; the
existing write/release mapping remains fixed. IBL and participating E/I
experiments remain conditional on real behavioral utility.

The automatically generated data-bound figure is
[`exp43_fast_slow_causal_decomposition_development.pdf`](../exp43_fast_slow_causal_decomposition_development.pdf).
