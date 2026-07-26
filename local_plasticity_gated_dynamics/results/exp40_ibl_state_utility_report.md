# Exp40 IBL factorized-state utility audit

## Decision

The outcome-exposed 30-animal cohort is a post-hoc development panel, not confirmatory evidence. The development gate **did not pass**. A disjoint new cohort was therefore neither frozen nor opened, and neural analysis remains locked.

All 30 animals and all 210 planned condition cells were retained. 27 animals formed a complete paired endpoint panel; 3 failed symmetrically across all seven conditions because their chronological test fold contained fewer than eight low-contrast choices.

## Registered development readout

| Condition | Parameters | Mean low-contrast choice NLL | Mean context NLL |
|---|---:|---:|---:|
| history_only | 7 | 0.483938 | 0.693147 |
| learned_hmm_mean | 8 | 0.482630 | 0.307239 |
| semimarkov_mean | 8 | 0.488656 | 0.235848 |
| semimarkov_release | 10 | 0.490924 | 0.235848 |
| semimarkov_concentration | 10 | 0.495793 | 0.235848 |
| factorized_state | 12 | 0.496734 | 0.235848 |
| oracle_context_mean | 8 | 0.506526 | 0.000000 |

Semi-Markov context decoding improved over the learned HMM by 0.071391 nats/trial (95% animal bootstrap [0.049444, 0.095949]); the bounded development conclusion is **support**.

The factorized state did not convert that decoding gain into held-out choice utility. Dev-selected baseline minus factorized NLL was -0.010723 [-0.022459, 0.001010], positive in 9/27 animals. Any positive gain is **inconclusive**; a registered 0.005 nats/trial meaningful gain is **oppose**.

Release clamp harm was 0.001786 [-0.004956, 0.009137] (**inconclusive**). Precision clamp harm was -0.005114 [-0.009175, -0.001242] (**oppose**).

## Post-outcome assay probe

After inspecting the registered development result, one bounded probe selected regularization on all dev choices and added stronger regularization. It changed no observer state, task endpoint, test fold, or baseline family. Mean factorized low-contrast NLL changed from 0.496734 to 0.475206. Its paired gain over the dev-selected baseline remained negative (-0.003617 nats/trial). This probe diagnoses readout variance but cannot rescue or confirm the hypothesis.

## Interpretation boundary

The supported decoding result establishes only that known task duration structure helps recover the experimenter's block label. It does not show that release probability or run-length precision improves animal choice prediction, implements a biological actuator, or recovers an independently identifiable sensory-noise state. The truth-context condition is an evaluation diagnostic, not a behavioral upper bound: animals act on subjective beliefs rather than the experimenter's label.

The next admissible confirmation task must independently manipulate environmental volatility and observation noise and must first show held-out utility on an outcome-blind development gate. Scaling to IBL neural data is not unlocked by context NLL alone.

## Artifact receipt

- Registered run: `results/runs/exp40_ibl_state_utility/seed_0000/20260726T152019.701984Z`
- Assay-probe run: `results/runs/exp40_ibl_state_utility/seed_0000/20260726T153200.935487Z`
- Registered metrics SHA-256: `2404bb4ff6ac09d85a80389d0913b8e10f014b8c84ca7067d0e39f22e3cbaea4`
- Probe metrics SHA-256: `68f0ecd4dd23bb8eb9fb4452d1f9f55cf9654cf48a057e14b163ce4e04cb27ec`
- Statistical unit: animal; one session per animal in this cohort.
- Multiplicity: Holm across the five bounded development claims.
- Confirmatory status: inconclusive/not run on new data.
