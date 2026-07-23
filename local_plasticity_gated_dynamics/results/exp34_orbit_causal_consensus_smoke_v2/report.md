# Exp34 ORBIT causal-consensus report

- Profile: `smoke`
- Users (statistical unit): 2
- Algorithmic seeds averaged within user: 3
- Scale decision: **scale-authorized**
- Claim classification: **inconclusive**

## User-level accuracy

| Condition | Mean task-video accuracy |
|---|---:|
| prototype | 0.5913 |
| gain | 0.6801 |
| delta | 0.5174 |
| temporal | 0.6419 |
| selection_fixed_best | 0.6419 |
| causal_consensus | 0.7388 |
| memoryless_consensus | 0.6419 |
| delayed_consensus | 0.6967 |
| oracle_per_frame | 0.7875 |

## Registered causal comparisons

- Causal consensus minus selection_fixed_best: +0.0968 (95% user bootstrap +0.0527, +0.1409; positive users 2/2).
- Causal consensus minus memoryless_consensus: +0.0968 (95% user bootstrap +0.0527, +0.1409; positive users 2/2).
- Causal consensus minus delayed_consensus: +0.0421 (95% user bootstrap +0.0413, +0.0429; positive users 2/2).
- Mean per-frame oracle headroom: 0.1061.
- Retained oracle headroom fraction: 0.913.
- Mean actuator disagreement: 0.5959.

The gate uses no query labels or future frames, but computes the full four-actuator bank. Its consensus signal is specific to ORBIT's one-object-per-video structure. Development users only authorize a frozen test run and never count as confirmatory evidence.
