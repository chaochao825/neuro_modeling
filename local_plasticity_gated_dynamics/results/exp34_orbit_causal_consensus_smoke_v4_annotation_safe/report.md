# Exp34 ORBIT causal-consensus report

- Profile: `smoke`
- Users (statistical unit): 2
- Algorithmic seeds averaged within user: 3
- Scale decision: **scale-authorized**
- Claim classification: **inconclusive**

## User-level accuracy

| Condition | User-equal mean | Official task-video mean |
|---|---:|---:|
| prototype | 0.5912 | 0.5888 |
| gain | 0.6806 | 0.6763 |
| delta | 0.5332 | 0.5215 |
| temporal | 0.6420 | 0.6413 |
| selection_fixed_best | 0.6420 | 0.6413 |
| instantaneous_majority | 0.6487 | 0.6462 |
| causal_consensus | 0.7374 | 0.7326 |
| memoryless_consensus | 0.6420 | 0.6413 |
| delayed_consensus | 0.6940 | 0.6892 |
| oracle_per_frame | 0.7902 | 0.7875 |

## Registered causal comparisons

- Causal consensus minus selection_fixed_best: +0.0954 (95% user bootstrap +0.0510, +0.1398; positive users 2/2).
- Causal consensus minus memoryless_consensus: +0.0954 (95% user bootstrap +0.0510, +0.1398; positive users 2/2).
- Causal consensus minus instantaneous_majority: +0.0887 (95% user bootstrap +0.0642, +0.1131; positive users 2/2).
- Causal consensus minus delayed_consensus: +0.0434 (95% user bootstrap +0.0422, +0.0446; positive users 2/2).
- Mean per-frame oracle headroom: 0.1076.
- Retained oracle headroom fraction: 0.886.
- Mean actuator disagreement: 0.5858.

The gate uses no query labels or future frames, but computes the full four-actuator bank. Its consensus signal is specific to ORBIT's one-object-per-video structure. Development users only authorize a frozen test run and never count as confirmatory evidence.
The official-style point estimate flattens task/video samples for benchmark comparability; all hypothesis uncertainty still uses user as the independent unit.
