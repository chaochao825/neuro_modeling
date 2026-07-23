# Exp34 ORBIT causal-consensus report

- Profile: `formal`
- Users (statistical unit): 17
- Algorithmic seeds averaged within user: 5
- Coverage: 17/17 users and 4250/4250 seed-user-tasks
- Officially excluded short query videos: 3
- Scale decision: **not-applicable**
- Claim classification: **support**

## User-level accuracy

| Condition | User-equal mean | Official task-video mean |
|---|---:|---:|
| prototype | 0.6830 | 0.6436 |
| gain | 0.6896 | 0.6520 |
| delta | 0.6377 | 0.5946 |
| temporal | 0.7032 | 0.6611 |
| selection_fixed_best | 0.6896 | 0.6520 |
| instantaneous_majority | 0.6936 | 0.6529 |
| causal_consensus | 0.7189 | 0.6743 |
| memoryless_consensus | 0.7032 | 0.6611 |
| delayed_consensus | 0.7123 | 0.6683 |
| oracle_per_frame | 0.7665 | 0.7241 |

## Registered causal comparisons

- Causal consensus minus selection_fixed_best: +0.0293 (95% user bootstrap +0.0155, +0.0437; positive users 15/17; exact sign-flip p=0.000946; Holm p=0.001892).
- Causal consensus minus memoryless_consensus: +0.0157 (95% user bootstrap +0.0039, +0.0279; positive users 11/17; exact sign-flip p=0.025452; Holm p=0.025452).
- Causal consensus minus instantaneous_majority: +0.0253 (95% user bootstrap +0.0150, +0.0362; positive users 16/17; exact sign-flip p=0.000046; Holm p=0.000183).
- Causal consensus minus delayed_consensus: +0.0066 (95% user bootstrap +0.0043, +0.0088; positive users 15/17; exact sign-flip p=0.000092; Holm p=0.000275).
- Mean per-frame oracle headroom: 0.0546.
- Retained oracle headroom fraction: 0.536.
- Mean actuator disagreement: 0.3006.

The gate uses no query labels or future frames, but computes the full four-actuator bank. Its consensus signal is specific to ORBIT's one-object-per-video structure. Development users only authorize a frozen test run and never count as confirmatory evidence.
The official-style point estimate flattens task/video samples for benchmark comparability; all hypothesis uncertainty still uses user as the independent unit.
