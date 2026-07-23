# Exp33 ORBIT streaming few-shot report

- Profile: `smoke`
- Users (statistical unit): 2
- Algorithmic seeds averaged within user: 3
- Scale decision: **scale-not-authorized**
- Claim classification: **inconclusive**

## Held-out accuracy

| Condition | Mean user/video accuracy |
|---|---:|
| prototype | 0.5929 |
| gain | 0.6802 |
| delta | 0.5192 |
| temporal | 0.6431 |
| train_fixed_best | 0.6431 |
| reward_only_local | 0.6114 |
| credit_shuffled_local | 0.6394 |
| oracle_per_frame | 0.7892 |

## Registered comparisons

- Reward-only local minus train-selected fixed: -0.0316 (95% user bootstrap -0.0750, +0.0117).
- Reward-only local minus credit-shuffled local: -0.0279 (95% user bootstrap -0.0512, -0.0047).
- Mean oracle headroom: 0.1063.
- Mean action disagreement: 0.5967.

This report does not treat frames, videos, tasks, or repeated seeds as independent participants. Development results cannot support a confirmatory claim.
