# Exp37 Bayesian Change-Aware Prefix Confirmation

Evidence status: prospectively frozen session-held CORe50 evaluation.

Independent unit: session (n=9); seeds, tasks, objects, and frames were averaged within session.

## Primary registered comparisons

- change_reset_over_cumulative_switch: +0.0000 (95% session bootstrap +0.0000, +0.0000; Holm p=1).
- change_reset_over_fixed_forgetting_switch: -0.5329 (95% session bootstrap -0.5432, -0.5220; Holm p=0.015625).
- change_reset_over_sliding_window_switch: -0.5238 (95% session bootstrap -0.5334, -0.5141; Holm p=0.015625).
- change_reset_over_cumulative_natural: +0.0000 (95% session bootstrap +0.0000, +0.0000; Holm p=1).

## Operational diagnostics

- Cohort median detection delay: not estimable.
- Natural false alarms: 0.000 per 1,000 frames.
- Frozen stop rule triggered: True.

## Preregistered verdict

**OPPOSE**

- complete_session_coverage: pass
- hidden_gain_over_cumulative: fail
- hidden_gain_over_fixed_forgetting: fail
- hidden_gain_over_sliding_window: fail
- natural_noninferiority: pass
- detection_delay: fail
- false_alarm_rate: pass
- timing_controls: fail

This verdict concerns temporal decision state under frozen CORe50 evidence. It is not an official continual-learning, SOTA, or biological claim.
