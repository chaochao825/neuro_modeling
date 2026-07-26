# Exp35 Prefix Reliability Audit

Evidence status: retrospective mechanism audit; claim upgrade is prohibited.

## ORBIT user-level accuracy

- prefix_probability_equal: 0.8190
- prefix_prototype_probability: 0.8168
- prefix_calibrated_stack: 0.8168
- prefix_vote_equal: 0.8158
- prefix_gain_probability: 0.8134
- selection_prefix_single: 0.8134
- oracle_action_per_frame: 0.8122
- prefix_temporal_probability: 0.8111
- prefix_delta_probability: 0.7770
- selection_fixed_best: 0.7601
- temporal: 0.7601
- current_calibrated_stack: 0.7596
- prefix_consistency: 0.7592
- lagged_prefix_consistency: 0.7585
- current_probability_equal: 0.7111
- instantaneous_majority: 0.7098
- gain: 0.6896
- prototype: 0.6830
- delta: 0.6377

## Paired comparisons

- prefix consistency minus lagged_prefix_consistency: +0.0007 (95% user bootstrap +0.0004, +0.0010; Holm p=0.0012207).
- prefix consistency minus prefix_vote_equal: -0.0566 (95% user bootstrap -0.0907, -0.0291; Holm p=0.00158691).
- prefix consistency minus prefix_probability_equal: -0.0599 (95% user bootstrap -0.0944, -0.0320; Holm p=0.000549316).
- prefix consistency minus prefix_calibrated_stack: -0.0576 (95% user bootstrap -0.0925, -0.0299; Holm p=0.00158691).
- prefix consistency minus selection_prefix_single: -0.0542 (95% user bootstrap -0.0868, -0.0264; Holm p=0.00274658).
- prefix consistency minus temporal: -0.0009 (95% user bootstrap -0.0083, +0.0073; Holm p=0.819824).

## Exploratory mechanism decomposition

- prefix_prototype_probability minus prototype: +0.1339 (descriptive 95% user bootstrap +0.0877, +0.1854).
- prefix_gain_probability minus gain: +0.1238 (descriptive 95% user bootstrap +0.0780, +0.1776).
- prefix_delta_probability minus delta: +0.1393 (descriptive 95% user bootstrap +0.0768, +0.2137).
- prefix_temporal_probability minus temporal: +0.0510 (descriptive 95% user bootstrap +0.0236, +0.0843).
- prefix_probability_equal minus selection_prefix_single: +0.0057 (descriptive 95% user bootstrap -0.0043, +0.0161).
- prefix_probability_equal minus prefix_vote_equal: +0.0032 (descriptive 95% user bootstrap -0.0027, +0.0091).
- prefix_probability_equal minus prefix_calibrated_stack: +0.0022 (descriptive 95% user bootstrap -0.0044, +0.0085).

## Falsification verdict

- Comparative claim: **oppose**. prefix consistency did not exceed the strongest registered causal baseline.
- Consistency-as-correctness interpretation: **oppose**; the exact stable-wrong control yielded consistency accuracy 0.000 and wrong-lock fraction 1.000.

A positive retrospective ORBIT difference is descriptive only. An untouched external cohort is still required for confirmation.
