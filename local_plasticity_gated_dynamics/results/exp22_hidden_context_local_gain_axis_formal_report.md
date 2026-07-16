# Exp22: local hidden-context gain-axis audit

This standalone snapshot reports development-only postsynaptic gain-axis proposals derived off-policy from a frozen neutral trajectory and evaluated on a frozen high-rank Dale E/I receiver.

- Registered independent seeds: 30
- Eligible primary paired seeds: L1=30, L2=30
- Retained failed/invalid cells: 0
- Raw-run commit: `771526ecebebe8ff0f3151b4ba33856cb0de3215`.
- Raw software-environment SHA-256: `73c7ce782099a35335422f927fbeaf98573b68a6d5afc015e9f36049b49285fe`.
- Analysis commit: `cf4548bd7bc5f0088d7dacb8c3c5faf392da4358`; analysis-script SHA-256: `1b44c632335d88daea12bc2058ef89b8d377770babcb5459bbb030ed77c82cf2`.
- Analysis Python: `3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]`.
- Eligibility requires the structurally complete 11-cell grid, shared base network/gate/readout/split/neutral-trajectory receipts, reused feedback-specific tapes across L1/L2, attained selected-norm budgets, and no test truth, BPTT, autograd, or recurrent learning.
- L1 and L2 are separate panels. Aligned, random-signed, shuffled, orthogonal, and oracle proposal cells are selected-norm budget-matched within each panel; frozen_zero has a zero update budget and is an on/off baseline, not a matched-budget cell. The unselected norm is diagnostic only. An L2 conclusion is therefore conditional on L2-budget matching and must not be generalized to L1, and vice versa.
- Oracle-third-factor and truth-free 90-degree orthogonal cells are respectively an upper bound and a negative control. Neither can establish the main proposal-alignment claim.
- Recurrent weights are frozen. Proposals are computed off-policy from frozen development trajectories, so no conclusion is a claim of online local plasticity, recurrent plasticity, or weight-transport freedom.
- Exp22 contains no tuned BPTT, GRU, full-feedback, or recurrent-plasticity on/off baseline. It therefore cannot close the P0 acceptance criteria.
- Aligned absolute accuracy and balanced accuracy are reported with seed-level confidence intervals separately from relative contrasts. Balanced accuracy has its own registered threshold and must support, together with all three relative primary contrasts, and the registered oracle noninferiority margin for the joint claim to support. Those joint components must use an identical eligible seed set; otherwise the joint result is inconclusive. Raw accuracy has no registered threshold and remains descriptive/inconclusive.
- Bootstrap intervals target the mean seed-level effect, whereas the exact sign test checks cross-seed directional consistency. Support or opposition requires both the corresponding interval and the Holm-adjusted sign test; an inconclusive row can therefore coexist with an adverse mean confidence interval.

## Conclusions

| proposition | panel | comparison | estimate | ci_low | ci_high | n_eligible | n_planned | conclusion | claim_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off_policy_gain_axis_proposal_behavior_advantage | l1 | aligned_local_l1_vs_frozen_zero | -0.0031653503826899504 | -0.006308065259380634 | -0.0005867660260667369 | 30 | 30 | inconclusive | held-out behavior after a development-trajectory off-policy three-factor proposal within the L1-budget-matched panel only; not evidence for the other norm, not closed-loop local learning, and recurrent weights are bitwise frozen |
| off_policy_gain_axis_proposal_behavior_advantage | l1 | aligned_local_l1_vs_random_signed_feedback | -0.0036745769344969166 | -0.007185790078415278 | -0.0005428111961479845 | 30 | 30 | inconclusive | held-out behavior after a development-trajectory off-policy three-factor proposal within the L1-budget-matched panel only; not evidence for the other norm, not closed-loop local learning, and recurrent weights are bitwise frozen |
| off_policy_gain_axis_proposal_behavior_advantage | l1 | aligned_local_l1_vs_shuffled_feedback | -0.0027248468659205236 | -0.00602097429819209 | 0.00018388814177015199 | 30 | 30 | inconclusive | held-out behavior after a development-trajectory off-policy three-factor proposal within the L1-budget-matched panel only; not evidence for the other norm, not closed-loop local learning, and recurrent weights are bitwise frozen |
| off_policy_gain_axis_proposal_behavior_advantage | l2 | aligned_local_l2_vs_frozen_zero | -0.20822164198893445 | -0.25883758916368044 | -0.1536477618134075 | 30 | 30 | oppose | held-out behavior after a development-trajectory off-policy three-factor proposal within the L2-budget-matched panel only; not evidence for the other norm, not closed-loop local learning, and recurrent weights are bitwise frozen |
| off_policy_gain_axis_proposal_behavior_advantage | l2 | aligned_local_l2_vs_random_signed_feedback | -0.2058814456125394 | -0.2581840322363662 | -0.1522531469217743 | 30 | 30 | oppose | held-out behavior after a development-trajectory off-policy three-factor proposal within the L2-budget-matched panel only; not evidence for the other norm, not closed-loop local learning, and recurrent weights are bitwise frozen |
| off_policy_gain_axis_proposal_behavior_advantage | l2 | aligned_local_l2_vs_shuffled_feedback | -0.2048569502145889 | -0.2559689709810463 | -0.15062376140300152 | 30 | 30 | oppose | held-out behavior after a development-trajectory off-policy three-factor proposal within the L2-budget-matched panel only; not evidence for the other norm, not closed-loop local learning, and recurrent weights are bitwise frozen |
| oracle_third_factor_upper_bound_gap | l1 | aligned_local_l1_vs_oracle_third_factor | -0.0021766787400862094 | -0.006317451393338425 | 0.0021468639239871578 | 30 | 30 | support | development-hidden-context oracle third factor is an upper bound; this L1-budget-matched margin result cannot establish the local mechanism or generalize to the other norm |
| oracle_third_factor_upper_bound_gap | l2 | aligned_local_l2_vs_oracle_third_factor | -0.025304452171745945 | -0.09048068166765819 | 0.03635734765695723 | 30 | 30 | inconclusive | development-hidden-context oracle third factor is an upper bound; this L2-budget-matched margin result cannot establish the local mechanism or generalize to the other norm |
| orthogonal_feedback_negative_control | l1 | aligned_local_l1_vs_orthogonal_feedback | -0.0030140319635616956 | -0.006072785265699692 | -0.000394302265178872 | 30 | 30 | inconclusive | fixed-feedback 90-degree orthogonal negative control within the L1-budget-matched panel only; diagnostic, truth-free, and never sufficient for the main claim |
| orthogonal_feedback_negative_control | l2 | aligned_local_l2_vs_orthogonal_feedback | -0.20617020472671163 | -0.25875899750189624 | -0.15160304664807817 | 30 | 30 | oppose | fixed-feedback 90-degree orthogonal negative control within the L2-budget-matched panel only; diagnostic, truth-free, and never sufficient for the main claim |
| aligned_absolute_accuracy_descriptive | l1 | aligned_local_l1_behavior_accuracy | 0.8851666666666667 | 0.8787500000000003 | 0.8915000000000001 | 30 | 30 | inconclusive | absolute held-out aligned accuracy reported separately from relative contrasts; descriptive because no accuracy threshold was registered |
| aligned_absolute_balanced_accuracy | l1 | aligned_local_l1_behavior_balanced_accuracy | 0.8851027396829374 | 0.8785345217346816 | 0.8916884811889655 | 30 | 30 | support | absolute held-out aligned balanced accuracy with an independent registered threshold; never combined with relative performance through an OR rule |
| aligned_absolute_accuracy_descriptive | l2 | aligned_local_l2_behavior_accuracy | 0.6801666666666667 | 0.6277458333333333 | 0.7371666666666665 | 30 | 30 | inconclusive | absolute held-out aligned accuracy reported separately from relative contrasts; descriptive because no accuracy threshold was registered |
| aligned_absolute_balanced_accuracy | l2 | aligned_local_l2_behavior_balanced_accuracy | 0.680046448076693 | 0.6258624663495822 | 0.7363009271422675 | 30 | 30 | oppose | absolute held-out aligned balanced accuracy with an independent registered threshold; never combined with relative performance through an OR rule |
| joint_off_policy_proposal_alignment_specificity | l1 | aligned_local_l1_joint_vs_frozen_random_shuffled_and_oracle_margin |  |  |  | 30 | 30 | inconclusive | joint held-out behavior specificity for a frozen-trajectory off-policy gain-axis proposal in the L1-budget-matched panel that is also within the registered oracle margin; all components require the identical seed-level eligibility set; does not generalize to the other norm and is explicitly not closed-loop local learning or recurrent plasticity |
| joint_off_policy_proposal_alignment_specificity | l2 | aligned_local_l2_joint_vs_frozen_random_shuffled_and_oracle_margin |  |  |  | 30 | 30 | oppose | joint held-out behavior specificity for a frozen-trajectory off-policy gain-axis proposal in the L2-budget-matched panel that is also within the registered oracle margin; all components require the identical seed-level eligibility set; does not generalize to the other norm and is explicitly not closed-loop local learning or recurrent plasticity |

## Ineligible seeds

None.

## Retained failed or invalid cells

None.

## Classification rule

Held-out balanced-accuracy contrasts are paired within seed. The three primary comparisons per
panel use deterministic seed bootstrap intervals for the mean effect, exact sign tests for
cross-seed directional consistency, and one fixed Holm family across both panels. Missing tests
enter as p=1 without shrinking the planned family. Both the corresponding mean interval and the
Holm-adjusted sign test must pass for support or opposition. The joint claim can support only
when the aligned off-policy proposal supports all comparisons against frozen, random-signed, and
shuffled feedback, meets the registered absolute balanced-accuracy threshold, and supports the
registered oracle noninferiority margin; in particular, failure to beat frozen or shuffled can
never support it. Oracle and orthogonal failures do not invalidate otherwise eligible non-oracle
comparisons, but an unavailable or inconclusive oracle margin leaves the joint claim
inconclusive. The primary, absolute, and oracle components must also have an identical eligible
seed set; their intersection is reported as the joint eligibility receipt. These controls remain
bounded diagnostics and cannot establish the mechanism alone. Failed and scientifically
ineligible seeds remain visible in the raw snapshot. The registered absolute balanced-accuracy
threshold, relative primary claims, and oracle margin are joined through AND, never OR. Every
panel-level conclusion is conditional on its selected-norm budget match and does not generalize
to the other norm.
