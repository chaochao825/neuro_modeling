# Exp39 Post-Hoc Claim-Boundary Audit

Status: **claim-ineligible analysis of the frozen formal artifacts**.
The Exp39 algorithm, tapes, settings, metrics, and registered verdict were not changed.

## Cell-wise held-out utility

| Cell | Seen IMM minus factorized NLL | Positive seeds |
|---|---:|---:|
| `011` | +0.042613 | 27/30 |
| `101` | +0.020958 | 22/30 |
| `110` | -0.004333 | 12/30 |
| `111` | +0.132794 | 30/30 |

The aggregate advantage is not a uniform cell-wise result: cell `110` is negative on average.

## Seed-level cross-loading audit

Rows are estimated log controller coordinates; columns are true manipulated factors.

| Estimate | true h | true Q | true R |
|---|---:|---:|---:|
| h | 0.088327 | 0.028228 | 0.091796 |
| q | 0.067182 | 0.539656 | 1.491769 |
| r | 0.075404 | 0.549543 | 1.518126 |

## Timing

The original frozen `early_nll` summary includes each sequence's first block, which is initialization rather than a transition. Both the original all-block panel and a post-hoc transition-only panel are shown below.

- `all_blocks_including_sequence_initialization` / `selected_fixed_minus_factorized_nll` / `early_nll`: +0.061113 (24/30 positive seeds).
- `all_blocks_including_sequence_initialization` / `selected_fixed_minus_factorized_nll` / `late_nll`: +0.353154 (30/30 positive seeds).
- `all_blocks_including_sequence_initialization` / `seen_mode_imm_minus_factorized_nll` / `early_nll`: -0.086698 (2/30 positive seeds).
- `all_blocks_including_sequence_initialization` / `seen_mode_imm_minus_factorized_nll` / `late_nll`: +0.081915 (29/30 positive seeds).
- `transition_blocks_only` / `selected_fixed_minus_factorized_nll` / `early_nll`: +0.055958 (24/30 positive seeds).
- `transition_blocks_only` / `selected_fixed_minus_factorized_nll` / `late_nll`: +0.353871 (30/30 positive seeds).
- `transition_blocks_only` / `seen_mode_imm_minus_factorized_nll` / `early_nll`: -0.087523 (4/30 positive seeds).
- `transition_blocks_only` / `seen_mode_imm_minus_factorized_nll` / `late_nll`: +0.083153 (29/30 positive seeds).

## Selected adaptation rates

- h: beta=0.002 in 26/30 seeds (~500-step nominal time scale).
- h: beta=0.01 in 4/30 seeds (~100-step nominal time scale).
- q: beta=0.5 in 30/30 seeds (~2-step nominal time scale).
- r: beta=0.2 in 30/30 seeds (~5-step nominal time scale).

## Claim boundary

| Claim | Conclusion | Eligibility |
|---|---|---|
| `registered_average_unseen_composition_utility` | **support** | confirmatory_exp39_frozen_gate |
| `uniform_cellwise_composition_utility` | **oppose** | post_hoc_descriptive |
| `clean_three_factor_parameter_decomposition` | **oppose** | post_hoc_descriptive |
| `faster_post_switch_release_than_seen_mode_imm` | **oppose** | registered_secondary_descriptive |
| `late_regime_adaptation_advantage` | **support** | post_hoc_descriptive |
| `real_behavior_or_neural_utility` | **inconclusive** | not_tested_by_exp39 |

The next experiment must therefore test matched Q/R marginals, explicit reduced-factor baselines, and early switch release. This audit cannot itself upgrade any claim.
