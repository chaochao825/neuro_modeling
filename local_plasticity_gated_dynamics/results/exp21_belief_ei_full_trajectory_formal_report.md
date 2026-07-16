# Exp21: belief-controlled high-rank E/I full-trajectory audit

This is a standalone snapshot. It does not modify the historical project-wide `results/summary.csv` or `results/report.md`.

- Registered independent seeds: 30
- Complete seeds: 30
- Retained failed/invalid seeds: 0 (run-level execution status)
- Claim-level scientifically ineligible conclusion rows: 0 of 5
- Exact v2 collection-provenance seeds: 30 of 30
- Trial-reset identifiable total-operator receipts: 30 of 30
- Trial-reset physical-x perturbation receipts: 30 of 30
- Episode-continuous v2 sensitivity receipts: 30 of 30 (reported only; never gates trial-reset conclusions)
- Raw-run commit: `cf4548bd7bc5f0088d7dacb8c3c5faf392da4358`.
- Raw software-environment SHA-256: `73c7ce782099a35335422f927fbeaf98573b68a6d5afc015e9f36049b49285fe`.
- Analysis commit: `cf4548bd7bc5f0088d7dacb8c3c5faf392da4358`; analysis-script SHA-256: `30dd639b847b16feb52650a26155cf61346855f4ab2f36e34fe78931e54b112b`.
- Analysis Python: `3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]`.
- Registered latent dimension: d=4 (fixed mechanism audit; no nested-CV dimension selection).
- Primary state policy: every trial starts from the zero receiver state.
- The episode-continuous receiver is a sensitivity analysis and is not used for these five primary conclusions.
- Rollouts are conditioned on observed future exogenous controls; they are not autonomous forecasts or probabilistic LDS likelihoods.
- The nonlinear endpoint metric is finite-amplitude and finite-time; the fixed-drive result is only a narrow combined-actuator endpoint-separation sanity probe. It does not support gate causality, low-dimensional or shared-manifold dynamics, or attractor claims.
- Claim-level eligibility is definition-specific. Total-control gain and closure require the trial-reset 19-column full_shared_neutral_cue operator; perturbation additionally requires joint_state_pca_physical_x_projection_v2 with full sampled-reference coverage. State-affine and fixed-drive measurements are not gated by unrelated total-operator, perturbation, or episode-sensitivity receipts. Historical v1 rows supplied directly to this audit remain readable for claims whose scientific definition did not change, while the formal collector selects exact v2 run configurations.

## Conclusions

| proposition | comparison | estimate | ci_low | ci_high | n_eligible | n_planned | conclusion | claim_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial_reset_total_control_gain_vs_raw_common | trial_reset_total_full_vs_raw_common | 0.255256754700553 | 0.24970460806312728 | 0.2613022149842124 | 30 | 30 | support | raw receiver sensory input is shared; positive gain is raw-common rollout RMSE minus total scalar-control predictor RMSE; registered d=4 mechanism audit without nested-CV latent-dimension selection |
| trial_reset_population_state_affine_gain_vs_routed_common | trial_reset_routed_state_affine_vs_routed_common | 0.0003575161547440198 | 8.60000652335899e-05 | 0.0006299215068946757 | 30 | 30 | support | routed sensory input and train PCA are shared; state transition A and affine bias may depend on population-gain belief, while exogenous input and epoch coefficients remain shared; registered d=4 mechanism audit without nested-CV latent-dimension selection |
| trial_reset_full_trajectory_closure | trial_reset_total_full_registered_threshold | 0.33006875729117807 |  |  | 30 | 30 | support | controlled affine predictor rollout conditioned on observed future exogenous controls; not an autonomous or probabilistic LDS score; registered d=4 mechanism audit without nested-CV latent-dimension selection |
| trial_reset_nonlinear_normal_recovery_relative_to_tangent | frozen_physical_network_perturbation_audit | -2.9773298112871585 |  |  | 30 | 30 | support | finite-amplitude recovery normal to the physical-x projection of the train-fitted joint latent subspace; not proof of a joint manifold, global or asymptotic Lyapunov stability, or an attractor; registered d=4 mechanism audit without nested-CV latent-dimension selection |
| fixed_drive_separated_endpoint_probe | combined_actuator_fixed_drive_training_anchor_probe | 0.1425800985538237 |  |  | 30 | 30 | support | narrow finite-horizon combined-actuator endpoint-separation sanity probe on training-derived anchors; does not support gate causality, low-dimensional or shared-manifold dynamics, or attractor claims; registered d=4 mechanism audit without nested-CV latent-dimension selection |

## Retained failed or invalid seeds

None.

## Claim-level scientific ineligibility

None; every conclusion row had all planned seed-level measurements.

## Classification rule

The two rollout-gain claims use the independent seed as the paired unit, deterministic bootstrap
confidence intervals, exact sign tests, and one fixed two-hypothesis Holm family. A missing or
scientifically ineligible hypothesis is entered as p=1 and still occupies its planned family
slot. Registered mechanism audits require complete, identifiable measurements for every planned
seed. Run-level failures and claim-level scientific ineligibility are reported separately;
either prevents support for the affected claim. A completely absent registered attempt aborts
collection.
