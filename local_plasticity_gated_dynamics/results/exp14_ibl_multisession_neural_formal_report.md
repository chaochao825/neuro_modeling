# Exp14 IBL multi-session neural audit

The endpoint is held-out one-step conditional Poisson likelihood, not a full latent-LDS marginal likelihood.

- Registered primary: `stimulus_pre / primary_past_safe` — **inconclusive**.
- Cohort: 20 sessions nested within 20 animals.
- Raw SHA-256: `8c14068e5855dc16fe3039ac86465fe9588ccda697b86bfce4340364f6de0ccd`
- Run-manifest SHA-256: `7d141ffddb666c755459d7cfffaa47ac03f6781464d26df10ce6898ff65491b4`
- Compact manifest SHA-256: `a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09`
- Compact bundle SHA-256: `f6cd351717986ede771a6bbbe755edeb3c30ef4bda48e86c8471bcf4364a41a4`
- Registered formal JSON SHA-256: `7717227beca59f6a39286a5bfc6ded10ed7b0896ce6f49409542f1743bf79680`
- Portable formal-config SHA-256: `106b99b861b0de82bfd020c23881e14a861b09ddc50f70752badb56b7a104913`
- Snapshot scope: `registered_core`
- Anatomical anchor policy: `fixed_region_order_union`; minimum session coverage=5.
- Allen macro mapping: `exp14_allen_macro_region_mapping_v1`; artifact SHA-256=`3bac702ed6b3ee5c21acbbfd929b077baa63226369ca8e1bef0b6faeb487fc23`.
- Allen ontology source SHA-256: `63654b8d35c7c1b5665636b645da774776ee8263658192f5dca1e815095e9147`.
- Allen ontology provenance SHA-256: `a01b7fa535e6de437ac46e8cf9de68a87d6a9b5587d055a3935476d956109fdc`.
- Macro mapping compact scope: `a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09`; acronyms=168; acronym-set SHA-256=`53c5a0cb3591749aad8fc2848b2627dfe93a4a747fa391e4ec68094211b8369b`.
- Missing-region handling: `pooled_training_fold_region_mean` (training folds only).
- Shared anchor basis: `cortex, thalamus, striatum, hippocampus, midbrain, hindbrain`.
- HMM restart selection: `eligible_converged_identifiable_then_likelihood`; registered restarts=5; every retained outer fit had at least one converged, identifiable eligible restart.

## Anatomical anchor coverage

| Region | Sessions present | Sessions missing | Fraction present |
|---|---:|---:|---:|
| cortex | 17 | 3 | 0.850 |
| thalamus | 6 | 14 | 0.300 |
| striatum | 5 | 15 | 0.250 |
| hippocampus | 11 | 9 | 0.550 |
| midbrain | 9 | 11 | 0.450 |
| hindbrain | 5 | 15 | 0.250 |

## Absolute model views

| View | Panel | Scope | Family | Animal-mean NLL/count | Pseudo-R² | Parameters |
|---|---|---|---|---:|---:|---:|
| movement_pre | full_trial_sensitivity | sensitivity_only | common | 0.758173 | -0.520089 | 16242 |
| movement_pre | full_trial_sensitivity | sensitivity_only | full | 0.758936 | -0.520953 | 17412 |
| movement_pre | full_trial_sensitivity | sensitivity_only | shared | 0.758157 | -0.520047 | 16272 |
| movement_pre | primary_past_safe | sensitivity_only | common | 0.790345 | -0.619806 | 14865 |
| movement_pre | primary_past_safe | sensitivity_only | full | 0.790373 | -0.618717 | 15333 |
| movement_pre | primary_past_safe | sensitivity_only | shared | 0.790276 | -0.619563 | 14877 |
| stimulus_pre | full_trial_sensitivity | sensitivity_only | common | 0.743783 | -0.556274 | 12159 |
| stimulus_pre | full_trial_sensitivity | sensitivity_only | full | 0.746702 | -0.564016 | 12393 |
| stimulus_pre | full_trial_sensitivity | sensitivity_only | shared | 0.744388 | -0.558565 | 12165 |
| stimulus_pre | primary_past_safe | registered_primary | common | 0.870270 | -0.962571 | 12159 |
| stimulus_pre | primary_past_safe | registered_primary | full | 0.871719 | -0.965277 | 12393 |
| stimulus_pre | primary_past_safe | registered_primary | shared | 0.871265 | -0.966292 | 12165 |

## Animal-primary paired comparisons

| View | Panel | Scope | Common - shared NLL/count (positive favors shared) [95% CI] | 90% retention ratio | Panel conclusion | Core conclusion |
|---|---|---|---:|---:|---|---|
| stimulus_pre | primary_past_safe | registered_primary | -0.000995 [-0.003061, 0.000157] | nan | inconclusive | inconclusive |
| stimulus_pre | full_trial_sensitivity | sensitivity_only | -0.000605 [-0.001943, 0.000155] | nan | inconclusive | inconclusive |
| movement_pre | primary_past_safe | sensitivity_only | 0.000069 [-0.000100, 0.000269] | nan | inconclusive | inconclusive |
| movement_pre | full_trial_sensitivity | sensitivity_only | 0.000016 [-0.000061, 0.000111] | nan | inconclusive | inconclusive |

Only the registered stimulus-pre/past-safe panel can update the core claim. Movement-pre and full-trial-covariate panels are sensitivity analyses even when their panel-level result is conclusive.
