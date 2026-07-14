# Exp17 tiny-recursive calibration

This is a train/inner-validation-only calibration artifact. The dataset adapter loaded its opaque capability store, but no test prediction array was requested and the hidden-target scorer was not called.

- Status: `frozen_validation_only`
- Selected candidate: `blank_low_diversity`
- Submitted seeds: `[1000, 1001, 1002]`
- Confirmation on an independently frozen test panel is still required.
- Claim conclusion: **inconclusive**.

## Validation-only candidate summary

| candidate | n_seeds_complete | mean_validation_blank_cell_accuracy | mean_validation_exact_accuracy | mean_train_blank_cell_accuracy | mean_parameter_count | mean_optimizer_steps | candidate_config_sha256 | candidate_config_hash_consistent | complete_on_all_submitted_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_tokens_deeper_refinement | 3 | 0.139233 | 0 | 0.136938 | 11776 | 792 | 1488d427787808433ab4723c884b92fe91c0e6196016b98f004aa738255e3fa0 | True | True |
| all_tokens_high_diversity | 3 | 0.150001 | 0 | 0.148054 | 11776 | 792 | 54e4cc5e43c933944dce3643146f2b3079adb246751497531063d307645499a7 | True | True |
| blank_high_diversity | 3 | 0.15035 | 0 | 0.15016 | 11776 | 792 | 71b58f0e08610d0ff07e0aa16ce691b9e1207e5f554c122923dc3534463d1896 | True | True |
| blank_low_diversity | 3 | 0.152084 | 0 | 0.193892 | 11776 | 800 | 5c666020f9525cdc385da1047cca8b9cabf69f90f84bf76b3ab3d990fac4cdba | True | True |
