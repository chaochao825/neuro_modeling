# Exp44 Piray--Daw Q/R behavioral-utility development report

The registered Experiment-1 development conjunction **fails**. 
This is development evidence only and cannot directly upgrade a claim.

## Primary held-out participant contrasts

| Baseline | NLL gain (baseline - factorized) | 95% bootstrap CI | Holm p |
|---|---:|---:|---:|
| fixed_gain | +0.001044 | [-0.006093, +0.010351] | 1 |
| total_uncertainty | +0.006105 | [-0.002445, +0.017697] | 0.974001 |
| hierarchical_particle | +0.007164 | [+0.000894, +0.012360] | 0.0814636 |
| autocovariance_qr | +0.023895 | [+0.013229, +0.031912] | 6.69839e-06 |
| oracle_qr | +0.093632 | [+0.071623, +0.112772] | 3.84095e-15 |

| Baseline | MSE gain (baseline - factorized) | 95% bootstrap CI | Holm p |
|---|---:|---:|---:|
| fixed_gain | +0.019918 | [-0.385452, +0.522493] | 1 |
| total_uncertainty | +0.324224 | [-0.153438, +0.951971] | 0.974001 |
| hierarchical_particle | +0.463559 | [+0.112026, +0.761277] | 0.0301828 |
| autocovariance_qr | +1.531729 | [+0.999207, +1.977896] | 3.15761e-08 |
| oracle_qr | +6.381001 | [+5.256223, +7.398235] | 5.02287e-23 |

## Gate clauses

- `nll_gain_vs_fixed`: fail
- `nll_gain_vs_total_uncertainty`: fail
- `mse_gain_vs_fixed`: fail
- `mse_gain_vs_total_uncertainty`: fail
- `directional_qr_effects`: fail
- `cellwise_noninferiority_vs_total`: pass
- `particle_gain_retention`: pass

Executed gain Q effect: -0.000702.
Executed gain R effect (low R - high R): +0.001130.

## Frozen full-Experiment-1 selections

- `fixed_gain`: `fixed_gain:f5a1f9326639`, response sigma 5.500429.
- `total_uncertainty`: `total_uncertainty:2ade3e47abb7`, response sigma 5.508262.
- `factorized_local_em`: `factorized_local_em:2fe494fe42f9`, response sigma 5.513679.
- `autocovariance_qr`: `autocovariance_qr:3764e1712f24`, response sigma 5.650875.
- `hierarchical_particle`: `hierarchical_particle:437b6f9cf06c`, response sigma 5.555557.
- `oracle_qr`: `oracle_qr:b964134d4145`, response sigma 6.064788.

## Scope

Experiment 1 and Experiment 2 use the same bag/bird stimulus tape. This run tests held-out participant behavior only. It does not test unseen streams, hidden changes, POPGym control, neural activity, or a participating E/I mechanism.

The stop rule keeps Experiment 2 and POPGym unexecuted.
