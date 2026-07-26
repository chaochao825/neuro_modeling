# Exp39 Outcome-Blind Execution Amendment

Amended before inspecting any formal predictive NLL, latent MSE, parameter
estimate, clamp effect, or gate result.

The frozen entry point executes 30 independent seeds serially. The first
launch wrote completion statuses for seven seeds but would leave most of the
210 server idle. That incomplete attempt is preserved without scientific
inspection.

The amendment dispatches the exact hash-frozen run_seed function to six CPU
processes and applies the exact hash-frozen summarize function after all
results return. It changes no configuration, seed, random-number derivation,
tape, model, baseline, selection grid, endpoint, threshold, multiplicity
family, or figure. Per-seed raw artifacts and failures remain mandatory.

The parallel wrapper has its own SHA-256 receipt and the core implementation
receipt must still validate before launch.

