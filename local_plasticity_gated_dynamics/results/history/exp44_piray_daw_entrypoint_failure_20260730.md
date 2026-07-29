# Exp44 pre-outcome entrypoint failure

Status: preserved engineering failure; no scientific outcome was exposed.

- Date: 2026-07-30 (Asia/Shanghai)
- Frozen commit and tag: `7e785adda71f15f38d0f2db073af13a2d4a97dc8`,
  `exp44-dev-v1-preoutcome-20260730`
- Intended stage: Piray--Daw Experiment 1 development evaluation
- Exit status: `1`
- Failure point: module import, before dataset loading, candidate construction,
  cross-validation, or metric computation
- Command: `python experiments/exp44_piray_daw_qr_behavior.py --config
  configs/development/exp44_piray_daw_qr_behavior_v1.json --output
  results/exp44_piray_daw_qr_behavior_development_v1`
- Error: `ModuleNotFoundError: No module named 'src'`

The script was importable in pytest because pytest placed the repository root on
`sys.path`, but the documented direct-file entrypoint did not. The remediation is
limited to the same project-root bootstrap used by existing experiment scripts,
plus a subprocess regression test that runs the direct entrypoint with `--help`.
No model equation, parameter grid, data split, endpoint, threshold, random seed,
or numerical path was changed. A new pre-outcome commit and tag must precede the
replacement run; this failed launch remains part of the project history.
