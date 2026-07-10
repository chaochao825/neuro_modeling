# Reproduction Notes

## Environment

Use Python 3.11. The formal server snapshot was produced with Python 3.11.15;
the project metadata intentionally rejects Python 3.12 and later.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './local_plasticity_gated_dynamics[dev]'
python -m pip install -e ./shared_dynamics_real_data
python -m pip install -r neural_multiscale_tests/requirements.txt
python -m pip install pytest ruff
```

The upstream public MAT files are not redistributed. Place them at
`minimal_computation_original/` by cloning the upstream repository:

```bash
git clone https://github.com/ChrisWLynn/Minimal_computation minimal_computation_original
```

## Tests

The root configuration collects all four subprojects without module-name
collisions:

```bash
python -m pytest -q
```

For isolated diagnosis, the equivalent per-subproject commands are:

```bash
python -m pytest -q local_plasticity_gated_dynamics/tests
python -m pytest -q shared_dynamics_real_data/tests
python -m pytest -q minimal_computation_python/tests
python -m pytest -q neural_multiscale_tests/tests

python -m ruff check local_plasticity_gated_dynamics \
  shared_dynamics_real_data minimal_computation_python neural_multiscale_tests
```

## Core local-plasticity experiments

From `local_plasticity_gated_dynamics/`, every experiment accepts a JSON
config, an optional seed subset, and a results root. Formal runs use fixed seeds
0--19 and retain failed attempts.

```bash
cd local_plasticity_gated_dynamics
python experiments/exp01_feedback_dimension_sweep.py \
  --config configs/formal/exp01_feedback_dimension_sweep.json \
  --results-root results
python experiments/exp04_phase_gating.py \
  --config configs/formal/exp04_phase_gating.json \
  --results-root results
python scripts/build_report.py --results-root results --plots
cd ..
```

The committed core directory is a compact evidence snapshot containing source,
tests, `raw_metrics.csv`, summaries, reports, and figures. The original
timestamped run directories remain in the source workspace; this publication
does not pretend that the compact copy contains them.

## Shared-dynamics real-data panel

The formal visual experiment uses 20 deterministic neuron-subset seeds, three
purged contiguous folds, 128 aligned units, dimensions `1,2,4,8,16,32`, and six
models. It never randomly splits time points. A single process can run all
seeds, or several processes can receive disjoint subsets of the configured
seed universe:

```bash
python -m shared_dynamics_real_data.run_visual_context \
  --config shared_dynamics_real_data/configs/formal.json \
  --data-root minimal_computation_original \
  --results-root shared_dynamics_real_data/results

# Example disjoint shard (repeat with non-overlapping subsets):
python -m shared_dynamics_real_data.run_visual_context \
  --config shared_dynamics_real_data/configs/formal.json \
  --data-root minimal_computation_original \
  --results-root shared_dynamics_real_data/results \
  --seeds 0,1,2,3,4

python -m shared_dynamics_real_data.build_report \
  --results-root shared_dynamics_real_data/results --profile formal
python -m shared_dynamics_real_data.figures.plot_visual_context \
  --results-root shared_dynamics_real_data/results \
  --output-base shared_dynamics_real_data/figures/visual_context_shared_dynamics
```

`build_report.py` refuses mixed analysis/data fingerprints, validates the
planned cross-product, and treats a missing cell or fold as a failed
seed-by-dimension panel. The committed `results/runs/` snapshot contains the
path-sanitized configs, manifests, JSONL/CSV raw metrics, statuses, and logs for
the completed smoke and formal shards.

## Minimal-computation selectors

Run the current block-Schur, MATLAB-control-flow-compatible C. elegans result:

```bash
cd minimal_computation_python
python -B run_reproduction.py --dataset c_elegans --neuron 13 --max-inputs 32 \
  --selector schur_entropy_drop --completion-mode matlab_residual \
  --failure-selection matlab_last
cd ..
```

The committed C. elegans Schur result reaches the residual criterion at 7
inputs. The older four lightweight JSON files were produced before the Schur
implementation. To run that historical approximation explicitly, use:

```bash
cd minimal_computation_python
python -B run_reproduction.py --dataset hippocampus --neuron 13 \
  --max-inputs 30 --sweep 1,2,3,5,8,13,21,30 \
  --selector residual_approximation --initialization warm_start \
  --completion-mode strict_optimizer_and_residual \
  --failure-selection best_error --no-binary-search
cd ..
```

Change `--dataset` and the sweep only when intentionally regenerating the
historical visual/C. elegans baselines. Do not label those outputs Schur or
MATLAB parity.

## Legacy H1--H5 calibration

```bash
cd neural_multiscale_tests
python -B run_simulations.py --quick --seed 7
python -B figures/goal_alignment_visual_summary_plot.py
python -B figures/local_to_global_mechanism_map_plot.py
python -B figures/integrated_goal_status_plot.py
cd ..
```

These are single-seed synthetic calibration diagnostics, not biological
inference. The current interpretation boundary is
`docs/integrated_method_audit_zh.md`.
