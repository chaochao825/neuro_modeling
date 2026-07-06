# Reproduction Notes

## Environment

The code has been tested with Python 3.12 plus NumPy and Matplotlib on Windows, and is intended to run with a standard Python environment on Linux as well.

For `neural_multiscale_tests`:

```bash
cd /path/to/neuro_modeling/neural_multiscale_tests
pip install -r requirements.txt
python -B run_simulations.py --quick --seed 7
python -B figures/goal_alignment_visual_summary_plot.py
python -B figures/local_to_global_mechanism_map_plot.py
python -B figures/integrated_goal_status_plot.py
python -B -m unittest discover -s tests
```

For `minimal_computation_python`:

```bash
cd /path/to/neuro_modeling
git clone https://github.com/ChrisWLynn/Minimal_computation minimal_computation_original
cd /path/to/neuro_modeling/minimal_computation_python
python -B run_reproduction.py --dataset c_elegans --neuron 13 --max-inputs 32 --sweep 1,2,3,5,8,13,21,32
python -B run_reproduction.py --dataset hippocampus --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
python -B run_reproduction.py --dataset visual_spontaneous --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
python -B run_reproduction.py --dataset visual_responding --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
python -B figures/plot_paper_comparison.py
python -B -m unittest discover -s tests
```

## Provenance

- `neural_multiscale_tests/reports/summary.json` and `decision_matrix.json` were regenerated with `--quick --seed 7`.
- The four Minimal_computation JSON files include a `run_config` field with the exact sweep and threshold used.
- `minimal_computation_original/` is intentionally not vendored here. Clone it from `https://github.com/ChrisWLynn/Minimal_computation` inside the `neuro_modeling/` repository root before rerunning Minimal_computation sweeps.
