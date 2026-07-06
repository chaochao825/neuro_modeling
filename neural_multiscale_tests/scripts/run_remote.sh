#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

runner=(python)
if [ -n "${NEURAL_TESTS_PYTHON:-}" ]; then
  runner=("$NEURAL_TESTS_PYTHON")
elif [ -n "${NEURAL_TESTS_CONDA_ENV:-}" ]; then
  if [ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "NEURAL_TESTS_CONDA_ENV was set but conda.sh was not found under \$HOME/miniconda3." >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  runner=(conda run -n "$NEURAL_TESTS_CONDA_ENV" python)
  echo "Using requested conda environment: $NEURAL_TESTS_CONDA_ENV"
elif python -c "import numpy" >/dev/null 2>&1; then
  runner=(python)
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # Prefer an existing conda environment with numpy over installing packages.
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  for env_name in base myenv lgn mixbit iMap_eda promptir; do
    if conda run -n "$env_name" python -c "import numpy" >/dev/null 2>&1; then
      runner=(conda run -n "$env_name" python)
      echo "Using conda environment: $env_name"
      break
    fi
  done
fi

"${runner[@]}" -c "import numpy" >/dev/null
"${runner[@]}" -c "import platform, numpy; print('Python: ' + platform.python_version()); print('NumPy: ' + numpy.__version__)"
"${runner[@]}" -m unittest discover -s tests
"${runner[@]}" run_simulations.py --quick --seed 7
