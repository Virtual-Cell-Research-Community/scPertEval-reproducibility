#!/usr/bin/env bash
#SBATCH --job-name=scperteval-protocol-env
#SBATCH --partition=cpu-single
#SBATCH --time=01:00:00
#SBATCH --mem=16gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

module load compiler/gnu/12.1
cd "${PROJECT_ROOT}"
mkdir -p logs

export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility-protocol-py311"
export UV_CONCURRENT_BUILDS=1
export UV_LINK_MODE=copy

echo "[$(date -Is)] synchronizing the locked protocol-behavior environment"
if ! "${UV_BIN}" sync --frozen; then
  echo "[$(date -Is)] initial environment synchronization failed; retrying once" >&2
  "${UV_BIN}" sync --frozen
fi
"${UV_BIN}" run --frozen python -c "import geomloss, matplotlib, scperteval, torch; print(scperteval.__version__, torch.__version__)"
echo "[$(date -Is)] protocol-behavior environment is ready"
