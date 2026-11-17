#!/usr/bin/env bash
#SBATCH --job-name=scperteval-pop-lift
#SBATCH --partition=cpu-single
#SBATCH --time=08:00:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
TRACKS=(essential genetic-context chemical-context)
TRACK="${TRACKS[${SLURM_ARRAY_TASK_ID}]}"

if [[ ! -f "${PROJECT_ROOT}/scripts/lift_production_predictions.py" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi

module load compiler/gnu/12.1
cd "${PROJECT_ROOT}"
mkdir -p logs
export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility-py311"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[$(date -Is)] materializing control-lifted populations for ${TRACK}"
"${UV_BIN}" run --frozen python scripts/lift_production_predictions.py \
  --project-root "${PROJECT_ROOT}" \
  --track "${TRACK}" \
  --workers "${SLURM_CPUS_PER_TASK:-1}"
echo "[$(date -Is)] completed ${TRACK}"
