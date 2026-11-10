#!/usr/bin/env bash
#SBATCH --job-name=scperteval-context-baselines
#SBATCH --partition=cpu-single
#SBATCH --time=04:00:00
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
if [[ ! -f "${PROJECT_ROOT}/configs/experiments/replogle-nadig-four-context-60-cell.json" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi

if [[ "${SLURM_ARRAY_TASK_ID}" -lt 4 ]]; then
  EXPERIMENT="replogle-nadig-four-context-60-cell"
  FOLD="${SLURM_ARRAY_TASK_ID}"
else
  EXPERIMENT="sciplex3-nine-drug-context-transfer"
  FOLD="$((SLURM_ARRAY_TASK_ID - 4))"
fi

module load compiler/gnu/12.1
cd "${PROJECT_ROOT}"
mkdir -p logs
export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility-py311"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[$(date -Is)] experiment=${EXPERIMENT} fold=${FOLD}"
"${UV_BIN}" run --frozen python -m models.baselines.context_baselines \
  --data "models/data/${EXPERIMENT}/raw.h5ad" \
  --split "configs/experiments/${EXPERIMENT}.json" \
  --fold "${FOLD}" \
  --capability population \
  --population-size 120 \
  --output-root models/baselines
