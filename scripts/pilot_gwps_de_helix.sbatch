#!/usr/bin/env bash
#SBATCH --job-name=scperteval-gwps-de-pilot
#SBATCH --partition=cpu-single
#SBATCH --time=04:00:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
DATASET_DEFAULT="${PROJECT_ROOT}/models/data/replogle-weissman-2022-k562-gwps/raw.h5ad"
DATASET="${SCPERTEVAL_GWPS_VIEW:-${DATASET_DEFAULT}}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi
if [[ ! -f "${DATASET}" ]]; then
  echo "GWPS model view does not exist: ${DATASET}" >&2
  exit 2
fi
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv is not executable at ${UV_BIN}" >&2
  exit 127
fi

module load compiler/gnu/12.1
cd "${PROJECT_ROOT}"
mkdir -p logs results/protocol_behavior/pilots/gwps-de

export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility-py311"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST} dataset=${DATASET}"
"${UV_BIN}" run --frozen reproduce-protocol-behavior \
  --pilot gwps-de \
  --dataset "${DATASET}" \
  --output-dir results/protocol_behavior/pilots/gwps-de \
  --workers "${SLURM_CPUS_PER_TASK:-16}"
echo "[$(date -Is)] GWPS DE pilot completed"
