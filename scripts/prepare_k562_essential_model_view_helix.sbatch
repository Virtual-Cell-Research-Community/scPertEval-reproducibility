#!/usr/bin/env bash
#SBATCH --job-name=scperteval-model-view-k562
#SBATCH --partition=cpu-single
#SBATCH --time=08:00:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
DATASET="replogle-weissman-2022-k562-essential"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

if [[ -z "${SCPERTEVAL_CANONICAL_H5AD:-}" || ! -f "${SCPERTEVAL_CANONICAL_H5AD}" ]]; then
  echo "Set SCPERTEVAL_CANONICAL_H5AD to the validated canonical raw-count H5AD." >&2
  exit 2
fi
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv is not executable at ${UV_BIN}" >&2
  exit 127
fi

module load compiler/gnu/12.1
cd "${PROJECT_ROOT}"
mkdir -p logs

export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[$(date -Is)] Creating selected-model view from canonical raw counts"
"${UV_BIN}" run --frozen python models/data/prepare_data.py \
  --input "${SCPERTEVAL_CANONICAL_H5AD}" \
  --dataset-name "${DATASET}" \
  --cell-type K562 \
  --compatible-models gears state

echo "[$(date -Is)] Freezing five-fold unseen-perturbation split"
"${UV_BIN}" run --frozen python models/data/prepare_split.py \
  "models/data/${DATASET}/raw.h5ad" \
  --dataset-id "${DATASET}" \
  --split-id "${DATASET}-unseen-perturbation-5-fold"

echo "[$(date -Is)] Model view and split manifest completed"
