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
EXPERIMENT="configs/experiments/replogle-weissman-2022-k562-essential-perturbation-generalization.json"
CANONICAL_DEFAULT="/mnt/sds-hd/sd22b002/projects/PerturbationData/derived/${DATASET}/adata.h5ad"
CANONICAL="${SCPERTEVAL_CANONICAL_H5AD:-${CANONICAL_DEFAULT}}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

if [[ ! -f "${PROJECT_ROOT}/${EXPERIMENT}" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi
if [[ ! -f "${CANONICAL}" ]]; then
  echo "Canonical K562 Essential H5AD does not exist: ${CANONICAL}" >&2
  exit 2
fi
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv is not executable at ${UV_BIN}" >&2
  exit 127
fi
if grep -q 'pending-cross-model-feature-coverage-audit' "${PROJECT_ROOT}/${EXPERIMENT}"; then
  echo "The essential cohort is not frozen: complete the TxPert-public and ESM2 coverage audit first." >&2
  exit 2
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

COMMON_ARGS=(
  --input "${CANONICAL}"
  --experiment-manifest "${EXPERIMENT}"
)

echo "[$(date -Is)] Validating frozen K562 Essential metadata"
"${UV_BIN}" run --frozen python -m models.data.prepare_data \
  "${COMMON_ARGS[@]}" \
  --validate-metadata-only

echo "[$(date -Is)] Creating selected-model view from canonical raw counts"
"${UV_BIN}" run --frozen python -m models.data.prepare_data \
  "${COMMON_ARGS[@]}" \
  --dataset-name "${DATASET}" \
  --cell-type K562

echo "[$(date -Is)] Freezing five-fold unseen-perturbation split"
"${UV_BIN}" run --frozen python -m models.data.prepare_split \
  "models/data/${DATASET}/raw.h5ad" \
  --dataset-id "${DATASET}" \
  --split-id "${DATASET}-unseen-perturbation-5-fold"

echo "[$(date -Is)] Model view and split manifest completed"
