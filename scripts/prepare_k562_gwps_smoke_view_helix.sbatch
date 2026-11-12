#!/usr/bin/env bash
# Materialize a lightweight adapter smoke view while preserving the full GWPS target/gene universe.
#SBATCH --job-name=scperteval-smoke-view-gwps
#SBATCH --partition=cpu-single
#SBATCH --time=02:00:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SOURCE_DATASET="replogle-weissman-2022-k562-gwps"
SMOKE_DATASET="${SOURCE_DATASET}-smoke"
EXPERIMENT="configs/experiments/${SOURCE_DATASET}-perturbation-generalization.json"
CANONICAL_DEFAULT="/mnt/sds-hd/sd22b002/projects/PerturbationData/derived/${SOURCE_DATASET}/adata.h5ad"
CANONICAL="${SCPERTEVAL_CANONICAL_H5AD:-${CANONICAL_DEFAULT}}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

if [[ ! -f "${PROJECT_ROOT}/${EXPERIMENT}" ]]; then
    echo "Submit this job from the scPertEval-reproducibility repository root." >&2
    exit 2
fi
if [[ ! -f "${CANONICAL}" ]]; then
    echo "Canonical GWPS H5AD does not exist: ${CANONICAL}" >&2
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

echo "[$(date -Is)] Materializing full-target reduced-cell GWPS smoke view"
"${UV_BIN}" run --frozen python -m models.data.prepare_data \
    --input "${CANONICAL}" \
    --perturbation-key target_gene \
    --control-key control_class \
    --control-value non-targeting \
    --experiment-manifest "${EXPERIMENT}" \
    --dataset-name "${SMOKE_DATASET}" \
    --cell-type K562 \
    --smoke \
    --n-perts 6935 \
    --cells-per-pert 5 \
    --control-cells 500 \
    --seed 42

echo "[$(date -Is)] Freezing the smoke view split"
"${UV_BIN}" run --frozen python -m models.data.prepare_split \
    "models/data/${SMOKE_DATASET}/raw.h5ad" \
    --dataset-id "${SMOKE_DATASET}" \
    --split-id "${SMOKE_DATASET}-unseen-perturbation-5-fold" \
    --seed 42

cmp \
    "models/data/${SOURCE_DATASET}/kfold_splits.json" \
    "models/data/${SMOKE_DATASET}/kfold_splits.json"

echo "[$(date -Is)] Full and smoke target assignments are byte-identical"
echo "[$(date -Is)] GWPS smoke view completed"
