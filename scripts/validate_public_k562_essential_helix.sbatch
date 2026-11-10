#!/usr/bin/env bash
#SBATCH --job-name=scperteval-public-k562
#SBATCH --partition=cpu-single
#SBATCH --time=04:00:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
DATASET="replogle-weissman-2022-k562-essential"
FILENAME="ReplogleWeissman2022_K562_essential.h5ad"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

if [[ ! -f "${PROJECT_ROOT}/configs/datasets/${DATASET}.toml" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi
if [[ -z "${SCPERTEVAL_SOURCE_FILE:-}" || ! -f "${SCPERTEVAL_SOURCE_FILE}" ]]; then
  echo "Set SCPERTEVAL_SOURCE_FILE to the verified public scPerturb v1.4 source H5AD." >&2
  exit 2
fi
if [[ -z "${SCPERTEVAL_REFERENCE_H5AD:-}" || ! -f "${SCPERTEVAL_REFERENCE_H5AD}" ]]; then
  echo "Set SCPERTEVAL_REFERENCE_H5AD to the internal canonical H5AD used for comparison." >&2
  exit 2
fi
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv is not executable at ${UV_BIN}" >&2
  exit 127
fi

module load compiler/gnu/12.1
cd "${PROJECT_ROOT}"
mkdir -p "data/raw/${DATASET}" "data/processed/${DATASET}" logs
ln -sfn "${SCPERTEVAL_SOURCE_FILE}" "data/raw/${DATASET}/${FILENAME}"

export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility-py311"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

OUTPUT="data/processed/${DATASET}/adata.h5ad"
cleanup() {
  rm -f -- "${OUTPUT}"
}
trap cleanup EXIT

echo "[$(date -Is)] Preparing ${DATASET} with the public workflow"
"${UV_BIN}" run --frozen python -m workflows.data prepare "${DATASET}" --force
echo "[$(date -Is)] Comparing the regenerated file with the frozen reference checksum"
if ! "${UV_BIN}" run --frozen python -m workflows.data validate "${DATASET}"; then
  echo "[$(date -Is)] File-level checksum differs; comparing logical AnnData content"
fi
"${UV_BIN}" run --frozen python -m workflows.semantic_h5ad \
  "${OUTPUT}" "${SCPERTEVAL_REFERENCE_H5AD}"
echo "[$(date -Is)] Public regeneration matched the canonical logical AnnData content"
