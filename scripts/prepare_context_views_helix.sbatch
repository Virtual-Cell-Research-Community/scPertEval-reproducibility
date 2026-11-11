#!/usr/bin/env bash
#SBATCH --job-name=scperteval-context-data
#SBATCH --partition=cpu-single
#SBATCH --time=08:00:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
: "${SCPERTEVAL_CANONICAL_ROOT:?set to a root containing derived/<id>/adata.h5ad or processed/<id>/adata.h5ad}"

if [[ ! -f "${PROJECT_ROOT}/configs/experiments/replogle-nadig-four-context-60-cell.json" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv is not executable at ${UV_BIN}" >&2
  exit 127
fi

canonical_path() {
  local dataset="$1"
  local candidate
  for candidate in \
    "${SCPERTEVAL_CANONICAL_ROOT}/derived/${dataset}/adata.h5ad" \
    "${SCPERTEVAL_CANONICAL_ROOT}/processed/${dataset}/adata.h5ad"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "No canonical H5AD found for ${dataset} under ${SCPERTEVAL_CANONICAL_ROOT}" >&2
  return 1
}

module load compiler/gnu/12.1
cd "${PROJECT_ROOT}"
mkdir -p logs
export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility-py311"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
  EXPERIMENT="replogle-nadig-four-context-60-cell"
  INPUTS=(
    "replogle-weissman-2022-k562-essential=$(canonical_path replogle-weissman-2022-k562-essential)"
    "replogle-weissman-2022-rpe1=$(canonical_path replogle-weissman-2022-rpe1)"
    "nadig-oconner-2024-hepg2=$(canonical_path nadig-oconner-2024-hepg2)"
    "nadig-oconner-2024-jurkat=$(canonical_path nadig-oconner-2024-jurkat)"
  )
else
  EXPERIMENT="sciplex3-nine-drug-context-transfer"
  INPUTS=(
    "srivatsan-trapnell-2020-sciplex3=$(canonical_path srivatsan-trapnell-2020-sciplex3)"
  )
fi

ARGS=()
for input in "${INPUTS[@]}"; do
  ARGS+=(--input "${input}")
done

echo "[$(date -Is)] Preparing ${EXPERIMENT}"
"${UV_BIN}" run --frozen python -m models.data.prepare_context \
  --config "configs/experiments/${EXPERIMENT}.json" \
  "${ARGS[@]}" \
  --output "models/data/${EXPERIMENT}/raw.h5ad" \
  --counts-output "models/data/${EXPERIMENT}/counts.h5ad"
echo "[$(date -Is)] Context model view complete"
