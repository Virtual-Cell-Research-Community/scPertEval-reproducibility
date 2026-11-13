#!/usr/bin/env bash
#SBATCH --job-name=scperteval-protocol
#SBATCH --partition=cpu-single
#SBATCH --time=08:00:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
MANIFEST="${SCPERTEVAL_PROTOCOL_MANIFEST:-configs/protocol_behavior/production-v1.json}"
OUTPUT="${SCPERTEVAL_PROTOCOL_OUTPUT:-results/protocol_behavior/production-v1}"
STAGE_CLASS="${STAGE_CLASS:-cpu}"

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv is not executable at ${UV_BIN}" >&2
  exit 127
fi

module load compiler/gnu/12.1
cd "${PROJECT_ROOT}"
mkdir -p logs "${OUTPUT}"

export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility-protocol-py311"
export SCPERTEVAL_DATA_ROOT="${SCPERTEVAL_DATA_ROOT:-/mnt/sds-hd/sd22b002/projects/scPertEval-reproducibility}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

STAGE="$(${UV_BIN} run --frozen reproduce-protocol-behavior \
  --manifest "${MANIFEST}" \
  --list-stages \
  --stage-class "${STAGE_CLASS}" \
  | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
if [[ -z "${STAGE}" ]]; then
  echo "No production stage exists for array index ${SLURM_ARRAY_TASK_ID}." >&2
  exit 2
fi

echo "[$(date -Is)] node=${SLURM_JOB_NODELIST} stage_class=${STAGE_CLASS} stage=${STAGE} gpu=${CUDA_VISIBLE_DEVICES:-none}"
"${UV_BIN}" run --frozen reproduce-protocol-behavior \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT}" \
  --stage "${STAGE}" \
  --workers "${SLURM_CPUS_PER_TASK:-16}"
echo "[$(date -Is)] completed stage=${STAGE}"
