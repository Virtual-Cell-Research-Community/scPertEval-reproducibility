#!/usr/bin/env bash
#SBATCH --job-name=scperteval-protocol-finalize
#SBATCH --partition=cpu-single
#SBATCH --time=01:00:00
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
MANIFEST="${SCPERTEVAL_PROTOCOL_MANIFEST:-configs/protocol_behavior/production-v1.json}"
OUTPUT="${SCPERTEVAL_PROTOCOL_OUTPUT:-results/protocol_behavior/production-v1}"

module load compiler/gnu/12.1
cd "${PROJECT_ROOT}"
mkdir -p logs "${OUTPUT}"

export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility-protocol-py311"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] rendering and validating production artifacts"
"${UV_BIN}" run --frozen reproduce-protocol-behavior \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT}" \
  --finalize
echo "[$(date -Is)] production workflow complete"
