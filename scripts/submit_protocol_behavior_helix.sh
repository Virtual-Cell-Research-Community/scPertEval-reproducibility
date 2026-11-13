#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
MANIFEST="${SCPERTEVAL_PROTOCOL_MANIFEST:-configs/protocol_behavior/production-v1.json}"

if [[ ! -f pyproject.toml ]]; then
  echo "Run this script from the scPertEval-reproducibility repository root." >&2
  exit 2
fi

mkdir -p logs
N_CPU="$(${UV_BIN} run --frozen reproduce-protocol-behavior --manifest "${MANIFEST}" --list-stages --stage-class cpu | wc -l)"
N_GPU="$(${UV_BIN} run --frozen reproduce-protocol-behavior --manifest "${MANIFEST}" --list-stages --stage-class gpu | wc -l)"
if [[ "${N_CPU}" -lt 1 || "${N_GPU}" -lt 1 ]]; then
  echo "The production manifest must contain both CPU and GPU compute stages." >&2
  exit 2
fi

CPU_JOB="$(sbatch --parsable \
  --array="0-$((N_CPU - 1))%4" \
  --export=ALL,STAGE_CLASS=cpu \
  scripts/protocol_behavior_stage_helix.sbatch)"
GPU_JOB="$(sbatch --parsable \
  --partition=gpu-single \
  --gres=gpu:1 \
  --time=1-00:00:00 \
  --array="0-$((N_GPU - 1))%2" \
  --export=ALL,STAGE_CLASS=gpu \
  scripts/protocol_behavior_stage_helix.sbatch)"
FINAL_JOB="$(sbatch --parsable \
  --dependency="afterok:${CPU_JOB}:${GPU_JOB}" \
  scripts/protocol_behavior_finalize_helix.sbatch)"

echo "submitted ${N_CPU} resumable CPU stages as array job ${CPU_JOB}"
echo "submitted ${N_GPU} resumable GPU Sinkhorn stages as array job ${GPU_JOB}"
echo "submitted render/validation job ${FINAL_JOB} after both arrays complete successfully"
