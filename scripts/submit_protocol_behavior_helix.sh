#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${SCPERTEVAL_PROTOCOL_MANIFEST:-configs/protocol_behavior/production-v1.json}"

if [[ ! -f pyproject.toml ]]; then
  echo "Run this script from the scPertEval-reproducibility repository root." >&2
  exit 2
fi

mkdir -p logs
N_DATASETS="$(python3 -c "import json; print(len(json.load(open('${MANIFEST}'))['datasets']))")"
N_DE_PER_DATASET="$(python3 -c "import json; m=json.load(open('${MANIFEST}')); print(len(m['de']['references']) * len(m['de']['methods']))")"
N_GPU_PROTOCOLS="$(python3 -c "import json; print(len(json.load(open('${MANIFEST}'))['gpu_protocols']))")"
N_CPU="$((N_DATASETS * (N_DE_PER_DATASET + 1)))"
N_GPU="$((N_DATASETS * (N_GPU_PROTOCOLS > 0 ? 1 : 0)))"
if [[ "${N_CPU}" -lt 1 || "${N_GPU}" -lt 1 ]]; then
  echo "The production manifest must contain both CPU and GPU compute stages." >&2
  exit 2
fi

ENV_JOB="$(sbatch --parsable scripts/protocol_behavior_environment_helix.sbatch)"
CPU_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --array="0-$((N_CPU - 1))%4" \
  --export=ALL,STAGE_CLASS=cpu \
  scripts/protocol_behavior_stage_helix.sbatch)"
GPU_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --partition=gpu-single \
  --gres=gpu:1 \
  --time=1-00:00:00 \
  --array="0-$((N_GPU - 1))%2" \
  --export=ALL,STAGE_CLASS=gpu \
  scripts/protocol_behavior_stage_helix.sbatch)"
FINAL_JOB="$(sbatch --parsable \
  --dependency="afterok:${CPU_JOB}:${GPU_JOB}" \
  scripts/protocol_behavior_finalize_helix.sbatch)"

echo "submitted locked-environment job ${ENV_JOB}"
echo "submitted ${N_CPU} resumable CPU stages as array job ${CPU_JOB} after ${ENV_JOB}"
echo "submitted ${N_GPU} resumable GPU Sinkhorn stages as array job ${GPU_JOB} after ${ENV_JOB}"
echo "submitted render/validation job ${FINAL_JOB} after both arrays complete successfully"
