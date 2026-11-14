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
FIRST_DATASET="$(python3 -c "import json; print(json.load(open('${MANIFEST}'))['datasets'][0]['id'])")"
N_DE="$((N_DATASETS * N_DE_PER_DATASET))"
N_STANDARD_DRF="$((N_DATASETS - 1))"
N_GPU="$((N_DATASETS * (N_GPU_PROTOCOLS > 0 ? 1 : 0)))"
if [[ "${N_DE}" -lt 1 || "${N_STANDARD_DRF}" -lt 1 || "${N_GPU}" -lt 1 ]]; then
  echo "The production manifest must contain both CPU and GPU compute stages." >&2
  exit 2
fi
if [[ "${FIRST_DATASET}" != "gwps-k562" ]]; then
  echo "The Helix resource split expects gwps-k562 to be the first manifest dataset." >&2
  exit 2
fi

ENV_JOB="$(sbatch --parsable scripts/protocol_behavior_environment_helix.sbatch)"
DE_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --array="0-$((N_DE - 1))%4" \
  --export=ALL,STAGE_CLASS=de \
  scripts/protocol_behavior_stage_helix.sbatch)"
STANDARD_DRF_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --array="1-$((N_DATASETS - 1))%4" \
  --mem=64gb \
  --time=01:00:00 \
  --export=ALL,STAGE_CLASS=drf-cpu \
  scripts/protocol_behavior_stage_helix.sbatch)"
GWPS_DRF_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --array=0 \
  --export=ALL,STAGE_CLASS=drf-cpu \
  scripts/protocol_behavior_stage_helix.sbatch)"
GPU_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --partition=gpu-single \
  --gres=gpu:1 \
  --time=1-00:00:00 \
  --array="0-$((N_GPU - 1))%2" \
  --export=ALL,STAGE_CLASS=drf-gpu \
  scripts/protocol_behavior_stage_helix.sbatch)"
FINAL_JOB="$(sbatch --parsable \
  --dependency="afterok:${DE_JOB}:${STANDARD_DRF_JOB}:${GWPS_DRF_JOB}:${GPU_JOB}" \
  scripts/protocol_behavior_finalize_helix.sbatch)"

echo "submitted locked-environment job ${ENV_JOB}"
echo "submitted ${N_DE} resumable DE stages as array job ${DE_JOB} after ${ENV_JOB}"
echo "submitted ${N_STANDARD_DRF} standard CPU DRF stages as array job ${STANDARD_DRF_JOB} after ${ENV_JOB}"
echo "submitted GWPS CPU DRF stage as array job ${GWPS_DRF_JOB} after ${ENV_JOB}"
echo "submitted ${N_GPU} resumable GPU Sinkhorn stages as array job ${GPU_JOB} after ${ENV_JOB}"
echo "submitted render/validation job ${FINAL_JOB} after all compute arrays complete successfully"
