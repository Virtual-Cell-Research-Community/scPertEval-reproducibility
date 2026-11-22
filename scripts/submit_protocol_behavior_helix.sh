#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${SCPERTEVAL_PROTOCOL_MANIFEST:-configs/protocol_behavior/production-v2.json}"

if [[ ! -f pyproject.toml ]]; then
  echo "Run this script from the scPertEval-reproducibility repository root." >&2
  exit 2
fi

mkdir -p logs
OUTPUT="${SCPERTEVAL_PROTOCOL_OUTPUT:-$(python3 -c "import json; m=json.load(open('${MANIFEST}')); print(m.get('output_dir', 'results/protocol_behavior/' + m['id'].removeprefix('protocol-behavior-')))")}"
STATUS="$(python3 -c "import json; print(json.load(open('${MANIFEST}')).get('status', 'ready'))")"
if [[ "${STATUS}" == planned-pending-* ]]; then
  echo "Refusing to submit ${MANIFEST}: status is ${STATUS}. Update the reviewed manifest before launch." >&2
  exit 2
fi
N_DATASETS="$(python3 -c "import json; print(len(json.load(open('${MANIFEST}'))['datasets']))")"
N_DE_PER_DATASET="$(python3 -c "import json; m=json.load(open('${MANIFEST}')); print(len(m['de']['references']) * len(m['de']['methods']))")"
N_GPU_PROTOCOLS="$(python3 -c "import json; print(len(json.load(open('${MANIFEST}'))['gpu_protocols']))")"
N_DE="$((N_DATASETS * N_DE_PER_DATASET))"
N_STANDARD_DRF="$((N_DATASETS - 1))"
N_GPU="$((N_DATASETS * (N_GPU_PROTOCOLS > 0 ? 1 : 0)))"
if [[ "${N_DE}" -lt 1 || "${N_STANDARD_DRF}" -lt 1 || "${N_GPU}" -lt 1 ]]; then
  echo "The production manifest must contain both CPU and GPU compute stages." >&2
  exit 2
fi
HIGH_MEMORY_STAGE="$(python3 -c "import json; m=json.load(open('${MANIFEST}')); print(m.get('high_memory_cpu_drf_stage', 'drf.cpu.' + m['datasets'][0]['id']))")"
COMMON_EXPORT="ALL,SCPERTEVAL_PROTOCOL_MANIFEST=${MANIFEST},SCPERTEVAL_PROTOCOL_OUTPUT=${OUTPUT}"

ENV_JOB="$(sbatch --parsable \
  --export="${COMMON_EXPORT}" \
  scripts/protocol_behavior_environment_helix.sbatch)"
DE_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --array="0-$((N_DE - 1))%4" \
  --export="${COMMON_EXPORT},STAGE_CLASS=de" \
  scripts/protocol_behavior_stage_helix.sbatch)"
STANDARD_DRF_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --array="0-$((N_STANDARD_DRF - 1))%4" \
  --mem=64gb \
  --time=01:00:00 \
  --export="${COMMON_EXPORT},STAGE_CLASS=drf-cpu,EXCLUDE_STAGE=${HIGH_MEMORY_STAGE}" \
  scripts/protocol_behavior_stage_helix.sbatch)"
HIGH_MEMORY_DRF_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --array=0 \
  --export="${COMMON_EXPORT},STAGE_ID=${HIGH_MEMORY_STAGE}" \
  scripts/protocol_behavior_stage_helix.sbatch)"
GPU_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}" \
  --partition=gpu-single \
  --gres=gpu:1 \
  --time=1-00:00:00 \
  --array="0-$((N_GPU - 1))%2" \
  --export="${COMMON_EXPORT},STAGE_CLASS=drf-gpu" \
  scripts/protocol_behavior_stage_helix.sbatch)"
FINAL_JOB="$(sbatch --parsable \
  --dependency="afterok:${DE_JOB}:${STANDARD_DRF_JOB}:${HIGH_MEMORY_DRF_JOB}:${GPU_JOB}" \
  --export="${COMMON_EXPORT}" \
  scripts/protocol_behavior_finalize_helix.sbatch)"

echo "submitted locked-environment job ${ENV_JOB}"
echo "submitted ${N_DE} resumable DE stages as array job ${DE_JOB} after ${ENV_JOB}"
echo "submitted ${N_STANDARD_DRF} standard CPU DRF stages as array job ${STANDARD_DRF_JOB} after ${ENV_JOB}"
echo "submitted high-memory CPU DRF stage ${HIGH_MEMORY_STAGE} as array job ${HIGH_MEMORY_DRF_JOB} after ${ENV_JOB}"
echo "submitted ${N_GPU} resumable GPU Sinkhorn stages as array job ${GPU_JOB} after ${ENV_JOB}"
echo "submitted render/validation job ${FINAL_JOB} after all compute arrays complete successfully"
