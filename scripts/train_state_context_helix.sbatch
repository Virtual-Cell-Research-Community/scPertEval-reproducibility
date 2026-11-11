#!/usr/bin/env bash
#SBATCH --job-name=scperteval-state-context
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ ! -f "${PROJECT_ROOT}/models/pixi.toml" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi

if [[ "${SLURM_ARRAY_TASK_ID}" -lt 4 ]]; then
  EXPERIMENT="replogle-nadig-four-context-50-cell"
  FOLD="${SLURM_ARRAY_TASK_ID}"
else
  EXPERIMENT="sciplex3-nine-drug-context-transfer"
  FOLD="$((SLURM_ARRAY_TASK_ID - 4))"
fi

cd "${PROJECT_ROOT}/models"
mkdir -p "${PROJECT_ROOT}/logs"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[$(date -Is)] experiment=${EXPERIMENT} fold=${FOLD} node=${SLURM_JOB_NODELIST} gpu=${CUDA_VISIBLE_DEVICES}"
pixi run --locked -e state python state/train_predict.py \
  --dataset "${EXPERIMENT}" \
  --split-manifest "../configs/experiments/${EXPERIMENT}.json" \
  --fold "${FOLD}" \
  --perturbation-key model_perturbation
