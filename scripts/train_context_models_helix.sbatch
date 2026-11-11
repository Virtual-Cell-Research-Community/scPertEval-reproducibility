#!/usr/bin/env bash
# Submit through submit_context_models_helix.sh. One model-specific array covers the seven
# leave-one-context-out folds and three frozen seeds without launching the whole study here.
#SBATCH --job-name=scperteval-context-model
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=192gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-20
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
: "${MODEL:?submit with MODEL=<state|cpa|biolord|scdisinfact|trvae>}"
case "${MODEL}" in
  state|cpa|biolord|scdisinfact|trvae) ;;
  *) echo "Unsupported context model: ${MODEL}" >&2; exit 2 ;;
esac
if [[ ! -f "${PROJECT_ROOT}/models/pixi.lock" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi

SEEDS=(42 43 44)
TASKS_PER_SEED=7
SEED_INDEX="$((SLURM_ARRAY_TASK_ID / TASKS_PER_SEED))"
TASK_INDEX="$((SLURM_ARRAY_TASK_ID % TASKS_PER_SEED))"
SEED="${SEEDS[${SEED_INDEX}]}"
if [[ "${TASK_INDEX}" -lt 4 ]]; then
  EXPERIMENT="replogle-nadig-four-context-50-cell"
  FOLD="${TASK_INDEX}"
else
  EXPERIMENT="sciplex3-nine-drug-context-transfer"
  FOLD="$((TASK_INDEX - 4))"
fi

cd "${PROJECT_ROOT}/models"
mkdir -p "${PROJECT_ROOT}/logs"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Beast and Beauty expose the NVIDIA driver through NixOS' OpenGL-driver path. This is a
# no-op on conventional Helix nodes, but keeps the same launcher usable on all three hosts.
if [[ -d /run/opengl-driver/lib ]]; then
  export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "[$(date -Is)] model=${MODEL} experiment=${EXPERIMENT} fold=${FOLD} seed=${SEED} node=${SLURM_JOB_NODELIST} gpu=${CUDA_VISIBLE_DEVICES}"
pixi run --locked -e "${MODEL}" python "${MODEL}/train_predict.py" \
  --dataset "${EXPERIMENT}" \
  --split-manifest "../configs/experiments/${EXPERIMENT}.json" \
  --fold "${FOLD}" \
  --perturbation-key model_perturbation \
  --seed "${SEED}"
