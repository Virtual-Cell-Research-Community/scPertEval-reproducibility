#!/usr/bin/env bash
# Submit through submit_context_models_helix.sh. One model-specific array covers the seven
# leave-one-context-out folds at seed 42 without launching the whole study here.
#SBATCH --job-name=scperteval-context-model
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=192gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
: "${MODEL:?submit with MODEL=<state|cpa|biolord|scdisinfact|trvae|sams-vae-s>}"
case "${MODEL}" in
  state|cpa|biolord|scdisinfact|trvae|sams-vae-s) ;;
  *) echo "Unsupported context model: ${MODEL}" >&2; exit 2 ;;
esac
if [[ ! -f "${PROJECT_ROOT}/models/pixi.lock" ]]; then
  echo "Submit this job from the scPertEval-reproducibility repository root." >&2
  exit 2
fi

SEED=42
TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
if [[ "${TASK_INDEX}" -lt 4 ]]; then
  if [[ "${MODEL}" == "trvae" || "${MODEL}" == "sams-vae-s" ]]; then
    echo "${MODEL} is excluded from the genetic track." >&2
    exit 2
  fi
  EXPERIMENT="replogle-nadig-four-context-50-cell"
  FOLD="${TASK_INDEX}"
else
  EXPERIMENT="sciplex3-nine-drug-context-transfer"
  FOLD="$((TASK_INDEX - 4))"
fi

ENVIRONMENT="${MODEL}"
ADAPTER="${MODEL}/train_predict.py"
if [[ "${MODEL}" == "sams-vae-s" ]]; then
  ADAPTER="sams_vae_s/train_predict.py"
fi

cd "${PROJECT_ROOT}/models"
mkdir -p "${PROJECT_ROOT}/logs"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Some locked model environments require a newer C++ runtime than Helix exposes through
# /lib64.  Prefer the environment's own runtime libraries so compiled packages such as
# pandas do not accidentally bind against the host libstdc++ before Pixi starts Python.
ENVIRONMENT_LIBRARY_PATH="${PROJECT_ROOT}/models/.pixi/envs/${ENVIRONMENT}/lib"
if [[ -d "${ENVIRONMENT_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${ENVIRONMENT_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# Beast and Beauty expose the NVIDIA driver through NixOS' OpenGL-driver path. This is a
# no-op on conventional Helix nodes, but keeps the same launcher usable on all three hosts.
if [[ -d /run/opengl-driver/lib ]]; then
  export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "[$(date -Is)] model=${MODEL} experiment=${EXPERIMENT} fold=${FOLD} seed=${SEED} node=${SLURM_JOB_NODELIST} gpu=${CUDA_VISIBLE_DEVICES}"
pixi run --locked -e "${ENVIRONMENT}" python "${ADAPTER}" \
  --dataset "${EXPERIMENT}" \
  --split-manifest "../configs/experiments/${EXPERIMENT}.json" \
  --fold "${FOLD}" \
  --perturbation-key model_perturbation \
  --seed "${SEED}"
