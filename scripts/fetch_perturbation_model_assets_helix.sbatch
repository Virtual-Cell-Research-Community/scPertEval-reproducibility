#!/usr/bin/env bash
# Fetch the two external asset bundles not already present on the shared SDS checkout.
# Task 0: scGPT pretrained checkpoint. Task 1: PRESAGE/TabICL embedding cache.
# Re-running is safe because both Pixi tasks declare their expected outputs.
#SBATCH --job-name=scperteval-assets
#SBATCH --partition=cpu-single
#SBATCH --time=04:00:00
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
PIXI_BIN="${PIXI_BIN:-${HOME}/.pixi/bin/pixi}"
if [[ ! -f "${PROJECT_ROOT}/models/pixi.toml" ]]; then
    echo "Submit this job from the scPertEval-reproducibility repository root." >&2
    exit 2
fi
if [[ ! -x "${PIXI_BIN}" ]]; then
    echo "Pixi is not executable at ${PIXI_BIN}." >&2
    exit 127
fi

cd "${PROJECT_ROOT}/models"
export CONDA_OVERRIDE_CUDA=12
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${TMPDIR:-/tmp}/pixi-cache-${USER}}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) "${PIXI_BIN}" run -e scgpt fetch-checkpoint ;;
    1) "${PIXI_BIN}" run -e presage fetch-cache ;;
    *) echo "Unknown asset task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
