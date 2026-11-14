#!/usr/bin/env bash
# Submit the 34 implemented single-seed context fits: four models across four genetic
# folds, and those four plus trVAE and SAMS-VAE(S) across three chemical folds.
# --canary submits the first genetic and chemical fold for shared models and the first
# chemical fold for chemical-only models. --test-only asks Slurm to validate without
# creating jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_ONLY=0
CANARY=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --test-only) TEST_ONLY=1 ;;
    --canary) CANARY=1 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
MODELS=("${@:-state cpa biolord scdisinfact trvae sams-vae-s}")
if [[ "${#MODELS[@]}" -eq 1 ]]; then
  read -ra MODELS <<< "${MODELS[0]}"
fi

cd "${PROJECT_ROOT}"
for model in "${MODELS[@]}"; do
  case "${model}" in
    state|cpa|biolord|scdisinfact|trvae|sams-vae-s) ;;
    *) echo "Unsupported context model: ${model}" >&2; exit 2 ;;
  esac
  array="0-6"
  if [[ "${model}" == "trvae" || "${model}" == "sams-vae-s" ]]; then
    array="4-6"
    if [[ "${CANARY}" == "1" ]]; then
      array="4"
    fi
  elif [[ "${CANARY}" == "1" ]]; then
    array="0,4"
  fi
  sbatch_args=(
    --array="${array}"
    --job-name="scperteval-${model}-context"
    --export="ALL,MODEL=${model}"
  )
  if [[ "${TEST_ONLY}" == "1" ]]; then
    sbatch_args+=(--test-only)
  fi
  sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/train_context_models_helix.sbatch"
done
