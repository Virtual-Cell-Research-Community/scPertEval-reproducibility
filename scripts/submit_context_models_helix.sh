#!/usr/bin/env bash
# Submit the 34 implemented single-seed context fits: four models across four genetic
# folds, and those four plus trVAE and SAMS-VAE(S) across three chemical folds.
# --canary submits the first genetic and chemical fold for shared models and the first
# chemical fold for chemical-only models. --test-only asks Slurm to validate without
# creating jobs. --staged submits those canaries plus family-local afterok arrays.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_ONLY=0
STAGED=0
CANARY=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --test-only) TEST_ONLY=1 ;;
    --staged) STAGED=1 ;;
    --canary) CANARY=1 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ "${STAGED}" == "1" && ("${TEST_ONLY}" == "1" || "${CANARY}" == "1") ]]; then
  echo "--staged cannot be combined with --test-only or --canary" >&2
  exit 2
fi
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
  canary_array="0,4"
  if [[ "${model}" == "trvae" || "${model}" == "sams-vae-s" ]]; then
    array="4-6"
    canary_array="4"
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
    --time=24:00:00
  )
  case "${model}" in
    state|cpa) sbatch_args+=(--mem=64G) ;;
    biolord|trvae|sams-vae-s) sbatch_args+=(--mem=96G) ;;
    scdisinfact) sbatch_args+=(--mem=192G) ;;
  esac
  if [[ "${TEST_ONLY}" == "1" ]]; then
    sbatch_args+=(--test-only)
  fi
  if [[ "${STAGED}" == "1" ]]; then
    canary_args=("${sbatch_args[@]}")
    for index in "${!canary_args[@]}"; do
      if [[ "${canary_args[$index]}" == --array=* ]]; then
        canary_args[$index]="--array=${canary_array}"
      fi
    done
    canary_id="$(sbatch --parsable "${canary_args[@]}" "${SCRIPT_DIR}/train_context_models_helix.sbatch")"
    canary_id="${canary_id%%;*}"
    echo "submitted ${model} context canary ${canary_id}"
    sbatch \
      --dependency="afterok:${canary_id}" \
      --kill-on-invalid-dep=yes \
      "${sbatch_args[@]}" \
      "${SCRIPT_DIR}/train_context_models_helix.sbatch"
  else
    sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/train_context_models_helix.sbatch"
  fi
done
