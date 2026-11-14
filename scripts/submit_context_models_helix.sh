#!/usr/bin/env bash
# Submit the 34 implemented single-seed context fits: four models across four genetic
# folds, and those four plus trVAE and SAMS-VAE(S) across three chemical folds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
  fi
  sbatch \
    --array="${array}" \
    --job-name="scperteval-${model}-context" \
    --export="ALL,MODEL=${model}" \
    "${SCRIPT_DIR}/train_context_models_helix.sbatch"
done
