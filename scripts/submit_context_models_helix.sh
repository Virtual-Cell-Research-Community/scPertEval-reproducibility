#!/usr/bin/env bash
# Prepare (but do not invoke from automation) the 31 currently implemented single-seed
# context fits. SAMS-VAE(S) adds three chemical fits after its pending adapter is implemented.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS=("${@:-state cpa biolord scdisinfact trvae}")
if [[ "${#MODELS[@]}" -eq 1 ]]; then
  read -ra MODELS <<< "${MODELS[0]}"
fi

cd "${PROJECT_ROOT}"
for model in "${MODELS[@]}"; do
  case "${model}" in
    state|cpa|biolord|scdisinfact|trvae) ;;
    *) echo "Unsupported context model: ${model}" >&2; exit 2 ;;
  esac
  array="0-6"
  if [[ "${model}" == "trvae" ]]; then
    array="4-6"
  fi
  sbatch \
    --array="${array}" \
    --job-name="scperteval-${model}-context" \
    --export="ALL,MODEL=${model}" \
    "${SCRIPT_DIR}/train_context_models_helix.sbatch"
done
