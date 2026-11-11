#!/usr/bin/env bash
# Prepare (but do not invoke from automation) 93 full context fits. Four models cover all
# seven folds and trVAE covers only the three chemical folds. Pass model names to submit a subset.
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
  array="0-20"
  if [[ "${model}" == "trvae" ]]; then
    # TASK_INDEX 4-6 within each seven-task seed block selects the chemical folds.
    array="4-6,11-13,18-20"
  fi
  sbatch \
    --array="${array}" \
    --job-name="scperteval-${model}-context" \
    --export="ALL,MODEL=${model}" \
    "${SCRIPT_DIR}/train_context_models_helix.sbatch"
done
