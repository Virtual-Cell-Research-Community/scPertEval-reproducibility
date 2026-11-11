#!/usr/bin/env bash
# Prepare (but do not invoke from automation) the five model arrays for the 105 full context
# fits: 5 models x 7 context folds x 3 seeds. Pass model names to submit a subset.
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
  sbatch \
    --job-name="scperteval-${model}-context" \
    --export="ALL,MODEL=${model}" \
    "${SCRIPT_DIR}/train_context_models_helix.sbatch"
done
