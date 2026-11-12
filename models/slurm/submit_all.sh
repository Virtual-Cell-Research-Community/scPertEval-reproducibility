#!/bin/bash
# Submit the frozen five-fold x three-seed GWPS grid.
#
# Usage: models/slurm/submit_all.sh [--test-only] [--smoke] [dataset] [model ...]
#   models/slurm/submit_all.sh
#   models/slurm/submit_all.sh --test-only
#   models/slurm/submit_all.sh --smoke replogle-weissman-2022-k562-gwps gears scgpt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_ONLY=0
SMOKE=0
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --test-only) TEST_ONLY=1 ;;
        --smoke) SMOKE=1 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
DATASET="${1:-replogle-weissman-2022-k562-gwps}"
if [[ $# -gt 0 ]]; then shift; fi
MODELS=("${@:-gears scgpt presage sclambda state tabicl}")
# The default above is a single unsplit string when no args are passed; split it properly.
if [ "${#MODELS[@]}" -eq 1 ]; then
    read -ra MODELS <<< "${MODELS[0]}"
fi

# cd into models/ before submitting: train.sbatch's #SBATCH --output=slurm/logs/... is a
# relative path, resolved against the submission-time working directory — not guaranteed to
# be models/ if this script is invoked from elsewhere.
cd "$MODELS_DIR"
mkdir -p slurm/logs
for MODEL in "${MODELS[@]}"; do
    if [[ "$MODEL" == "tabicl" || "$SMOKE" == "1" ]]; then
        ARRAY="0-4"
    else
        ARRAY="0-14"
    fi
    if [[ "$SMOKE" == "1" ]]; then
        ARRAY="0"
    fi
    SBATCH_ARGS=(
        --job-name="$MODEL-$DATASET"
        --array="$ARRAY"
        --export=ALL,MODEL="$MODEL",DATASET="$DATASET",MODELS_DIR="$MODELS_DIR",FOLD_COUNT=5,SEEDS=42:43:44,SMOKE="$SMOKE"
    )
    if [[ "$TEST_ONLY" == "1" ]]; then
        SBATCH_ARGS+=(--test-only)
    fi
    sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/train.sbatch"
done
