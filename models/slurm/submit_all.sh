#!/bin/bash
# Submit the frozen five-fold, single-seed essential-screen grid for implemented adapters.
#
# Usage: models/slurm/submit_all.sh [--test-only] [--smoke] [--afterok JOB_ID] [dataset] [model ...]
#   models/slurm/submit_all.sh
#   models/slurm/submit_all.sh --test-only
#   models/slurm/submit_all.sh --smoke replogle-weissman-2022-k562-essential gears scgpt
#   models/slurm/submit_all.sh --smoke --afterok 123456
#
# Full arrays default to one concurrent task per model. This bounds simultaneous large SDS
# reads while still allowing the five implemented model families to make progress in parallel. Override
# deliberately with MAX_CONCURRENT=<n> after measuring storage and node pressure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_ONLY=0
SMOKE=0
AFTEROK=""
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --test-only) TEST_ONLY=1 ;;
        --smoke) SMOKE=1 ;;
        --afterok)
            [[ -n "${2:-}" ]] || { echo "--afterok requires a job id" >&2; exit 2; }
            AFTEROK="$2"
            shift
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
DATASET="${1:-replogle-weissman-2022-k562-essential}"
if [[ $# -gt 0 ]]; then shift; fi
MODELS=("${@:-gears scgpt presage sclambda tabicl}")
# The default above is a single unsplit string when no args are passed; split it properly.
if [ "${#MODELS[@]}" -eq 1 ]; then
    read -ra MODELS <<< "${MODELS[0]}"
fi
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONCURRENT must be a positive integer" >&2
    exit 2
fi

# cd into models/ before submitting: train.sbatch's #SBATCH --output=slurm/logs/... is a
# relative path, resolved against the submission-time working directory — not guaranteed to
# be models/ if this script is invoked from elsewhere.
cd "$MODELS_DIR"
mkdir -p slurm/logs
for MODEL in "${MODELS[@]}"; do
    ARRAY="0-4"
    if [[ "$SMOKE" == "1" ]]; then
        ARRAY="0"
    else
        ARRAY="${ARRAY}%${MAX_CONCURRENT}"
    fi
    SBATCH_ARGS=(
        --job-name="$MODEL-$DATASET"
        --array="$ARRAY"
        --export=ALL,MODEL="$MODEL",DATASET="$DATASET",MODELS_DIR="$MODELS_DIR",FOLD_COUNT=5,SEEDS=42,SMOKE="$SMOKE"
    )
    if [[ "$MODEL" == "sclambda" && "$SMOKE" != "1" ]]; then
        SBATCH_ARGS+=(--mem=256G)
    fi
    if [[ "$SMOKE" == "1" ]]; then
        SBATCH_ARGS+=(--cpus-per-task=8 --mem=64G --time=08:00:00)
    fi
    if [[ -n "$AFTEROK" ]]; then
        SBATCH_ARGS+=(--dependency="afterok:$AFTEROK")
    fi
    if [[ "$TEST_ONLY" == "1" ]]; then
        SBATCH_ARGS+=(--test-only)
    fi
    sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/train.sbatch"
done
