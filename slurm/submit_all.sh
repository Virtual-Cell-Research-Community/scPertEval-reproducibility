#!/bin/bash
# Submits one 5-fold array job (train.sbatch) per model, all against the same dataset.
#
# Usage: models/slurm/submit_all.sh [dataset] [model ...]
#   models/slurm/submit_all.sh                              # all 6 models, replogle22k562
#   models/slurm/submit_all.sh replogle22k562 gears scgpt    # just these two
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${1:-replogle22k562}"
shift || true
MODELS=("${@:-gears scgpt presage sclambda state tabicl}")
# The default above is a single unsplit string when no args are passed; split it properly.
if [ "${#MODELS[@]}" -eq 1 ]; then
    read -ra MODELS <<< "${MODELS[0]}"
fi

# cd into models/ before submitting: train.sbatch's #SBATCH --output=slurm/logs/... is a
# relative path, resolved against the submission-time working directory — not guaranteed to
# be models/ if this script is invoked from elsewhere.
cd "$MODELS_DIR"
for MODEL in "${MODELS[@]}"; do
    sbatch --job-name="$MODEL" --export=ALL,MODEL="$MODEL",DATASET="$DATASET",MODELS_DIR="$MODELS_DIR" "$SCRIPT_DIR/train.sbatch"
done
