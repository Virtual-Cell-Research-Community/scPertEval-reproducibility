#!/usr/bin/env bash
# Wait for a persistent tmux canary to terminate. Release folds 1--4 only when
# that canary exited successfully; otherwise leave the model family paused.

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <canary-tmux-session> <model> <gpu-index>" >&2
    exit 2
fi

CANARY_SESSION="$1"
MODEL="$2"
GPU="$3"
POLL_SECONDS="${POLL_SECONDS:-30}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! [[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "POLL_SECONDS must be a positive integer" >&2
    exit 2
fi
if ! tmux has-session -t "${CANARY_SESSION}" 2>/dev/null; then
    echo "Canary tmux session does not exist: ${CANARY_SESSION}" >&2
    exit 2
fi

echo "[$(date -Is)] waiting for canary session ${CANARY_SESSION}"
while [[ "$(tmux display-message -p -t "${CANARY_SESSION}" '#{pane_dead}')" != "1" ]]; do
    sleep "${POLL_SECONDS}"
done

canary_status="$(tmux display-message -p -t "${CANARY_SESSION}" '#{pane_dead_status}')"
if [[ "${canary_status}" != "0" ]]; then
    echo "[$(date -Is)] canary failed with exit status ${canary_status}; ${MODEL} remains paused" >&2
    exit "${canary_status}"
fi

echo "[$(date -Is)] canary passed; releasing ${MODEL} folds 1--4 on GPU ${GPU}"
cd "${PROJECT_ROOT}"
GPU="${GPU}" FOLDS="1 2 3 4" SEEDS=42 \
    scripts/run_perturbation_model_workstation.sh "${MODEL}"
