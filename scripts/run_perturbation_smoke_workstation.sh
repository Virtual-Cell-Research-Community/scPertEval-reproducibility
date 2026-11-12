#!/usr/bin/env bash
# Run the one-fold GWPS vertical smoke pass on a two-GPU workstation.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"
DATASET="${DATASET:-replogle-weissman-2022-k562-gwps}"
PIXI_BIN="${PIXI_BIN:-${HOME}/.pixi/bin/pixi}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs-smoke}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/perturbation-smoke-workstation}"

if [[ ! -x "${PIXI_BIN}" ]]; then
    echo "Pixi is not executable at ${PIXI_BIN}." >&2
    exit 127
fi
if [[ ! -f "${MODELS_DIR}/data/${DATASET}/raw.h5ad" ]]; then
    echo "Prepared model view is missing for ${DATASET}." >&2
    exit 2
fi

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
cd "${MODELS_DIR}"

echo "[$(date -Is)] Fetching/checking scGPT and PRESAGE/TabICL assets"
"${PIXI_BIN}" run -e scgpt fetch-checkpoint || exit $?
"${PIXI_BIN}" run -e presage fetch-cache || exit $?

run_model() {
    local gpu="$1"
    local model="$2"
    shift 2
    echo "[$(date -Is)] Starting ${model} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PIXI_BIN}" run -e "${model}" \
        python "${model}/train_predict.py" \
        --dataset "${DATASET}" \
        --fold 0 \
        --seed 42 \
        --runs-root "${RUNS_ROOT}" \
        "$@" \
        >"${LOG_ROOT}/${model}.out" \
        2>"${LOG_ROOT}/${model}.err"
    local status=$?
    echo "[$(date -Is)] ${model} exited with status ${status}"
    return "${status}"
}

run_group_zero() {
    local status=0
    run_model 0 gears --epochs 2 || status=1
    run_model 0 presage --epochs 2 --batch-size 4 || status=1
    run_model 0 sclambda --epochs 10 || status=1
    return "${status}"
}

run_group_one() {
    local status=0
    run_model 1 scgpt --epochs 2 || status=1
    run_model 1 state --max-steps 20 --val-freq 10 --batch-size 4 --num-workers 0 || status=1
    run_model 1 tabicl || status=1
    return "${status}"
}

run_group_zero &
PID_ZERO=$!
run_group_one &
PID_ONE=$!

STATUS=0
wait "${PID_ZERO}" || STATUS=1
wait "${PID_ONE}" || STATUS=1
echo "[$(date -Is)] GWPS vertical smoke pass completed with status ${STATUS}"
exit "${STATUS}"
