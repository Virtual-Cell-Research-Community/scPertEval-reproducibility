#!/usr/bin/env bash
# Run the one-fold GWPS vertical smoke pass on a two-GPU workstation.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"
DATASET="${DATASET:-replogle-weissman-2022-k562-gwps}"
PIXI_BIN="${PIXI_BIN:-${HOME}/.pixi/bin/pixi}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs-smoke}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/perturbation-smoke-workstation}"
GPU_ZERO="${GPU_ZERO:-0}"
GPU_ONE="${GPU_ONE:-1}"
SERIAL="${SERIAL:-0}"
FETCH_ASSETS="${FETCH_ASSETS:-1}"
NVIDIA_DRIVER_LIBRARY_PATH="${NVIDIA_DRIVER_LIBRARY_PATH:-}"

# NixOS exposes the host NVIDIA driver here, but does not add it to the
# library search path of ordinary shells.  Conda/Pixi CUDA runtimes still
# need the host's libcuda.so in order to initialize a GPU.
if [[ -z "${NVIDIA_DRIVER_LIBRARY_PATH}" && -d /run/opengl-driver/lib ]]; then
    NVIDIA_DRIVER_LIBRARY_PATH=/run/opengl-driver/lib
fi

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

if [[ "${FETCH_ASSETS}" == "1" ]]; then
    echo "[$(date -Is)] Fetching/checking scGPT and PRESAGE/TabICL assets"
    "${PIXI_BIN}" run -e scgpt fetch-checkpoint || exit $?
    "${PIXI_BIN}" run -e presage fetch-cache || exit $?
else
    echo "[$(date -Is)] Reusing preverified model assets"
fi

run_model() {
    local gpu="$1"
    local model="$2"
    shift 2
    echo "[$(date -Is)] Starting ${model} on GPU ${gpu}"
    LD_LIBRARY_PATH="${NVIDIA_DRIVER_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
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
    run_model "${GPU_ZERO}" gears --epochs 2 || status=1
    run_model "${GPU_ZERO}" presage --epochs 2 --batch-size 4 || status=1
    run_model "${GPU_ZERO}" sclambda --epochs 10 || status=1
    return "${status}"
}

run_group_one() {
    local status=0
    run_model "${GPU_ONE}" scgpt --epochs 2 || status=1
    run_model "${GPU_ONE}" state --max-steps 20 --val-freq 10 --batch-size 4 --num-workers 0 || status=1
    run_model "${GPU_ONE}" tabicl || status=1
    return "${status}"
}

STATUS=0
if [[ "${SERIAL}" == "1" ]]; then
    run_group_zero || STATUS=1
    run_group_one || STATUS=1
else
    run_group_zero &
    PID_ZERO=$!
    run_group_one &
    PID_ONE=$!
    wait "${PID_ZERO}" || STATUS=1
    wait "${PID_ONE}" || STATUS=1
fi
echo "[$(date -Is)] GWPS vertical smoke pass completed with status ${STATUS}"
exit "${STATUS}"
