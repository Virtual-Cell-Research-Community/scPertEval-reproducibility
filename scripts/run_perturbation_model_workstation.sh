#!/usr/bin/env bash
# Run one implemented model's frozen essential-screen folds on a persistent workstation GPU.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <model> [adapter arguments ...]" >&2
    echo "models: gears presage scgpt sclambda tabicl txpert-public genepert fmlp-genept fmlp-esm2 cellflow" >&2
    exit 2
fi

MODEL="$1"
shift
case "${MODEL}" in
    gears|presage|scgpt|sclambda|tabicl|txpert-public|genepert|fmlp-genept|fmlp-esm2|cellflow) ;;
    *) echo "unsupported model: ${MODEL}" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"
DATASET="${DATASET:-replogle-weissman-2022-k562-essential}"
PIXI_BIN="${PIXI_BIN:-${HOME}/.pixi/bin/pixi}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/home/pschaefer/sds-hd/sd22b002/projects/scPertEval-reproducibility}"
if [[ -d "${SHARED_PROJECT_ROOT}" ]]; then
    DEFAULT_RUNS_ROOT="${SHARED_PROJECT_ROOT}/runs"
    DEFAULT_LOG_ROOT="${SHARED_PROJECT_ROOT}/logs/perturbation-production-workstation"
else
    DEFAULT_RUNS_ROOT="${PROJECT_ROOT}/runs"
    DEFAULT_LOG_ROOT="${PROJECT_ROOT}/logs/perturbation-production-workstation"
fi
RUNS_ROOT="${RUNS_ROOT:-${DEFAULT_RUNS_ROOT}}"
LOG_ROOT="${LOG_ROOT:-${DEFAULT_LOG_ROOT}}"
GPU="${GPU:-0}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SEEDS="${SEEDS:-42}"
FORCE_INCOMPLETE="${FORCE_INCOMPLETE:-0}"
NVIDIA_DRIVER_LIBRARY_PATH="${NVIDIA_DRIVER_LIBRARY_PATH:-}"

# NixOS keeps the host NVIDIA driver outside ordinary shells. Pixi's CUDA
# libraries still need the host libcuda.so in their runtime search path.
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
if ! [[ "${GPU}" =~ ^[0-9]+$ ]]; then
    echo "GPU must be a non-negative integer." >&2
    exit 2
fi
if [[ "${FORCE_INCOMPLETE}" != "0" && "${FORCE_INCOMPLETE}" != "1" ]]; then
    echo "FORCE_INCOMPLETE must be 0 or 1." >&2
    exit 2
fi

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
cd "${MODELS_DIR}"
echo "[$(date -Is)] host=$(hostname) dataset=${DATASET} model=${MODEL} runs_root=${RUNS_ROOT} logs_root=${LOG_ROOT}"

read -ra FOLD_VALUES <<< "${FOLDS}"
if [[ "${MODEL}" == "tabicl" || "${MODEL}" == "genepert" ]]; then
    SEED_VALUES=(42)
else
    read -ra SEED_VALUES <<< "${SEEDS}"
fi
if [[ ${#FOLD_VALUES[@]} -eq 0 || ${#SEED_VALUES[@]} -eq 0 ]]; then
    echo "FOLDS and SEEDS must not be empty." >&2
    exit 2
fi

STATUS_LOG="${LOG_ROOT}/${MODEL}-status.tsv"
if [[ ! -f "${STATUS_LOG}" ]]; then
    printf 'timestamp\tmodel\tfold\tseed\tstatus\texit_code\n' >"${STATUS_LOG}"
fi

ENVIRONMENT="${MODEL}"
ADAPTER="${MODEL}/train_predict.py"
MODEL_ARGS=()
case "${MODEL}" in
    txpert-public)
        ADAPTER="txpert_public/train_predict.py"
        ;;
    fmlp-genept|fmlp-esm2)
        ENVIRONMENT="fmlp"
        ADAPTER="fmlp/train_predict.py"
        MODEL_ARGS+=(--model "${MODEL}")
        ;;
    cellflow)
        MODEL_ARGS+=(--embedding-path "fmlp/cache/esm2-15b/esm2_t48_15B_UR50D_mean.pickle")
        ;;
esac

for seed in "${SEED_VALUES[@]}"; do
    for fold in "${FOLD_VALUES[@]}"; do
        if ! [[ "${fold}" =~ ^[0-4]$ && "${seed}" =~ ^[0-9]+$ ]]; then
            echo "invalid fold/seed pair: fold=${fold} seed=${seed}" >&2
            exit 2
        fi

        run_dir="${RUNS_ROOT}/${DATASET}/${MODEL}/fold-${fold}/seed-${seed}"
        log_path="${LOG_ROOT}/${MODEL}-fold-${fold}-seed-${seed}.log"
        extra_args=()
        if [[ "${FORCE_INCOMPLETE}" == "1" && -f "${run_dir}/resolved-run.json" && ! -f "${run_dir}/completion.json" ]]; then
            extra_args+=(--force)
        fi

        echo "[$(date -Is)] Starting ${MODEL} fold=${fold} seed=${seed} on GPU ${GPU}" | tee -a "${log_path}"
        set +e
        LD_LIBRARY_PATH="${NVIDIA_DRIVER_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            CUDA_VISIBLE_DEVICES="${GPU}" "${PIXI_BIN}" run -e "${ENVIRONMENT}" \
            python "${ADAPTER}" \
            "${MODEL_ARGS[@]}" \
            --dataset "${DATASET}" \
            --fold "${fold}" \
            --seed "${seed}" \
            --runs-root "${RUNS_ROOT}" \
            "${extra_args[@]}" \
            "$@" \
            >>"${log_path}" 2>&1
        status=$?
        set -e
        if [[ ${status} -eq 0 ]]; then
            state=complete-or-verified-skip
        else
            state=failed
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -Is)" "${MODEL}" "${fold}" "${seed}" "${state}" "${status}" \
            | tee -a "${STATUS_LOG}" "${log_path}"
        if [[ ${status} -ne 0 ]]; then
            exit "${status}"
        fi
    done
done

echo "[$(date -Is)] ${MODEL} workstation sweep finished successfully" | tee -a "${LOG_ROOT}/${MODEL}-sweep.log"
