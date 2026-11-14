#!/bin/bash
# Generic per-model, per-fold training job. submit_all.sh supplies one five-task array
# at seed 42. It also sets MODEL/DATASET through --export; this is not meant to be submitted
# directly without those values.
#
# Every models/*/train_predict.py already exposes --dataset/--fold with full-scale defaults
# (epochs/max_steps/batch_size etc. — see each script's own docstring for how those defaults
# were chosen), so this script is deliberately just "cd into models/, pixi run the right env" —
# no per-model resource/hyperparameter branching here.
#
# GPU type is deliberately NOT pinned (--gres=gpu:1, any type) rather than e.g. h200 only:
# only 12 h200s exist cluster-wide (codon-gpu-001 x8, codon-gpu-003/hl-codon-gpu-020 x4) vs.
# ~48 a100 + 8 l40s — pinning to h200 would cap us at 12 concurrent jobs out of 30 (6 models x
# 5 folds) for no benefit that matters here. Trade-off: training-time/GPU-hours are no longer
# strictly apples-to-apples across folds that land on different GPU generations — accepted
# deliberately for throughput; each benchmark_fold*.json (once added) should still record
# which GPU it ran on so this is auditable after the fact, not silently lost.
#
# V100 was previously excluded here (--exclude=codon-gpu-[017-019]): confirmed via
# slurm/check_gpu_compat.sbatch that flash-attn 2.x kernels launch fine on l40s/a100/h200
# (compute capability 8.9/8.0/9.0), but couldn't confirm V100 at all, and flash-attn 2.x is
# known to have dropped Volta (compute capability 7.0) support, so scgpt would likely
# hard-crash there. Dropped as of 2026-07-17: those three nodes are no longer in `sinfo`'s
# node list at all (fully decommissioned, not just drained/down as they were when this was
# written), and `sinfo -N | grep -i v100` now returns nothing cluster-wide — sbatch rejects
# --exclude naming nodes that don't exist ("Invalid node name specified"), and there's no
# longer any V100 node for the exclusion to matter for. If V100 nodes ever reappear under a
# different name, re-add an --exclude for them.
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
# Retain a conservative request until every essential-screen adapter has a measured peak.
# 192 GiB leaves room for adapter-specific copies and matches the validated context launcher.
# Sixteen CPUs avoid monopolizing an entire GPU node for preprocessing that is predominantly
# I/O- and GPU-bound.
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_task%a_%A.out
#SBATCH --error=slurm/logs/%x_task%a_%A.err

set -euo pipefail

: "${MODEL:?set via sbatch --export=ALL,MODEL=<gears|scgpt|presage|sclambda|tabicl>}"
DATASET="${DATASET:-replogle-weissman-2022-k562-essential}"
FOLD_COUNT="${FOLD_COUNT:-5}"
SEEDS="${SEEDS:-42}"
# SLURM copies the submitted script to a spool dir before running it, so a
# ${BASH_SOURCE[0]}-relative path resolves to the wrong place (and $SLURM_SUBMIT_DIR depends
# on the submitting shell's cwd, which callers of this script aren't guaranteed to control) —
# MODELS_DIR is set explicitly at submission time instead (see submit_all.sh).
: "${MODELS_DIR:?set via sbatch --export=ALL,MODELS_DIR=/abs/path/to/models}"
cd "$MODELS_DIR"

if [[ "$MODEL" == "tabicl" ]]; then
    FOLD="$SLURM_ARRAY_TASK_ID"
    SEED=42
else
    IFS=: read -ra SEED_VALUES <<< "$SEEDS"
    FOLD=$((SLURM_ARRAY_TASK_ID % FOLD_COUNT))
    SEED_INDEX=$((SLURM_ARRAY_TASK_ID / FOLD_COUNT))
    SEED="${SEED_VALUES[$SEED_INDEX]:?array index has no corresponding training seed}"
fi

EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA_ARGS=(--runs-root "${MODELS_DIR}/../runs-smoke")
    case "$MODEL" in
        gears) EXTRA_ARGS+=(--epochs 2) ;;
        presage) EXTRA_ARGS+=(--epochs 2 --batch-size 4) ;;
        scgpt) EXTRA_ARGS+=(--epochs 2) ;;
        sclambda) EXTRA_ARGS+=(--epochs 10) ;;
        tabicl) ;;
    esac
fi

echo "model=$MODEL dataset=$DATASET fold=$FOLD seed=$SEED smoke=${SMOKE:-0} node=$SLURM_JOB_NODELIST gpu=$CUDA_VISIBLE_DEVICES"
pixi run -e "$MODEL" python "$MODEL/train_predict.py" \
    --dataset "$DATASET" \
    --fold "$FOLD" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}"
