#!/bin/bash
# Computes every no-training baseline (no_change, mean, linear, interpolated) against every fold,
# one process:
# baselines/baselines.py's own CLI already loops baselines/folds internally and reads raw.h5ad
# once, so this is a single non-array CPU job -- not a per-model, per-fold array like
# train.sbatch (there's no separate model to train per fold here).
#
# No GPU needed: baselines.py is pure AnnData/numpy/sklearn (PCA + ridge regression), same
# "runs fine from any model's environment" contract as data/prepare_data.py/prepare_split.py --
# run via the `gears` env below for no reason beyond it being the first one defined. Every
# feature in pixi.toml declares system-requirements.cuda = "12" though (even ones with no
# actual GPU dependency, like gears' own pytorch_geometric-only stack), so pixi refuses to
# resolve/install it on a CPU-only node ("virtual package '__cuda' does not match") without
# CONDA_OVERRIDE_CUDA mocking that virtual package -- as pixi's own error suggests.
#
# Usage: sbatch --export=ALL,MODELS_DIR="$(pwd)" slurm/baselines.sbatch
#   (or, for a non-default dataset: sbatch --export=ALL,MODELS_DIR="$(pwd)",DATASET=smoke_k562 slurm/baselines.sbatch)
#SBATCH --account=dugourd
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=baselines
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

DATASET="${DATASET:-replogle22k562}"
# Same spool-dir caveat as train.sbatch: SLURM copies the submitted script to a spool dir
# before running it, so a ${BASH_SOURCE[0]}-relative path resolves to the wrong place --
# MODELS_DIR is set explicitly at submission time instead.
: "${MODELS_DIR:?set via sbatch --export=ALL,MODELS_DIR=/abs/path/to/models}"
cd "$MODELS_DIR"

echo "dataset=$DATASET node=$SLURM_JOB_NODELIST"
CONDA_OVERRIDE_CUDA=12.0 pixi run -e gears python baselines/baselines.py --dataset "$DATASET"
