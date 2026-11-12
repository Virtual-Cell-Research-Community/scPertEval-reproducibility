#!/bin/bash
# Computes the three ranked, deterministic GWPS baselines against every fold in one process.
# The runner reads raw.h5ad once and writes validated, checksum-addressed run bundles using the
# same layout and safe-resumption contract as the substantive models. The technical-duplicate
# oracle is intentionally opt-in and is not part of this ranked-baseline job.
#
# Usage: sbatch --export=ALL,MODELS_DIR="$(pwd)" slurm/baselines.sbatch
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=baselines
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

DATASET="${DATASET:-replogle-weissman-2022-k562-gwps}"
# Same spool-dir caveat as train.sbatch: SLURM copies the submitted script to a spool dir
# before running it, so a ${BASH_SOURCE[0]}-relative path resolves to the wrong place --
# MODELS_DIR is set explicitly at submission time instead.
: "${MODELS_DIR:?set via sbatch --export=ALL,MODELS_DIR=/abs/path/to/models}"
cd "$MODELS_DIR"

REPOSITORY_ROOT="$(cd .. && pwd)"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
export UV_PYTHON="3.11"
export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/scPertEval-reproducibility-py311"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "dataset=$DATASET node=$SLURM_JOB_NODELIST"
cd "$REPOSITORY_ROOT"
"$UV_BIN" run --frozen python -m models.baselines.perturbation_baselines --dataset "$DATASET"
