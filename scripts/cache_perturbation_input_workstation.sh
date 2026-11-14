#!/usr/bin/env bash
# Materialize a disposable, checksum-verified workstation cache of the frozen
# K562-essential model view, then point the runtime checkout at it. Authoritative
# inputs and every run artifact remain on SDS.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-replogle-weissman-2022-k562-essential}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/home/pschaefer/sds-hd/sd22b002/projects/scPertEval-reproducibility}"
CACHE_ROOT="${CACHE_ROOT:-/var/tmp/scperteval-model-data}"
EXPECTED_SHA256="${EXPECTED_SHA256:-673ed91b5ad7bd04b16422a1273ed2e095d5670acefa130c414df9efdc97b70a}"

source_dir="${SHARED_PROJECT_ROOT}/models/data/${DATASET}"
cache_dir="${CACHE_ROOT}/${DATASET}"
runtime_link="${PROJECT_ROOT}/models/data/${DATASET}"

if [[ ! -f "${source_dir}/raw.h5ad" ]]; then
    echo "Frozen SDS input is missing: ${source_dir}/raw.h5ad" >&2
    exit 2
fi
if [[ -e "${runtime_link}" && ! -L "${runtime_link}" ]]; then
    echo "Refusing to replace non-symlink runtime data path: ${runtime_link}" >&2
    exit 2
fi

mkdir -p "${cache_dir}"
rsync -a --delete "${source_dir}/" "${cache_dir}/"

actual_sha256="$(sha256sum "${cache_dir}/raw.h5ad" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    echo "Cached input checksum mismatch: expected ${EXPECTED_SHA256}, got ${actual_sha256}" >&2
    exit 1
fi

ln -sfn "${cache_dir}" "${runtime_link}"
echo "[$(date -Is)] cached ${DATASET} at ${cache_dir}"
echo "[$(date -Is)] verified sha256=${actual_sha256}"
echo "[$(date -Is)] runtime link ${runtime_link} -> $(readlink "${runtime_link}")"
