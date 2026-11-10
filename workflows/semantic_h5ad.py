"""Serialization-independent checksums for prepared AnnData objects."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

FORMAT_VERSION = "scperteval-semantic-h5ad-v1"


def _put(digest: Any, value: bytes | str) -> None:
    payload = value.encode() if isinstance(value, str) else value
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _hash_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    _put(digest, json.dumps([str(value) for value in frame.columns], ensure_ascii=False))
    _put(digest, json.dumps([str(dtype) for dtype in frame.dtypes], ensure_ascii=False))
    _put(digest, json.dumps([str(name) for name in frame.index.names], ensure_ascii=False))
    values = pd.util.hash_pandas_object(frame, index=True, categorize=False).to_numpy(dtype="<u8")
    _put(digest, values.tobytes())
    return digest.hexdigest()


def _hash_array(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    _put(digest, json.dumps(array.shape))
    _put(digest, str(array.dtype))
    if array.dtype.kind in "biufc":
        canonical = array.astype(array.dtype.newbyteorder("<"), copy=False)
        _put(digest, np.ascontiguousarray(canonical).tobytes())
    else:
        _put(digest, json.dumps(array.tolist(), default=str, ensure_ascii=False, allow_nan=True))
    return digest.hexdigest()


def _hash_sparse(value: sp.spmatrix) -> str:
    matrix = value.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.sha256()
    _put(digest, json.dumps(matrix.shape))
    _put(digest, str(matrix.dtype))
    _put(digest, np.ascontiguousarray(matrix.data.astype(matrix.dtype.newbyteorder("<"), copy=False)).tobytes())
    _put(digest, np.ascontiguousarray(matrix.indices.astype("<i8", copy=False)).tobytes())
    _put(digest, np.ascontiguousarray(matrix.indptr.astype("<i8", copy=False)).tobytes())
    return digest.hexdigest()


def _hash_value(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return _hash_frame(value)
    if sp.issparse(value):
        return _hash_sparse(value)
    if isinstance(value, Mapping):
        digest = hashlib.sha256()
        for key in sorted(value):
            _put(digest, str(key))
            _put(digest, _hash_value(value[key]))
        return digest.hexdigest()
    if isinstance(value, (list, tuple)):
        digest = hashlib.sha256()
        for item in value:
            _put(digest, _hash_value(item))
        return digest.hexdigest()
    return _hash_array(value)


def _hash_matrix(matrix: Any, *, block_rows: int) -> str:
    digest = hashlib.sha256()
    _put(digest, json.dumps(matrix.shape))
    for start in range(0, matrix.shape[0], block_rows):
        block = matrix[start : start + block_rows]
        _put(digest, _hash_sparse(block) if sp.issparse(block) else _hash_array(block))
    return digest.hexdigest()


def semantic_h5ad_components(path: Path, *, block_rows: int = 4096) -> dict[str, str]:
    """Return deterministic hashes of every logical AnnData component.

    HDF5 chunking, compression, object ordering, and other writer details are intentionally
    excluded. Sparse matrices are canonicalized to sorted CSR blocks before hashing.
    """
    data = ad.read_h5ad(path, backed="r")
    try:
        components = {
            "X": _hash_matrix(data.X, block_rows=block_rows),
            "obs": _hash_frame(data.obs),
            "var": _hash_frame(data.var),
            "uns": _hash_value(dict(data.uns)),
        }
        for slot_name in ["layers", "obsm", "varm", "obsp", "varp"]:
            slot = getattr(data, slot_name)
            for key in sorted(slot.keys()):
                value = slot[key]
                if hasattr(value, "shape") and len(value.shape) == 2 and value.shape[0] > block_rows:
                    component = _hash_matrix(value, block_rows=block_rows)
                else:
                    component = _hash_value(value)
                components[f"{slot_name}/{key}"] = component
        if data.raw is not None:
            components["raw/X"] = _hash_matrix(data.raw.X, block_rows=block_rows)
            components["raw/var"] = _hash_frame(data.raw.var)
        return components
    finally:
        data.file.close()


def semantic_h5ad_sha256(path: Path, *, block_rows: int = 4096) -> str:
    """Hash all logical components of an H5AD independently of HDF5 serialization."""
    components = semantic_h5ad_components(path, block_rows=block_rows)
    digest = hashlib.sha256()
    _put(digest, FORMAT_VERSION)
    for name in sorted(components):
        _put(digest, name)
        _put(digest, components[name])
    return digest.hexdigest()


def compare_semantic_h5ad(left: Path, right: Path, *, block_rows: int = 4096) -> dict[str, str]:
    """Require every logical component to match and return the shared component hashes."""
    left_components = semantic_h5ad_components(left, block_rows=block_rows)
    right_components = semantic_h5ad_components(right, block_rows=block_rows)
    if left_components != right_components:
        names = sorted(set(left_components) | set(right_components))
        differing = [name for name in names if left_components.get(name) != right_components.get(name)]
        raise ValueError(f"H5AD semantic components differ: {', '.join(differing)}")
    return left_components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path, nargs="?")
    args = parser.parse_args()

    left_digest = semantic_h5ad_sha256(args.left)
    print(f"{args.left}: {left_digest}")
    if args.right is not None:
        right_digest = semantic_h5ad_sha256(args.right)
        print(f"{args.right}: {right_digest}")
        if left_digest != right_digest:
            compare_semantic_h5ad(args.left, args.right)
            raise AssertionError("overall semantic digests differ despite equal components")
        print("semantic H5AD content matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
