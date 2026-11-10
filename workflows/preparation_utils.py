"""Bounded-memory helpers shared by the public preparation workflows."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.interpolate import UnivariateSpline

CELL_CYCLE_SHA256 = "eedd82fb247f5fbbca2b98c4b7709d236320c4bb2d3cec001b86c86a1c7cd848"


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file without reading it all into memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_raw_counts(matrix: sp.spmatrix | np.ndarray) -> None:
    """Reject matrices whose non-zero values are not count-like integers."""
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    sample = values[:1_000_000]
    if sample.size and (np.nanmin(sample) < 0 or not np.allclose(sample, np.rint(sample))):
        raise ValueError("X does not contain non-negative raw counts")


def add_crispri_residual_expression(
    adata: AnnData,
    *,
    target_col: str = "target_gene",
    control_label: str = "non-targeting",
    out_key: str = "residual_expression",
    eps: float = 1e-9,
) -> None:
    """Add per-cell target expression relative to non-targeting controls."""
    targets = adata.obs[target_col].to_numpy()
    control_mask = targets == control_label
    if not np.any(control_mask):
        raise ValueError(f"no control cells where {target_col} == {control_label!r}")

    control_mean = np.asarray(adata.X[control_mask].mean(axis=0)).ravel().astype(np.float64)
    var_index = pd.Index(adata.var_names)
    residual = np.full(adata.n_obs, np.nan, dtype=np.float32)
    for target in pd.unique(targets[~control_mask]):
        if target not in var_index:
            continue
        column = var_index.get_loc(target)
        rows = targets == target
        values = adata.X[rows, column]
        values = values.toarray().ravel() if sp.issparse(values) else np.asarray(values).ravel()
        residual[rows] = ((values + eps) / (control_mean[column] + eps)).astype(np.float32)
    adata.obs[out_key] = residual


def column_variance(matrix: sp.spmatrix, blocksize: int = 1_000) -> np.ndarray:
    """Compute column variances in bounded-memory dense blocks."""
    values = []
    for start in range(0, matrix.shape[1], blocksize):
        block = matrix[:, start : start + blocksize].toarray()
        values.extend(block.std(axis=0, dtype=np.float64) ** 2)
    return np.asarray(values)


def clipped_standardized_std(
    matrix: sp.spmatrix,
    means: np.ndarray,
    predicted_std: np.ndarray,
    clip: float,
) -> np.ndarray:
    """Calculate clipped standardized deviations in one sparse column traversal."""
    columns = matrix.tocsc(copy=True)
    columns.sum_duplicates()
    columns.eliminate_zeros()
    n_obs, n_vars = columns.shape
    result = np.empty(n_vars, dtype=np.float32)

    for index in range(n_vars):
        start, end = columns.indptr[index : index + 2]
        values = columns.data[start:end].astype(np.float32, copy=True)
        values -= means[index]
        values /= predicted_std[index]
        np.clip(values, -clip, clip, out=values)

        zero_value = np.float32(-means[index] / predicted_std[index])
        zero_value = np.clip(zero_value, -clip, clip)
        implicit_zeros = n_obs - values.size
        total = np.sum(values, dtype=np.float64) + implicit_zeros * float(zero_value)
        total_squared = np.sum(values * values, dtype=np.float64) + implicit_zeros * float(zero_value**2)
        variance = total_squared / n_obs - (total / n_obs) ** 2
        result[index] = np.sqrt(max(variance, 0.0))
    return result


def add_vst_statistics(adata: AnnData, eps: float = 1e-8) -> None:
    """Add the historical VST statistics used by the benchmark preparation."""
    frame = pd.DataFrame(index=adata.var_names)
    frame["mean"] = adata.X.mean(axis=0).A1
    frame["var"] = column_variance(adata.X)
    frame["mean"] = np.where(frame["mean"] <= 0, eps, frame["mean"])
    frame["var"] = np.where(frame["var"] <= 0, eps, frame["var"])
    frame["log10_mean"] = np.log10(frame["mean"])
    frame["log10_var"] = np.log10(frame["var"])

    order = np.argsort(frame["log10_mean"].to_numpy())
    spline = UnivariateSpline(frame["log10_mean"].to_numpy()[order], frame["log10_var"].to_numpy()[order])
    frame["log10_predicted_var"] = spline(frame["log10_mean"])
    frame["predicted_var"] = 10 ** frame["log10_predicted_var"]
    frame["predicted_std"] = frame["predicted_var"] ** 0.5
    frame["std_var"] = clipped_standardized_std(
        adata.X,
        frame["mean"].to_numpy(np.float32),
        frame["predicted_std"].to_numpy(np.float32),
        adata.n_obs**0.5,
    )
    frame["std_var_rank"] = (frame["std_var"].rank(method="dense", ascending=False) - 1).astype(int)

    columns = [
        "mean",
        "var",
        "log10_mean",
        "log10_var",
        "log10_predicted_var",
        "predicted_var",
        "predicted_std",
        "std_var",
        "std_var_rank",
    ]
    adata.var = adata.var.drop(columns=[name for name in columns if name in adata.var]).join(frame[columns])


def read_cell_cycle_genes(path: Path) -> tuple[list[str], list[str]]:
    """Read and verify the pinned Regev cell-cycle reference."""
    digest = file_checksum(path)
    if digest != CELL_CYCLE_SHA256:
        raise ValueError(f"unexpected checksum for {path}: {digest}")
    genes = pd.read_csv(path)
    if genes.columns.tolist() != ["gene", "phase"]:
        raise ValueError(f"unexpected cell-cycle reference schema in {path}")
    return (
        genes.loc[genes["phase"] == "s", "gene"].tolist(),
        genes.loc[genes["phase"] == "g2m", "gene"].tolist(),
    )
