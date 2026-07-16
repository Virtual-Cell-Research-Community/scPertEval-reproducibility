"""Corrections to cell-gears==0.0.2's own internals (this package's pinned, older gears
version — separate from the gears==0.1.2 used by models/gears — see models/gears/patch.py for
that one) — speed fixes and genuine bugs alike, applied via monkeypatching from
train_predict.py before any PertData work happens. Kept in its own module for the same
reasons documented there.

``apply()`` covers two kinds of thing:

- ``restore_series_nonzero`` — a pandas/scipy version-compatibility shim, byte-for-byte the
  same fix as models/gears/patch.py's version (same underlying issue, same two packages).
- Two drop-in-replacement speed fixes. ``get_dropout_non_zero_genes`` is byte-for-byte
  identical between gears==0.1.2 and cell-gears==0.0.2 (confirmed by diffing both installed
  copies), so the same fix applies verbatim. ``create_dataset_file`` differs from
  gears/patch.py's version in ways specific to this older API:
    - it RETURNS the processed dict (``self.dataset_processed = self.create_dataset_file()``)
      rather than setting self.dataset_processed as a side effect.
    - create_cell_graph here concatenates a one-hot perturbation-indicator column into ``x``
      itself (shape (n_genes, 2): expression + indicator) instead of storing pert_idx as
      separate Data metadata (as gears==0.1.2 does) — this package's PertData has no
      set_pert_genes()/pert_list mechanism at all, so there's no equivalent to replicate there.
    - get_pert_idx takes an extra ``adata_`` argument (unused internally beyond being part of
      the original signature — kept for interface compatibility, not because we need it).
  Both are drop-in replacements — identical outputs to the originals (verified by exact/
  statistical equivalence checks on the real full-scale replogle22k562 data), just avoiding
  avoidable per-item Python overhead. Neither changes any algorithm, DE method, or
  random-sampling semantics.

``load_or_build_pert_data`` is a different kind of thing and isn't applied via ``apply()`` —
it's a wrapper *we* call instead of ``PertData.new_data_process()`` directly, correcting a real
gap in this package's own caching (see its own docstring), not a monkeypatch of its namespace.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

import gears.pertdata as _pertdata
from gears import PertData
from gears.utils import print_sys


def restore_series_nonzero() -> None:
    """Restore ``pandas.Series.nonzero()``, dropped in pandas 1.0.

    Byte-for-byte the same fix as models/gears/patch.py's version — see its docstring for the
    full writeup (cell-gears indexes a scipy sparse matrix with a raw pandas boolean Series;
    current scipy calls ``.nonzero()`` directly on the indexer; pandas dropped that method from
    ``Series`` in 1.0. Not a bug in either library in isolation, just a gap that opened up
    between two libraries this package's code predates).
    """
    if not hasattr(pd.Series, "nonzero"):
        pd.Series.nonzero = lambda self: self.to_numpy().nonzero()


def fast_get_dropout_non_zero_genes(adata):
    """Drop-in replacement for ``gears.data_utils.get_dropout_non_zero_genes``.

    Identical fix to models/gears/patch.py's version of the same function — the two
    packages' copies of this function are byte-for-byte identical, including the same O(n^2)
    set-membership bug (profiled there at 81% of the function's runtime on real data) and the
    same redundant mean-expression recomputation. See that module's docstring for the full
    writeup; not repeated here beyond this pointer since the fix itself is verbatim the same.
    """
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis=0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(
        len(adata.obs.condition.unique()), adata.X.toarray().shape[1]
    )
    ctrl = mean_expression[np.where(pert_list == "ctrl")[0]]

    pert_full_id2pert = dict(adata.obs[["condition_name", "condition"]].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    ctrl_zero = np.where(np.array(ctrl)[0] == 0)[0]

    for pert in adata.uns["rank_genes_groups_cov_all"].keys():
        p = pert_full_id2pert[pert]
        X = condition2mean_expression[p]

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, ctrl_zero)
        non_dropouts = np.concatenate((non_zero, true_zeros))

        non_zero_set = set(non_zero.tolist())
        non_dropouts_set = set(non_dropouts.tolist())

        top = adata.uns["rank_genes_groups_cov_all"][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts_set][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero_set][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)

    adata.uns["top_non_dropout_de_20"] = top_non_dropout_de_20
    adata.uns["non_dropout_gene_idx"] = non_dropout_gene_idx
    adata.uns["non_zeros_gene_idx"] = non_zeros_gene_idx
    adata.uns["top_non_zero_de_20"] = top_non_zero_de_20

    return adata


def fast_create_dataset_file(self):
    """Drop-in replacement for ``PertData.create_dataset_file``/``create_cell_graph_dataset``
    (cell-gears==0.0.2). Same batching strategy as models/gears/patch.py's version — densify
    once per category (and the control pool once for the whole dataset), draw all of a
    category's random control assignments in one ``np.random.randint(..., size=n)`` call,
    build one batched tensor instead of N small ones — adapted for this version's ``x``
    layout (expression concatenated with a one-hot perturbation-indicator column) and its
    "return the dict" calling convention (``self.dataset_processed = self.create_dataset_file()``
    in this package's own ``new_data_process``, unlike gears==0.1.2's side-effect convention).

    The one-hot perturbation-indicator column is identical for every cell within a
    perturbation category (it only depends on pert_idx, which is category-constant in the
    original too — computed once per category there, same as here), so it's built once per
    category and broadcast across that category's cells via ``torch.stack`` rather than
    rebuilt from scratch per cell.

    Verified equivalent to the original the same way as models/gears/patch.py's version: for
    'ctrl', output is exactly identical; for perturbed categories, de_idx/y match exactly,
    the one-hot indicator column matches exactly, and every sampled expression row is
    confirmed to be a genuine row of self.ctrl_adata.

    ``y_t[i].clone()`` below is load-bearing — see models/gears/patch.py's version of this
    function for the full writeup (same bug, same fix): pickling a tensor *view* serializes
    its entire backing storage, not just the visible slice, so without cloning, every cell's
    Data.y would drag along a reference to its whole category's dense block. Confirmed
    empirically on the gears==0.1.2 side: this alone bloated a 43GB cell_graphs.pkl (should
    be ~15GB) and OOM'd a 64GB-capped job on load. ``x`` here doesn't need the same treatment
    — ``torch.stack`` (unlike plain indexing) always allocates a fresh output and copies its
    inputs into it, so ``torch.stack([x_t[i], pert_feats_t], dim=1)`` is already a standalone
    tensor with no shared storage back to ``x_t``.
    """
    ctrl_X = self.ctrl_adata.X
    ctrl_dense = np.asarray(ctrl_X.todense() if hasattr(ctrl_X, "todense") else ctrl_X, dtype=np.float32)
    n_ctrl, n_genes = ctrl_dense.shape

    dl = {}
    for pert_category in tqdm(self.adata.obs["condition"].unique()):
        adata_ = self.adata[self.adata.obs["condition"] == pert_category]
        de_genes = adata_.uns["rank_genes_groups_cov_all"]
        n = adata_.n_obs
        X_ = adata_.X
        target_dense = np.asarray(X_.todense() if hasattr(X_, "todense") else X_, dtype=np.float32)

        num_de_genes = 20
        if pert_category != "ctrl":
            pert_idx = self.get_pert_idx(pert_category, adata_)
            pert_de_category = adata_.obs["condition_name"][0]
            de_idx = np.where(
                adata_.var_names.isin(np.array(de_genes[pert_de_category][:num_de_genes]))
            )[0]
            ctrl_idx = np.random.randint(0, n_ctrl, size=n)
            x_dense = ctrl_dense[ctrl_idx]
            y_dense = target_dense
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            x_dense = target_dense
            y_dense = target_dense

        pert_feats = np.zeros(n_genes, dtype=np.float32)
        if pert_idx is not None:
            for p in pert_idx:
                pert_feats[int(np.abs(p))] = 1.0
        pert_feats_t = torch.from_numpy(pert_feats)

        x_t = torch.from_numpy(x_dense)
        y_t = torch.from_numpy(y_dense)

        dl[pert_category] = [
            Data(
                x=torch.stack([x_t[i], pert_feats_t], dim=1),
                edge_index=None,
                edge_attr=None,
                y=y_t[i].clone().unsqueeze(0),
                de_idx=de_idx,
                pert=pert_category,
            )
            for i in range(n)
        ]
    print_sys("Done!")
    return dl


def load_or_build_pert_data(pert_data: PertData, dataset_name: str, adata: ad.AnnData) -> None:
    """PertData.new_data_process() (the entry point for a custom, non-benchmark dataset —
    what we use) always rebuilds its per-cell PyG graph cache from scratch, even though it
    writes to a fold-independent path (data_pyg/cell_graphs.pkl) that every fold's
    train_predict.py invocation would otherwise rebuild identically. PertData.load() (this
    package's own cell-gears==0.0.2 — older than models/gears's 0.1.2, see module docstring)
    checks for and reuses that cache first; new_data_process() doesn't. This replicates
    load()'s cache-check (and, unlike load()'s own cache-hit branch, also sets ctrl_adata/
    gene_names in both branches — load() only sets those on a cache miss, a latent gap in this
    version too) so fold 1-4 reuse fold 0's cache instead of rebuilding it. Note this
    cell-gears version has no set_pert_genes()/pert_list mechanism at all (unlike 0.1.2) —
    nothing to replicate there, this script never uses it.

    Written with the temp-file-then-rename pattern (atomic on POSIX) so a fold job starting
    mid-write from another concurrently-running fold never reads a partially-written cache —
    doesn't eliminate wasted duplicate work under a race (all folds might still see "no cache
    yet" if launched at the same time), just guarantees no corruption if that happens.
    """
    dataset_name = dataset_name.lower()
    save_data_folder = Path(pert_data.data_path) / dataset_name
    save_data_folder.mkdir(parents=True, exist_ok=True)
    pyg_path = save_data_folder / "data_pyg"
    pyg_path.mkdir(parents=True, exist_ok=True)
    dataset_fname = pyg_path / "cell_graphs.pkl"
    adata_fname = save_data_folder / "perturb_processed.h5ad"
    pert_data.dataset_name = dataset_name
    pert_data.dataset_path = str(save_data_folder)
    if dataset_fname.is_file() and adata_fname.is_file():
        print(f"Local copy of pyg dataset detected at {dataset_fname}, loading instead of rebuilding...")
        pert_data.adata = ad.read_h5ad(adata_fname)
        pert_data.ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        pert_data.gene_names = pert_data.adata.var.gene_name
        with open(dataset_fname, "rb") as f:
            pert_data.dataset_processed = pickle.load(f)
        return
    pert_data.new_data_process(dataset_name=dataset_name, adata=adata)
    tmp_fname = dataset_fname.with_suffix(".pkl.tmp")
    with open(tmp_fname, "wb") as f:
        pickle.dump(pert_data.dataset_processed, f)
    tmp_fname.replace(dataset_fname)


def apply() -> None:
    """Monkeypatch cell-gears' own PertData/data_utils internals with the faster, equivalent
    versions above, and restore the pandas/scipy compatibility shim. Must be called before any
    PertData work happens — same reasoning as models/gears/patch.py's apply().
    """
    restore_series_nonzero()
    _pertdata.get_dropout_non_zero_genes = fast_get_dropout_non_zero_genes
    _pertdata.PertData.create_dataset_file = fast_create_dataset_file
