"""Corrections to GEARS' own internals (gears==0.1.2) — speed fixes and genuine bugs alike —
applied via monkeypatching from train_predict.py before any PertData work happens. Kept in
its own module so train_predict.py stays a straightforward reading of GEARS' documented API,
with the patched-vs-vanilla behavior and the reasoning for it isolated in one place.

``apply()`` covers two kinds of thing:

- A pandas/scipy version-compatibility shim (``restore_series_nonzero``) — not a GEARS bug at
  all, but a gap that opened up between two *other* libraries GEARS' code silently depends on.
- Two drop-in-replacement speed fixes (``fast_get_dropout_non_zero_genes``,
  ``fast_create_dataset_file``) — identical outputs to the originals (verified by exact/
  statistical equivalence checks against gears.data_utils.get_dropout_non_zero_genes and
  gears.pertdata.PertData.create_dataset_file on the real full-scale replogle22k562 data),
  just avoiding avoidable per-item Python overhead. Neither changes GEARS' own algorithm, DE
  method, or random-sampling semantics — see each function's docstring for what, specifically,
  was slow and why.

``load_or_build_pert_data`` is a different kind of thing and isn't applied via ``apply()`` —
it's a wrapper *we* call instead of ``PertData.new_data_process()`` directly, correcting a real
gap in GEARS' own caching (see its own docstring), not a monkeypatch of GEARS' namespace.

Two more genuine bugs, same family as ``load_or_build_pert_data``'s but upstream of it: GEARS'
own Dataverse-download helpers (``dataverse_download``, ``tar_data_download_wrapper``) are also
check-then-act with non-atomic writes, and every one of GEARS' three reference downloads
(``gene2go_all.pkl``, ``essential_all_data_pert_genes.pkl``, ``go_essential_all.tar.gz``) goes
through them, sharing the same ``PERT_DATA_DIR`` across all 5 concurrently-launched fold
processes. ``_atomic_download``/``_atomic_tar_download_and_extract`` fix that the same way
``load_or_build_pert_data`` fixes the PyG graph cache race — patched into GEARS' own namespace
via ``apply()`` since, unlike the graph cache, these are called from deep inside GEARS' own
``PertData.__init__``/``set_pert_genes``/``model_initialize``, not somewhere we can wrap directly.
"""

from __future__ import annotations

import os
import pickle
import shutil
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
import torch
from torch_geometric.data import Data
from tqdm import tqdm

import gears.pertdata as _pertdata
import gears.utils as _utils
from gears import PertData
from gears.utils import print_sys


def restore_series_nonzero() -> None:
    """Restore ``pandas.Series.nonzero()``, dropped in pandas 1.0.

    Not a GEARS bug — a compatibility gap between two *other* libraries GEARS' code (e.g.
    ``self.adata[self.adata.obs['condition'] == 'ctrl']``, boolean-indexing a scipy sparse
    matrix with a raw pandas Series) silently depends on. Older scipy coerced such an indexer
    via ``np.asarray(...)`` first; current scipy instead calls ``.nonzero()`` directly on
    whatever it's handed, assuming ndarray-like behavior. pandas dropped that exact method
    from ``Series`` back in 1.0 (in favor of ``.to_numpy().nonzero()``). Neither library did
    anything wrong in isolation — GEARS' code just predates both of these independent changes.
    """
    if not hasattr(pd.Series, "nonzero"):
        pd.Series.nonzero = lambda self: self.to_numpy().nonzero()


def fast_get_dropout_non_zero_genes(adata):
    """Drop-in replacement for ``gears.data_utils.get_dropout_non_zero_genes``.

    Two changes, both purely mechanical — same output as the original, verified by exact
    equality of every ``adata.uns`` key it writes, checked against the original on the real
    full-scale replogle22k562 data:

    1. The original recomputes each condition's mean expression a SECOND time inside the
       main loop (``np.mean(adata[adata.obs.condition == p].X, axis=0)``, full AnnData
       boolean subsetting) even though the identical quantity was already computed more
       cheaply just above via raw sparse-matrix row indexing (``condition2mean_expression``).
       Reused instead of recomputed.
    2. The original checks gene-index membership (``i in non_dropouts`` / ``i in non_zero``)
       against plain numpy arrays inside a Python list comprehension over up to ~8357 genes
       per perturbation — an O(n) linear scan per check, O(n^2) per perturbation. Cast to
       ``set`` first for O(1) lookups. This dominates the original function's cost: profiled
       at 105.6s of 129.6s (81%) on the real full-scale data.

    Also drops a few lines of GEARS' own dead code at the end of the original (recomputes
    non_zero/zero/true_zeros/non_dropouts from whichever condition happened to run last in
    the loop, then never uses any of it — confirmed by reading every downstream use of
    ``adata.uns``; removing it doesn't change any output).
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


def fast_create_dataset_file(self) -> None:
    """Drop-in replacement for ``PertData.create_dataset_file``/``create_cell_graph_dataset``.

    The original builds one ``torch_geometric.data.Data`` object per CELL (212,073 for the
    real full-scale replogle22k562 data) inside a pure-Python loop, doing a separate
    ``.toarray()`` + ``torch.Tensor()`` call per cell, plus — for non-control cells — drawing
    one random control cell via individual AnnData row-slicing per target cell. Measured at
    ~1.69ms/cell (~6 min total) via a representative 20-condition probe on real data.

    This batches all of that per perturbation category: densify the category's cells once,
    draw all of that category's random control assignments in a single
    ``np.random.randint(..., size=n)`` call and gather them from a control matrix densified
    once up front for the whole dataset (``self.ctrl_adata`` is already capped at
    ``MAX_CELLS_CONTROL`` — small enough to hold dense), and build one batched tensor instead
    of N small ones. Only the final per-cell ``Data(...)`` object is still constructed one at
    a time (torch_geometric's own collation expects a list of Data objects), but by then it's
    just wrapping pre-made tensor slices, not doing any conversion work itself.

    Verified equivalent to the original: for the (non-random) 'ctrl' category, output is
    exactly identical (x, y, de_idx, pert_idx all match); for perturbed categories, de_idx/
    pert_idx/y match exactly and every sampled x row is confirmed to come from
    ``self.ctrl_adata`` (statistically equivalent random control assignment — the original
    isn't seeded either, so exact cell-to-cell random pairing was never a guarantee to
    preserve, only the sampling distribution).

    The ``.clone()`` on each per-cell ``x``/``y`` slice below is load-bearing, not
    defensive-programming noise: ``x_t[i]``/``y_t[i]`` are *views* into the category's
    shared batched tensor, and when pickled (this whole dict eventually goes through
    ``pickle.dump`` in ``new_data_process``), PyTorch serializes a view's entire backing
    storage, not just the visible slice — so without ``.clone()``, every single cell's Data
    object drags a reference to its whole category's dense block along with it. Caught this
    empirically: the first version (no ``.clone()``) produced a 43GB cell_graphs.pkl for the
    real full-scale replogle22k562 data (vs. ~15GB expected from the raw tensor sizes alone),
    and loading that file back OOM'd a 64GB-capped job. A 100-row isolated repro confirmed the
    mechanism directly: pickling 100 unclonced row-views of a shared parent tensor produced a
    334MB file; the same 100 rows individually ``.clone()``'d produced 3.4MB — a 100x
    difference for identical data.
    """
    print_sys("Creating dataset file...")
    ctrl_X = self.ctrl_adata.X
    ctrl_dense = np.asarray(ctrl_X.todense() if hasattr(ctrl_X, "todense") else ctrl_X, dtype=np.float32)
    n_ctrl = ctrl_dense.shape[0]

    self.dataset_processed = {}
    for pert_category in tqdm(self.adata.obs["condition"].unique()):
        adata_ = self.adata[self.adata.obs["condition"] == pert_category]
        num_de_genes = 20
        if "rank_genes_groups_cov_all" in adata_.uns:
            de_genes = adata_.uns["rank_genes_groups_cov_all"]
            de = True
        else:
            de = False
            num_de_genes = 1

        n = adata_.n_obs
        X_ = adata_.X
        target_dense = np.asarray(X_.todense() if hasattr(X_, "todense") else X_, dtype=np.float32)

        if pert_category != "ctrl":
            pert_idx = self.get_pert_idx(pert_category)
            pert_de_category = adata_.obs["condition_name"][0]
            if de:
                de_idx = np.where(
                    adata_.var_names.isin(np.array(de_genes[pert_de_category][:num_de_genes]))
                )[0]
            else:
                de_idx = [-1] * num_de_genes
            ctrl_idx = np.random.randint(0, n_ctrl, size=n)
            x_dense = ctrl_dense[ctrl_idx]
            y_dense = target_dense
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            x_dense = target_dense
            y_dense = target_dense

        x_t = torch.from_numpy(x_dense)
        y_t = torch.from_numpy(y_dense)
        pert_idx_val = [-1] if pert_idx is None else pert_idx

        self.dataset_processed[pert_category] = [
            Data(
                x=x_t[i].clone().unsqueeze(1),
                pert_idx=pert_idx_val,
                y=y_t[i].clone().unsqueeze(0),
                de_idx=de_idx,
                pert=pert_category,
            )
            for i in range(n)
        ]
    print_sys("Done!")


def _atomic_download(url: str, save_path: str) -> None:
    """Drop-in replacement for ``gears.utils.dataverse_download`` (check-then-act, writes
    directly to ``save_path`` via a truncating ``open(save_path, 'wb')``) — every one of GEARS'
    three Dataverse-hosted reference files (``gene2go_all.pkl`` in ``PertData.__init__``,
    ``essential_all_data_pert_genes.pkl`` in ``set_pert_genes()``, ``go_essential_all.tar.gz``
    via ``get_similarity_network()``) goes through it, and all three share the same
    ``PERT_DATA_DIR`` across every fold's ``train_predict.py`` invocation. ``submit_all.sh``
    launches all 5 folds concurrently, so on a cold cache every one of them can race through the
    original's ``os.path.exists(save_path)`` check simultaneously, then all open the same
    destination path in ``'wb'`` mode at once — each writer's own buffered chunks interleave
    with the others' at essentially arbitrary offsets, producing neither download intact.
    Downstream ``pickle.load``/``tarfile.open`` on the result then raises, non-deterministically,
    only on the very first cold run.

    Same fix as ``load_or_build_pert_data``: write to a private per-attempt temp file in the
    same directory (so the final rename is same-filesystem, hence atomic) and only
    ``os.replace()`` it into ``save_path`` once the download is known-complete. A concurrent
    reader's ``os.path.exists(save_path)`` check therefore only ever sees the complete old state
    (nothing) or the complete new one, never a partially-written file. Replacing a *file*
    destination is atomic even if another process's download already won the race and landed
    there first — POSIX ``rename()`` on a file target always succeeds, silently overwriting with
    what is, here, byte-identical content (unlike the directory case below).
    """
    if os.path.exists(save_path):
        print_sys("Found local copy...")
        return

    print_sys("Downloading...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(save_path) or ".", prefix=".download-")
    try:
        with os.fdopen(fd, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    except BaseException:
        os.remove(tmp_path)
        raise
    finally:
        progress_bar.close()
    os.replace(tmp_path, save_path)


def _atomic_tar_download_and_extract(url: str, save_path: str, data_path: str) -> None:
    """Drop-in replacement for ``gears.utils.tar_data_download_wrapper`` — covers the same
    check-then-act gap as ``_atomic_download`` above, but for the *extraction* step
    specifically. The tarball download itself is already safe once ``dataverse_download`` is
    patched to ``_atomic_download`` (see ``apply()`` below); this wraps the
    ``tarfile.extractall()`` call in the same atomic-rename pattern instead of leaving it as a
    second, independent race, where multiple folds extract into the same shared ``data_path``
    concurrently and one fold's ``pd.read_csv`` could catch another fold's extraction mid-write.

    Extracts into a private temp directory first, then renames the tar's own top-level output
    directory into place at ``save_path`` in one atomic step. Unlike ``_atomic_download``'s
    file-to-file replace, a directory-to-directory ``os.replace()`` raises ``OSError`` if the
    destination already exists and is non-empty (POSIX ``rename()``'s own semantics for
    directories) — so a process that loses this particular race gets an expected, harmless error
    here instead of a corrupted destination; treated as a no-op, since by the time that happens
    the winning process's identical extraction is already in place.
    """
    if os.path.exists(save_path):
        print_sys("Found local copy...")
        return

    _atomic_download(url, save_path + ".tar.gz")
    print_sys("Extracting tar file...")

    tmp_root = tempfile.mkdtemp(dir=data_path, prefix=".extract-")
    try:
        with tarfile.open(save_path + ".tar.gz") as tar:
            tar.extractall(path=tmp_root)
        extracted = Path(tmp_root) / Path(save_path).name
        try:
            extracted.replace(save_path)
        except OSError:
            if not os.path.isdir(save_path):
                raise
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print_sys("Done!")


def load_or_build_pert_data(pert_data: PertData, dataset_name: str, adata: ad.AnnData) -> None:
    """PertData.new_data_process() (the entry point for a custom, non-benchmark dataset —
    what we use, since we're not one of GEARS' 4 named datasets) always rebuilds its per-cell
    PyG graph cache from scratch. PertData.load() (GEARS' own path for its named benchmarks)
    checks for and reuses an existing data_pyg/cell_graphs.pkl first — but new_data_process()
    has no equivalent check, even though it writes to that exact same fold-independent path.
    Since every fold's train_predict.py invocation builds a fresh PertData from the identical
    (fold-independent) adata, that graph construction is fully redundant across folds. This
    replicates load()'s cache-check (and, unlike load()'s own cache-hit branch, also sets
    ctrl_adata/gene_names in both branches — load() only sets those in its cache-miss branch,
    a latent gap in GEARS' own code) so fold 1-4 reuse fold 0's cache instead of rebuilding it.

    On a cache miss, new_data_process() is never allowed to write directly to the shared
    adata_fname/dataset_fname paths above, even though that's what it does internally
    (``pickle.dump(..., open(dataset_fname, "wb"))`` and ``adata.write_h5ad(...)``, both
    non-atomic, both at the exact paths this function's own cache-hit check inspects). A
    prior version of this function let that happen and then atomically rewrote just
    dataset_fname afterward — which sounds safe but isn't: a concurrently-starting fold's
    ``is_file()`` check can observe adata_fname (written first, non-atomically, by
    new_data_process() itself) already complete and dataset_fname mid-write (exists, but only
    partially written — pickling a multi-GB object is not instantaneous), and try to load a
    truncated cache. Confirmed this is a real gap, not just theoretical, by tracing
    new_data_process()'s own source line by line.

    Instead, ``pert_data.data_path`` is pointed at a private, per-attempt temp directory for
    the whole build, so new_data_process()'s own writes land somewhere no other fold ever
    looks. Only once both outputs are known-complete are they moved into the shared location,
    each via its own atomic rename (``Path.replace()``). At every instant, a concurrent
    reader's ``is_file()`` check therefore sees either the complete old state or the complete
    new one for each file — and since the cache-hit check requires *both* files present,
    there's no window (not even between the two renames) where an in-progress build could be
    mistaken for a finished one. A crash mid-build just leaves an orphaned temp directory
    (cleaned up below) rather than a corrupted shared file.
    """
    dataset_name = dataset_name.lower()
    original_data_path = pert_data.data_path
    save_data_folder = Path(original_data_path) / dataset_name
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
        pert_data.set_pert_genes()
        pert_data.ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        pert_data.gene_names = pert_data.adata.var.gene_name
        with open(dataset_fname, "rb") as f:
            pert_data.dataset_processed = pickle.load(f)
        return

    tmp_root = Path(tempfile.mkdtemp(prefix=f".build-{dataset_name}-", dir=str(save_data_folder.parent)))
    try:
        pert_data.data_path = str(tmp_root)
        pert_data.new_data_process(dataset_name=dataset_name, adata=adata)
        tmp_adata_fname = tmp_root / dataset_name / "perturb_processed.h5ad"
        tmp_dataset_fname = tmp_root / dataset_name / "data_pyg" / "cell_graphs.pkl"
        tmp_adata_fname.replace(adata_fname)
        tmp_dataset_fname.replace(dataset_fname)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        pert_data.data_path = original_data_path
        pert_data.dataset_path = str(save_data_folder)


def apply() -> None:
    """Monkeypatch GEARS' own PertData/data_utils internals with the faster, equivalent
    versions above, and restore the pandas/scipy compatibility shim. Must be called before
    any PertData work happens: new_data_process() looks up get_dropout_non_zero_genes as a
    module global at call time, and create_dataset_file is looked up on the instance/class at
    call time, so patching the module-level names is enough — no need to touch already-created
    objects, but this must run before new_data_process() is actually called.

    ``dataverse_download``/``tar_data_download_wrapper`` are patched in both
    ``gears.utils`` (where ``get_similarity_network`` resolves them as its own module globals
    at call time, regardless of who imported ``get_similarity_network`` itself — e.g.
    ``gears.gears``) and ``gears.pertdata`` (which imported its own bound name for
    ``dataverse_download`` at import time, used directly by ``PertData.__init__`` and
    ``set_pert_genes()``). ``zip_data_download_wrapper`` is left unpatched — it's only reachable
    via ``PertData.load()``, GEARS' path for its own named benchmark datasets, which
    ``load_or_build_pert_data`` never calls.
    """
    restore_series_nonzero()
    _pertdata.get_dropout_non_zero_genes = fast_get_dropout_non_zero_genes
    _pertdata.PertData.create_dataset_file = fast_create_dataset_file
    _utils.dataverse_download = _atomic_download
    _utils.tar_data_download_wrapper = _atomic_tar_download_and_extract
    _pertdata.dataverse_download = _atomic_download
