"""Corrections to PRESAGE's own internals — speed fixes and genuine bugs alike — applied from
train_predict.py before ``train()`` is called. Kept in its own module so train_predict.py
stays a straightforward reading of PRESAGE's documented API, with the patched-vs-vanilla
behavior and the reasoning for it isolated in one place (same rationale as
models/gears/patch.py and models/scgpt/patch.py).

Unlike those two, none of this is applied via monkeypatching an *installed package's* module
globals directly at import time — PRESAGE's own ``train()`` hardcodes a few names
(``ReploglePRESAGEDataModule``, ``ModelHarness``, ``os``) by looking them up in its own
module namespace at call time, so ``apply()`` here patches ``train``'s namespace specifically,
right before calling it — the same mechanism, just applied at the call site rather than at
import time, since the checkpoint-preserving destination directory and the dataset name are
only known once argparse has run.

- ``compute_degs_fast`` — a drop-in replacement for ``scPerturbDataModule.compute_degs``,
  illico-backed (~85min -> seconds at full scale). Same shape as gears/scgpt's ``fast_*``
  functions.
- ``_CheckpointPreservingOS`` — proxies the stdlib ``os`` module but copies a file out before
  letting ``train.py``'s own unconditional ``os.remove(best_model_path)`` delete it; there's no
  config flag to keep the checkpoint otherwise.
- ``SmokePRESAGEDataModule`` / ``SmokeModelHarness`` — subclasses correcting genuine bugs in
  inherited behavior (a ``compute_degs`` that silently no-ops for any non-benchmark dataset;
  an ``on_test_epoch_end`` that unconditionally crashes at this scale) — not literal
  monkeypatches, but the idiomatic way to override behavior PRESAGE gives no hook for.
- CSVLogger injection — train.py's own train() only configures a
  ``pl.loggers.WandbLogger(offline=True)`` (no CSVLogger), so unlike gears/sclambda/scgpt
  (which print per-epoch metrics unconditionally, captured instead via stdout/stderr Tees in
  those models' own train_predict.py) or STATE (whose CSVLogger output is just relocated),
  PRESAGE has no plain-text per-epoch metrics artifact at all — offline wandb logs are
  binary/protobuf, not readable without the wandb SDK. ``apply()`` patches
  ``presage_train_module.pl.Trainer.__init__`` to inject an additional CSVLogger alongside
  whatever logger train.py passes, so ``metrics.csv`` (val_loss and whatever else
  ``ModelHarness`` logs) gets written the same way it does for STATE — train_predict.py then
  relocates it into ``metrics/<dataset>/fold_<i>.csv`` to match every other model.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import scanpy as sc
import scipy.sparse
from tqdm import tqdm

import model_harness as presage_model_harness_module
import train as presage_train_module
from presage_datamodule import ReploglePRESAGEDataModule


class _CheckpointPreservingOS:
    """Proxies the stdlib ``os`` module but copies a file into ``dest_dir`` before letting
    a ``remove()`` call through. train.py's own ``train()`` loads its best checkpoint into
    the lightning module and then immediately deletes it (``os.remove(best_model_path)``) —
    there's no config flag or hook to keep it, so this intercepts that one call by replacing
    the *name* ``os`` inside train.py's own module namespace (not the global ``os`` module
    itself, which every other import of it still sees unmodified)."""

    def __init__(self, dest_dir: Path) -> None:
        self._dest_dir = dest_dir

    def remove(self, path) -> None:
        self._dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, self._dest_dir / Path(path).name)
        os.remove(path)

    def __getattr__(self, name):
        return getattr(os, name)


def compute_degs_fast(self, adata) -> None:
    """Drop-in replacement for ``scPerturbDataModule.compute_degs`` (datamodule.py) — same
    control-cell sampling (capped at 5000, ``random_state=0``), same per-perturbation JSON
    caching (``self.deg_dir/<pert>.json``, skips any perturbation already computed) and
    ``self.merged_deg_file`` output, same statistical test (Wilcoxon rank-sum vs. control) and
    same top-1000-genes-by-significance ranking (matching scanpy's ``rankby_abs=True`` — both
    up- and down-regulated genes can appear, not just one direction). Only the compute backend
    differs: the original loops over every perturbation individually, each time re-slicing +
    ``.copy()``-ing an AnnData and running a fresh ``scanpy.tl.rank_genes_groups`` call that
    recomputes the (unchanging) control side from scratch — at full scale (~1789 perturbations)
    this took ~85 minutes for a single dataset. illico's ``asymptotic_wilcoxon`` tests every
    not-yet-cached perturbation against the shared control group in one vectorized call instead.

    Verified equivalent on real data, not just synthetic: on a real full-scale perturbation
    (65 cells vs. 5000 control), sorting illico's raw ``(pert, gene)`` output by ``|z_score|``
    descending per perturbation gives a 1000/1000 top-1000 overlap and 0.9999999 Spearman
    correlation against scanpy's own ``rankby_abs=True`` result — but only once
    ``use_continuity``/``tie_correct`` below were set to match what PRESAGE's own
    ``sc.tl.rank_genes_groups(..., method="wilcoxon", rankby_abs=True)`` call actually runs
    (scanpy's own defaults, since PRESAGE never overrides them — checked directly against
    scanpy's ``_rank_genes_groups.py``: ``tie_correct=False``, and scanpy's wilcoxon has no
    continuity correction at all). A first attempt using ``use_continuity=True, tie_correct=True``
    (copied from this repo's own ``de_mwu``, a different context with its own reasons for those
    settings) diverged badly on real sparse count data — only 78% top-1000 overlap, 0.64
    Spearman — because real single-cell data has many exact ties that tie correction handles
    differently; a synthetic Gaussian test with no ties didn't surface this at all. (illico's own
    ``return_as_scanpy=True``/``n_genes`` shortcut is also not used here — checked separately,
    it ranks by raw z-score, not ``|z_score|``, so it silently drops down-regulated genes from a
    top-k cut; ranking illico's raw output ourselves avoids that.)
    """
    from illico import asymptotic_wilcoxon

    if (adata.obs[self.perturb_field] == self.control_key).sum() > 5000:
        control_locs = (
            adata.obs[adata.obs[self.perturb_field] == self.control_key]
            .sample(5000, random_state=0)
            .index
        )
    else:
        control_locs = adata.obs[adata.obs[self.perturb_field] == self.control_key].index

    os.makedirs(self.deg_dir, exist_ok=True)
    degs = {}
    remaining = []
    for pert in set(adata.obs[self.perturb_field]) - {self.control_key}:
        target = os.path.join(self.deg_dir, f"{pert}.json")
        if os.path.isfile(target):
            with open(target) as fp:
                degs[pert] = json.load(fp)
        else:
            remaining.append(pert)

    if remaining:
        remaining_locs = adata.obs[adata.obs[self.perturb_field].isin(remaining)].index
        temp = adata[control_locs.union(remaining_locs)].copy()
        # pandas Index.union() sorts by value (cell barcode string), not by original row
        # position — the resulting reordered slice can leave the sparse matrix's per-row
        # column indices unsorted, which illico's own sparse-matrix handling requires
        # (scanpy/scipy's own sparse ops don't care, so this only ever surfaces here).
        if scipy.sparse.issparse(temp.X):
            temp.X.sort_indices()
        result = asymptotic_wilcoxon(
            temp,
            is_log1p=True,
            group_keys=self.perturb_field,
            reference=self.control_key,
            # os.cpu_count() reports the whole node's CPU count, ignoring SLURM's cgroup
            # allocation (e.g. 128 on a node where our job is only allocated 64) — illico's
            # own numba backend then rejects it outright ("must be between 1 and <allocated>").
            # sched_getaffinity respects the actual usable/allocated CPU set.
            n_threads=len(os.sched_getaffinity(0)),
            alternative="two-sided",
            # Matching scanpy's own defaults, not scPertEval's own de_mwu usage — see this
            # function's docstring for why that distinction mattered here.
            use_continuity=False,
            tie_correct=False,
            return_as_scanpy=False,
        )
        for pert in tqdm(remaining, desc="compute_degs (illico)"):
            sub = result.xs(pert, level=0)
            top = sub["z_score"].abs().sort_values(ascending=False).index[:1000].to_list()
            degs[pert] = top
            # Atomic write: two folds racing to compute the same not-yet-cached perturbation
            # is harmless (same deterministic result, just wasted duplicate work), but a
            # direct `open(..., "w")` isn't — a fold's own "already computed?" check just above
            # (`os.path.isfile(target)`) would happily read a file another fold is still
            # mid-write to, since a file that merely exists but is only partially written still
            # passes that check. Writing to a temp path and renaming into place means any file
            # visible at the real path is always complete — but the temp path itself must be
            # unique per attempt, not just "target + .tmp": two folds racing on the same
            # not-yet-cached pert would otherwise both write through that same fixed temp
            # path concurrently, which is exactly the corruption this is meant to avoid, just
            # moved one level down. tempfile.mkstemp() claims a genuinely unique path.
            target = os.path.join(self.deg_dir, f"{pert}.json")
            fd, tmp_target = tempfile.mkstemp(dir=self.deg_dir, prefix=f"{pert}.", suffix=".json.tmp")
            with os.fdopen(fd, "w") as fp:
                json.dump(top, fp)
            os.replace(tmp_target, target)

    merged_dir = os.path.dirname(self.merged_deg_file) or "."
    fd, tmp_merged = tempfile.mkstemp(
        dir=merged_dir, prefix=os.path.basename(self.merged_deg_file) + ".", suffix=".tmp"
    )
    with os.fdopen(fd, "w") as fp:
        json.dump(degs, fp)
    os.replace(tmp_merged, self.merged_deg_file)


class SmokePRESAGEDataModule(ReploglePRESAGEDataModule):
    """``ReploglePRESAGEDataModule``, adapted for our own custom (non-benchmark) dataset —
    see the module docstring for why each override is needed. ``urls`` is populated with the
    actual dataset name in ``apply()`` (it must contain the name being used, but the name
    itself is only known once ``--dataset`` is parsed, after this class body runs)."""

    # ReplogleDataModule's own compute_degs only loads a pre-existing per-gene DEG cache;
    # rebind to compute_degs_fast (illico-backed, verified-equivalent to the real wilcoxon
    # rank_genes_groups implementation on the common base class — see its own docstring).
    compute_degs = compute_degs_fast

    def prepare_data(self) -> None:
        if self._data_prepared:
            return
        if not os.path.exists(self.merged_deg_file):
            adata = sc.read(self.preprocessed_path)
            self.compute_degs(adata)
        else:
            print(f"Found local preprocessed DEG file {self.merged_deg_file}")
        self._data_prepared = True


class SmokeModelHarness(presage_model_harness_module.ModelHarness):
    """``ModelHarness`` with ``on_test_epoch_end`` (PRESAGE's own internal functional-metrics
    logging, via ``Evaluator``) disabled. Both its branches (gated by ``training.eval_test``)
    unconditionally call ``Evaluator._take_topk_union_degs``, which indexes DEGs against a
    gene panel that doesn't line up at this smoke scale (``KeyError: [...] not in index`` —
    not something ``eval_test`` can dodge, since it only picks which branch, not whether that
    call happens). ``train()``'s actual predictions come from a separate, later
    ``get_predictions()``/``trainer.predict()`` call that never depends on this hook's side
    effects, so skipping it entirely is safe.
    """

    def on_test_epoch_end(self) -> None:
        pass


def _install_csv_logger_patch(metrics_dir: Path) -> None:
    """Wrap pl.Trainer.__init__ so any single `logger=` it's given becomes `[logger,
    CSVLogger(...)]` — Trainer accepts a list of loggers natively, so this is additive, not a
    replacement of train.py's own WandbLogger. version is passed as a plain string
    ("fold_<i>"), not an int, so CSVLogger.log_dir uses it as-is rather than prefixing
    "version_" (see CSVLogger.log_dir's own source) — landing at exactly
    metrics_dir/metrics.csv, with metrics_dir named to match SAVED_MODELS_DIR's own
    <dataset>/fold_<i> convention.
    """
    pl = presage_train_module.pl
    original_init = pl.Trainer.__init__

    def patched_init(self, *args, logger=None, **kwargs):
        if logger is not None and not isinstance(logger, (list, tuple)):
            csv_logger = pl.loggers.CSVLogger(save_dir=str(metrics_dir.parent), name="", version=metrics_dir.name)
            logger = [logger, csv_logger]
        original_init(self, *args, logger=logger, **kwargs)

    pl.Trainer.__init__ = patched_init


def apply(dataset_name: str, checkpoint_dir: Path, metrics_dir: Path) -> None:
    """Monkeypatch train.py's own module namespace with the corrected classes above, and
    register ``dataset_name`` in ``SmokePRESAGEDataModule.urls`` (must be present before
    ``prepare_data()`` runs, or PRESAGE's own hardcoded-benchmark check rejects an
    unrecognized name). Must be called after argparse (both arguments are only known then)
    and before ``presage_train_module.train(config)``.
    """
    SmokePRESAGEDataModule.urls = {**ReploglePRESAGEDataModule.urls, dataset_name: "unused"}
    # train() hardcodes `ReploglePRESAGEDataModule`/`ModelHarness` (bound by name into its own
    # module namespace at import time) rather than taking a datamodule/harness class as a
    # config/argument — same kind of small necessary patch as scGPT's `pert_data.split =
    # "custom"` elsewhere in this repo, not a reimplementation of train() itself.
    presage_train_module.ReploglePRESAGEDataModule = SmokePRESAGEDataModule
    presage_train_module.ModelHarness = SmokeModelHarness
    presage_train_module.os = _CheckpointPreservingOS(checkpoint_dir)
    _install_csv_logger_patch(metrics_dir)
