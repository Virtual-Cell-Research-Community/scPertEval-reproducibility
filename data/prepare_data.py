"""Prepare a Miller et al. 2025-consistent raw dataset for models/, optionally reduced to a
smoke-test scale.

Loads the canonical, scPertEval-processed dataset from ``models/data/source/<source>_processed_complete.h5ad``
— downloaded via ``docs/user-guide/datasets.md``'s ``gsutil cp gs://scperteval/processed/...``
step directly into ``models/data/source/`` rather than the project's own top-level ``data/``
dir, so this reproduction scaffold's data dependency is self-contained under ``models/``. It's
scPertEval's own generic "normalize + trim" prep (see ``docs/notebooks/02_preparing_a_dataset.ipynb``):
``filter_cells(min_genes=200)``, ``filter_genes(min_cells=3)``, ``normalize_total(1e4)`` +
``log1p``. This script then applies the parts of
Miller et al. 2025's B.2 protocol (shiftbioscience/Perturbation-Models-Outperform-Baselines's
``data/replogle22k562/get_data.py``) that scPertEval's own prep doesn't already share with it —
see ``docs/user-guide/reproducing-literature-protocols.md`` for the full comparison this closes:

- Drop perturbations whose own target gene isn't itself a measured feature. GEARS/scGPT's data
  pipeline (``gears.utils.create_cell_graph_dataset_for_prediction``) hard-crashes on these at
  predict time, and even where it doesn't crash (``gears.pertdata.PertData.get_pert_idx``'s own
  ``try/except`` at train time, which falls back to ``pert_idx=[-1]``) it silently trains on
  that cell's real post-perturbation expression while flagging the graph as if no gene had been
  targeted at all. Not something Miller's own ``get_data.py`` resolves either — it only prints a
  ``Missing genes`` diagnostic after the fact and leaves the cells in — but dropped here rather
  than left to crash or silently mistrain a model.
- Per-perturbation downsampling: non-control conditions capped at ``round(mean)`` cells (over
  the surviving population), control capped at ``MAX_CELLS_CONTROL`` (8192, Miller's own value).
- Gene panel restricted to the top-``N_HVGS`` (8192, Miller's value) HVGs union every remaining
  perturbed gene (guaranteed already measured, since the drop above removed anything that
  wasn't) — reduces training cost without ever losing a perturbation's own target gene.

Not applied: Miller's own >=12-cells (then >=4-cells post-downsampling) perturbation filters —
both are a no-op for replogle22k562 specifically (its minimum per-perturbation count is already
30, well above either threshold) — kept as explicit checks below rather than silently skipped,
so a future dataset where they'd actually matter doesn't fall through unnoticed.

Usage::

    python models/data/prepare_data.py
    python models/data/prepare_data.py --smoke --dataset-name smoke_k562 --n-perts 12 --cells-per-pert 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ad.settings.allow_write_nullable_strings = True

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source"

MIN_CELLS_PER_PERT = 12  # Miller's first filter, pre-downsampling
MIN_CELLS_PER_PERT_POST = 4  # Miller's second filter, post-downsampling
MAX_CELLS_CONTROL = 8192  # Miller's control cap
N_HVGS = 8192  # Miller's HVG count
SEED = 42


def _target_genes(condition: str) -> list[str]:
    """A condition's own perturbed gene(s) — ``"control"`` has none, a combo splits on ``"+"``
    (scPertEval's own combination convention, docs/user-guide/datasets.md)."""
    return [] if condition == "control" else condition.split("+")


def apply_miller_fixes(adata: ad.AnnData, seed: int = SEED) -> ad.AnnData:
    """The parts of Miller et al. 2025's B.2 protocol scPertEval's own prep doesn't already
    apply — see the module docstring for what each step does and why."""
    rng = np.random.default_rng(seed)

    pert = adata.obs["perturbation"].to_numpy()
    var_set = set(adata.var_names)
    measured = np.array([all(g in var_set for g in _target_genes(c)) for c in pert])
    adata = adata[measured].copy()
    pert = adata.obs["perturbation"].to_numpy()

    counts = pd.Series(pert).value_counts()
    eligible = counts[(counts >= MIN_CELLS_PER_PERT) | (counts.index == "control")].index
    adata = adata[np.isin(pert, eligible)].copy()
    pert = adata.obs["perturbation"].to_numpy()

    counts = pd.Series(pert).value_counts()
    max_cells = round(counts.drop("control", errors="ignore").mean())
    keep_idx = []
    for cond, n in counts.items():
        idx = np.where(pert == cond)[0]
        cap = MAX_CELLS_CONTROL if cond == "control" else max_cells
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep_idx.append(idx)
    adata = adata[np.sort(np.concatenate(keep_idx))].copy()
    pert = adata.obs["perturbation"].to_numpy()

    counts = pd.Series(pert).value_counts()
    eligible = counts[(counts >= MIN_CELLS_PER_PERT_POST) | (counts.index == "control")].index
    adata = adata[np.isin(pert, eligible)].copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVGS)
    hvg_genes = set(adata.var_names[adata.var["highly_variable"]])
    perturbed_genes = {g for c in adata.obs["perturbation"].unique() for g in _target_genes(c)}
    genes_to_keep = sorted(hvg_genes | perturbed_genes)
    adata = adata[:, genes_to_keep].copy()
    # highly_variable_genes() adds several non-string .var columns (means, dispersions, ...)
    # only needed transiently to pick genes_to_keep above — drop them rather than carry them
    # into the object-dtype normalization below, which anndata can't write for booleans/floats.
    adata.var = adata.var[[]]

    return adata


def smoke_subsample(
    adata: ad.AnnData, n_perts: int, cells_per_pert: int, control_cells: int, seed: int
) -> ad.AnnData:
    """Further reduce an already Miller-fixed population to a tiny smoke-test scale — the same
    selection logic this script used to apply directly to the raw processed file, now layered
    on top of the shared Miller-consistent population instead."""
    rng = np.random.default_rng(seed)
    pert = adata.obs["perturbation"].to_numpy()

    pert_counts = pd.Series(pert).value_counts()
    pert_counts = pert_counts[pert_counts.index != "control"]
    eligible = np.sort(pert_counts[pert_counts >= cells_per_pert].index.to_numpy())
    chosen_perts = rng.choice(eligible, size=min(n_perts, len(eligible)), replace=False)

    keep_idx = []
    control_idx = np.where(pert == "control")[0]
    keep_idx.append(rng.choice(control_idx, size=min(control_cells, len(control_idx)), replace=False))
    for p in chosen_perts:
        idx = np.where(pert == p)[0]
        keep_idx.append(rng.choice(idx, size=min(cells_per_pert, len(idx)), replace=False))
    keep_idx = np.sort(np.concatenate(keep_idx))
    return adata[keep_idx].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", default="replogle22k562", help="data/<source>_processed_complete.h5ad to prepare (default: replogle22k562)"
    )
    parser.add_argument(
        "--dataset-name", default=None, help="output name under models/data/<name>/ (default: --source's value)"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="additionally subsample to a tiny smoke-test scale (see --n-perts etc.)"
    )
    parser.add_argument("--n-perts", type=int, default=12, help="[--smoke] number of perturbations to keep (default: 12)")
    parser.add_argument("--cells-per-pert", type=int, default=100, help="[--smoke] per-perturbation cell cap (default: 100)")
    parser.add_argument("--control-cells", type=int, default=500, help="[--smoke] control cell cap (default: 500)")
    parser.add_argument("--cell-type", default="K562", help="cell_type label to assign (default: K562)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    args = parser.parse_args()

    source_path = SOURCE_DIR / f"{args.source}_processed_complete.h5ad"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} missing — run `gsutil cp gs://scperteval/processed/{source_path.name} {SOURCE_DIR}/` "
            "first (see docs/user-guide/datasets.md)"
        )
    dataset_name = args.dataset_name or args.source
    out_path = HERE / dataset_name / "raw.h5ad"

    adata = ad.read_h5ad(source_path)
    adata = apply_miller_fixes(adata, seed=args.seed)
    print(f"after Miller fixes: {adata.shape}, {adata.obs['perturbation'].nunique()} perturbations")

    if args.smoke:
        adata = smoke_subsample(adata, args.n_perts, args.cells_per_pert, args.control_cells, args.seed)
        print(f"after smoke subsample: {adata.shape}, {adata.obs['perturbation'].nunique()} perturbations")

    pert = adata.obs["perturbation"].to_numpy()
    adata.obs["condition"] = np.where(pert == "control", "ctrl", [f"{g}+ctrl" for g in pert])
    adata.obs["cell_type"] = args.cell_type
    adata.var["gene_name"] = adata.var_names
    adata.obs.index = pd.Index(np.asarray(adata.obs.index, dtype=object), name=adata.obs.index.name)
    adata.var.index = pd.Index(np.asarray(adata.var.index, dtype=object), name=adata.var.index.name)
    for col in adata.obs.columns:
        adata.obs[col] = np.asarray(adata.obs[col], dtype=object)
    for col in adata.var.columns:
        adata.var[col] = np.asarray(adata.var[col], dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"wrote {adata.shape} to {out_path}")


if __name__ == "__main__":
    main()
