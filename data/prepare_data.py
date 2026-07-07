"""Subsample a scPertEval dataset into a small GEARS-formatted AnnData for a smoke test.

Shared across every model (GEARS, scGPT, ...): they all read this same raw file so their
held-out test genes line up and their predictions are comparable against one ground truth.

GEARS expects ``adata.obs['condition']`` formatted as ``"ctrl"`` for control cells and
``"<GENE>+ctrl"`` for single-gene perturbations, plus ``adata.obs['cell_type']`` and
``adata.var['gene_name']``. See ``PertData.new_data_process`` in snap-stanford/GEARS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ad.settings.allow_write_nullable_strings = True

SOURCE = Path(__file__).resolve().parents[2] / "data" / "replogle22k562_processed_complete.h5ad"
OUT = Path(__file__).resolve().parent / "smoke_k562_raw.h5ad"
N_PERTS = 12
CELLS_PER_PERT = 100
CONTROL_CELLS = 500
CELL_TYPE = "K562"
SEED = 0


def main() -> None:
    adata = ad.read_h5ad(SOURCE)
    rng = np.random.default_rng(SEED)

    pert_counts = adata.obs["perturbation"].value_counts()
    pert_counts = pert_counts[pert_counts.index != "control"]
    # Sort explicitly rather than relying on value_counts()'s tie-break order for equal
    # counts, which isn't guaranteed stable across pandas versions — rng.choice's draw
    # depends on this array's order, so an unstable order silently changes which genes
    # get chosen even with the same seed.
    eligible = np.sort(pert_counts[pert_counts >= CELLS_PER_PERT].index.to_numpy())
    # Require the perturbed gene to itself be in the measured panel: cell-gears==0.0.2
    # (scGPT's pinned GEARS version) looks up each perturbation's target gene in
    # var['gene_name'] with no fallback and crashes (IndexError) if it's absent; newer
    # cell-gears tolerates this (try/except + a broader gene vocabulary), but excluding
    # these keeps the raw data usable by every GEARS version uniformly.
    eligible = eligible[np.isin(eligible, adata.var_names)]
    chosen_perts = rng.choice(eligible, size=min(N_PERTS, len(eligible)), replace=False)

    keep_idx = []
    obs = adata.obs
    control_idx = np.where(obs["perturbation"].to_numpy() == "control")[0]
    keep_idx.append(rng.choice(control_idx, size=min(CONTROL_CELLS, len(control_idx)), replace=False))
    for p in chosen_perts:
        idx = np.where(obs["perturbation"].to_numpy() == p)[0]
        keep_idx.append(rng.choice(idx, size=min(CELLS_PER_PERT, len(idx)), replace=False))
    keep_idx = np.sort(np.concatenate(keep_idx))

    sub = adata[keep_idx].copy()
    pert = sub.obs["perturbation"].to_numpy()
    sub.obs["condition"] = np.where(pert == "control", "ctrl", [f"{g}+ctrl" for g in pert])
    sub.obs["cell_type"] = CELL_TYPE
    sub.var["gene_name"] = sub.var_names
    sub.obs.index = pd.Index(np.asarray(sub.obs.index, dtype=object), name=sub.obs.index.name)
    sub.var.index = pd.Index(np.asarray(sub.var.index, dtype=object), name=sub.var.index.name)
    for col in sub.obs.columns:
        sub.obs[col] = np.asarray(sub.obs[col], dtype=object)
    for col in sub.var.columns:
        sub.var[col] = np.asarray(sub.var[col], dtype=object)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(OUT)
    print(f"wrote {sub.shape} to {OUT}")
    print(f"perturbations: {len(chosen_perts)} + control, cells/pert cap {CELLS_PER_PERT}")


if __name__ == "__main__":
    sys.exit(main())
