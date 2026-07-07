"""Compute a k-fold, full-coverage gene-level cross-validation split (Miller et al. 2025's
design uses 5 folds; `N_FOLDS` is set lower here for a fast-iterating smoke test), replacing
GEARS' own one-shot `prepare_split(split="simulation", ...)`.

GEARS' `simulation` split draws one random 75/~7.5/~17.5 train/val/test partition of genes —
see `gears.data_utils.DataSplitter.get_simulation_split` — and stops there: whatever ~25% of
genes lands in "test" is the only slice of the perturbation space ever scored. Rerunning with a
different seed gives an independent, non-complementary partition with no guarantee of covering
every gene. This script instead partitions every perturbed gene into `N_FOLDS` disjoint blocks
so each gene is the test fold exactly once — a model trained and scored once per fold, with the
folds' test predictions concatenated afterwards, is scored on every perturbation in the dataset,
not an arbitrary slice of it.

The split is gene-level, not condition- or cell-level (`"ABCF1+ctrl"` moves as a unit), the same
principle GEARS' own split uses — only the *partitioning strategy* changes, from one random draw
to k full-coverage folds. No GEARS import is needed for this step (it only needs the perturbation
labels), so it does not need the `gears` pixi environment.

Outputs, per input raw `.h5ad`:
- ``<name>_kfold_splits.json`` — human-readable train/val/test gene lists per fold.
- ``<name>_folds/fold_<i>.pkl`` — the same split, one pickle per fold, in the
  ``{"train": [...], "val": [...], "test": [...]}`` format GEARS' own
  ``pert_data.prepare_split(split="custom", split_dict_path=...)`` expects (condition strings,
  ``ctrl`` folded into train) — so `train_predict.py` scripts keep using GEARS' own
  `PertData`/dataloader machinery for everything except the random split itself.

Usage::

    python models/data/prepare_split.py models/data/smoke_k562_raw.h5ad
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np

N_FOLDS = 3
SEED = 42
VAL_FRAC_OF_REMAINDER = 0.125  # 12.5% of the 80% non-test remainder = 10% of the total


def kfold_gene_splits(genes: list[str], n_folds: int = N_FOLDS, seed: int = SEED) -> list[dict[str, list[str]]]:
    """k-fold train/val/test gene assignment: every gene is the test fold exactly once.

    Parameters
    ----------
    genes : list of str
        Unique perturbed gene symbols (``control`` excluded).
    n_folds : int
        Number of folds (``N_FOLDS`` by default — 5 for Miller et al. 2025's design, lower
        here for a fast-iterating smoke test).
    seed : int
        Random seed for the one-time shuffle.

    Returns
    -------
    list of dict
        One ``{"train": [...], "val": [...], "test": [...]}`` per fold (gene symbols, sorted).
    """
    rng = np.random.default_rng(seed)
    # Sort first: dict/set iteration order isn't guaranteed stable, and rng.permutation's
    # draw depends on the input array's order, so an unstable order would silently change
    # the split even with the same seed.
    shuffled = rng.permutation(sorted(genes))
    fold_bounds = np.array_split(np.arange(len(shuffled)), n_folds)

    folds = []
    for test_idx in fold_bounds:
        test = shuffled[test_idx]
        remainder = np.delete(shuffled, test_idx)
        n_val = int(round(len(remainder) * VAL_FRAC_OF_REMAINDER))
        val, train = remainder[:n_val], remainder[n_val:]
        folds.append(
            {"train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist())}
        )
    return folds


def to_gears_custom_split(fold: dict[str, list[str]]) -> dict[str, list[str]]:
    """A fold's gene lists as GEARS' ``split="custom"`` condition-string format."""
    return {
        "train": [f"{g}+ctrl" for g in fold["train"]] + ["ctrl"],
        "val": [f"{g}+ctrl" for g in fold["val"]],
        "test": [f"{g}+ctrl" for g in fold["test"]],
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <raw.h5ad>")
    raw_path = Path(sys.argv[1])
    stem = raw_path.stem.removesuffix("_raw")
    out_json = raw_path.parent / f"{stem}_kfold_splits.json"
    fold_dir = raw_path.parent / f"{stem}_folds"

    adata = ad.read_h5ad(raw_path)
    genes = sorted({c.split("+")[0] for c in adata.obs["condition"].unique() if c != "ctrl"})
    folds = kfold_gene_splits(genes)

    out_json.write_text(json.dumps(folds, indent=2) + "\n")
    print(f"wrote {len(folds)}-fold split over {len(genes)} genes to {out_json}")

    fold_dir.mkdir(parents=True, exist_ok=True)
    for i, fold in enumerate(folds):
        split_dict = to_gears_custom_split(fold)
        with open(fold_dir / f"fold_{i}.pkl", "wb") as f:
            pickle.dump(split_dict, f)
        print(f"  fold {i}: train={len(split_dict['train'])} val={len(split_dict['val'])} test={len(split_dict['test'])}")


if __name__ == "__main__":
    sys.exit(main())
