"""Compute a k-fold, full-coverage gene-level cross-validation split (default 5 folds, Miller
et al. 2025's own design; pass ``--n-folds 3`` for a fast-iterating smoke test), replacing
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

Outputs, alongside the input raw.h5ad in its own models/data/<dataset>/ folder:
- ``kfold_splits.json`` — human-readable train/val/test gene lists per fold.
- ``folds/fold_<i>.pkl`` — the same split, one pickle per fold, in the
  ``{"train": [...], "val": [...], "test": [...]}`` format GEARS' own
  ``pert_data.prepare_split(split="custom", split_dict_path=...)`` expects (condition strings,
  ``ctrl`` folded into train) — so `train_predict.py` scripts keep using GEARS' own
  `PertData`/dataloader machinery for everything except the random split itself.
- ``folds/fold_<i>_presage.json`` — the same fold again, in PRESAGE's own
  ``scPerturbDataModule.setup``/``pert_to_ind`` split format (bare gene symbols, ``"control"``
  instead of ``ctrl``, folded into train) — see ``to_presage_custom_split``. Same gene membership
  as ``fold_<i>.pkl``, just relabeled, so gears/scgpt/presage are all scored on identical folds.

Usage::

    python models/data/prepare_split.py models/data/replogle22k562/raw.h5ad
    python models/data/prepare_split.py models/data/smoke_k562/raw.h5ad --n-folds 3
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np

N_FOLDS = 5
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


def to_presage_custom_split(fold: dict[str, list[str]]) -> dict[str, list[str]]:
    """A fold's gene lists as PRESAGE's split format.

    Unlike GEARS' ``"GENE+ctrl"`` condition strings, PRESAGE's own preprocessing
    (``PRESAGEDataModule.prepare_data``: ``.replace("+", "_").replace("ctrl", "").strip("_")``,
    empty string -> ``"control"``) reduces a single-gene condition to the bare gene symbol and
    the control condition to ``"control"`` — a combo would become ``"GENE1_GENE2"``, but this
    split is gene-level/single-gene only, so that case doesn't arise here.
    """
    return {
        "train": [*fold["train"], "control"],
        "val": list(fold["val"]),
        "test": list(fold["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_path", help="path to a prepared raw.h5ad (models/data/<dataset>/raw.h5ad)")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS, help=f"number of CV folds (default: {N_FOLDS}, Miller's own value)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    out_json = raw_path.parent / "kfold_splits.json"
    fold_dir = raw_path.parent / "folds"

    adata = ad.read_h5ad(raw_path)
    genes = sorted({c.split("+")[0] for c in adata.obs["condition"].unique() if c != "ctrl"})
    folds = kfold_gene_splits(genes, n_folds=args.n_folds, seed=args.seed)

    out_json.write_text(json.dumps(folds, indent=2) + "\n")
    print(f"wrote {len(folds)}-fold split over {len(genes)} genes to {out_json}")

    fold_dir.mkdir(parents=True, exist_ok=True)
    for i, fold in enumerate(folds):
        split_dict = to_gears_custom_split(fold)
        with open(fold_dir / f"fold_{i}.pkl", "wb") as f:
            pickle.dump(split_dict, f)

        presage_split = to_presage_custom_split(fold)
        (fold_dir / f"fold_{i}_presage.json").write_text(json.dumps(presage_split, indent=2) + "\n")

        print(f"  fold {i}: train={len(split_dict['train'])} val={len(split_dict['val'])} test={len(split_dict['test'])}")


if __name__ == "__main__":
    main()
