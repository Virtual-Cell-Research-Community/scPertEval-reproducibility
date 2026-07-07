"""Smoke test: train GEARS on a small subsample and write a scPertEval-compatible predictions.h5ad.

Uses our own 5-fold cross-validation split (models/data/prepare_split.py) rather than GEARS'
own ``simulation`` split — see that script's docstring for why (full gene coverage across
folds vs. one arbitrary random holdout). GEARS' own ``PertData``/dataloader machinery is
otherwise unchanged: our fold is handed to it via ``split="custom"``. Predictions are one
averaged profile per held-out perturbation (GEARS.predict returns a point estimate, not
per-cell samples).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ad.settings.allow_write_nullable_strings = True

# cell-gears 0.1.2 indexes scipy sparse matrices with a raw pandas boolean Series
# (e.g. `adata.X[adata.obs.condition == 'ctrl']`). Older scipy quietly coerced the
# Series via `np.asarray` first; current scipy calls `.nonzero()` on it directly,
# and pandas dropped `Series.nonzero()` back in pandas 1.0. Restore it.
if not hasattr(pd.Series, "nonzero"):
    pd.Series.nonzero = lambda self: self.to_numpy().nonzero()

from gears import GEARS, PertData  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "smoke_k562_raw.h5ad"
FOLD_DIR = HERE.parent / "data" / "smoke_k562_folds"
PERT_DATA_DIR = HERE / "pert_data"
OUT_DIR = HERE / "smoke_data"
EPOCHS = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0, help="which fold_<i>.pkl to train/predict on")
    args = parser.parse_args()

    fold_path = FOLD_DIR / f"fold_{args.fold}.pkl"
    out_path = OUT_DIR / f"smoke_k562_predictions_fold{args.fold}.h5ad"
    adata = ad.read_h5ad(RAW)

    pert_data = PertData(str(PERT_DATA_DIR))
    pert_data.new_data_process(dataset_name="smoke_k562", adata=adata)
    pert_data.prepare_split(split="custom", split_dict_path=str(fold_path))
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    model = GEARS(pert_data, device="cuda")
    model.model_initialize(hidden_size=64)
    model.train(epochs=EPOCHS)

    test_perts = sorted(pert_data.set2conditions["test"])
    print(f"test perturbations ({len(test_perts)}): {test_perts}")

    single_gene_perts = [c.split("+")[0] for c in test_perts if c != "ctrl"]
    pred_lists = [[g] for g in single_gene_perts]
    preds = model.predict(pred_lists)

    gene_names = np.asarray(pert_data.gene_names.values)
    rows, conditions = [], []
    for gene, pred_key in zip(single_gene_perts, preds, strict=True):
        rows.append(preds[pred_key])
        conditions.append(gene)

    pred_adata = ad.AnnData(
        X=np.vstack(rows).astype(np.float32),
        obs={"perturbation": conditions},
    )
    pred_adata.var_names = pd.Index(gene_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_adata.write_h5ad(out_path)
    print(f"wrote {pred_adata.shape} predictions to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
