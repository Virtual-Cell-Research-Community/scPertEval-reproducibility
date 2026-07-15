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
DATA_DIR = HERE.parent / "data"
PERT_DATA_DIR = HERE / "pert_data"
OUT_DIR = HERE / "smoke_data"
SAVED_MODELS_DIR = HERE / "saved_models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="smoke_k562",
        help="models/data/<dataset>/{raw.h5ad,folds/} to train/predict on (default: smoke_k562)",
    )
    parser.add_argument("--fold", type=int, default=0, help="which fold_<i>.pkl to train/predict on")
    # GEARS has no early stopping (gears/gears.py's train() always runs the full fixed
    # epoch count) — 20 matches train()'s own default, and cellsimbench's own gears.yaml
    # config uses every other architecture default unchanged, only epochs (10) differs.
    # Smoke scale needs a much lower value passed explicitly, e.g. --epochs 2.
    parser.add_argument("--epochs", type=int, default=20, help="training epochs (default: 20, full-scale)")
    # batch_size=32 already matches both smoke and full scale (cell-level batching, plenty
    # of cells even in the smoke subsample) — cellsimbench's gears.yaml uses the same value.
    parser.add_argument("--batch-size", type=int, default=32, help="training batch size (default: 32)")
    # cellsimbench's gears.yaml uses 512; only affects eval throughput, not results, so
    # it's safe as the default at every scale.
    parser.add_argument("--test-batch-size", type=int, default=512, help="eval batch size (default: 512)")
    args = parser.parse_args()

    raw = DATA_DIR / args.dataset / "raw.h5ad"
    fold_path = DATA_DIR / args.dataset / "folds" / f"fold_{args.fold}.pkl"
    out_path = OUT_DIR / f"{args.dataset}_predictions_fold{args.fold}.h5ad"
    adata = ad.read_h5ad(raw)

    pert_data = PertData(str(PERT_DATA_DIR))
    pert_data.new_data_process(dataset_name=args.dataset, adata=adata)
    pert_data.prepare_split(split="custom", split_dict_path=str(fold_path))
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    model = GEARS(pert_data, device="cuda")
    model.model_initialize(hidden_size=64)
    model.train(epochs=args.epochs)
    # GEARS' own save_model() does a single-level os.mkdir(path), which raises if the
    # parent doesn't exist yet. Namespaced by dataset too — otherwise a smoke run and a
    # full run at the same fold index would silently overwrite each other's checkpoint.
    saved_model_dir = SAVED_MODELS_DIR / args.dataset / f"fold_{args.fold}"
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(saved_model_dir))

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
