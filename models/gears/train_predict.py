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
import contextlib
import io
import random
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

ad.settings.allow_write_nullable_strings = True

import patch as _patch
from gears import GEARS, PertData

_patch.apply()

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from workflows.model_runs import add_point_run_arguments, begin_point_model_run

DATA_DIR = HERE.parent / "data"
PERT_DATA_DIR = HERE / "pert_data"

_EPOCH_MSE_RE = re.compile(r"Epoch (\d+): Train Overall MSE: ([\d.]+) Validation Overall MSE: ([\d.]+)\.")
_EPOCH_DE_MSE_RE = re.compile(r"Train Top 20 DE MSE: ([\d.]+) Validation Top 20 DE MSE: ([\d.]+)\.")


@contextlib.contextmanager
def _tee_stderr():
    """GEARS' own train() reports per-epoch metrics only via gears.utils.print_sys, which
    prints to stderr (not stdout) — see that function's own one-line body. Capture it here
    while still passing everything through to the real stderr live, rather than swallowing
    progress output for the run's duration."""
    buf = io.StringIO()
    real_stderr = sys.stderr

    class _Tee:
        def write(self, s):
            real_stderr.write(s)
            buf.write(s)

        def flush(self):
            real_stderr.flush()

    sys.stderr = _Tee()
    try:
        yield buf
    finally:
        sys.stderr = real_stderr


def _parse_gears_metrics(captured: str) -> list[dict]:
    """GEARS prints two lines per epoch (see gears.py's train()): overall MSE first, then
    top-20 DE MSE — always in that order, one pair per epoch, so zipping the two regexes'
    matches positionally is safe."""
    overall = _EPOCH_MSE_RE.findall(captured)
    de = _EPOCH_DE_MSE_RE.findall(captured)
    rows = []
    for (epoch, train_mse, val_mse), (train_mse_de, val_mse_de) in zip(overall, de):
        rows.append(
            {
                "epoch": int(epoch),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "train_mse_de": float(train_mse_de),
                "val_mse_de": float(val_mse_de),
            }
        )
    return rows


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
    add_point_run_arguments(parser)
    args = parser.parse_args()

    run = begin_point_model_run(
        args,
        repository_root=REPOSITORY_ROOT,
        model_id="gears",
        environment="models/pixi.lock#gears",
        configuration={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
        },
    )
    if run.skipped:
        print(f"skipping verified completed run: {run.handle.directory}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    raw = run.raw_path
    fold_path = DATA_DIR / args.dataset / "folds" / f"fold_{args.fold}.pkl"
    out_path = run.prediction_path
    adata = ad.read_h5ad(raw)

    pert_data = PertData(str(PERT_DATA_DIR))
    _patch.load_or_build_pert_data(pert_data, args.dataset, adata)
    pert_data.prepare_split(split="custom", split_dict_path=str(fold_path))
    # model_initialize() below (via get_similarity_network -> get_coexpression_network_from_train)
    # caches a co-expression network keyed by self.split among other things, but actually
    # COMPUTES it from this fold's own set2conditions['train'] — genuinely fold-specific, unlike
    # the PyG graph cache above (which is correctly fold-agnostic). GEARS' own cache filename
    # never includes a fold index at all, so left as "custom" it would build the network from
    # whichever fold happens to run first and every other fold would silently (and incorrectly)
    # reuse it. prepare_split() above requires the literal "custom" (hardcoded allowed-values
    # check) so this can't be passed in directly; override it after, once already validated —
    # get_dataloader()'s own branching only special-cases the literal strings "no_split"/
    # "no_test", so any other value here (including this one) is safe for it.
    pert_data.split = f"custom_fold{args.fold}"
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    model = GEARS(pert_data, device="cuda")
    model.model_initialize(hidden_size=64)
    with _tee_stderr() as captured:
        model.train(epochs=args.epochs)
    metrics_path = run.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_parse_gears_metrics(captured.getvalue())).to_csv(metrics_path, index=False)
    # GEARS' own save_model() does a single-level os.mkdir(path), which raises if the
    # parent doesn't exist yet. Namespaced by dataset too — otherwise a smoke run and a
    # full run at the same fold index would silently overwrite each other's checkpoint.
    saved_model_dir = run.checkpoint_dir
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(saved_model_dir))

    test_perts = sorted(pert_data.set2conditions["test"])
    print(f"test perturbations ({len(test_perts)}): {test_perts}")

    single_gene_perts = [c.split("+")[0] for c in test_perts if c != "ctrl"]
    # GEARS' own perturbation-similarity graph (model.pert_list) is built from a fixed,
    # pre-downloaded GO-annotation reference (PertData.set_pert_genes's gene2go_all.pkl) —
    # independent of our own data, and doesn't cover every gene we measure. GEARS.predict()
    # hard-crashes (ValueError) on any gene it doesn't cover. The frozen cohort was selected
    # against this reference, so any miss is a cohort/reference drift error, not something an
    # adapter may silently drop after training.
    ungraphed = [g for g in single_gene_perts if g not in model.pert_list]
    if ungraphed:
        raise ValueError(
            f"frozen common cohort contains {len(ungraphed)} targets outside GEARS' graph: "
            f"{ungraphed}"
        )
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
    pred_adata.write_h5ad(out_path, compression="gzip")
    checkpoints = sorted(path for path in saved_model_dir.rglob("*") if path.is_file())
    if not checkpoints:
        raise FileNotFoundError(f"GEARS produced no checkpoint under {saved_model_dir}")
    run.complete(implementation_revision="cell-gears==0.1.2", checkpoints=checkpoints)
    print(f"wrote {pred_adata.shape} predictions to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
