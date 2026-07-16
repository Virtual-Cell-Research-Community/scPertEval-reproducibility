"""Smoke test: train PRESAGE on a small subsample and write a scPertEval-compatible
predictions.h5ad.

Uses our own 5-fold cross-validation split (models/data/prepare_split.py) rather than
PRESAGE's own named-benchmark/JSON-split convention — see that script's
``to_presage_custom_split`` for how a fold's gene lists get relabeled into PRESAGE's format
(bare gene symbols, ``"control"`` for the control condition). Same underlying gene
partition as models/gears and models/scgpt, so all three models are scored on identical
folds.

PRESAGE's own ``PRESAGEDataModule``/``ReploglePRESAGEDataModule`` is built around its own
named benchmark datasets (a hardcoded ``urls`` whitelist; ``prepare_data`` downloads+extracts
a zip, then normalizes/log1ps/HVG-filters it) and, in ``train.py``'s actual class
(``ReploglePRESAGEDataModule``, MRO puts ``ReplogleDataModule`` after ``PRESAGEDataModule``),
``compute_degs`` only *loads* a pre-existing per-gene DEG cache shipped for those datasets
rather than computing DEGs — silently empty for any dataset that isn't one of theirs, which
later crashes ``ModelHarness.ind_to_pert``'s ``assert key in self.degs``. The corrections for
all of this — a working ``compute_degs``, a datamodule/harness that accept a custom dataset,
and the checkpoint-preserving ``os`` proxy — live in ``patch.py`` (``_patch.apply()`` below),
kept separate so this script stays a straightforward reading of PRESAGE's own ``train()`` entry
point (Trainer/EarlyStopping/checkpointing/prediction-extraction logic, all untouched).

``model.pathway_files`` uses PRESAGE's own curated list of external knowledge-source
embeddings (``sample.knowledge_experimental.txt``, vendored via presage-src), fetched into
models/presage/cache/ by `pixi run -e presage fetch-cache`. The file's own lines are
CWD-relative (``../cache/other_embeddings/X.pkl``, meant for a process running from
PRESAGE's own repo root) — ``build_pathway_files`` rewrites them into absolute paths into
our own cache instead of relying on that assumption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ad.settings.allow_write_nullable_strings = True

import datamodule as presage_datamodule_module  # noqa: E402
import train as presage_train_module  # noqa: E402

import patch as _patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
PERT_DATA_DIR = HERE / "pert_data"
CACHE_DIR = HERE / "cache"  # populated by `pixi run -e presage fetch-cache`
OUT_DIR = HERE / "smoke_data"
SAVED_MODELS_DIR = HERE / "saved_models"


def build_pathway_files(cache_dir: Path, out_path: Path) -> Path:
    """Rewrite PRESAGE's vendored sample pathway-source list as absolute paths into our
    own fetched cache, one per line, in the format ``model.pathway_files`` expects."""
    sample_file = (
        Path(presage_datamodule_module.__file__).parent
        / "presage_sample_files"
        / "prior_files"
        / "sample.knowledge_experimental.txt"
    )
    resolved = []
    for line in sample_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # lines look like "../cache/other_embeddings/X.pkl" or "../cache/pathway_embeddings/Y.pkl"
        rel = line.split("cache/", 1)[1]
        resolved.append(str(cache_dir / "cache" / rel))
    out_path.write_text("\n".join(resolved) + "\n")
    return out_path


def build_config(
    dataset: str,
    data_dir: Path,
    seed_path: Path,
    pathway_files: Path,
    predictions_file: Path,
    epochs: int,
    batch_size: int,
) -> dict:
    """A flat, dotted-key config matching train_presage.py's own CLI defaults (its argparse
    is bypassed entirely — this dict goes straight into PRESAGE's own ``train()``, which
    groups dotted keys into ``{"model": ..., "data": ..., "training": ..., ...}`` itself),
    with the handful of overrides needed to point it at our own data/split/cache/output."""
    return {
        "model.lr": 2.4e-3,
        "model.weight_decay": 2.17e-13,
        "model.momentum": 0.9,
        "model.optimizer": "Adam",
        "data.noisy_pseudobulk": False,
        "model.n_nmf_embedding": 128,
        "model.node2vec_walk_length": 100,
        "model.node2vec_context_size": 5,
        "model.node2vec_walks_per_node": 3,
        "model.node2vec_num_negative_samples": 3,
        "model.node2vec_p": 1.0,
        "model.node2vec_q": 2.0,
        "model.node2vec_batchsize": 32,
        "model.pathway_files": str(pathway_files),
        "model.embedding_files": "None",
        "model.dim_red_alg": "Node2Vec",
        "model.cosine_loss_scale": 1.0,
        "model.contrastive_loss_scale": 0,
        "model.contrastive_nneigh": 5,
        "model.contrastive_neigh_metric": "minkowski",
        "model.univariate_cosine_loss_scale": 0.0,
        "model.mse_loss_scale": 1.0,
        "model.vector_norm_loss_scale": 1.0,
        "model.added_singles_loss_scale": 1.0,
        "model.nonlinear_offset_sparsity_loss_scale": 0.001,
        "model.item_hidden_size": 512,
        "model.item_nlayers": 0,
        "model.pathway_item_hidden_size": 128,
        "model.pathway_item_nlayers": 2,
        "model.pathway_pool_type": "sum",
        "model.pathway_weight_type": "gat",
        "model.pool_nlayers": 2,
        "model.softmax_temperature": 0.15,
        "model.gat_weight": 0.85,
        "model.linear_coefficient_hidden_size": 32,
        "model.linear_coefficient_nlayers": 3,
        "model.linear_coefficient_features": "both",
        "model.batch_norm": True,
        "model.pathway_pma_layer_norm": False,
        "model.set_hidden_size": 32,
        "model.set_nlayers": 2,
        "model.pca_trainable_decoder": False,
        "model.pca_loss_scale": 1.0,
        # not part of the CLI's own offline default (False) — no wandb login available here
        "training.offline": True,
        # CLI default is True, which routes into PRESAGE's own internal functional-metrics
        # suite (top-K union DEG eval etc, model_harness.py's on_test_epoch_end). We don't
        # use any of that — scPertEval scores the raw predictions.h5ad itself downstream —
        # and it's fragile at this smoke scale (KeyErrors indexing DEGs against a gene panel
        # that doesn't line up at this size). False routes to the lighter evaluator(..., False)
        # path, which is what actually produces the predictions we read out below.
        "training.eval_test": False,
        "model.learnable_gene_embedding": False,
        "hyperparameter_tune.sweep": False,
        "data.dataset": dataset,
        "data.data_dir": str(data_dir),
        "data.seed": str(seed_path),
        "data.latent_dim": "None",
        "model.n_neigh_prune": "5",
        "model.min_cos": -1.0,
        "model.min_genes_per_kg": 10,
        "model.input_preparation": "prep_gene_embeddings",
        "data.source": "scperturb",
        "model.gex_coexpression_style": "coexpression",
        "training.predictions_file": str(predictions_file),
        "training.embedding_file": None,
        "training.attention_file": None,
        "model.harness": "ModelHarness",
        "data.use_pseudobulk": True,
        "data.preprocessing_zscore": False,
        # CLI default is 16. With use_pseudobulk=True (one row per perturbation),
        # train_dataloader()'s drop_last=True drops the only (partial) batch and trains
        # on zero batches every epoch once batch_size exceeds the fold's training-gene
        # count — true at smoke scale (~7 training genes), not at full scale. Smoke runs
        # must pass a smaller --batch-size explicitly (e.g. 4).
        "data.batch_size": batch_size,
        # CLI default is 10000; early stopping (patience=10 on val_loss, already active
        # via train()'s own EarlyStopping callback) is expected to end training well
        # before that. 1000 matches both PRESAGE's own hyperparameter-sweep config and
        # cellsimbench's presage.yaml. Smoke scale needs a much lower value passed
        # explicitly so iteration stays fast.
        "training.max_epochs": epochs,
        "training.precision": "32",
        "hyperparameter_tune.tag": "",
        "training.devices": 1,
        "training.seed": 42,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="smoke_k562",
        help="models/data/<dataset>/{raw.h5ad,folds/} to train/predict on (default: smoke_k562)",
    )
    parser.add_argument("--fold", type=int, default=0, help="which fold_<i>_presage.json to train/predict on")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="training.max_epochs (default: 1000, full-scale; early stopping usually ends training well before this)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="data.batch_size (default: 32, full-scale; see build_config)"
    )
    args = parser.parse_args()

    raw = DATA_DIR / args.dataset / "raw.h5ad"
    seed_path = DATA_DIR / args.dataset / "folds" / f"fold_{args.fold}_presage.json"
    out_path = OUT_DIR / f"{args.dataset}_predictions_fold{args.fold}.h5ad"

    if not (CACHE_DIR / "cache" / "pathway_embeddings").exists():
        raise FileNotFoundError(f"{CACHE_DIR} has no gene-embedding cache — run `pixi run -e presage fetch-cache` first")

    # presage.py/train.py hardcode a few CWD-relative paths of their own (the Node2Vec
    # embeddings it computes from *our* training data cache to "./cache/pathway_embeddings/",
    # not to be confused with CACHE_DIR above — the Zenodo download; ModelCheckpoint's
    # dirpath="./saved_models"; wandb's own run directory) — chdir here so all of that lands
    # under models/presage/ instead of leaking into the shared models/ workspace root.
    os.chdir(HERE)

    dataset_dir = PERT_DATA_DIR / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(raw)
    adata.var["measured_gene"] = True
    # sc.tl.rank_genes_groups (PRESAGE's own compute_degs) assumes this uns key is present
    # once .X is log-transformed (scanpy issue #2239) — our raw file is already on that
    # scale (models/data/prepare_data.py), just never went through sc.pp.log1p itself.
    adata.uns["log1p"] = {"base": None}
    # Evaluator.__init__ hard-requires obs["gene"] (never actually set anywhere in PRESAGE's
    # own vendored code — assumed to already exist on scPerturb.org-sourced data). Same
    # information as obs["perturbation"] here, so just alias it.
    adata.obs["gene"] = adata.obs["perturbation"]
    preprocessed_path = dataset_dir / f"{args.dataset}_processed.h5ad"
    adata.write_h5ad(preprocessed_path)

    pathway_files_path = build_pathway_files(CACHE_DIR, dataset_dir / "pathway_files.txt")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_csv = OUT_DIR / f"{args.dataset}_predictions_fold{args.fold}.csv"

    config = build_config(
        dataset=args.dataset,
        data_dir=PERT_DATA_DIR,
        seed_path=seed_path,
        pathway_files=pathway_files_path,
        predictions_file=predictions_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    # Namespaced by dataset too — otherwise a smoke run and a full run at the same fold
    # index would silently overwrite each other's checkpoint.
    _patch.apply(args.dataset, SAVED_MODELS_DIR / args.dataset / f"fold_{args.fold}")
    presage_train_module.train(config)

    # datamodule.setup("test") deliberately bundles train perturbations in alongside test
    # ones ("train are added here so it computes the geometric eval on train" — datamodule.py),
    # so the CSV train() writes has a row per test *and* train gene. Keep only the held-out
    # test-fold genes, matching gears/scgpt's own predictions.h5ad convention.
    test_genes = json.loads(seed_path.read_text())["test"]
    preds = pd.read_csv(predictions_csv, index_col=0).loc[test_genes]
    pred_adata = ad.AnnData(X=preds.to_numpy().astype(np.float32), obs={"perturbation": preds.index.to_numpy()})
    pred_adata.var_names = pd.Index(preds.columns)
    pred_adata.write_h5ad(out_path)
    print(f"wrote {pred_adata.shape} predictions to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
