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
import shutil
import sys
import tempfile
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
OUT_DIR = HERE / "predictions"
SAVED_MODELS_DIR = HERE / "saved_models"
METRICS_DIR = HERE / "metrics"


def build_pathway_files(cache_dir: Path, out_path: Path) -> Path:
    """Rewrite PRESAGE's vendored sample pathway-source list as absolute paths into our
    own fetched cache, one per line, in the format ``model.pathway_files`` expects.

    Content only depends on cache_dir, not on args.fold — identical for every fold of a
    dataset — so this skips rebuilding if out_path already exists, and writes via a
    temp-file-then-atomic-rename when it does rebuild, so a concurrently-running fold can
    never read a partially-written file (see main()'s preprocessed_path handling below for
    the same pattern, applied to a case where it actually bit us once). The temp path itself
    comes from tempfile.mkstemp() rather than a fixed "out_path + suffix" name — two folds
    racing to rebuild this (both missing it at the same time) must not end up writing through
    the *same* temp path concurrently, which would reintroduce the exact corruption this is
    meant to avoid, just one level down."""
    if out_path.exists():
        return out_path
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
    fd, tmp_out_path = tempfile.mkstemp(dir=str(out_path.parent), prefix=out_path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fp:
        fp.write("\n".join(resolved) + "\n")
    os.replace(tmp_out_path, out_path)
    return out_path


def load_presage_official_config(cache_dir: Path) -> dict:
    """Reproduce PRESAGE's own official config-loading recipe exactly, straight from the
    JSON files ``pixi run -e presage fetch-cache`` already fetches — the single source of
    truth for these values instead of a hand-copied second version that can drift from it.

    Traced directly from Genentech/PRESAGE's own ``notebooks/PRESAGE_example.ipynb`` and
    ``src/train_presage.py`` (this env's vendored CLI script): load ``defaults_config.json``,
    layer ``singles_config.json`` on top (the general settings PRESAGE's own example applies
    for single-gene-perturbation tasks — exactly our task type, as opposed to combinatorial),
    then layer a dataset-specific config on top of that (dataset wins on conflict) — here
    ``replogle_k562_essential_config.json``, the closest official match to our own Replogle
    K562 essential-genome data (models/data/prepare_data.py), applied regardless of the
    --dataset flag since smoke/full-scale are both resamples of this one underlying source.
    Underscore keys in the singles/dataset layers are translated to dotted ones
    (``key.replace("_", ".", 1)``) exactly as the notebook's own merge step does."""
    configs_dir = cache_dir / "configs"
    config = json.loads((configs_dir / "defaults_config.json").read_text())
    singles = json.loads((configs_dir / "singles_config.json").read_text())
    dataset_specific = json.loads((configs_dir / "replogle_k562_essential_config.json").read_text())
    singles.update(dataset_specific)
    config.update(
        {
            key.replace("_", ".", 1): value
            for key, value in singles.items()
            if value is not None and key not in {"config", "data_config"}
        }
    )
    return config


def build_config(
    dataset: str,
    data_dir: Path,
    seed_path: Path,
    pathway_files: Path,
    predictions_file: Path,
    epochs: int,
    batch_size: int,
    cache_dir: Path,
) -> dict:
    """PRESAGE's own official config (load_presage_official_config above), plus the
    handful of train_presage.py argparse-only defaults that aren't in any of its shipped
    config JSON (model.set_hidden_size/set_nlayers, data.latent_dim, training.devices),
    plus the overrides needed to point it at our own data/split/cache/output."""
    config = load_presage_official_config(cache_dir)
    config.update(
        {
            "model.set_hidden_size": 32,
            "model.set_nlayers": 2,
            "data.latent_dim": "None",
            "training.devices": 1,
            "model.pathway_files": str(pathway_files),
            # not part of the CLI's own offline default (False) — no wandb login available here
            "training.offline": True,
            # CLI/official default is True, which routes into PRESAGE's own internal functional-metrics
            # suite (top-K union DEG eval etc, model_harness.py's on_test_epoch_end). We don't
            # use any of that — scPertEval scores the raw predictions.h5ad itself downstream —
            # and it's fragile at this smoke scale (KeyErrors indexing DEGs against a gene panel
            # that doesn't line up at this size). False routes to the lighter evaluator(..., False)
            # path, which is what actually produces the predictions we read out below.
            "training.eval_test": False,
            # legacy unused parameter — PRESAGE's own example notebook sets this by hand too,
            # outside of any config file.
            "model.learnable_gene_embedding": False,
            "data.dataset": dataset,
            "data.data_dir": str(data_dir),
            "data.seed": str(seed_path),
            "data.source": "scperturb",
            "training.predictions_file": str(predictions_file),
            # official default (from replogle_k562_essential_config.json) is 16. With
            # use_pseudobulk=True (one row per perturbation), train_dataloader()'s
            # drop_last=True drops the only (partial) batch and trains on zero batches
            # every epoch once batch_size exceeds the fold's training-gene count — true at
            # smoke scale (~7 training genes). Smoke runs must pass a smaller --batch-size
            # explicitly (e.g. 4).
            "data.batch_size": batch_size,
            # official default (singles_config.json) is 1000, matching this script's own
            # --epochs default already; early stopping (patience=10 on val_loss, already
            # active via train()'s own EarlyStopping callback) is expected to end training
            # well before that. Smoke scale needs a much lower value passed explicitly.
            "training.max_epochs": epochs,
        }
    )
    return config


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
        "--batch-size",
        type=int,
        default=16,
        help="data.batch_size (default: 16, PRESAGE's own official replogle_k562_essential override; see build_config)",
    )
    args = parser.parse_args()

    raw = DATA_DIR / args.dataset / "raw.h5ad"
    seed_path = DATA_DIR / args.dataset / "folds" / f"fold_{args.fold}_presage.json"
    out_path = OUT_DIR / args.dataset / f"fold_{args.fold}.h5ad"

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
    if not preprocessed_path.exists():
        # Fold-independent (none of the transforms above depend on args.fold), so skip
        # rebuilding once another fold has already written it, and never write directly to
        # this shared path: anndata's h5ad write isn't atomic, so a fold reading this file
        # while another fold's write is still in progress could see a truncated, unreadable
        # file — not hypothetical, reproduced exactly this for STATE's own equivalent shared
        # h5ad by killing a job mid-write (see state/train_predict.py's prepare_dataset()).
        # tempfile.mkstemp() claims a genuinely unique temp path — two folds racing on a cold
        # cache (both missing preprocessed_path at the same time) must not both write through
        # the same fixed "preprocessed_path + suffix" name concurrently, which would
        # reintroduce the exact corruption this is meant to avoid, just one level down.
        fd, tmp_preprocessed_path = tempfile.mkstemp(
            dir=str(preprocessed_path.parent), prefix=preprocessed_path.name + ".", suffix=".tmp"
        )
        os.close(fd)  # just reserving a unique name; write_h5ad needs to open the path itself
        adata.write_h5ad(tmp_preprocessed_path)
        os.replace(tmp_preprocessed_path, preprocessed_path)

    pathway_files_path = build_pathway_files(CACHE_DIR, dataset_dir / "pathway_files.txt")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_csv = out_path.parent / f"fold_{args.fold}.csv"

    config = build_config(
        dataset=args.dataset,
        data_dir=PERT_DATA_DIR,
        seed_path=seed_path,
        pathway_files=pathway_files_path,
        predictions_file=predictions_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cache_dir=CACHE_DIR,
    )
    # Namespaced by dataset too — otherwise a smoke run and a full run at the same fold
    # index would silently overwrite each other's checkpoint.
    fold_metrics_dir = METRICS_DIR / args.dataset / f"fold_{args.fold}"
    _patch.apply(args.dataset, SAVED_MODELS_DIR / args.dataset / f"fold_{args.fold}", fold_metrics_dir)
    presage_train_module.train(config)
    # CSVLogger (injected by _patch.apply above) writes to fold_metrics_dir/metrics.csv —
    # relocate to a flat file matching predictions/saved_models' own <dataset>/fold_<i>
    # convention, same as STATE's own metrics.csv relocation.
    shutil.move(str(fold_metrics_dir / "metrics.csv"), str(METRICS_DIR / args.dataset / f"fold_{args.fold}.csv"))

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
