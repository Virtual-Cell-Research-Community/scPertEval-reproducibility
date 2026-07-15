"""Smoke test: run royerlab/tfm-perturbation's TabICL baseline on a small subsample and write
a scPertEval-compatible predictions.h5ad.

tfm-perturbation (https://github.com/royerlab/tfm-perturbation) ships two unrelated tasks under
one repo. Its `*_cellmatch` models (``tabpfn_cellmatch.py``, ``tabicl_cellmatch.py``,
``nb_cellmatch.py``) do cross-cell-type transfer — match cells between a labelled source cell
type and held-out target cell types via NN/OT, a different problem from ours. `TabICLMultiviewModel`
(``tabicl_multiview.py``) is the one that matches: for each held-out perturbation in one cell
type, predict its (PCA-reduced) control-subtracted pseudobulk logFC from the *perturbed gene's
own embedding* — using TabICL as an in-context tabular regressor over (gene embedding -> logFC)
pairs drawn from the fold's training perturbations, no gradient training or pretraining of our
own required (TabICL's own checkpoint is pretrained; ICL only). "Multiview" means doing this once
per embedding modality (ESM2, STRING, ...) and fusing the per-modality predictions — see
``tabicl_multiview_base.yaml`` in their repo for the full knob list; only the few overridden
below deviate from those defaults.

TabPFNMultiviewModel (same file, swaps in TabPFNRegressor for TabICLRegressor — otherwise
identical logic, TabICLMultiviewModel is its parent class) was tried first, since "TabPFN" is
the more widely-known name for this class of model. It was dropped: TabPFN's own checkpoint
download now gates on accepting a license through an interactive browser flow
(ux.priorlabs.ai/TABPFN_TOKEN) — a hard requirement with no batch/CI-friendly path in this
environment. TabICL is the same kind of tabular in-context-learning foundation model, its
checkpoint (``jingang/TabICL`` on Hugging Face) downloads with a plain, ungated
``huggingface_hub.hf_hub_download`` — so it's used here instead, no functional loss for this
benchmark's purposes.

Two things fall out of reuse already in this repo:

- The embedding pickles TabICLMultiviewModel needs as ``modality_pickle_paths`` are the exact
  same PRESAGE gene-embedding cache models/presage/cache/ already has (`pixi run -e presage
  fetch-cache`) — this model reuses PRESAGE's own precomputed embeddings as its tabular
  features, replacing PRESAGE's GNN with a pretrained tabular foundation model on the same
  inputs. MODALITY_PICKLE_PATHS below mirrors their own paper config
  (``experiments/configs/modality/top_plus_perturbseq_k562_essential.yaml``), minus the
  cross-dataset Perturb-seq pseudobulk modalities (RPE1/HEPG2/Jurkat pseudobulk features) —
  we don't have those other datasets processed through their pipeline, only our own K562 smoke
  subsample.
- ``models/data/smoke_k562_folds/fold_<i>_presage.json`` (built by
  ``models/data/prepare_split.py``'s ``to_presage_custom_split`` for models/presage) is already
  in the exact flat ``{"train": [...], "val": [...], "test": [...]}`` schema
  ``tfm_perturbation.utils.load_split_json`` expects — reused as-is, no new split-conversion
  needed, so gears/scgpt/presage/sclambda/state/tabicl are all scored on identical held-out genes.

``presage_data_root=None`` skips their PRESAGE-cache-backed auxiliary eval files (geneset/
perturbation-cluster/ncells files) — optional inputs their own code already treats as "None" when
absent; scPertEval scores the raw predictions.h5ad itself downstream, same reasoning as PRESAGE's
own ``training.eval_test=False`` and STATE's ``--predict-only``.

No PCA-component or TabICL-regressor overrides are set — ``x_pca_n_components``/
``y_pca_n_components`` default to 128 but the model itself caps them to
``min(requested, n_train_perturbations, n_features)`` (our fold's train context is only ~7-8
perturbations), so the shipped defaults already do the right thing at this scale without
hand-tuning. Same "take the official preset as-is" reasoning as STATE's ``model=pertsets``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from tfm_perturbation.datamodules.presage import PresageAnnDataDataModule
from tfm_perturbation.models.transcriptomics.tabicl_multiview import TabICLMultiviewModel

ad.settings.allow_write_nullable_strings = True

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
PRESAGE_CACHE = HERE.parent / "presage" / "cache" / "cache"
RUN_DIR = HERE / "runs"
OUT_DIR = HERE / "smoke_data"
CONTROL_LABEL = "control"

# Mirrors tfm-perturbation's own experiments/configs/modality/top_plus_perturbseq_k562_essential.yaml,
# minus the cross-dataset Perturb-seq pseudobulk modalities (perturbseq_rpe1_essential/hepg2/jurkat)
# — those need other datasets run through their pipeline, which we don't have here.
MODALITY_PICKLE_PATHS = {
    "funk_cellprofiler": str(PRESAGE_CACHE / "other_embeddings" / "funk.cellprofiler.embeddings.pkl"),
    "biogpt_gene2biogpt": str(PRESAGE_CACHE / "other_embeddings" / "biogpt_emb_gene2biogpt.pkl"),
    "esm_gene2esm": str(PRESAGE_CACHE / "other_embeddings" / "esm_emb_gene2esm.pkl"),
    "stringdb_human_medium": str(PRESAGE_CACHE / "pathway_embeddings" / "stringdb.human.medium.pkl"),
    "stringdb_human_highest": str(PRESAGE_CACHE / "pathway_embeddings" / "stringdb.human.highest.pkl"),
    "crispr_gene_effect_depmap": str(PRESAGE_CACHE / "other_embeddings" / "CRISPRGeneEffectDepMap.pkl"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="smoke_k562",
        help="models/data/<dataset>_raw.h5ad + <dataset>_folds/ to train/predict on (default: smoke_k562)",
    )
    parser.add_argument("--fold", type=int, default=0, help="which fold in <dataset>_folds/ to train/predict on")
    args = parser.parse_args()

    raw = DATA_DIR / f"{args.dataset}_raw.h5ad"
    split_json_path = DATA_DIR / f"{args.dataset}_folds" / f"fold_{args.fold}_presage.json"
    # Namespaced by dataset too — otherwise a smoke run and a full run at the same fold
    # index would silently overwrite each other's artifact.
    run_dir = RUN_DIR / args.dataset / f"fold_{args.fold}"

    datamodule = PresageAnnDataDataModule(
        dataset=args.dataset,
        data_file_path=str(raw),
        split_json_path=str(split_json_path),
        perturbation_col="perturbation",
        control_label=CONTROL_LABEL,
    )

    model = TabICLMultiviewModel(
        split_json_path=str(split_json_path),
        control_label=CONTROL_LABEL,
        perturbation_key="perturbation",
        # Our smoke adata has no batch/plate column — falls back to global-control logFC.
        batch_key=None,
        modality_pickle_paths=MODALITY_PICKLE_PATHS,
        presage_data_root=None,
        degs_cache_dir=str(run_dir / "degs"),
    )

    model.fit(output_dir=run_dir, datamodule=datamodule)

    with open(run_dir / "tabicl_multiview_artifact.pkl", "rb") as f:
        artifact = pickle.load(f)
    test = artifact["fused"]["test"]
    gene_names = artifact["metadata"]["gene_names"]

    pred_adata = ad.AnnData(
        X=np.asarray(test["predictions"], dtype=np.float32),
        obs={"perturbation": np.asarray(test["perturbations"])},
    )
    pred_adata.var_names = pd.Index(gene_names)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.dataset}_predictions_fold{args.fold}.h5ad"
    pred_adata.write_h5ad(out_path)
    print(f"wrote {pred_adata.shape} predictions to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
