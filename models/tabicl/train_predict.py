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
- ``models/data/<dataset>/folds/fold_<i>_tabicl.json`` (built by
  ``models/data/prepare_split.py``'s ``to_tabicl_custom_split``) is in
  the exact flat ``{"train": [...], "val": [...], "test": [...]}`` schema
  ``tfm_perturbation.utils.load_split_json`` expects. Its test genes are identical to every
  other model's; because TabICLv2 performs no checkpoint selection, the frozen validation genes
  are merged into its in-context development examples rather than unnecessarily discarded.

``presage_data_root=None`` skips their PRESAGE-cache-backed auxiliary eval files (geneset/
perturbation-cluster/ncells files) — optional inputs their own code already treats as "None" when
absent; scPertEval scores the raw predictions.h5ad itself downstream, same reasoning as PRESAGE's
own ``training.eval_test=False`` and STATE's ``--predict-only``.

The pinned upstream ``TabICLMultiviewModel.fit`` nevertheless performs a single-cell Wilcoxon
DEG calculation and PRESAGE-style metric evaluation before it serializes its artifact. That work
is deliberately bypassed here through ``fit_prediction_only``: it is not part of model fitting,
does not change the fused predictions, scales pathologically on the full GWPS object, and would
duplicate the shared scPertEval evaluation applied to every model after predictions are frozen.

No PCA-component or TabICL-regressor overrides are set — ``x_pca_n_components``/
``y_pca_n_components`` default to 128 but the model itself caps them to
``min(requested, n_train_perturbations, n_features)`` (our fold's train context is only ~7-8
perturbations), so the shipped defaults already do the right thing at this scale without
hand-tuning. Same "take the official preset as-is" reasoning as STATE's ``model=pertsets``.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from tfm_perturbation.datamodules.presage import PresageAnnDataDataModule
from tfm_perturbation.models.transcriptomics.tabicl_multiview import TabICLMultiviewModel

ad.settings.allow_write_nullable_strings = True

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.tabicl.prediction_only import fit_prediction_only
from models.tabicl.split_compat import initialize_with_nonempty_validation
from workflows.model_runs import add_point_run_arguments, begin_point_model_run

DATA_DIR = HERE.parent / "data"
PRESAGE_CACHE = HERE.parent / "presage" / "cache" / "cache"
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


def _report_timing(stage: str, seconds: float) -> None:
    print(f"[TabICLv2][timing] {stage}={seconds:.3f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="smoke_k562",
        help="models/data/<dataset>/{raw.h5ad,folds/} to train/predict on (default: smoke_k562)",
    )
    parser.add_argument("--fold", type=int, default=0, help="which fold in <dataset>_folds/ to train/predict on")
    add_point_run_arguments(parser, deterministic=True)
    args = parser.parse_args()
    total_start = time.perf_counter()

    run = begin_point_model_run(
        args,
        repository_root=REPOSITORY_ROOT,
        model_id="tabicl",
        environment="models/pixi.lock#tabicl",
        configuration={
            "training_mode": "in_context",
            "fusion_weighting_mode": "none",
            "validation_filtering": False,
            "upstream_evaluation": "skipped_prediction_only",
            "stage_timing": True,
        },
        deterministic=True,
    )
    if run.skipped:
        print(f"skipping verified completed run: {run.handle.directory}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    raw = run.raw_path
    split_dir = DATA_DIR / args.dataset / "folds"
    split_json_path = split_dir / f"fold_{args.fold}_tabicl.json"
    validation_split_json_path = split_dir / f"fold_{args.fold}_presage.json"
    # Namespaced by dataset too — otherwise a smoke run and a full run at the same fold
    # index would silently overwrite each other's artifact.
    run_dir = run.checkpoint_dir / "tabicl"

    datamodule = PresageAnnDataDataModule(
        dataset=args.dataset,
        data_file_path=str(raw),
        split_json_path=str(validation_split_json_path),
        perturbation_col="perturbation",
        control_label=CONTROL_LABEL,
    )
    setup_start = time.perf_counter()
    initialize_with_nonempty_validation(
        datamodule,
        validation_split_path=validation_split_json_path,
        model_split_path=split_json_path,
    )
    datamodule_setup_seconds = time.perf_counter() - setup_start
    _report_timing("datamodule_setup", datamodule_setup_seconds)

    model = TabICLMultiviewModel(
        split_json_path=str(split_json_path),
        control_label=CONTROL_LABEL,
        perturbation_key="perturbation",
        # Our smoke adata has no batch/plate column — falls back to global-control logFC.
        batch_key=None,
        modality_pickle_paths=MODALITY_PICKLE_PATHS,
        # Keep validation out of model selection: all modalities are fused equally and no
        # validation-R2 component filtering is applied. The full development pool can
        # therefore be used as in-context examples without changing a selected checkpoint.
        fusion_weighting_mode="none",
        ica_r2_threshold=0.0,
        presage_data_root=None,
        degs_cache_dir=str(run_dir / "degs"),
    )

    prediction_fit_start = time.perf_counter()
    fit_prediction_only(model, output_dir=run_dir, datamodule=datamodule)
    prediction_fit_seconds = time.perf_counter() - prediction_fit_start
    _report_timing("prediction_fit", prediction_fit_seconds)

    serialization_start = time.perf_counter()
    with open(run_dir / "tabicl_multiview_artifact.pkl", "rb") as f:
        artifact = pickle.load(f)
    test = artifact["fused"]["test"]
    gene_names = artifact["metadata"]["gene_names"]

    # TabICLMultiviewModel predicts logFC (perturbed pseudobulk minus control pseudobulk) —
    # tabicl_multiview.py's _build_rna_logfc_adata subtracts the control profile before ever
    # handing data to the model, and the fused/test/predictions array is that same delta space,
    # never converted back. scPertEval's own scoring (context.py's centroid()) subtracts
    # ds.control_mean() from *both* ground truth and prediction for centered protocols, so it
    # needs predictions already in absolute-expression units — same reasoning, same fix, as
    # PRESAGE's train_predict.py (also a native delta-predictor whose control_mean addback was
    # missing). Recomputed directly from raw.h5ad rather than pulled from the artifact, since
    # the artifact's own control_profile isn't exposed in "fused"/"metadata" and gene_names here
    # is the model's full output panel, matching raw.h5ad's var_names.
    adata = ad.read_h5ad(raw)
    control_mean = np.asarray(adata.X[(adata.obs["perturbation"] == "control").to_numpy()].mean(axis=0)).ravel()
    control_mean = pd.Series(control_mean, index=adata.var_names).reindex(gene_names).to_numpy()

    pred_adata = ad.AnnData(
        X=(np.asarray(test["predictions"], dtype=np.float32) + control_mean.astype(np.float32)),
        obs={"perturbation": np.asarray(test["perturbations"])},
    )
    pred_adata.var_names = pd.Index(gene_names)

    out_path = run.prediction_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_adata.write_h5ad(out_path, compression="gzip")
    prediction_serialization_seconds = time.perf_counter() - serialization_start
    _report_timing("prediction_serialization", prediction_serialization_seconds)
    elapsed_before_completion_seconds = time.perf_counter() - total_start
    pd.DataFrame(
        [
            {
                "training_mode": "in_context",
                "gradient_updates": 0,
                "modalities": len(MODALITY_PICKLE_PATHS),
                "upstream_evaluation": "skipped_prediction_only",
                "datamodule_setup_seconds": datamodule_setup_seconds,
                "prediction_fit_seconds": prediction_fit_seconds,
                "prediction_serialization_seconds": prediction_serialization_seconds,
                "elapsed_before_completion_seconds": elapsed_before_completion_seconds,
            }
        ]
    ).to_csv(run.metrics_path, index=False)
    run.complete(
        implementation_revision="royerlab/tfm-perturbation@c7b2786c2aee946db1da442c39ad1d9bdb303d14",
        extra_artifacts={"in_context_artifact": run_dir / "tabicl_multiview_artifact.pkl"},
    )
    _report_timing("elapsed_before_completion", elapsed_before_completion_seconds)
    print(f"wrote {pred_adata.shape} predictions to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
