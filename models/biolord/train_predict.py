"""Train one joint bioLORD model for a frozen leave-one-context-out fold."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.biolord.context_config import (
    assign_biolord_splits,
    relabel_biolord_predictions,
    select_heldout_controls,
)
from workflows.predictions import build_prediction_manifest
from workflows.run_contract import begin_run, complete_run, repository_revision

ROOT = REPOSITORY_ROOT
DATA_ROOT = ROOT / "models" / "data"
RUN_ROOT = ROOT / "runs"

IMPLEMENTATION_REVISION = "nitzanlab/biolord@b7688790e49728d7f8d0906b980a629de484b19b"

MODULE_PARAMS = {
    "decoder_width": 1024,
    "decoder_depth": 4,
    "attribute_nn_width": 512,
    "attribute_nn_depth": 2,
    "n_latent_attribute_categorical": 4,
    "gene_likelihood": "normal",
    "reconstruction_penalty": 1e2,
    "unknown_attribute_penalty": 1e1,
    "unknown_attribute_noise_param": 1e-1,
    "attribute_dropout_rate": 0.1,
    "use_batch_norm": False,
    "use_layer_norm": False,
}

TRAINER_PARAMS = {
    "n_epochs_warmup": 0,
    "latent_lr": 1e-4,
    "latent_wd": 1e-4,
    "decoder_lr": 1e-4,
    "decoder_wd": 1e-4,
    "attribute_nn_lr": 1e-2,
    "attribute_nn_wd": 4e-8,
    "step_size_lr": 45,
    "cosine_scheduler": True,
    "scheduler_final_lr": 1e-5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--perturbation-key", default="model_perturbation")
    parser.add_argument("--context-key", default="context")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=10)
    parser.add_argument("--prediction-population-size", type=int, default=100)
    parser.add_argument("--runs-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    split = json.loads(args.split_manifest.read_text())
    if args.dataset != split["id"]:
        raise ValueError(f"--dataset {args.dataset!r} must match split id {split['id']!r}")
    try:
        fold_spec = split["folds"][args.fold]
    except IndexError as error:
        raise ValueError(f"fold {args.fold} is outside the context split") from error
    fold_id = str(fold_spec["id"])
    raw = DATA_ROOT / args.dataset / "raw.h5ad"
    data_manifest = args.data_manifest or raw.with_suffix(".manifest.json")
    for required in [raw, data_manifest, args.split_manifest]:
        if not required.is_file():
            raise FileNotFoundError(required)

    module_params = {**MODULE_PARAMS, "seed": args.seed}
    handle = begin_run(
        args.runs_root,
        experiment=args.dataset,
        model="biolord",
        fold=fold_id,
        seed=args.seed,
        configuration={
            "n_latent": 32,
            "module": module_params,
            "trainer": TRAINER_PARAMS,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "early_stop_patience": args.early_stop_patience,
            "check_val_every_n_epoch": args.check_val_every_n_epoch,
            "prediction_population_size": args.prediction_population_size,
            "perturbation_key": args.perturbation_key,
            "context_key": args.context_key,
        },
        inputs={"model_view_manifest": data_manifest, "split_manifest": args.split_manifest},
        command=[sys.executable, *sys.argv],
        repo_root=ROOT,
        force=args.force,
    )
    if handle.skipped:
        print(f"skipping verified completed run: {handle.directory}")
        return

    import biolord

    data = assign_biolord_splits(
        ad.read_h5ad(raw),
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
    )
    biolord.Biolord.setup_anndata(
        data,
        ordered_attributes_keys=None,
        categorical_attributes_keys=[args.context_key, args.perturbation_key],
    )
    model = biolord.Biolord(
        adata=data,
        n_latent=32,
        model_name=f"{args.dataset}-{fold_id}-seed-{args.seed}",
        module_params=module_params,
        train_classifiers=False,
        split_key="biolord_split",
        train_split="train",
        valid_split="valid",
        test_split="test",
    )
    model.train(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        device=1 if torch.cuda.is_available() else "auto",
        batch_size=args.batch_size,
        plan_kwargs=TRAINER_PARAMS,
        early_stopping=True,
        early_stopping_patience=args.early_stop_patience,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_workers=1,
        enable_checkpointing=False,
    )
    checkpoint_dir = handle.directory / "checkpoints" / "model"
    model.save(checkpoint_dir, overwrite=True)

    controls = select_heldout_controls(
        data,
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
        population_size=args.prediction_population_size,
        seed=args.seed,
    )
    predictions = model.compute_prediction_adata(
        data,
        controls,
        target_attributes=[args.perturbation_key],
        add_attributes=[args.context_key],
    )
    predictions = relabel_biolord_predictions(
        predictions,
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        population_size=args.prediction_population_size,
    )
    predictions.X = np.asarray(predictions.X, dtype=np.float32)
    prediction_path = handle.directory / "predictions" / "predictions.h5ad"
    predictions.write_h5ad(prediction_path, compression="gzip")
    prediction_manifest = build_prediction_manifest(
        prediction_path=prediction_path,
        truth_path=raw,
        dataset_manifest_path=data_manifest,
        split_manifest_path=args.split_manifest,
        capability="population",
        model_id="biolord",
        implementation_revision=IMPLEMENTATION_REVISION,
        adapter_revision=repository_revision(ROOT),
        fold=args.fold,
        seed=args.seed,
        environment="models/pixi.lock#biolord",
        command=" ".join([sys.executable, *sys.argv]),
        minimum_cells_per_condition=50,
    )
    prediction_manifest_path = prediction_path.with_suffix(".manifest.json")
    prediction_manifest_path.write_text(json.dumps(prediction_manifest, indent=2) + "\n")

    complete_run(
        handle,
        {
            "checkpoint": checkpoint_dir / "model.pt",
            "metrics": checkpoint_dir / "history.csv",
            "prediction": prediction_path,
            "prediction_manifest": prediction_manifest_path,
        },
        precomputed_checksums={"prediction": prediction_manifest["artifact"]["sha256"]},
    )
    print(f"wrote {predictions.shape} predictions to {prediction_path}")


if __name__ == "__main__":
    main()
