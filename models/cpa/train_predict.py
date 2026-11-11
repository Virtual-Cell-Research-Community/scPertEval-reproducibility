"""Train one joint CPA model for a frozen leave-one-context-out fold."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.cpa.context_config import assign_cpa_splits, build_counterfactual_queries
from workflows.predictions import build_prediction_manifest
from workflows.run_contract import begin_run, complete_run, repository_revision

ROOT = REPOSITORY_ROOT
DATA_ROOT = ROOT / "models" / "data"
RUN_ROOT = ROOT / "runs"


MODEL_PARAMS = {
    "n_latent": 64,
    "recon_loss": "gauss",
    "doser_type": "linear",
    "n_hidden_encoder": 128,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": True,
    "dropout_rate_encoder": 0.0,
    "dropout_rate_decoder": 0.1,
    "variational": False,
}

TRAINER_PARAMS = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 0,
    "mixup_alpha": 0.0,
    "adv_steps": None,
    "n_hidden_adv": 64,
    "n_layers_adv": 3,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 5.0,
    "lr": 3e-4,
    "wd": 4e-7,
    "adv_lr": 3e-4,
    "adv_wd": 4e-7,
    "adv_loss": "cce",
    "doser_lr": 3e-4,
    "doser_wd": 4e-7,
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,
    "step_size_lr": 10,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--perturbation-key", default="model_perturbation")
    parser.add_argument("--context-key", default="context")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=5)
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

    handle = begin_run(
        args.runs_root,
        experiment=args.dataset,
        model="cpa",
        fold=fold_id,
        seed=args.seed,
        configuration={
            "model": MODEL_PARAMS,
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

    import cpa

    data = assign_cpa_splits(
        ad.read_h5ad(raw),
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
    )
    cpa.CPA.setup_anndata(
        data,
        perturbation_key=args.perturbation_key,
        control_group="control",
        dosage_key="cpa_dose",
        categorical_covariate_keys=[args.context_key],
        is_count_data=False,
        max_comb_len=1,
    )
    model = cpa.CPA(
        adata=data,
        split_key="cpa_split",
        train_split="train",
        valid_split="valid",
        test_split="test",
        seed=args.seed,
        **MODEL_PARAMS,
    )
    checkpoint_dir = handle.directory / "checkpoints" / "model"
    model.train(
        max_epochs=args.max_epochs,
        use_gpu=torch.cuda.is_available(),
        batch_size=args.batch_size,
        plan_kwargs=TRAINER_PARAMS,
        early_stopping_patience=args.early_stop_patience,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        save_path=str(checkpoint_dir),
    )

    queries = build_counterfactual_queries(
        data,
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
        population_size=args.prediction_population_size,
        seed=args.seed,
    )
    # The query cells retain the held-out controls in X, but their perturbation labels
    # were changed above. Re-register them so CPA rebuilds the encoded perturbation and
    # dosage matrices instead of silently predicting the control condition.
    cpa.CPA.setup_anndata(
        queries,
        perturbation_key=args.perturbation_key,
        control_group="control",
        dosage_key="cpa_dose",
        categorical_covariate_keys=[args.context_key],
        is_count_data=False,
        max_comb_len=1,
    )
    model.predict(queries, batch_size=args.batch_size, n_samples=1)
    predictions = ad.AnnData(
        X=np.asarray(queries.obsm["CPA_pred"], dtype=np.float32),
        obs=pd.DataFrame(
            {"perturbation": queries.obs["perturbation"].astype(str).to_numpy()},
            index=queries.obs_names.copy(),
        ),
        var=pd.DataFrame(index=data.var_names.copy()),
    )
    prediction_path = handle.directory / "predictions" / "predictions.h5ad"
    predictions.write_h5ad(prediction_path, compression="gzip")
    prediction_manifest = build_prediction_manifest(
        prediction_path=prediction_path,
        truth_path=raw,
        dataset_manifest_path=data_manifest,
        split_manifest_path=args.split_manifest,
        capability="population",
        model_id="cpa",
        implementation_revision="theislab/cpa@7cda37e9cca2078af1b2373dc6f9422a5bc6f87d",
        adapter_revision=repository_revision(ROOT),
        fold=args.fold,
        seed=args.seed,
        environment="models/pixi.lock#cpa",
        command=" ".join([sys.executable, *sys.argv]),
        minimum_cells_per_condition=50,
    )
    prediction_manifest_path = prediction_path.with_suffix(".manifest.json")
    prediction_manifest_path.write_text(json.dumps(prediction_manifest, indent=2) + "\n")

    checkpoint_path = checkpoint_dir / "model.pt"
    metrics_path = checkpoint_dir / "history.csv"
    complete_run(
        handle,
        {
            "checkpoint": checkpoint_path,
            "metrics": metrics_path,
            "prediction": prediction_path,
            "prediction_manifest": prediction_manifest_path,
        },
    )
    print(f"wrote {predictions.shape} predictions to {prediction_path}")


if __name__ == "__main__":
    main()
