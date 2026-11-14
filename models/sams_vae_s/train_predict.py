"""Train the published PerturBench SAMS-VAE(S) on one chemical context fold."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.state.context_config import (
    model_conditions,
    observed_population_sizes,
    sample_heldout_controls,
)
from models.trvae.context_config import build_prediction_adata
from workflows.predictions import build_prediction_manifest
from workflows.run_contract import begin_run, complete_run, repository_revision

DATA_ROOT = REPOSITORY_ROOT / "models" / "data"
RUN_ROOT = REPOSITORY_ROOT / "runs"
IMPLEMENTATION_REVISION = "altoslabs/perturbench@c84038bc1ea409aa54f3832cfa6f34f5059adf0c"
ENVIRONMENT = "models/pixi.lock#sams-vae-s"

PUBLISHED_CONFIGURATION = {
    "disable_e_dist": True,
    "disable_sparsity": True,
    "dropout": 0.4,
    "generative_counterfactual": True,
    "hidden_dim_cond": 329,
    "hidden_dim_x": 2560,
    "inject_covariates_decoder": True,
    "inject_covariates_encoder": True,
    "latent_dim": 32,
    "learning_rate": 0.000341765289234546,
    "mask_prior_probability": 0.05618183575503891,
    "n_layers_decoder": 3,
    "n_layers_encoder_e": 2,
    "n_layers_encoder_x": 4,
    "softplus_output": True,
    "weight_decay": 6.943657713062555e-08,
    "batch_size": 256,
    "minimum_epochs": 5,
    "maximum_epochs": 500,
    "early_stopping_patience": 50,
}


def context_split_masks(
    data: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    context_key: str = "context",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen development and validation masks for SAMS-VAE(S)."""
    try:
        fold_spec = split["folds"][fold]
    except IndexError as error:
        raise ValueError(f"fold {fold} is outside the context split") from error
    labels = data.obs[perturbation_key].astype(str)
    contexts = data.obs[context_key].astype(str)
    test_context = str(fold_spec["test_context"])
    train_contexts = set(map(str, fold_spec["train_contexts"]))
    validation_conditions = set(map(str, split["validation_conditions"]))
    controls = labels == "control"
    validation = contexts.isin(train_contexts) & labels.isin(validation_conditions) & ~controls
    development = (contexts.isin(train_contexts) & ~validation) | ((contexts == test_context) & controls)
    if not development.any() or not validation.any():
        raise ValueError("SAMS-VAE(S) requires nonempty development and validation partitions")
    if np.any(development.to_numpy() & validation.to_numpy()):
        raise ValueError("SAMS-VAE(S) development and validation partitions overlap")
    return development.to_numpy(), validation.to_numpy()


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def encoded_arrays(
    data: ad.AnnData,
    mask: np.ndarray,
    *,
    conditions: list[str],
    contexts: list[str],
    perturbation_key: str,
    context_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode expression, perturbation, and context exactly as PerturBench's pipeline."""
    subset = data[mask]
    expression = np.asarray(_dense(subset.X), dtype=np.float32)
    condition_index = {label: index for index, label in enumerate(conditions)}
    context_index = {label: index for index, label in enumerate(contexts)}
    perturbations = np.zeros((subset.n_obs, len(conditions)), dtype=np.float32)
    labels = subset.obs[perturbation_key].astype(str).to_numpy()
    for row, label in enumerate(labels):
        if label != "control":
            if label not in condition_index:
                raise ValueError(f"unknown SAMS-VAE(S) condition {label!r}")
            perturbations[row, condition_index[label]] = 1.0
    covariates = np.zeros((subset.n_obs, len(contexts)), dtype=np.float32)
    for row, label in enumerate(subset.obs[context_key].astype(str)):
        if label not in context_index:
            raise ValueError(f"unknown SAMS-VAE(S) context {label!r}")
        covariates[row, context_index[label]] = 1.0
    return expression, perturbations, covariates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--perturbation-key", default="model_perturbation")
    parser.add_argument("--context-key", default="context")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=PUBLISHED_CONFIGURATION["maximum_epochs"])
    parser.add_argument("--batch-size", type=int, default=PUBLISHED_CONFIGURATION["batch_size"])
    parser.add_argument("--early-stop-patience", type=int, default=PUBLISHED_CONFIGURATION["early_stopping_patience"])
    parser.add_argument(
        "--prediction-population-size",
        type=int,
        help="fixed smoke-test override; default matches every observed treated population",
    )
    parser.add_argument("--runs-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import lightning
    import torch
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from omegaconf import OmegaConf
    from perturbench.data.types import Batch
    from perturbench.modelcore.models.sams_vae import SparseAdditiveVAE
    from torch.utils.data import DataLoader, Dataset

    class ArrayDataset(Dataset):
        def __init__(self, expression: np.ndarray, perturbations: np.ndarray, covariates: np.ndarray) -> None:
            self.expression = torch.from_numpy(expression)
            self.perturbations = torch.from_numpy(perturbations)
            self.covariates = torch.from_numpy(covariates)

        def __len__(self) -> int:
            return len(self.expression)

        def __getitem__(self, index: int) -> Batch:
            return Batch(
                gene_expression=self.expression[index],
                perturbations=self.perturbations[index],
                covariates={args.context_key: self.covariates[index]},
            )

    def collate(items: list[Batch]) -> Batch:
        return Batch(
            gene_expression=torch.stack([item.gene_expression for item in items]),
            perturbations=torch.stack([item.perturbations for item in items]),
            covariates={
                args.context_key: torch.stack([item.covariates[args.context_key] for item in items])
            },
        )

    class MetricHistory(Callback):
        def __init__(self) -> None:
            self.rows: list[dict[str, float | int]] = []

        def on_validation_epoch_end(self, trainer, _model) -> None:
            if trainer.sanity_checking:
                return
            row: dict[str, float | int] = {"epoch": trainer.current_epoch + 1}
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    row[str(key)] = float(value.detach().cpu())
            self.rows.append(row)

    if args.max_epochs <= 0 or args.batch_size <= 0 or args.early_stop_patience < 0:
        raise ValueError("epochs and batch size must be positive, and patience must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    lightning.seed_everything(args.seed, workers=True)

    split = json.loads(args.split_manifest.read_text())
    if args.dataset != split["id"]:
        raise ValueError(f"--dataset {args.dataset!r} must match split id {split['id']!r}")
    fold_spec = split["folds"][args.fold]
    fold_id = str(fold_spec["id"])
    test_context = str(fold_spec["test_context"])
    raw = DATA_ROOT / args.dataset / "raw.h5ad"
    data_manifest = args.data_manifest or raw.with_suffix(".manifest.json")
    for required in [raw, data_manifest, args.split_manifest]:
        if not required.is_file():
            raise FileNotFoundError(required)

    configuration = {
        **PUBLISHED_CONFIGURATION,
        "maximum_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.early_stop_patience,
        "training_seed": args.seed,
        "prediction_population_size": args.prediction_population_size or "match-observed",
        "fit_scope": "one-joint-model-across-all-perturbations-per-fold",
        "checkpoint_selection": "minimum validation loss",
    }
    handle = begin_run(
        args.runs_root,
        experiment=args.dataset,
        model="sams-vae-s",
        fold=fold_id,
        seed=args.seed,
        configuration=configuration,
        inputs={"model_view_manifest": data_manifest, "split_manifest": args.split_manifest},
        command=[sys.executable, *sys.argv],
        repo_root=REPOSITORY_ROOT,
        force=args.force,
    )
    if handle.skipped:
        print(f"skipping verified completed run: {handle.directory}")
        return

    data = ad.read_h5ad(raw)
    conditions = model_conditions(split, capability="point")
    contexts = sorted(map(str, split["contexts"]))
    development_mask, validation_mask = context_split_masks(
        data,
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
    )
    development = encoded_arrays(
        data,
        development_mask,
        conditions=conditions,
        contexts=contexts,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
    )
    validation = encoded_arrays(
        data,
        validation_mask,
        conditions=conditions,
        contexts=contexts,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
    )
    counts = development[1].sum(axis=0, dtype=np.float64)
    context = {
        "perturbation_counts": pd.DataFrame({"count": counts}, index=conditions),
        "perturbation_uniques": set(conditions),
        "covariate_uniques": {args.context_key: contexts},
    }
    model = SparseAdditiveVAE(
        n_genes=data.n_vars,
        n_perts=len(conditions),
        transform=None,
        context=context,
        evaluation=OmegaConf.create({}),
        n_layers_encoder_x=PUBLISHED_CONFIGURATION["n_layers_encoder_x"],
        n_layers_encoder_e=PUBLISHED_CONFIGURATION["n_layers_encoder_e"],
        n_layers_decoder=PUBLISHED_CONFIGURATION["n_layers_decoder"],
        hidden_dim_x=PUBLISHED_CONFIGURATION["hidden_dim_x"],
        hidden_dim_cond=PUBLISHED_CONFIGURATION["hidden_dim_cond"],
        latent_dim=PUBLISHED_CONFIGURATION["latent_dim"],
        dropout=PUBLISHED_CONFIGURATION["dropout"],
        inject_covariates_encoder=PUBLISHED_CONFIGURATION["inject_covariates_encoder"],
        inject_covariates_decoder=PUBLISHED_CONFIGURATION["inject_covariates_decoder"],
        mask_prior_probability=PUBLISHED_CONFIGURATION["mask_prior_probability"],
        lr=PUBLISHED_CONFIGURATION["learning_rate"],
        wd=PUBLISHED_CONFIGURATION["weight_decay"],
        softplus_output=PUBLISHED_CONFIGURATION["softplus_output"],
        generative_counterfactual=PUBLISHED_CONFIGURATION["generative_counterfactual"],
        disable_sparsity=PUBLISHED_CONFIGURATION["disable_sparsity"],
        disable_e_dist=PUBLISHED_CONFIGURATION["disable_e_dist"],
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ArrayDataset(*development),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        ArrayDataset(*validation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    checkpoint = ModelCheckpoint(
        dirpath=handle.directory / "checkpoints",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    history = MetricHistory()
    trainer = lightning.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        min_epochs=min(PUBLISHED_CONFIGURATION["minimum_epochs"], args.max_epochs),
        max_epochs=args.max_epochs,
        precision="32-true",
        deterministic=False,
        check_val_every_n_epoch=1,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val_loss", mode="min", patience=args.early_stop_patience),
            history,
        ],
        logger=False,
        enable_model_summary=False,
        log_every_n_steps=10,
        default_root_dir=handle.directory,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError("SAMS-VAE(S) did not produce a best validation checkpoint")
    checkpoint_path = Path(checkpoint.best_model_path)
    history_path = handle.directory / "logs" / "metrics.csv"
    pd.DataFrame(history.rows).to_csv(history_path, index=False)

    best_model = SparseAdditiveVAE.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    best_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    population_sizes = (
        {condition: args.prediction_population_size for condition in model_conditions(split, capability="population")}
        if args.prediction_population_size is not None
        else observed_population_sizes(
            data,
            split,
            args.fold,
            perturbation_key=args.perturbation_key,
            context_key=args.context_key,
        )
    )
    controls = sample_heldout_controls(
        data,
        split,
        args.fold,
        max(population_sizes.values()),
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
        seed=args.seed,
    )
    control_values = np.asarray(_dense(controls.X), dtype=np.float32)
    condition_index = {label: index for index, label in enumerate(conditions)}
    context_index = {label: index for index, label in enumerate(contexts)}
    native_predictions: dict[str, np.ndarray] = {}
    torch.manual_seed(args.seed)
    with torch.no_grad():
        for condition in model_conditions(split, capability="population"):
            size = population_sizes[condition]
            perturbations = torch.zeros((size, len(conditions)), dtype=torch.float32, device=device)
            perturbations[:, condition_index[condition]] = 1.0
            covariates = torch.zeros((size, len(contexts)), dtype=torch.float32, device=device)
            covariates[:, context_index[test_context]] = 1.0
            batch = Batch(
                gene_expression=torch.from_numpy(control_values[:size]).to(device),
                perturbations=perturbations,
                covariates={args.context_key: covariates},
            )
            native_predictions[condition] = best_model.predict(batch).detach().cpu().numpy().astype(np.float32)
    predictions = build_prediction_adata(
        native_predictions,
        data.var,
        test_context=test_context,
        population_sizes=population_sizes,
    )
    prediction_path = handle.directory / "predictions" / "predictions.h5ad"
    predictions.write_h5ad(prediction_path, compression="gzip")
    prediction_manifest = build_prediction_manifest(
        prediction_path=prediction_path,
        truth_path=raw,
        dataset_manifest_path=data_manifest,
        split_manifest_path=args.split_manifest,
        capability="population",
        model_id="sams-vae-s",
        implementation_revision=IMPLEMENTATION_REVISION,
        adapter_revision=repository_revision(REPOSITORY_ROOT),
        fold=args.fold,
        seed=args.seed,
        environment=ENVIRONMENT,
        command=" ".join([sys.executable, *sys.argv]),
        minimum_cells_per_condition=min(50, args.prediction_population_size or 50),
    )
    prediction_manifest_path = prediction_path.with_suffix(".manifest.json")
    prediction_manifest_path.write_text(json.dumps(prediction_manifest, indent=2) + "\n")
    complete_run(
        handle,
        {
            "checkpoint": checkpoint_path,
            "metrics": history_path,
            "prediction": prediction_path,
            "prediction_manifest": prediction_manifest_path,
        },
        precomputed_checksums={"prediction": prediction_manifest["artifact"]["sha256"]},
    )
    print(f"wrote {predictions.shape} predictions to {prediction_path}")


if __name__ == "__main__":
    main()
