"""Train one joint scDisInFact model for a frozen leave-one-context-out fold."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.scdisinfact.context_config import (
    build_prediction_adata,
    select_heldout_controls,
    select_scdisinfact_training_data,
)
from models.scdisinfact.joint_patch import apply_joint_condition_mmd_patch
from models.state.context_config import model_conditions
from workflows.predictions import build_prediction_manifest
from workflows.run_contract import (
    begin_run,
    complete_run,
    repository_revision,
    resume_incomplete_run,
)

ROOT = REPOSITORY_ROOT
DATA_ROOT = ROOT / "models" / "data"
RUN_ROOT = ROOT / "runs"

IMPLEMENTATION_REVISION = "ZhangLabGT/scDisInFact@9466ecf257d923f879cb39712c530d24826d88bc"
MODEL_PARAMS = {
    "Ks": [8, 2],
    "reg_mmd_comm": 1e-4,
    "reg_mmd_diff": 1e-4,
    "reg_kl_comm": 1e-5,
    "reg_kl_diff": 1e-2,
    "reg_class": 1.0,
    "reg_gl": 1.0,
    "interval": 10,
    "lr": 5e-4,
    "reconstruction_loss": "NB",
}


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def _condition_centroids(model, *, chunk_size: int = 2048) -> torch.Tensor:
    """Encode training cells once and reproduce upstream condition-centroid arithmetic."""
    n_conditions = len(model.data_dict["matching_dict"]["cond_names"][0])
    latent_width = model.Ks["diff_factors"][0]
    sums = torch.zeros((n_conditions, latent_width), device=model.device)
    counts = torch.zeros(n_conditions, device=model.device)
    with torch.no_grad():
        for dataset in model.data_dict["datasets"]:
            for start in range(0, len(dataset), chunk_size):
                stop = min(start + chunk_size, len(dataset))
                labels = dataset.diff_labels[0][start:stop].to(model.device)
                inferred = model.inference(
                    counts=dataset.counts_norm[start:stop].to(model.device),
                    batch_ids=dataset.batch_id[start:stop, None].to(model.device),
                )
                sums.index_add_(0, labels, inferred["mu_d"][0])
                counts.index_add_(0, labels, torch.ones(len(labels), device=model.device))
    if torch.any(counts == 0):
        raise ValueError("scDisInFact has a condition without training cells")
    return sums / counts[:, None]


def _predict_native_counts(
    model,
    controls: ad.AnnData,
    conditions: list[str],
    *,
    perturbation_key: str,
    context_key: str,
) -> dict[str, np.ndarray]:
    """Vectorize upstream predict_counts while caching its training latent centroids."""
    control_counts = _dense(controls.X)
    totals = np.sum(control_counts, axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("held-out controls contain an empty count row")
    size_factors = totals / 100.0
    normalized = np.log1p(control_counts / size_factors)
    normalized = torch.as_tensor(
        model.data_dict["scaler"].transform(normalized),
        dtype=torch.float32,
        device=model.device,
    )

    condition_names = np.asarray(model.data_dict["matching_dict"]["cond_names"][0]).astype(str)
    condition_ids = {name: index for index, name in enumerate(condition_names)}
    batch_names = np.asarray(model.data_dict["matching_dict"]["batch_name"]).astype(str)
    contexts = controls.obs[context_key].astype(str).unique()
    if len(contexts) != 1:
        raise ValueError(f"prediction controls span multiple contexts: {contexts.tolist()}")
    batch_matches = np.flatnonzero(batch_names == contexts[0])
    if len(batch_matches) != 1:
        raise ValueError(f"held-out context {contexts[0]!r} is absent from scDisInFact batches")
    batch_ids = torch.full(
        (controls.n_obs, 1),
        int(batch_matches[0]),
        dtype=torch.long,
        device=model.device,
    )
    control_labels = controls.obs[perturbation_key].astype(str).unique()
    if control_labels.tolist() != ["control"]:
        raise ValueError(f"prediction source must contain controls only, got {control_labels.tolist()}")
    control_id = condition_ids["control"]

    centroids = _condition_centroids(model)
    model.eval()
    with torch.no_grad():
        inferred = model.inference(counts=normalized, batch_ids=batch_ids)
        common = inferred["mu_c"]
        control_specific = inferred["mu_d"][0]
        predictions = {}
        for condition in conditions:
            if condition not in condition_ids:
                raise ValueError(f"condition {condition!r} is absent from scDisInFact training labels")
            specific = control_specific + centroids[condition_ids[condition]] - centroids[control_id]
            generated = model.generative(z_c=common, z_d=[specific], batch_ids=batch_ids)
            predictions[condition] = (
                generated["mu"].detach().cpu().numpy() * size_factors
            ).astype(np.float32)
    return predictions


def _postprocess_existing(
    args: argparse.Namespace,
    *,
    fold_id: str,
    counts_path: Path,
    counts_manifest: Path,
) -> None:
    """Validate and finalize saved native-count predictions without retraining."""
    handle = resume_incomplete_run(
        args.runs_root,
        experiment=args.dataset,
        model="scdisinfact",
        fold=fold_id,
        seed=args.seed,
    )
    checkpoint_path = handle.directory / "checkpoints" / "model.pt"
    metrics_path = handle.directory / "checkpoints" / "training-losses.npz"
    prediction_path = handle.directory / "predictions" / "predictions.h5ad"
    resolved = json.loads((handle.directory / "resolved-run.json").read_text())
    prediction_manifest = build_prediction_manifest(
        prediction_path=prediction_path,
        truth_path=counts_path,
        dataset_manifest_path=counts_manifest,
        split_manifest_path=args.split_manifest,
        capability="population",
        model_id="scdisinfact",
        implementation_revision=IMPLEMENTATION_REVISION,
        adapter_revision=repository_revision(ROOT),
        fold=args.fold,
        seed=args.seed,
        environment="models/pixi.lock#scdisinfact",
        command=" ".join(resolved["command"]),
        minimum_cells_per_condition=50,
        expression_space="raw_counts",
        evaluation_transform="normalize-total-10000-log1p",
    )
    prediction_manifest_path = prediction_path.with_suffix(".manifest.json")
    prediction_manifest_path.write_text(json.dumps(prediction_manifest, indent=2) + "\n")
    complete_run(
        handle,
        {
            "checkpoint": checkpoint_path,
            "metrics": metrics_path,
            "prediction": prediction_path,
            "prediction_manifest": prediction_manifest_path,
        },
        precomputed_checksums={"prediction": prediction_manifest["artifact"]["sha256"]},
    )
    print(f"finalized existing predictions at {prediction_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--perturbation-key", default="model_perturbation")
    parser.add_argument("--context-key", default="context")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prediction-population-size", type=int, default=100)
    parser.add_argument("--runs-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--counts-manifest", type=Path)
    parser.add_argument("--truth-manifest", type=Path)
    parser.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="validate and finalize existing checkpoint/predictions without training",
    )
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
    test_context = str(fold_spec["test_context"])
    data_dir = DATA_ROOT / args.dataset
    counts_path = data_dir / "counts.h5ad"
    truth_path = data_dir / "raw.h5ad"
    counts_manifest = args.counts_manifest or counts_path.with_suffix(".manifest.json")
    truth_manifest = args.truth_manifest or truth_path.with_suffix(".manifest.json")
    for required in [
        counts_path,
        truth_path,
        counts_manifest,
        truth_manifest,
        args.split_manifest,
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)

    if args.postprocess_existing:
        _postprocess_existing(
            args,
            fold_id=fold_id,
            counts_path=counts_path,
            counts_manifest=counts_manifest,
        )
        return

    configuration = {
        "model": MODEL_PARAMS,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "prediction_population_size": args.prediction_population_size,
        "perturbation_key": args.perturbation_key,
        "context_key": args.context_key,
        "fit_scope": "one-joint-model-across-all-perturbations-per-fold",
        "batch_condition_grouping": args.context_key,
    }
    handle = begin_run(
        args.runs_root,
        experiment=args.dataset,
        model="scdisinfact",
        fold=fold_id,
        seed=args.seed,
        configuration=configuration,
        inputs={
            "native_counts_manifest": counts_manifest,
            "evaluation_view_manifest": truth_manifest,
            "split_manifest": args.split_manifest,
        },
        command=[sys.executable, *sys.argv],
        repo_root=ROOT,
        force=args.force,
    )
    if handle.skipped:
        print(f"skipping verified completed run: {handle.directory}")
        return

    import scDisInFact

    apply_joint_condition_mmd_patch()

    full_counts = ad.read_h5ad(counts_path)
    training = select_scdisinfact_training_data(
        full_counts,
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
    )
    controls = select_heldout_controls(
        full_counts,
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
        population_size=args.prediction_population_size,
        seed=args.seed,
    )
    data_dict = scDisInFact.create_scdisinfact_dataset(
        training.X,
        training.obs.copy(),
        condition_key=[args.perturbation_key],
        batch_key=args.context_key,
        batch_cond_key=args.context_key,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    constructor_params = {key: value for key, value in MODEL_PARAMS.items() if key != "reconstruction_loss"}
    model = scDisInFact.scdisinfact(
        data_dict=data_dict,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        **constructor_params,
    )
    losses = model.train_model(
        nepochs=args.max_epochs,
        recon_loss=MODEL_PARAMS["reconstruction_loss"],
    )

    checkpoint_path = handle.directory / "checkpoints" / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "configuration": configuration,
            "implementation_revision": IMPLEMENTATION_REVISION,
        },
        checkpoint_path,
    )
    metrics_path = handle.directory / "checkpoints" / "training-losses.npz"
    np.savez_compressed(
        metrics_path,
        total=np.asarray(losses[0]),
        reconstruction=np.asarray(losses[1]),
        kl_common=np.asarray(losses[2]),
        kl_differential=np.asarray(losses[3]),
        mmd_common=np.asarray(losses[4]),
        mmd_differential=np.asarray(losses[5]),
        classification=np.asarray(losses[6]),
        group_lasso=np.asarray(losses[7]),
    )

    native_predictions = _predict_native_counts(
        model,
        controls,
        model_conditions(split, capability="population"),
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
    )
    predictions = build_prediction_adata(
        native_predictions,
        full_counts.var,
        test_context=test_context,
        population_size=args.prediction_population_size,
    )
    prediction_path = handle.directory / "predictions" / "predictions.h5ad"
    predictions.write_h5ad(prediction_path, compression="gzip")
    prediction_manifest = build_prediction_manifest(
        prediction_path=prediction_path,
        truth_path=counts_path,
        dataset_manifest_path=counts_manifest,
        split_manifest_path=args.split_manifest,
        capability="population",
        model_id="scdisinfact",
        implementation_revision=IMPLEMENTATION_REVISION,
        adapter_revision=repository_revision(ROOT),
        fold=args.fold,
        seed=args.seed,
        environment="models/pixi.lock#scdisinfact",
        command=" ".join([sys.executable, *sys.argv]),
        minimum_cells_per_condition=50,
        expression_space="raw_counts",
        evaluation_transform="normalize-total-10000-log1p",
    )
    prediction_manifest_path = prediction_path.with_suffix(".manifest.json")
    prediction_manifest_path.write_text(json.dumps(prediction_manifest, indent=2) + "\n")

    complete_run(
        handle,
        {
            "checkpoint": checkpoint_path,
            "metrics": metrics_path,
            "prediction": prediction_path,
            "prediction_manifest": prediction_manifest_path,
        },
        precomputed_checksums={"prediction": prediction_manifest["artifact"]["sha256"]},
    )
    print(f"wrote {predictions.shape} predictions to {prediction_path}")


if __name__ == "__main__":
    main()
