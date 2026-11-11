"""Train one joint trVAE model for a frozen leave-one-context-out fold."""

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

from models.state.context_config import model_conditions
from models.trvae.context_config import (
    build_prediction_adata,
    select_heldout_controls,
    select_trvae_training_data,
)
from models.trvae.scarchon_patch import apply_scarchon_patches
from workflows.predictions import build_prediction_manifest
from workflows.run_contract import begin_run, complete_run, repository_revision

ROOT = REPOSITORY_ROOT
DATA_ROOT = ROOT / "models" / "data"
RUN_ROOT = ROOT / "runs"

IMPLEMENTATION_REVISION = "theislab/trvaep==0.1.0+scArchon@42f6c2c202936966ae10c522f784881128b7efbb"
MODEL_PARAMS = {
    "encoder_layer_sizes": [128, 32],
    "decoder_layer_sizes": [32, 128],
    "latent_dim": 10,
    "alpha": 1e-4,
    "use_mmd": True,
    "beta": 10.0,
}


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def _predict_populations(
    model,
    controls: ad.AnnData,
    conditions: list[str],
    *,
    perturbation_key: str,
) -> dict[str, np.ndarray]:
    """Encode controls once, then decode the same population under every perturbation."""
    values = np.asarray(_dense(controls.X), dtype=np.float32)
    source_labels = controls.obs[perturbation_key].astype(str).to_numpy()
    if set(source_labels) != {"control"}:
        raise ValueError("trVAE prediction source must contain controls only")
    known = set(map(str, model.label_encoder.classes_))
    missing = sorted(set(conditions) - known)
    if missing:
        raise ValueError(f"trVAE training labels do not include conditions: {missing}")

    source_ids = model.label_encoder.transform(source_labels)
    latent = model.get_latent(values, source_ids)
    predictions = {}
    for condition in conditions:
        target_ids = model.label_encoder.transform(np.repeat(condition, controls.n_obs))
        predictions[condition] = np.asarray(
            model.reconstruct(latent, target_ids, use_latent=True),
            dtype=np.float32,
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--perturbation-key", default="model_perturbation")
    parser.add_argument("--context-key", default="context")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
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
    test_context = str(fold_spec["test_context"])
    raw = DATA_ROOT / args.dataset / "raw.h5ad"
    data_manifest = args.data_manifest or raw.with_suffix(".manifest.json")
    for required in [raw, data_manifest, args.split_manifest]:
        if not required.is_file():
            raise FileNotFoundError(required)

    configuration = {
        "model": MODEL_PARAMS,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "prediction_population_size": args.prediction_population_size,
        "perturbation_key": args.perturbation_key,
        "context_key": args.context_key,
        "fit_scope": "one-joint-model-across-all-perturbations-per-fold",
        "early_stopping": False,
    }
    handle = begin_run(
        args.runs_root,
        experiment=args.dataset,
        model="trvae",
        fold=fold_id,
        seed=args.seed,
        configuration=configuration,
        inputs={"model_view_manifest": data_manifest, "split_manifest": args.split_manifest},
        command=[sys.executable, *sys.argv],
        repo_root=ROOT,
        force=args.force,
    )
    if handle.skipped:
        print(f"skipping verified completed run: {handle.directory}")
        return

    apply_scarchon_patches()
    from trvaep.model import CVAE, Trainer

    data = ad.read_h5ad(raw)
    training = select_trvae_training_data(
        data,
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
    )
    controls = select_heldout_controls(
        data,
        split,
        args.fold,
        perturbation_key=args.perturbation_key,
        context_key=args.context_key,
        population_size=args.prediction_population_size,
        seed=args.seed,
    )
    n_conditions = training.obs[args.perturbation_key].astype(str).nunique()
    model = CVAE(
        training.n_vars,
        num_classes=n_conditions,
        encoder_layer_sizes=MODEL_PARAMS["encoder_layer_sizes"].copy(),
        decoder_layer_sizes=MODEL_PARAMS["decoder_layer_sizes"].copy(),
        latent_dim=MODEL_PARAMS["latent_dim"],
        alpha=MODEL_PARAMS["alpha"],
        use_mmd=MODEL_PARAMS["use_mmd"],
        beta=MODEL_PARAMS["beta"],
    )
    trainer = Trainer(
        model,
        training,
        condition_key=args.perturbation_key,
        seed=args.seed,
    )
    # scArchon's vendored trainer disables its early-stopping block, so use a patience above
    # the requested epoch count with the official upstream package to reproduce that behavior.
    trainer.train_trvae(
        n_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_patience=args.max_epochs + 1,
    )

    checkpoint_path = handle.directory / "checkpoints" / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_classes": model.label_encoder.classes_.tolist(),
            "configuration": configuration,
            "implementation_revision": IMPLEMENTATION_REVISION,
        },
        checkpoint_path,
    )
    metrics_path = handle.directory / "checkpoints" / "training-history.json"
    metrics_path.write_text(
        json.dumps({key: list(map(float, values)) for key, values in trainer.logs.items()}, indent=2)
        + "\n"
    )

    torch.manual_seed(args.seed)
    native_predictions = _predict_populations(
        model,
        controls,
        model_conditions(split, capability="population"),
        perturbation_key=args.perturbation_key,
    )
    predictions = build_prediction_adata(
        native_predictions,
        data.var,
        test_context=test_context,
        population_size=args.prediction_population_size,
    )
    prediction_path = handle.directory / "predictions" / "predictions.h5ad"
    predictions.write_h5ad(prediction_path, compression="gzip")
    prediction_manifest = build_prediction_manifest(
        prediction_path=prediction_path,
        truth_path=raw,
        dataset_manifest_path=data_manifest,
        split_manifest_path=args.split_manifest,
        capability="population",
        model_id="trvae",
        implementation_revision=IMPLEMENTATION_REVISION,
        adapter_revision=repository_revision(ROOT),
        fold=args.fold,
        seed=args.seed,
        environment="models/pixi.lock#trvae",
        command=" ".join([sys.executable, *sys.argv]),
        minimum_cells_per_condition=50,
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
    )
    print(f"wrote {predictions.shape} predictions to {prediction_path}")


if __name__ == "__main__":
    main()
