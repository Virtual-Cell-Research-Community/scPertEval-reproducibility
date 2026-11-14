"""Train Miller et al.'s CellFlow configuration on one unseen-gene fold.

Unlike CellSimBench's reporting wrapper, this adapter retains native cell-level predictions.
It samples the frozen 512-control test pool with replacement when necessary so every held-out
perturbation has exactly as many predictions as observed treated cells.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from workflows.model_runs import add_point_run_arguments, begin_population_model_run

IMPLEMENTATION_REVISION = (
    "shiftbioscience/Perturbation-Models-Outperform-Baselines@"
    "15361aec8392fb45a75c8c9da88ca5e6510fca97;"
    "theislab/CellFlow@0f55ba20c49bcee60da979bd5a9fb4ab16420b1e"
)
ENVIRONMENT = "models/pixi.toml#cellflow"

MILLER_CONFIGURATION = {
    "n_pca_components": 50,
    "n_control_cells_train": 2048,
    "n_control_cells_val": 256,
    "n_control_cells_test": 512,
    "condition_embedding_dim": 1024,
    "cond_output_dropout": 0.9,
    "layers_before_pool_dims": [1024, 1024],
    "layers_before_pool_dropout": 0.5,
    "layers_after_pool_dims": [1024, 1024],
    "layers_after_pool_dropout": 0.2,
    "pooling": "attention_token",
    "max_combination_length": 2,
    "time_freqs": 1024,
    "time_encoder_dims": [2048, 2048, 2048],
    "hidden_dims": [4096, 4096, 4096],
    "decoder_dims": [4096, 4096, 4096],
    "decoder_dropout": 0.2,
    "epsilon": 0.1,
    "tau_a": 1.0,
    "tau_b": 1.0,
    "num_iterations": 200_000,
    "valid_freq": 200_000,
    "batch_size": 1024,
    "learning_rate": 5e-5,
    "multi_steps": 20,
}


def load_embeddings(path: Path, targets: list[str]) -> dict[str, np.ndarray]:
    """Load one finite, common-dimensional ESM2 vector for every frozen target."""
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if isinstance(value, pd.DataFrame):
        value = {str(index): row.to_numpy() for index, row in value.iterrows()}
    if not isinstance(value, dict):
        raise TypeError("CellFlow embedding asset must contain a gene-indexed dictionary or DataFrame")
    missing = sorted(set(targets) - set(map(str, value)))
    if missing:
        raise ValueError(f"CellFlow embedding asset is missing {len(missing)} frozen targets: {missing[:10]}")
    embeddings = {target: np.asarray(value[target], dtype=np.float32) for target in targets}
    dimensions = {embedding.shape for embedding in embeddings.values()}
    if len(dimensions) != 1 or next(iter(dimensions), ()) == ():
        raise ValueError(f"CellFlow embeddings do not share one vector dimension: {sorted(dimensions)}")
    if not all(np.isfinite(embedding).all() for embedding in embeddings.values()):
        raise ValueError("CellFlow embedding asset contains NaN or infinite values")
    embeddings["control"] = np.zeros(next(iter(embeddings.values())).shape, dtype=np.float32)
    return embeddings


def annotate_cellflow(data: ad.AnnData, embeddings: dict[str, np.ndarray]) -> None:
    """Add the exact covariate columns consumed by Miller's CellFlow wrapper."""
    labels = data.obs["perturbation"].astype(str)
    controls = data.obs["is_control"].astype(bool)
    data.obs["condition"] = labels.where(~controls, "control").to_numpy()
    data.obs["gene_target_1"] = labels.where(~controls, "control").to_numpy()
    data.obs["gene_target_2"] = "control"
    data.obs["is_control"] = controls.to_numpy()
    data.uns["gene_embeddings"] = embeddings


def split_cellflow_data(
    data: ad.AnnData,
    fold: dict,
    *,
    seed: int,
    n_control_train: int,
    n_control_val: int,
    n_control_test: int,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Reproduce Miller's independent seeded control subsamples and train/val split."""
    labels = data.obs["perturbation"].astype(str)
    controls = data.obs["is_control"].astype(bool).to_numpy()
    control_indices = np.flatnonzero(controls)
    rng = np.random.RandomState(seed)

    def control_sample(size: int) -> np.ndarray:
        if len(control_indices) <= size:
            return control_indices.copy()
        return rng.choice(control_indices, size=size, replace=False)

    train_control = control_sample(n_control_train)
    val_control = control_sample(n_control_val)
    test_control = control_sample(n_control_test)
    train_treated = np.flatnonzero((~controls) & labels.isin(fold["train"]).to_numpy())
    val_treated = np.flatnonzero((~controls) & labels.isin(fold["val"]).to_numpy())
    return (
        data[np.concatenate([train_control, train_treated])].copy(),
        data[np.concatenate([val_control, val_treated])].copy(),
        data[test_control].copy(),
    )


def _prepare_model(cellflow_model, configuration: dict, seed: int) -> None:
    import optax
    from cellflow.utils import match_linear

    before_pool = {
        "genetic_perturbation": {
            "layer_type": "mlp",
            "dims": configuration["layers_before_pool_dims"],
            "dropout_rate": configuration["layers_before_pool_dropout"],
        }
    }
    after_pool = {
        "layer_type": "mlp",
        "dims": configuration["layers_after_pool_dims"],
        "dropout_rate": configuration["layers_after_pool_dropout"],
    }
    match_fn = functools.partial(
        match_linear,
        epsilon=configuration["epsilon"],
        tau_a=configuration["tau_a"],
        tau_b=configuration["tau_b"],
    )
    optimizer = optax.MultiSteps(optax.adam(configuration["learning_rate"]), configuration["multi_steps"])
    cellflow_model.prepare_model(
        condition_mode="deterministic",
        regularization=0.0,
        condition_embedding_dim=configuration["condition_embedding_dim"],
        cond_output_dropout=configuration["cond_output_dropout"],
        pooling=configuration["pooling"],
        layers_before_pool=before_pool,
        layers_after_pool=after_pool,
        time_freqs=configuration["time_freqs"],
        time_encoder_dims=configuration["time_encoder_dims"],
        time_encoder_dropout=0.0,
        hidden_dims=configuration["hidden_dims"],
        hidden_dropout=0.0,
        decoder_dims=configuration["decoder_dims"],
        decoder_dropout=configuration["decoder_dropout"],
        conditioning="concatenation",
        probability_path={"constant_noise": 1.0},
        match_fn=match_fn,
        linear_projection_before_concatenation=False,
        layer_norm_before_concatenation=False,
        optimizer=optimizer,
        seed=seed,
    )


def predict_populations(
    model,
    controls: ad.AnnData,
    truth: ad.AnnData,
    test_targets: list[str],
    reference: ad.AnnData,
    *,
    seed: int,
) -> ad.AnnData:
    """Predict and reconstruct one exactly size-matched population per target."""
    from cellflow.preprocessing import project_pca

    test_labels = truth.obs["perturbation"].astype(str)
    rng = np.random.default_rng(seed)
    pcs = np.asarray(reference.varm["PCs"], dtype=np.float32)
    means = np.asarray(reference.varm["X_mean"], dtype=np.float32).reshape(1, -1)
    population_sizes = {target: int((test_labels == target).sum()) for target in test_targets}
    prediction_batch_size = max(population_sizes.values())
    replace = prediction_batch_size > controls.n_obs
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    for target in test_targets:
        population_size = population_sizes[target]
        # Keep one input shape across conditions. JAX otherwise recompiles the prediction
        # graph for every distinct observed population size. Drawing the largest required
        # batch and retaining its first ``population_size`` rows has the same sampling
        # distribution as drawing that smaller population directly.
        indices = rng.choice(controls.n_obs, size=prediction_batch_size, replace=replace)
        source = controls[indices].copy()
        project_pca(source, ref_adata=reference)
        covariate_data = pd.DataFrame(
            {
                "condition": [target],
                "gene_target_1": [target],
                "gene_target_2": ["control"],
                "is_control": [False],
            }
        )
        prediction = model.predict(
            adata=source,
            sample_rep="X_pca",
            condition_id_key="condition",
            covariate_data=covariate_data,
        )
        pca_values = np.asarray(prediction[target], dtype=np.float32).squeeze()
        pca_values = np.atleast_2d(pca_values)
        arrays.append((pca_values[:population_size] @ pcs.T + means).astype(np.float32))
        labels.extend([target] * population_size)
    values = np.concatenate(arrays)
    obs = pd.DataFrame(
        {"perturbation": labels},
        index=[f"cellflow-{index}" for index in range(len(labels))],
    )
    return ad.AnnData(X=values, obs=obs, var=pd.DataFrame(index=reference.var_names.copy()))


def _training_metrics(model, num_iterations: int) -> pd.DataFrame:
    logs = model.trainer.training_logs
    losses = list(map(float, logs.get("loss", [])))
    rows = [{"iteration": index + 1, "loss": loss} for index, loss in enumerate(losses)]
    if not rows:
        rows = [{"iteration": num_iterations, "loss": np.nan}]
    for key, values in logs.items():
        if key == "loss" or not values:
            continue
        rows[-1][key] = float(np.asarray(values[-1]))
    return pd.DataFrame(rows)


def _atomic_write_h5ad(data: ad.AnnData, path: Path) -> None:
    """Write a large H5AD without leaving a corrupt destination after interruption."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        data.write_h5ad(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="replogle22k562-essential")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--num-iterations", type=int, default=MILLER_CONFIGURATION["num_iterations"])
    parser.add_argument("--batch-size", type=int, default=MILLER_CONFIGURATION["batch_size"])
    parser.add_argument("--valid-freq", type=int)
    parser.add_argument("--pca-components", type=int, default=MILLER_CONFIGURATION["n_pca_components"])
    add_point_run_arguments(parser)
    args = parser.parse_args()

    import cellflow
    from cellflow.model import CellFlow
    from cellflow.preprocessing import centered_pca, project_pca

    configuration = {
        **MILLER_CONFIGURATION,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "valid_freq": args.valid_freq or args.num_iterations,
        "n_pca_components": args.pca_components,
        "validation_usage": "metrics at training end; not checkpoint selection",
        "population_size_policy": "match observed test cells",
    }
    run = begin_population_model_run(
        args,
        repository_root=REPOSITORY_ROOT,
        model_id="cellflow",
        environment=ENVIRONMENT,
        configuration=configuration,
        additional_inputs={"esm2_embeddings": args.embedding_path},
    )
    if run.skipped:
        print(f"verified completed run; skipping {run.handle.directory}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    data = ad.read_h5ad(run.raw_path)
    split = json.loads(run.split_manifest_path.read_text())
    fold = split["folds"][args.fold]
    targets = list(map(str, split["perturbations"]))
    embeddings = load_embeddings(args.embedding_path, targets)
    annotate_cellflow(data, embeddings)
    train, validation, test_controls = split_cellflow_data(
        data,
        fold,
        seed=args.seed,
        n_control_train=configuration["n_control_cells_train"],
        n_control_val=configuration["n_control_cells_val"],
        n_control_test=configuration["n_control_cells_test"],
    )

    n_components = min(args.pca_components, train.n_obs - 1, train.n_vars - 1)
    if n_components < 2:
        raise ValueError("CellFlow requires at least two PCA components")
    centered_pca(train, method="scanpy", keep_centered_data=False, n_comps=n_components)
    project_pca(validation, ref_adata=train)

    model = CellFlow(train, solver="otfm")
    model.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"genetic_perturbation": ("gene_target_1", "gene_target_2")},
        perturbation_covariate_reps={"genetic_perturbation": "gene_embeddings"},
        max_combination_length=configuration["max_combination_length"],
        null_value=0.0,
    )
    model.prepare_validation_data(
        train,
        name="train",
        n_conditions_on_log_iteration=1,
        n_conditions_on_train_end=1,
    )
    model.prepare_validation_data(
        validation,
        name="val",
        n_conditions_on_log_iteration=1,
        n_conditions_on_train_end=1,
    )
    _prepare_model(model, configuration, args.seed)
    callbacks = [
        cellflow.training.Metrics(metrics=["mmd", "e_distance"]),
        cellflow.training.PCADecodedMetrics(ref_adata=train, metrics=["r_squared"]),
    ]
    model.train(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        callbacks=callbacks,
        valid_freq=configuration["valid_freq"],
    )

    run.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run.checkpoint_dir), file_prefix="cellflow", overwrite=True)
    model_path = run.checkpoint_dir / "cellflow_CellFlow.pkl"
    reference_path = run.checkpoint_dir / "pca_reference.h5ad"
    _atomic_write_h5ad(train, reference_path)
    prediction = predict_populations(
        model,
        test_controls,
        data,
        list(map(str, fold["test"])),
        train,
        seed=args.seed,
    )
    run.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_h5ad(prediction, run.prediction_path)
    run.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _training_metrics(model, args.num_iterations).to_csv(run.metrics_path, index=False)
    run.complete(
        implementation_revision=IMPLEMENTATION_REVISION,
        checkpoints=[model_path, reference_path],
        evaluation_transform="identity",
    )
    print(f"completed {run.handle.directory}")


if __name__ == "__main__":
    main()
