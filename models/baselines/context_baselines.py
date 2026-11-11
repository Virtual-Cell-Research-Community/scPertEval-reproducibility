"""Leakage-free point baselines for leave-one-context-out experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from workflows.predictions import build_prediction_manifest, expected_test_conditions
from workflows.run_contract import begin_run, complete_run, repository_revision

HERE = Path(__file__).resolve().parent
ROOT = REPOSITORY_ROOT


def _mean(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel()


def _context_linear_prediction(
    data: ad.AnnData,
    *,
    train_contexts: list[str],
    test_context: str,
    model_conditions: list[str],
    ridge: float,
) -> list[np.ndarray]:
    """Joint ridge model using condition identity and observed control-state PCs."""
    contexts = data.obs["context"].astype(str).to_numpy()
    perturbations = data.obs["model_perturbation"].astype(str).to_numpy()
    control_means = {
        context: _mean(data.X[(contexts == context) & (perturbations == "control")])
        for context in [*train_contexts, test_context]
    }

    train_controls = np.vstack([control_means[context] for context in train_contexts])
    center = train_controls.mean(axis=0)
    centered_controls = train_controls - center
    _, _, components = np.linalg.svd(centered_controls, full_matrices=False)
    n_context_components = min(max(len(train_contexts) - 1, 0), components.shape[0])
    components = components[:n_context_components]

    n_conditions = len(model_conditions)
    design_rows = []
    response_rows = []
    for context_index, context in enumerate(train_contexts):
        context_scores = centered_controls[context_index] @ components.T
        for condition_index, condition in enumerate(model_conditions):
            condition_mask = (contexts == context) & (perturbations == condition)
            if not condition_mask.any():
                raise ValueError(f"training context {context!r} has no cells for {condition!r}")
            row = np.zeros(1 + n_conditions + n_context_components, dtype=np.float64)
            row[0] = 1.0
            row[1 + condition_index] = 1.0
            row[1 + n_conditions :] = context_scores
            design_rows.append(row)
            response_rows.append(_mean(data.X[condition_mask]))

    design = np.vstack(design_rows)
    response = np.vstack(response_rows)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ response)

    test_scores = (control_means[test_context] - center) @ components.T
    predictions = []
    for condition_index in range(n_conditions):
        row = np.zeros(design.shape[1], dtype=np.float64)
        row[0] = 1.0
        row[1 + condition_index] = 1.0
        row[1 + n_conditions :] = test_scores
        predictions.append((row @ coefficients).astype(np.float32))
    return predictions


def predict_context_baselines(
    data: ad.AnnData,
    split: dict,
    fold: int | str,
    *,
    capability: str = "point",
    population_size: int = 100,
    seed: int = 42,
    linear_ridge: float = 1.0,
) -> dict[str, ad.AnnData]:
    """Predict held-out profiles or populations without using held-out treated cells."""
    fold_spec = split["folds"][fold] if isinstance(fold, int) else next(
        item for item in split["folds"] if item["id"] == fold
    )
    test_context = str(fold_spec["test_context"])
    train_contexts = list(map(str, fold_spec["train_contexts"]))
    expected = expected_test_conditions(split, fold, capability)
    model_conditions = [condition.removeprefix(f"{test_context}::") for condition in expected]

    contexts = data.obs["context"].astype(str).to_numpy()
    perturbations = data.obs["model_perturbation"].astype(str).to_numpy()
    test_control_mask = (contexts == test_context) & (perturbations == "control")
    test_control = _mean(data.X[test_control_mask])
    no_change_rows = []
    transfer_rows = []
    if capability == "population":
        control_indices = np.flatnonzero(test_control_mask)
        if len(control_indices) < population_size:
            raise ValueError(
                f"held-out context {test_context!r} has {len(control_indices)} controls; "
                f"expected at least {population_size}"
            )
        rng = np.random.default_rng(seed)
        selected_controls = np.sort(rng.choice(control_indices, population_size, replace=False))
        control_population = data.X[selected_controls]
        control_population_dense = (
            control_population.toarray() if sp.issparse(control_population) else np.asarray(control_population)
        )
    for condition in model_conditions:
        deltas = []
        for context in train_contexts:
            control = _mean(data.X[(contexts == context) & (perturbations == "control")])
            treated_mask = (contexts == context) & (perturbations == condition)
            if not treated_mask.any():
                raise ValueError(f"training context {context!r} has no cells for {condition!r}")
            deltas.append(_mean(data.X[treated_mask]) - control)
        mean_delta = np.mean(deltas, axis=0)
        if capability == "population":
            no_change_rows.append(control_population)
            transfer_rows.append(control_population_dense + mean_delta)
        elif capability == "point":
            no_change_rows.append(test_control)
            transfer_rows.append(test_control + mean_delta)
        else:
            raise ValueError(f"unknown prediction capability: {capability}")

    def bundle(rows: list) -> ad.AnnData:
        if capability == "population":
            matrix = (
                sp.vstack(rows, format="csr")
                if rows and all(sp.issparse(row) for row in rows)
                else np.vstack([row.toarray() if sp.issparse(row) else row for row in rows]).astype(np.float32)
            )
            labels = [condition for condition in expected for _ in range(population_size)]
        else:
            matrix = np.vstack(rows).astype(np.float32)
            labels = expected
        result = ad.AnnData(
            X=matrix,
            obs=pd.DataFrame(
                {"perturbation": labels},
                index=[f"prediction-{index}" for index in range(len(labels))],
            ),
            var=pd.DataFrame(index=data.var_names.copy()),
        )
        return result

    predictions = {"no-change": bundle(no_change_rows), "fold-mean": bundle(transfer_rows)}
    if capability == "point":
        predictions["linear"] = bundle(
            _context_linear_prediction(
                data,
                train_contexts=train_contexts,
                test_context=test_context,
                model_conditions=model_conditions,
                ridge=linear_ridge,
            )
        )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--data-manifest",
        type=Path,
        help="prepared model-view manifest (default: <data stem>.manifest.json)",
    )
    parser.add_argument("--capability", choices=["point", "population"], default="population")
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-ridge", type=float, default=1.0)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["no-change", "fold-mean", "linear"],
        help="subset to materialize (default: both population baselines, or all point baselines)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    fold: int | str = int(args.fold) if args.fold.isdigit() else args.fold
    fold_spec = split["folds"][fold] if isinstance(fold, int) else next(
        item for item in split["folds"] if item["id"] == fold
    )
    fold_id = str(fold_spec.get("id", fold))
    data_manifest = args.data_manifest or args.data.with_suffix(".manifest.json")
    if not data_manifest.is_file():
        raise FileNotFoundError(data_manifest)
    default_models = ["no-change", "fold-mean", *(["linear"] if args.capability == "point" else [])]
    models = args.models or default_models
    if args.capability == "population" and "linear" in models:
        raise ValueError("the context linear baseline is point-capable, not population-capable")
    revision = repository_revision(ROOT)
    handles = {
        model: begin_run(
            args.runs_root,
            experiment=split["id"],
            model=model,
            fold=fold_id,
            seed=args.seed,
            configuration={
                "capability": args.capability,
                "population_size": args.population_size,
                "linear_ridge": args.linear_ridge if model == "linear" else None,
            },
            inputs={"model_view_manifest": data_manifest, "split_manifest": args.split},
            command=[sys.executable, *sys.argv],
            repo_root=ROOT,
            force=args.force,
        )
        for model in models
    }
    if all(handle.skipped for handle in handles.values()):
        print(f"all verified baseline runs already complete for {split['id']} {fold_id} seed {args.seed}")
        return 0

    predictions = predict_context_baselines(
        ad.read_h5ad(args.data),
        split,
        fold,
        capability=args.capability,
        population_size=args.population_size,
        seed=args.seed,
        linear_ridge=args.linear_ridge,
    )
    for model, prediction in predictions.items():
        if model not in models:
            continue
        handle = handles[model]
        if handle.skipped:
            print(f"skipping verified completed run: {handle.directory}")
            continue
        output = handle.directory / "predictions" / "predictions.h5ad"
        prediction.write_h5ad(output, compression="gzip")
        manifest = build_prediction_manifest(
            prediction_path=output,
            truth_path=args.data,
            dataset_manifest_path=data_manifest,
            split_manifest_path=args.split,
            capability=args.capability if model != "linear" else "point",
            model_id=model,
            implementation_revision=revision,
            adapter_revision=revision,
            fold=fold,
            seed=args.seed,
            environment="uv.lock",
            command=" ".join([sys.executable, *sys.argv]),
            minimum_cells_per_condition=args.population_size if args.capability == "population" else None,
        )
        manifest_path = output.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        complete_run(
            handle,
            {
                "prediction": output,
                "prediction_manifest": manifest_path,
            },
            precomputed_checksums={"prediction": manifest["artifact"]["sha256"]},
        )
        print(f"wrote {prediction.shape} {model} predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
