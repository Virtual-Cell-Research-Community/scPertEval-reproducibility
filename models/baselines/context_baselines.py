"""Leakage-free point baselines for leave-one-context-out experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from workflows.predictions import expected_test_conditions

HERE = Path(__file__).resolve().parent


def _mean(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel()


def predict_context_baselines(
    data: ad.AnnData,
    split: dict,
    fold: int | str,
    *,
    capability: str = "point",
    population_size: int = 120,
    seed: int = 42,
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

    return {"no-change": bundle(no_change_rows), "fold-mean": bundle(transfer_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--capability", choices=["point", "population"], default="population")
    parser.add_argument("--population-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    fold: int | str = int(args.fold) if args.fold.isdigit() else args.fold
    predictions = predict_context_baselines(
        ad.read_h5ad(args.data),
        split,
        fold,
        capability=args.capability,
        population_size=args.population_size,
        seed=args.seed,
    )
    fold_tag = fold if isinstance(fold, int) else next(
        index for index, item in enumerate(split["folds"]) if item["id"] == fold
    )
    for model, prediction in predictions.items():
        output = args.output_root / model / "predictions" / split["id"] / f"fold_{fold_tag}.h5ad"
        output.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_h5ad(output, compression="gzip")
        print(f"wrote {prediction.shape} {model} predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
