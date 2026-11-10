"""Leakage-free point baselines for leave-one-context-out experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from workflows.predictions import expected_test_conditions

HERE = Path(__file__).resolve().parent


def _mean(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel()


def predict_context_baselines(
    data: ad.AnnData,
    split: dict,
    fold: int | str,
) -> dict[str, ad.AnnData]:
    """Predict held-out context profiles from its controls and training-context effects."""
    fold_spec = split["folds"][fold] if isinstance(fold, int) else next(
        item for item in split["folds"] if item["id"] == fold
    )
    test_context = str(fold_spec["test_context"])
    train_contexts = list(map(str, fold_spec["train_contexts"]))
    expected = expected_test_conditions(split, fold, "point")
    model_conditions = [condition.removeprefix(f"{test_context}::") for condition in expected]

    contexts = data.obs["context"].astype(str).to_numpy()
    perturbations = data.obs["model_perturbation"].astype(str).to_numpy()
    test_control = _mean(data.X[(contexts == test_context) & (perturbations == "control")])
    no_change_rows = []
    transfer_rows = []
    for condition in model_conditions:
        no_change_rows.append(test_control)
        deltas = []
        for context in train_contexts:
            control = _mean(data.X[(contexts == context) & (perturbations == "control")])
            treated_mask = (contexts == context) & (perturbations == condition)
            if not treated_mask.any():
                raise ValueError(f"training context {context!r} has no cells for {condition!r}")
            deltas.append(_mean(data.X[treated_mask]) - control)
        transfer_rows.append(test_control + np.mean(deltas, axis=0))

    def bundle(rows: list[np.ndarray]) -> ad.AnnData:
        result = ad.AnnData(
            X=np.vstack(rows).astype(np.float32),
            obs=pd.DataFrame({"perturbation": expected}, index=[f"prediction-{index}" for index in range(len(rows))]),
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
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    fold: int | str = int(args.fold) if args.fold.isdigit() else args.fold
    predictions = predict_context_baselines(ad.read_h5ad(args.data), split, fold)
    fold_tag = fold if isinstance(fold, int) else next(
        index for index, item in enumerate(split["folds"]) if item["id"] == fold
    )
    for model, prediction in predictions.items():
        output = args.output_root / model / "predictions" / split["id"] / f"fold_{fold_tag}.h5ad"
        output.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_h5ad(output)
        print(f"wrote {prediction.shape} {model} predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
