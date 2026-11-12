"""Pure data helpers for joint trVAE leave-one-context-out runs."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from models.state.context_config import observed_population_sizes, sample_heldout_controls


def densify_expression(data: ad.AnnData) -> ad.AnnData:
    """Materialize ``X`` in memory for trvaep 0.1.0's dense-only data loader.

    The caller passes a fold-local AnnData copy, so this never changes the canonical sparse
    view on disk and avoids duplicating the rest of the AnnData object a second time.
    """
    if sparse.issparse(data.X):
        data.X = data.X.toarray()
    else:
        data.X = np.asarray(data.X)
    return data


def _fold(split: dict, fold: int) -> dict:
    try:
        return split["folds"][fold]
    except IndexError as error:
        raise ValueError(f"fold {fold} is outside 0..{len(split['folds']) - 1}") from error


def select_trvae_training_data(
    data: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    context_key: str = "context",
) -> ad.AnnData:
    """Keep all training contexts plus held-out controls, without treatment leakage."""
    required = {perturbation_key, context_key}
    missing = sorted(required - set(data.obs.columns))
    if missing:
        raise ValueError(f"trVAE input is missing obs fields: {', '.join(missing)}")

    fold_spec = _fold(split, fold)
    test_context = str(fold_spec["test_context"])
    train_contexts = set(map(str, fold_spec["train_contexts"]))
    labels = data.obs[perturbation_key].astype(str)
    contexts = data.obs[context_key].astype(str)
    keep = contexts.isin(train_contexts) | ((contexts == test_context) & (labels == "control"))
    result = data[keep.to_numpy()].copy()
    heldout = result.obs[context_key].astype(str) == test_context
    if set(result.obs.loc[heldout, perturbation_key].astype(str)) != {"control"}:
        raise ValueError("trVAE training data leak held-out-context treatments")
    expected_contexts = train_contexts | {test_context}
    observed_contexts = set(result.obs[context_key].astype(str))
    if observed_contexts != expected_contexts:
        raise ValueError(
            "trVAE training data do not cover the frozen contexts; "
            f"missing={sorted(expected_contexts - observed_contexts)}"
        )
    return result


def select_heldout_controls(
    data: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    context_key: str = "context",
    population_size: int | None = None,
    seed: int = 42,
) -> ad.AnnData:
    """Select enough controls for the largest observed held-out population."""
    if population_size is None:
        sizes = observed_population_sizes(
            data,
            split,
            fold,
            perturbation_key=perturbation_key,
            context_key=context_key,
        )
        population_size = max(sizes.values())
    return sample_heldout_controls(
        data,
        split,
        fold,
        population_size,
        perturbation_key=perturbation_key,
        context_key=context_key,
        seed=seed,
    )


def build_prediction_adata(
    predictions: dict[str, np.ndarray],
    var: pd.DataFrame,
    *,
    test_context: str,
    population_size: int | None = None,
    population_sizes: dict[str, int] | None = None,
) -> ad.AnnData:
    """Package trVAE populations, which already share the evaluation expression space."""
    if not predictions:
        raise ValueError("no trVAE predictions were supplied")
    matrices = []
    rows = []
    names = []
    for condition in sorted(predictions):
        expected_size = (
            population_sizes[condition]
            if population_sizes is not None
            else population_size
        )
        if expected_size is None:
            raise ValueError("population_sizes are required for match-observed predictions")
        matrix = np.asarray(predictions[condition], dtype=np.float32)[:expected_size]
        if matrix.shape != (expected_size, len(var)):
            raise ValueError(
                f"condition {condition!r} has shape {matrix.shape}; "
                f"expected {(expected_size, len(var))}"
            )
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("trVAE predictions must be finite and non-negative")
        matrices.append(matrix)
        archival = f"{test_context}::{condition}"
        for index in range(expected_size):
            names.append(f"{archival}::prediction-{index}")
            rows.append(
                {
                    "perturbation": archival,
                    "model_perturbation": condition,
                    "context": test_context,
                }
            )
    return ad.AnnData(
        X=np.concatenate(matrices, axis=0),
        obs=pd.DataFrame(rows, index=pd.Index(names)),
        var=var.copy(),
    )
