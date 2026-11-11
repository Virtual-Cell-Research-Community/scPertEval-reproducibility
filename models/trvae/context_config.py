"""Pure data helpers for joint trVAE leave-one-context-out runs."""

from __future__ import annotations

import zlib

import anndata as ad
import numpy as np
import pandas as pd


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
    population_size: int = 100,
    seed: int = 42,
) -> ad.AnnData:
    """Select a deterministic held-out-context control population."""
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    test_context = str(_fold(split, fold)["test_context"])
    labels = data.obs[perturbation_key].astype(str)
    contexts = data.obs[context_key].astype(str)
    indices = np.flatnonzero(((contexts == test_context) & (labels == "control")).to_numpy())
    if len(indices) < population_size:
        raise ValueError(
            f"held-out context {test_context!r} has {len(indices)} controls; need {population_size}"
        )
    context_seed = zlib.crc32(test_context.encode())
    rng = np.random.default_rng(np.asarray([seed, context_seed], dtype=np.uint32))
    selected = np.sort(rng.choice(indices, population_size, replace=False))
    return data[selected].copy()


def build_prediction_adata(
    predictions: dict[str, np.ndarray],
    var: pd.DataFrame,
    *,
    test_context: str,
    population_size: int,
) -> ad.AnnData:
    """Package trVAE populations, which already share the evaluation expression space."""
    if not predictions:
        raise ValueError("no trVAE predictions were supplied")
    matrices = []
    rows = []
    names = []
    for condition in sorted(predictions):
        matrix = np.asarray(predictions[condition], dtype=np.float32)
        if matrix.shape != (population_size, len(var)):
            raise ValueError(
                f"condition {condition!r} has shape {matrix.shape}; "
                f"expected {(population_size, len(var))}"
            )
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("trVAE predictions must be finite and non-negative")
        matrices.append(matrix)
        archival = f"{test_context}::{condition}"
        for index in range(population_size):
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
