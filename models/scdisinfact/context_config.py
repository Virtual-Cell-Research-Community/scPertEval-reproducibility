"""Pure data helpers for joint scDisInFact leave-one-context-out runs."""

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


def select_scdisinfact_training_data(
    data: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    context_key: str = "context",
) -> ad.AnnData:
    """Keep every training-context cell and held-out-context controls only."""
    required = {perturbation_key, context_key}
    missing = sorted(required - set(data.obs.columns))
    if missing:
        raise ValueError(f"scDisInFact input is missing obs fields: {', '.join(missing)}")

    fold_spec = _fold(split, fold)
    test_context = str(fold_spec["test_context"])
    train_contexts = set(map(str, fold_spec["train_contexts"]))
    labels = data.obs[perturbation_key].astype(str)
    contexts = data.obs[context_key].astype(str)
    keep = contexts.isin(train_contexts) | ((contexts == test_context) & (labels == "control"))
    result = data[keep.to_numpy()].copy()

    observed_contexts = set(result.obs[context_key].astype(str))
    expected_contexts = train_contexts | {test_context}
    if observed_contexts != expected_contexts:
        raise ValueError(
            "scDisInFact training data do not cover the frozen contexts; "
            f"missing={sorted(expected_contexts - observed_contexts)}"
        )
    heldout = result.obs[context_key].astype(str) == test_context
    if set(result.obs.loc[heldout, perturbation_key].astype(str)) != {"control"}:
        raise ValueError("scDisInFact training data leak held-out-context treatments")
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
    """Select a deterministic control population from the held-out context."""
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


def counts_to_evaluation_view(counts: np.ndarray, target_sum: float = 10_000.0) -> np.ndarray:
    """Map positive predicted counts to the shared normalize-total/log1p view."""
    values = np.asarray(counts, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("predicted counts must be a two-dimensional matrix")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("predicted counts must be finite and non-negative")
    totals = values.sum(axis=1, keepdims=True, dtype=np.float64)
    if np.any(totals <= 0):
        raise ValueError("predicted count rows must have positive totals")
    normalized = values * (target_sum / totals)
    return np.log1p(normalized).astype(np.float32)


def build_prediction_adata(
    predictions: dict[str, np.ndarray],
    var: pd.DataFrame,
    *,
    test_context: str,
    population_size: int,
) -> ad.AnnData:
    """Package per-condition native-count predictions in the common evaluation space."""
    if not predictions:
        raise ValueError("no scDisInFact predictions were supplied")
    matrices = []
    rows = []
    names = []
    for condition in sorted(predictions):
        matrix = np.asarray(predictions[condition])
        if matrix.shape != (population_size, len(var)):
            raise ValueError(
                f"condition {condition!r} has shape {matrix.shape}; "
                f"expected {(population_size, len(var))}"
            )
        matrices.append(counts_to_evaluation_view(matrix))
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
