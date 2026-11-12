"""Pure data helpers for joint bioLORD leave-one-context-out runs."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from models.state.context_config import (
    model_conditions,
    observed_population_sizes,
    sample_heldout_controls,
)


def _fold(split: dict, fold: int) -> dict:
    try:
        return split["folds"][fold]
    except IndexError as error:
        raise ValueError(f"fold {fold} is outside 0..{len(split['folds']) - 1}") from error


def assign_biolord_splits(
    data: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    context_key: str = "context",
) -> ad.AnnData:
    """Assign training, validation, and held-out-treatment cells for one fold."""
    required = {perturbation_key, context_key}
    missing = sorted(required - set(data.obs.columns))
    if missing:
        raise ValueError(f"bioLORD input is missing obs fields: {', '.join(missing)}")

    fold_spec = _fold(split, fold)
    test_context = str(fold_spec["test_context"])
    train_contexts = set(map(str, fold_spec["train_contexts"]))
    labels = data.obs[perturbation_key].astype(str)
    contexts = data.obs[context_key].astype(str)
    controls = labels == "control"
    assignments = np.full(data.n_obs, "train", dtype=object)
    assignments[(contexts == test_context) & ~controls] = "test"
    assignments[
        contexts.isin(train_contexts)
        & labels.isin(set(map(str, split["validation_conditions"])))
        & ~controls
    ] = "valid"

    result = data.copy()
    result.obs["biolord_split"] = pd.Categorical(
        assignments,
        categories=["train", "valid", "test"],
    )
    counts = result.obs["biolord_split"].value_counts()
    if any(int(counts.get(part, 0)) == 0 for part in ["train", "valid", "test"]):
        raise ValueError(f"bioLORD fold contains an empty split: {counts.to_dict()}")
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
    """Select enough held-out controls for the largest observed treated population."""
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


def relabel_biolord_predictions(
    prediction: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    population_size: int | None = None,
    population_sizes: dict[str, int] | None = None,
) -> ad.AnnData:
    """Keep frozen test conditions and assign canonical archival identifiers."""
    if perturbation_key not in prediction.obs:
        raise ValueError(f"bioLORD predictions are missing obs[{perturbation_key!r}]")
    test_context = str(_fold(split, fold)["test_context"])
    allowed = set(model_conditions(split, capability="population"))
    labels = prediction.obs[perturbation_key].astype(str)
    result = prediction[labels.isin(allowed)].copy()
    result.obs["perturbation"] = test_context + "::" + result.obs[perturbation_key].astype(str)
    if population_sizes is not None:
        sizes = population_sizes
    elif population_size is not None:
        sizes = {condition: population_size for condition in allowed}
    else:
        raise ValueError("population_sizes are required for match-observed predictions")
    model_labels = result.obs[perturbation_key].astype(str)
    keep = np.zeros(result.n_obs, dtype=bool)
    for condition, size in sizes.items():
        indices = np.flatnonzero((model_labels == condition).to_numpy())
        keep[indices[:size]] = True
    result = result[keep].copy()
    within_population = result.obs.groupby("perturbation", observed=True).cumcount()
    result.obs_names = pd.Index(
        [
            f"{label}::prediction-{index}"
            for label, index in zip(result.obs["perturbation"], within_population, strict=True)
        ]
    )
    counts = result.obs["perturbation"].value_counts()
    expected = {f"{test_context}::{condition}" for condition in allowed}
    expected_counts = {f"{test_context}::{condition}": size for condition, size in sizes.items()}
    actual_counts = {condition: int(counts.get(condition, 0)) for condition in expected}
    if actual_counts != expected_counts:
        raise ValueError(
            "bioLORD predictions do not cover the frozen populations; "
            f"expected {expected_counts}, got {actual_counts}"
        )
    return result
