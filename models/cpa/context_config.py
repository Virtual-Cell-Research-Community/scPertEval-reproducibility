"""Pure data helpers for joint CPA leave-one-context-out runs."""

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


def assign_cpa_splits(
    data: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    context_key: str = "context",
) -> ad.AnnData:
    """Assign the frozen train/validation/test populations for one context fold."""
    required = {perturbation_key, context_key}
    missing = sorted(required - set(data.obs.columns))
    if missing:
        raise ValueError(f"CPA input is missing obs fields: {', '.join(missing)}")

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
    result.obs["cpa_split"] = pd.Categorical(
        assignments,
        categories=["train", "valid", "test"],
    )
    result.obs["cpa_dose"] = 1.0
    counts = result.obs["cpa_split"].value_counts()
    if any(int(counts.get(part, 0)) == 0 for part in ["train", "valid", "test"]):
        raise ValueError(f"CPA fold contains an empty split: {counts.to_dict()}")
    return result


def build_counterfactual_queries(
    data: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    context_key: str = "context",
    population_size: int | None = None,
    seed: int = 42,
) -> ad.AnnData:
    """Build one held-out-control query per observed treated cell and condition."""
    fold_spec = _fold(split, fold)
    test_context = str(fold_spec["test_context"])
    conditions = model_conditions(split, capability="population")
    sizes = (
        {condition: population_size for condition in conditions}
        if population_size is not None
        else observed_population_sizes(
            data,
            split,
            fold,
            perturbation_key=perturbation_key,
            context_key=context_key,
        )
    )
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("population sizes must be positive")
    controls = sample_heldout_controls(
        data,
        split,
        fold,
        max(sizes.values()),
        perturbation_key=perturbation_key,
        context_key=context_key,
        seed=seed,
    )

    queries = []
    for condition in conditions:
        size = sizes[condition]
        query = controls[:size].copy()
        query.obs[perturbation_key] = condition
        query.obs[context_key] = test_context
        query.obs["cpa_dose"] = 1.0
        query.obs["cpa_split"] = "test"
        query.obs["perturbation"] = f"{test_context}::{condition}"
        query.obs_names = pd.Index(
            [f"{test_context}::{condition}::prediction-{index}" for index in range(size)]
        )
        queries.append(query)
    return ad.concat(queries, join="inner", merge="same", index_unique=None)
