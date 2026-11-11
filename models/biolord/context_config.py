"""Pure data helpers for joint bioLORD leave-one-context-out runs."""

from __future__ import annotations

import zlib

import anndata as ad
import numpy as np
import pandas as pd

from models.state.context_config import model_conditions


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
    population_size: int = 100,
    seed: int = 42,
) -> ad.AnnData:
    """Select a deterministic, bounded control population from the held-out context."""
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


def relabel_biolord_predictions(
    prediction: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    population_size: int = 100,
) -> ad.AnnData:
    """Keep frozen test conditions and assign canonical archival identifiers."""
    if perturbation_key not in prediction.obs:
        raise ValueError(f"bioLORD predictions are missing obs[{perturbation_key!r}]")
    test_context = str(_fold(split, fold)["test_context"])
    allowed = set(model_conditions(split, capability="population"))
    labels = prediction.obs[perturbation_key].astype(str)
    result = prediction[labels.isin(allowed)].copy()
    result.obs["perturbation"] = test_context + "::" + result.obs[perturbation_key].astype(str)
    within_population = result.obs.groupby("perturbation", observed=True).cumcount()
    result.obs_names = pd.Index(
        [
            f"{label}::prediction-{index}"
            for label, index in zip(result.obs["perturbation"], within_population, strict=True)
        ]
    )
    counts = result.obs["perturbation"].value_counts()
    expected = {f"{test_context}::{condition}" for condition in allowed}
    if set(counts.index) != expected or not (counts == population_size).all():
        raise ValueError(
            "bioLORD predictions do not cover the frozen populations; "
            f"expected {population_size} cells for {len(expected)} conditions, got {counts.to_dict()}"
        )
    return result
