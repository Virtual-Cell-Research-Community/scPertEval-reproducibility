"""Pure split helpers for STATE leave-one-context-out training."""

from __future__ import annotations

import warnings
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _fold(split: dict, fold: int) -> dict:
    try:
        return split["folds"][fold]
    except IndexError as error:
        raise ValueError(f"fold {fold} is outside 0..{len(split['folds']) - 1}") from error


def model_conditions(split: dict, *, capability: str) -> list[str]:
    """Return model-space condition labels, optionally excluding undersized populations."""
    if "perturbations" in split:
        return sorted(map(str, split["perturbations"]))
    if "drugs" not in split or "doses" not in split:
        raise ValueError("context split must declare perturbations or drugs and doses")
    excluded = {
        (item["drug"], float(item["dose"]))
        for item in split.get("population_exclusions", [])
    }
    conditions = []
    for drug in split["drugs"]:
        for dose in split["doses"]:
            if capability == "population" and (drug["label"], float(dose)) in excluded:
                continue
            conditions.append(f"{drug['canonical_label']}::{float(dose):g}nM")
    return sorted(conditions)


def observed_population_sizes(
    data: ad.AnnData,
    split: dict,
    fold: int,
    *,
    perturbation_key: str = "model_perturbation",
    context_key: str = "context",
) -> dict[str, int]:
    """Return the held-out treated-cell count for every population condition."""
    required = {perturbation_key, context_key}
    missing = sorted(required - set(data.obs.columns))
    if missing:
        raise ValueError(f"context data are missing obs fields: {', '.join(missing)}")
    test_context = str(_fold(split, fold)["test_context"])
    allowed = model_conditions(split, capability="population")
    in_context = data.obs[context_key].astype(str) == test_context
    labels = data.obs.loc[in_context, perturbation_key].astype(str)
    counts = labels[labels.isin(allowed)].value_counts().to_dict()
    missing_conditions = sorted(set(allowed) - set(counts))
    if missing_conditions:
        raise ValueError(f"held-out context is missing population conditions: {missing_conditions}")
    return {condition: int(counts[condition]) for condition in allowed}


def sample_heldout_controls(
    data: ad.AnnData,
    split: dict,
    fold: int,
    population_size: int,
    *,
    perturbation_key: str = "model_perturbation",
    context_key: str = "context",
    seed: int = 42,
) -> ad.AnnData:
    """Sample held-out controls, with replacement only when the requested size requires it."""
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    test_context = str(_fold(split, fold)["test_context"])
    labels = data.obs[perturbation_key].astype(str)
    contexts = data.obs[context_key].astype(str)
    indices = np.flatnonzero(((contexts == test_context) & (labels == "control")).to_numpy())
    if not len(indices):
        raise ValueError(f"held-out context {test_context!r} has no control cells")
    context_seed = zlib.crc32(test_context.encode())
    rng = np.random.default_rng(np.asarray([seed, context_seed], dtype=np.uint32))
    replace = len(indices) < population_size
    selected = rng.choice(indices, population_size, replace=replace)
    if not replace:
        selected = np.sort(selected)
    # Replacement sampling temporarily repeats source observation names. The archival query
    # identifiers assigned immediately below are unique, so suppress AnnData's transient warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Observation names are not unique")
        result = data[selected].copy()
    result.obs["source_obs_name"] = result.obs_names.astype(str)
    result.obs_names = pd.Index(
        [f"{test_context}::control-sample-{index}" for index in range(population_size)]
    )
    return result


def build_context_toml(
    split: dict,
    fold: int,
    out_path: Path,
    dataset_name: str,
    dataset_dir: Path,
) -> Path:
    """Write a few-shot split that retains held-out controls but withholds its treatments."""
    fold_spec = _fold(split, fold)
    test_context = str(fold_spec["test_context"])
    train_contexts = list(map(str, fold_spec["train_contexts"]))
    validation = sorted(map(str, split["validation_conditions"]))
    test = model_conditions(split, capability="point")

    def values(items: list[str]) -> str:
        return ", ".join(f'"{item}"' for item in items)

    lines = [
        "[datasets]",
        f'{dataset_name} = "{dataset_dir}"',
        "",
        "[training]",
        f'{dataset_name} = "train"',
        "",
        "[zeroshot]",
        "",
        "[fewshot]",
    ]
    for context in train_contexts:
        lines.extend(
            [
                f'[fewshot."{dataset_name}.{context}"]',
                f"val = [{values(validation)}]",
                "test = []",
                "",
            ]
        )
    lines.extend(
        [
            f'[fewshot."{dataset_name}.{test_context}"]',
            "val = []",
            f"test = [{values(test)}]",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))
    return out_path


def relabel_context_predictions(
    prediction: ad.AnnData,
    split: dict,
    fold: int,
    *,
    capability: str = "population",
    perturbation_key: str = "perturbation",
) -> ad.AnnData:
    """Keep held-out-context predictions and assign archival evaluation identifiers."""
    fold_spec = _fold(split, fold)
    test_context = str(fold_spec["test_context"])
    if "cell_type" not in prediction.obs:
        raise ValueError("STATE context predictions are missing obs['cell_type']")
    if perturbation_key not in prediction.obs:
        raise ValueError(f"STATE context predictions are missing obs[{perturbation_key!r}]")
    labels = prediction.obs[perturbation_key].astype(str)
    allowed = set(model_conditions(split, capability=capability))
    mask = (prediction.obs["cell_type"].astype(str) == test_context) & labels.isin(allowed)
    result = prediction[mask].copy()
    result.obs["perturbation"] = test_context + "::" + result.obs[perturbation_key].astype(str)
    observed = set(result.obs["perturbation"].astype(str))
    expected = {f"{test_context}::{condition}" for condition in allowed}
    if observed != expected:
        raise ValueError(
            "STATE context predictions do not cover the frozen test conditions; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return result
