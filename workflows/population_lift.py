"""Lift centroid predictions into matched cell populations using control residuals."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from workflows.predictions import expected_test_conditions
from workflows.preparation_utils import file_checksum


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def _rng(seed: int, *labels: str) -> np.random.Generator:
    entropy = [seed, *(zlib.crc32(label.encode()) for label in labels)]
    return np.random.default_rng(np.asarray(entropy, dtype=np.uint32))


def _sample_indices(
    obs: pd.DataFrame,
    control_indices: np.ndarray,
    treated_indices: np.ndarray,
    *,
    condition: str,
    seed: int,
    batch_key: str,
) -> np.ndarray:
    """Sample controls, matching the treated batch composition when possible."""
    if batch_key in obs and len(treated_indices):
        control_batches = obs.iloc[control_indices][batch_key].astype(str).to_numpy()
        treated_batches = obs.iloc[treated_indices][batch_key].astype(str)
        if set(treated_batches.unique()) <= set(control_batches):
            selected: list[int] = []
            for batch, count in treated_batches.value_counts(sort=False).sort_index().items():
                candidates = control_indices[control_batches == str(batch)]
                chosen = _rng(seed, condition, str(batch)).choice(
                    candidates,
                    size=int(count),
                    replace=len(candidates) < int(count),
                )
                selected.extend(map(int, chosen))
            return np.asarray(selected, dtype=np.int64)
    return _rng(seed, condition).choice(
        control_indices,
        size=len(treated_indices),
        replace=len(control_indices) < len(treated_indices),
    ).astype(np.int64)


def _nonnegative_with_target_mean(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Project onto nonnegative expression while retaining the point-model centroid."""
    result = np.maximum(values, 0).astype(np.float32, copy=False)
    observed = result.mean(axis=0, dtype=np.float64)
    positive = (target > 0) & (observed > 0)
    result[:, positive] *= (target[positive] / observed[positive]).astype(np.float32)
    result[:, target <= 0] = 0
    return result


def lift_prediction_population(
    truth: ad.AnnData,
    prediction: ad.AnnData,
    split: dict,
    fold: int,
    *,
    seed: int = 42,
    batch_key: str = "batch",
) -> ad.AnnData:
    """Broadcast each predicted mean effect over deterministically sampled controls.

    The sampled control residuals are centered before adding the model prediction, so the
    resulting population retains the original point prediction as its exact centroid. Negative
    log-expression values are clipped and each gene is rescaled to restore that centroid.
    """
    if "perturbation" not in prediction.obs:
        raise ValueError("prediction is missing obs['perturbation']")
    if not prediction.var_names.is_unique or not truth.var_names.is_unique:
        raise ValueError("truth and prediction require unique gene identifiers")
    if set(map(str, prediction.var_names)) != set(map(str, truth.var_names)):
        raise ValueError("truth and prediction do not contain the same gene set")

    fold_spec = split["folds"][fold]
    if "test_context" in fold_spec:
        required = {"context", "evaluation_condition"}
        missing = sorted(required - set(truth.obs.columns))
        if missing:
            raise ValueError(f"context-transfer truth is missing obs fields: {', '.join(missing)}")
        context_mask = truth.obs["context"].astype(str) == str(fold_spec["test_context"])
        local = truth[context_mask.to_numpy(), prediction.var_names].copy()
        labels = local.obs["evaluation_condition"].astype(str).to_numpy()
    else:
        if "perturbation" not in truth.obs:
            raise ValueError("perturbation-generalization truth is missing obs['perturbation']")
    conditions = expected_test_conditions(split, fold, "population")
    if "test_context" not in fold_spec:
        eligible = truth.obs["perturbation"].astype(str).isin(["control", *conditions])
        local = truth[eligible.to_numpy(), prediction.var_names].copy()
        labels = local.obs["perturbation"].astype(str).to_numpy()

    control_indices = np.flatnonzero(labels == "control")
    if not len(control_indices):
        raise ValueError("truth contains no eligible control cells")
    prediction_labels = prediction.obs["perturbation"].astype(str).to_numpy()

    matrices: list[np.ndarray] = []
    observations: list[pd.DataFrame] = []
    for condition in conditions:
        treated_indices = np.flatnonzero(labels == condition)
        if not len(treated_indices):
            raise ValueError(f"truth contains no treated cells for {condition!r}")
        point_indices = np.flatnonzero(prediction_labels == condition)
        if not len(point_indices):
            raise ValueError(f"prediction contains no rows for {condition!r}")

        point = _dense(prediction.X[point_indices]).mean(axis=0, dtype=np.float64)
        if np.any(point < 0) or not np.isfinite(point).all():
            raise ValueError(f"prediction centroid for {condition!r} is not finite nonnegative expression")
        selected = _sample_indices(
            local.obs,
            control_indices,
            treated_indices,
            condition=condition,
            seed=seed,
            batch_key=batch_key,
        )
        controls = _dense(local.X[selected]).astype(np.float32, copy=False)
        residuals = controls - controls.mean(axis=0, dtype=np.float64)
        lifted = _nonnegative_with_target_mean(point + residuals, point)
        matrices.append(lifted)

        obs = pd.DataFrame(
            {
                "perturbation": condition,
                "source_control": local.obs_names[selected].astype(str),
            },
            index=[f"{condition}-lifted-{index}" for index in range(len(selected))],
        )
        if batch_key in local.obs:
            obs[batch_key] = local.obs.iloc[selected][batch_key].astype(str).to_numpy()
        observations.append(obs)

    return ad.AnnData(
        X=np.concatenate(matrices).astype(np.float32, copy=False),
        obs=pd.concat(observations),
        var=pd.DataFrame(index=prediction.var_names.copy()),
    )


def lifted_manifest(
    *,
    output_path: Path,
    source_prediction_path: Path,
    truth_path: Path,
    split_manifest_path: Path,
    model_id: str,
    fold: int,
    seed: int,
    source_prediction_sha256: str | None = None,
    truth_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
) -> dict:
    """Build the small provenance manifest consumed by the evaluation driver."""
    output = ad.read_h5ad(output_path, backed="r")
    try:
        n_obs, n_vars = output.shape
    finally:
        output.file.close()
    return {
        "schema_version": 1,
        "capability": "population",
        "model": {"id": model_id},
        "split": {
            "fold": fold,
            "manifest_sha256": split_manifest_sha256 or file_checksum(split_manifest_path),
        },
        "expression": {
            "space": "log1p_expression",
            "evaluation_space": "log1p_expression",
            "evaluation_transform": "identity",
        },
        "population": {
            "construction": "centroid-preserving-control-residual-lift",
            "control_sampling_seed": seed,
            "batch_matched_when_available": True,
        },
        "source": {
            "prediction_sha256": source_prediction_sha256 or file_checksum(source_prediction_path),
            "truth_sha256": truth_sha256 or file_checksum(truth_path),
        },
        "artifact": {
            "path": output_path.name,
            "format": "h5ad",
            "sha256": file_checksum(output_path),
            "n_obs": n_obs,
            "n_vars": n_vars,
            "perturbation_key": "perturbation",
        },
    }


def manifest_is_current(
    manifest_path: Path,
    *,
    source_prediction_sha256: str,
    truth_sha256: str,
    split_manifest_sha256: str,
) -> bool:
    """Return whether a completed lift still matches all three source artifacts."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        output_path = manifest_path.parent / manifest["artifact"]["path"]
        return (
            output_path.is_file()
            and manifest["source"]["prediction_sha256"] == source_prediction_sha256
            and manifest["source"]["truth_sha256"] == truth_sha256
            and manifest["split"]["manifest_sha256"] == split_manifest_sha256
        )
    except (KeyError, OSError, json.JSONDecodeError):
        return False
