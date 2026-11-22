"""Score retained prediction bundles with the frozen Atlas protocol universe."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scperteval as spe
from scperteval.types import Protocol

from .atlas_runtime import target_key_for_task
from .truth_coverage_runtime import REASON_MAP, task_protocols

__all__ = ["evaluate_prediction", "prepare_prediction_fold", "task_protocols"]


def prepare_prediction_fold(
    truth: ad.AnnData,
    *,
    task: str,
    fold_label: str,
    protocols: Sequence[Protocol],
    minimum_cells: int = 50,
    workers: int = 0,
):
    """Warm truth-derived state once so every model in a fold reuses it."""
    return spe.prepare(
        truth,
        list(protocols),
        min_cells=minimum_cells,
        seed=42,
        control_label="control",
        perturbation_key="perturbation",
        target_key=target_key_for_task(task, truth.obs.columns),
        workers=workers,
        name=f"{task}__{fold_label}",
    )


def evaluate_prediction(
    prepared,
    prediction: ad.AnnData,
    *,
    task: str,
    fold_label: str,
    model: str,
    training_seed: int,
    expected_units: Sequence[str],
    protocols: Sequence[Protocol],
    held_out_context: str | None = None,
) -> pd.DataFrame:
    """Return one tidy score row per protocol and evaluation unit.

    Prediction artifacts are already in the shared evaluation expression view. Point-model
    profiles enter through their frozen control-residual population lift, whose centroid is
    exactly the original point prediction. Native population models use their retained cells.
    """
    labels = prediction.obs["perturbation"].astype(str)
    observed = sorted(labels.unique().tolist())
    expected = sorted(map(str, expected_units))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(f"prediction does not match frozen evaluation units: missing={missing}, extra={extra}")

    frames: list[pd.DataFrame] = []
    for protocol in protocols:
        result = spe.score(prepared, protocol, prediction, de_method="t-test")
        frame = result.per_perturbation[["perturbation", "score", "undefined_reason"]].copy()
        frame["undefined_reason"] = (
            frame["undefined_reason"].fillna("").map(lambda reason: REASON_MAP.get(str(reason), str(reason)))
        )
        if sorted(frame["perturbation"].astype(str).tolist()) != expected:
            raise ValueError(f"{protocol.name} did not return the frozen evaluation-unit cohort")
        frame = frame.rename(columns={"perturbation": "evaluation_unit"})
        frame.insert(0, "task", task)
        frame.insert(1, "fold", fold_label)
        frame.insert(2, "held_out_context", held_out_context)
        frame.insert(3, "model", model)
        frame.insert(4, "training_seed", training_seed)
        frame.insert(5, "protocol_id", protocol.name)
        frame["representation"] = protocol.representation
        frame["better"] = protocol.better
        frame["score_defined"] = np.isfinite(frame["score"].to_numpy(float))
        frames.append(frame)

    detail = pd.concat(frames, ignore_index=True)
    if detail.duplicated(["protocol_id", "evaluation_unit"]).any():
        raise ValueError("evaluation emitted duplicate protocol/evaluation-unit rows")
    return detail
