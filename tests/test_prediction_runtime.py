from __future__ import annotations

import anndata as ad
import numpy as np
from scperteval.protocols import metrics
from scperteval.types import Protocol

from analyses.ranking_sensitivity.prediction_runtime import (
    evaluate_prediction,
    prepare_prediction_fold,
)


def test_evaluate_prediction_scores_concrete_protocols_with_tidy_identity_columns():
    rng = np.random.default_rng(4)
    labels = ["control"] * 20
    truth_parts = [rng.poisson(1, size=(20, 6)).astype(np.float32)]
    prediction_parts = []
    prediction_labels = []
    for index, perturbation in enumerate(["pertA", "pertB"]):
        values = rng.poisson(1, size=(12, 6)).astype(np.float32)
        values[:, index] += 4
        truth_parts.append(values)
        labels.extend([perturbation] * len(values))
        prediction_parts.append(values.copy())
        prediction_labels.extend([perturbation] * len(values))
    truth = ad.AnnData(np.vstack(truth_parts))
    truth.var_names = [f"g{i}" for i in range(truth.n_vars)]
    truth.obs["perturbation"] = labels
    prediction = ad.AnnData(np.vstack(prediction_parts))
    prediction.var_names = truth.var_names.copy()
    prediction.obs["perturbation"] = prediction_labels
    protocols = [
        Protocol("atlas-mse", metrics.mse, representation="centroid", better="lower", perfect=0.0),
        Protocol("atlas-pearson", metrics.pearson, representation="centroid"),
    ]
    prepared = prepare_prediction_fold(
        truth,
        task="synthetic",
        fold_label="fold-0",
        protocols=protocols,
        minimum_cells=10,
        workers=1,
    )
    result = evaluate_prediction(
        prepared,
        prediction,
        task="synthetic",
        fold_label="fold-0",
        model="model-a",
        training_seed=42,
        expected_units=["pertA", "pertB"],
        protocols=protocols,
    )
    assert len(result) == 4
    assert result["score_defined"].all()
    assert set(result["protocol_id"]) == {"atlas-mse", "atlas-pearson"}
    assert set(result["evaluation_unit"]) == {"pertA", "pertB"}
    assert set(result["model"]) == {"model-a"}
    assert set(result["better"]) == {"lower", "higher"}


def test_evaluate_prediction_rejects_wrong_cohort():
    truth = ad.AnnData(np.ones((30, 2), dtype=np.float32))
    truth.var_names = ["g0", "g1"]
    truth.obs["perturbation"] = ["control"] * 10 + ["pertA"] * 10 + ["pertB"] * 10
    prediction = ad.AnnData(np.ones((10, 2), dtype=np.float32))
    prediction.var_names = truth.var_names.copy()
    prediction.obs["perturbation"] = ["pertA"] * 10
    protocol = Protocol("atlas-mse", metrics.mse, representation="centroid", better="lower", perfect=0.0)
    prepared = prepare_prediction_fold(
        truth,
        task="synthetic",
        fold_label="fold-0",
        protocols=[protocol],
        minimum_cells=5,
        workers=1,
    )
    try:
        evaluate_prediction(
            prepared,
            prediction,
            task="synthetic",
            fold_label="fold-0",
            model="model-a",
            training_seed=42,
            expected_units=["pertA", "pertB"],
            protocols=[protocol],
        )
    except ValueError as error:
        assert "missing=['pertB']" in str(error)
    else:
        raise AssertionError("expected a frozen-cohort mismatch")
