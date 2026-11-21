from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scripts.lift_production_predictions import _truth_identity
from workflows.population_lift import lift_prediction_population


@pytest.mark.parametrize("manifest_key", ["artifact", "output"])
def test_truth_identity_accepts_both_prepared_data_manifest_schemas(
    tmp_path, manifest_key: str
) -> None:
    truth = tmp_path / "raw.h5ad"
    truth.write_bytes(b"truth")
    checksum = "a" * 64
    truth.with_suffix(".manifest.json").write_text(
        f'{{"{manifest_key}": {{"sha256": "{checksum}"}}}}\n'
    )

    assert _truth_identity(truth) == f"sha256:{checksum}"


def test_truth_identity_falls_back_to_file_stat_without_manifest_checksum(tmp_path) -> None:
    truth = tmp_path / "raw.h5ad"
    truth.write_bytes(b"truth")
    truth.with_suffix(".manifest.json").write_text('{"schema_version": 1}\n')

    status = truth.stat()
    assert _truth_identity(truth) == f"stat-v1:{status.st_size}:{status.st_mtime_ns}"


def test_point_predictions_lift_to_deterministic_centroid_preserving_populations() -> None:
    truth = AnnData(
        X=np.asarray(
            [
                [1.0, 2.0],
                [2.0, 1.0],
                [1.5, 1.5],
                [3.0, 2.0],
                [3.5, 2.5],
                [2.5, 3.0],
                [2.0, 2.5],
                [2.5, 2.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {
                "perturbation": ["control"] * 3 + ["G1"] * 3 + ["G2"] * 2,
                "batch": ["b1", "b1", "b2", "b1", "b1", "b2", "b1", "b2"],
            },
            index=[f"truth-{index}" for index in range(8)],
        ),
        var=pd.DataFrame(index=["a", "b"]),
    )
    point = AnnData(
        X=np.asarray([[4.0, 3.0], [-0.5, 3.5]], dtype=np.float32),
        obs=pd.DataFrame(
            {"perturbation": ["G1", "G2"]},
            index=["prediction-G1", "prediction-G2"],
        ),
        var=truth.var.copy(),
    )
    split = {
        "id": "test-split",
        "dataset": "test",
        "prediction_population_policy": {"mode": "match-observed"},
        "folds": [{"test": ["G1", "G2"]}],
    }

    first = lift_prediction_population(truth, point, split, 0, seed=42)
    second = lift_prediction_population(truth, point, split, 0, seed=42)

    np.testing.assert_allclose(first.X, second.X)
    assert first.obs["source_control"].tolist() == second.obs["source_control"].tolist()
    assert first.obs["perturbation"].value_counts().to_dict() == {"G1": 3, "G2": 2}
    assert first.obs.loc[first.obs["perturbation"] == "G1", "batch"].value_counts().to_dict() == {
        "b1": 2,
        "b2": 1,
    }
    np.testing.assert_allclose(
        np.asarray(first[first.obs["perturbation"] == "G1"].X).mean(axis=0),
        point.X[0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(first[first.obs["perturbation"] == "G2"].X).mean(axis=0),
        np.maximum(point.X[1], 0),
        rtol=1e-6,
    )
    assert np.asarray(first.X).min() >= 0
