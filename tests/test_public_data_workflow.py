from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from models.baselines.context_baselines import predict_context_baselines
from models.data.prepare_context import prepare_context_view
from models.data.prepare_data import prepare_canonical_raw
from models.data.prepare_split import build_split_manifest, kfold_gene_splits
from workflows.data import MANIFEST_ROOT, load_manifest, source_artifacts
from workflows.evaluate import evaluate_bundle, matched_draw, rank_models, summarize_evaluations
from workflows.predictions import build_prediction_manifest, expected_test_conditions
from workflows.preparation import harmonize_scperturb, merge_disambiguated_gene_names
from workflows.preparation_utils import CELL_CYCLE_SHA256, assert_raw_counts, file_checksum
from workflows.semantic_h5ad import compare_semantic_h5ad, semantic_h5ad_sha256

ROOT = Path(__file__).resolve().parents[1]
DATASET_IDS = {
    "replogle-weissman-2022-k562-essential",
    "replogle-weissman-2022-k562-gwps",
    "replogle-weissman-2022-rpe1",
    "nadig-oconner-2024-hepg2",
    "nadig-oconner-2024-jurkat",
    "srivatsan-trapnell-2020-sciplex3",
}


def test_dataset_manifests_pin_sources_and_reference_outputs() -> None:
    assert {path.stem for path in MANIFEST_ROOT.glob("*.toml")} == DATASET_IDS
    for dataset_id in DATASET_IDS:
        manifest = load_manifest(dataset_id)
        assert manifest["schema_version"] == 1
        assert manifest["contract"]["x"] == "raw_counts"
        assert len(manifest["reference_output"]["sha256"]) == 64
        assert manifest["reference_output"]["n_obs"] > 0
        assert manifest["reference_output"]["n_vars"] > 0
        for artifact in source_artifacts(manifest):
            assert artifact["url"].startswith("https://")
            assert artifact["size_bytes"] > 0
            assert len(artifact["checksum"]) in {32, 64}


def test_cell_cycle_reference_is_pinned() -> None:
    path = ROOT / "data" / "references" / "regev_lab_cell_cycle_genes.csv"
    assert file_checksum(path) == CELL_CYCLE_SHA256


def test_context_transfer_cohorts_are_frozen_and_internally_consistent() -> None:
    genetic = json.loads(
        (ROOT / "configs" / "experiments" / "replogle-nadig-four-context-60-cell.json").read_text()
    )
    assert genetic["n_perturbations"] == len(genetic["perturbations"]) == 322
    assert genetic["perturbations"] == sorted(set(genetic["perturbations"]))
    assert genetic["n_test_populations"] == len(genetic["contexts"]) * genetic["n_perturbations"]
    assert len(genetic["folds"]) == len(genetic["contexts"]) == 4
    assert all(len(expected_test_conditions(genetic, fold["id"], "population")) == 322 for fold in genetic["folds"])

    chemical = json.loads(
        (ROOT / "configs" / "experiments" / "sciplex3-nine-drug-context-transfer.json").read_text()
    )
    assert chemical["n_drugs"] == len(chemical["drugs"]) == 9
    assert chemical["n_point_conditions"] == len(chemical["drugs"]) * len(chemical["doses"]) == 36
    assert chemical["n_population_conditions"] == 33
    assert len(chemical["population_exclusions"]) == 3
    assert len(chemical["folds"]) == len(chemical["contexts"]) == 3
    for fold in chemical["folds"]:
        assert len(expected_test_conditions(chemical, fold["id"], "point")) == 36
        assert len(expected_test_conditions(chemical, fold["id"], "population")) == 33


def test_harmonize_scperturb_preserves_raw_counts_and_standardizes_obs() -> None:
    obs = pd.DataFrame(
        {
            "batch": ["b1", "b1"],
            "gene": ["non-targeting", "G1"],
            "gene_id": ["", "ENSG1"],
            "guide_id": ["ctrl", "guide-1"],
        },
        index=["cell-1", "cell-2"],
    )
    source = AnnData(
        X=sp.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32)),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    result = harmonize_scperturb(source)
    assert_raw_counts(result.X)
    assert result.obs["perturbation"].astype(str).tolist() == ["non-targeting", "G1"]
    assert result.obs["is_control"].tolist() == [True, False]
    assert result.obs["guide_target"].astype(str).tolist() == ["non-targeting", "G1"]
    assert result.obs["n_guides"].tolist() == [1, 1]


def test_merge_disambiguated_gene_names_sums_duplicate_symbol_columns() -> None:
    source = AnnData(
        X=sp.csr_matrix(np.array([[1, 2, 4]], dtype=np.float32)),
        obs=pd.DataFrame(index=["cell"]),
        var=pd.DataFrame(index=["G1_ENSG1", "G1_ENSG2", "G2"]),
    )
    result = merge_disambiguated_gene_names(source)
    assert result.var_names.tolist() == ["G1", "G2"]
    assert result.X.toarray().tolist() == [[3.0, 4.0]]
    assert result.var["n_merged"].tolist() == [2, 1]


def test_all_toml_files_parse() -> None:
    for path in MANIFEST_ROOT.glob("*.toml"):
        assert tomllib.loads(path.read_text())["id"] == path.stem
    for path in (ROOT / "configs" / "models").glob("*.toml"):
        manifest = tomllib.loads(path.read_text())
        assert manifest["id"] == path.stem
        assert manifest["capability"] in {"point", "population"}


def test_canonical_raw_model_view_maps_controls_and_normalizes_without_mutating_source() -> None:
    source = AnnData(
        X=sp.csr_matrix(np.array([[1, 1], [2, 0], [0, 4]], dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "perturbation": ["non-targeting", "G1", "G2"],
                "is_control": [True, False, False],
            },
            index=["control", "g1", "g2"],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    result = prepare_canonical_raw(source, min_genes=1, min_cells=1, target_sum=10)

    assert source.obs["perturbation"].tolist() == ["non-targeting", "G1", "G2"]
    assert source.X.toarray().tolist() == [[1.0, 1.0], [2.0, 0.0], [0.0, 4.0]]
    assert result.obs["perturbation"].tolist() == ["control", "G1", "G2"]
    expected = np.log1p(np.array([[5.0, 5.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32))
    np.testing.assert_allclose(result.X.toarray(), expected, rtol=1e-6)


def test_unseen_perturbation_split_manifest_is_explicit_and_full_coverage(tmp_path: Path) -> None:
    genes = ["G1", "G2", "G3", "G4", "G5"]
    folds = kfold_gene_splits(genes, n_folds=5, seed=42)
    manifest = build_split_manifest(
        split_id="example-unseen-perturbation-5-fold",
        dataset_id="example",
        raw_path=tmp_path / "raw.h5ad",
        raw_sha256="a" * 64,
        n_obs=100,
        n_vars=20,
        genes=genes,
        folds=folds,
        seed=42,
    )

    assert manifest["n_perturbations"] == 5
    assert manifest["model_view"]["expression_space"] == "log1p_expression"
    assert sorted(gene for fold in manifest["folds"] for gene in fold["test"]) == genes


def test_semantic_h5ad_checksum_ignores_hdf5_compression(tmp_path: Path) -> None:
    adata = AnnData(
        X=sp.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32)),
        obs=pd.DataFrame({"perturbation": pd.Categorical(["control", "G1"])}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    plain = tmp_path / "plain.h5ad"
    compressed = tmp_path / "compressed.h5ad"
    adata.write_h5ad(plain)
    adata.write_h5ad(compressed, compression="gzip")

    assert plain.read_bytes() != compressed.read_bytes()
    assert semantic_h5ad_sha256(plain) == semantic_h5ad_sha256(compressed)
    compare_semantic_h5ad(plain, compressed)


def test_semantic_h5ad_mismatch_reports_annotation_columns(tmp_path: Path) -> None:
    left = AnnData(
        X=sp.csr_matrix(np.eye(2, dtype=np.float32)),
        obs=pd.DataFrame({"score": [1.0, 2.0], "label": ["a", "b"]}, index=["c1", "c2"]),
        var=pd.DataFrame({"selected": [True, False]}, index=["g1", "g2"]),
    )
    right = left.copy()
    right.obs["score"] = [1.0, 2.5]
    right.var["selected"] = right.var["selected"].astype(str)
    left_path = tmp_path / "left.h5ad"
    right_path = tmp_path / "right.h5ad"
    left.write_h5ad(left_path)
    right.write_h5ad(right_path)

    with pytest.raises(ValueError) as error:
        compare_semantic_h5ad(left_path, right_path)
    message = str(error.value)
    assert "H5AD semantic components differ: obs, var" in message
    assert "obs.score" in message
    assert "different_values=1" in message
    assert "var.selected" in message
    assert "dtype=bool/object" in message


def test_prediction_manifest_validates_fold_conditions_and_gene_set(tmp_path: Path) -> None:
    truth = AnnData(
        X=sp.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32)),
        obs=pd.DataFrame({"perturbation": ["control", "G1"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    prediction = AnnData(
        X=np.array([[0.5, 1.5]], dtype=np.float32),
        obs=pd.DataFrame({"perturbation": ["G1"]}, index=["prediction-G1"]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    truth_path = tmp_path / "truth.h5ad"
    prediction_path = tmp_path / "prediction.h5ad"
    dataset_manifest = tmp_path / "example.toml"
    split_manifest = tmp_path / "split.json"
    truth.write_h5ad(truth_path)
    prediction.write_h5ad(prediction_path)
    dataset_manifest.write_text('id = "example"\n')
    split_manifest.write_text(
        json.dumps({"schema_version": 1, "id": "split", "folds": [{"train": [], "val": [], "test": ["G1"]}]})
    )

    manifest = build_prediction_manifest(
        prediction_path=prediction_path,
        truth_path=truth_path,
        dataset_manifest_path=dataset_manifest,
        split_manifest_path=split_manifest,
        capability="point",
        model_id="model",
        implementation_revision="upstream-rev",
        adapter_revision="adapter-rev",
        fold=0,
        seed=42,
        environment="test",
        command="test command",
    )

    assert manifest["capability"] == "point"
    assert manifest["artifact"]["n_obs"] == 1
    assert len(manifest["artifact"]["sha256"]) == 64


def test_matched_population_draw_and_rank_aggregation_are_deterministic() -> None:
    labels = ["control"] * 4 + ["G1"] * 4 + ["G2"] * 4
    data = AnnData(
        X=np.arange(24, dtype=np.float32).reshape(12, 2),
        obs=pd.DataFrame({"perturbation": labels}, index=[f"cell-{index}" for index in range(12)]),
        var=pd.DataFrame(index=["a", "b"]),
    )
    first = matched_draw(data, ["G1", "G2"], 2, 42, retain_controls=True)
    second = matched_draw(data, ["G1", "G2"], 2, 42, retain_controls=True)
    assert first.obs_names.tolist() == second.obs_names.tolist()
    assert first.obs["perturbation"].value_counts().to_dict() == {"control": 4, "G1": 2, "G2": 2}

    detail = pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "dataset": ["d"] * 4,
            "split": ["s"] * 4,
            "capability": ["point"] * 4,
            "variant": ["v"] * 4,
            "factor": ["reference"] * 4,
            "level": ["reference"] * 4,
            "protocol": ["pearson_ctrl"] * 4,
            "better": ["higher"] * 4,
            "perturbation": ["G1", "G2", "G1", "G2"],
            "score": [0.9, 0.7, 0.8, 0.6],
            "effect_strength": [2.0, 1.0, 2.0, 1.0],
        }
    )
    summary = summarize_evaluations(detail)
    ranking = rank_models(summary)
    assert set(summary["aggregation"]) == {"mean", "median", "effect-weighted-mean"}
    assert (ranking.loc[ranking["model"] == "m1", "rank"] == 1).all()


def test_evaluation_grid_scores_a_fixed_point_bundle(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    labels = ["control"] * 12 + ["G1"] * 12 + ["G2"] * 12
    truth = AnnData(
        X=rng.normal(size=(36, 8)).astype(np.float32),
        obs=pd.DataFrame({"perturbation": labels}, index=[f"cell-{index}" for index in range(36)]),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(8)]),
    )
    prediction = AnnData(
        X=np.vstack(
            [
                truth.X[truth.obs["perturbation"] == "G1"].mean(axis=0),
                truth.X[truth.obs["perturbation"] == "G2"].mean(axis=0),
            ]
        ),
        obs=pd.DataFrame({"perturbation": ["G1", "G2"]}, index=["prediction-G1", "prediction-G2"]),
        var=truth.var.copy(),
    )
    truth_path = tmp_path / "truth.h5ad"
    prediction_path = tmp_path / "prediction.h5ad"
    split_path = tmp_path / "split.json"
    grid_path = tmp_path / "grid.json"
    truth.write_h5ad(truth_path)
    prediction.write_h5ad(prediction_path)
    split_path.write_text(
        json.dumps(
            {
                "id": "split",
                "dataset": "example",
                "folds": [{"train": [], "val": [], "test": ["G1", "G2"]}],
            }
        )
    )
    grid_path.write_text(
        json.dumps(
            {
                "minimum_observed_cells": 12,
                "population": {"draw_size": 60, "draw_seeds": [42]},
                "variants": [
                    {
                        "id": "reference",
                        "factor": "reference",
                        "level": "control-centered-pearson-all-genes",
                        "protocol": "pearson_ctrl",
                        "de_method": "t-test",
                        "better": "higher",
                        "capabilities": ["point", "population"],
                    }
                ],
            }
        )
    )

    detail = evaluate_bundle(
        truth_path=truth_path,
        prediction_path=prediction_path,
        split_manifest_path=split_path,
        grid_path=grid_path,
        model_id="example-model",
        capability="point",
        fold=0,
    )

    assert detail["perturbation"].tolist() == ["G1", "G2"]
    assert detail["score"].notna().all()
    assert detail["model"].unique().tolist() == ["example-model"]


def test_genetic_context_view_freezes_model_and_evaluation_condition_ids(tmp_path: Path) -> None:
    inputs = {}
    for context in ["context-a", "context-b"]:
        labels = ["non-targeting"] * 2 + ["G1"] * 2 + ["G2"] * 2
        canonical = AnnData(
            X=sp.csr_matrix(np.ones((6, 201), dtype=np.float32)),
            obs=pd.DataFrame(
                {
                    "perturbation": labels,
                    "is_control": [True, True, False, False, False, False],
                    "batch": ["b1"] * 6,
                },
                index=[f"{context}-cell-{index}" for index in range(6)],
            ),
            var=pd.DataFrame(index=[f"gene-{index}" for index in range(201)]),
        )
        path = tmp_path / f"{context}.h5ad"
        canonical.write_h5ad(path)
        inputs[context] = path
    config = {
        "id": "context-test",
        "contexts": ["context-a", "context-b"],
        "perturbations": ["G1", "G2"],
    }

    result = prepare_context_view(
        config,
        inputs,
        minimum_cells=2,
        max_cells=2,
        max_controls=2,
        seed=42,
    )

    assert result.shape == (12, 201)
    assert set(result.obs["model_perturbation"].astype(str)) == {"control", "G1", "G2"}
    assert set(result.obs.loc[~result.obs["is_control"], "evaluation_condition"].astype(str)) == {
        "context-a::G1",
        "context-a::G2",
        "context-b::G1",
        "context-b::G2",
    }

    split = {
        **config,
        "folds": [
            {
                "id": "holdout-context-b",
                "test_context": "context-b",
                "train_contexts": ["context-a"],
            }
        ],
    }
    predictions = predict_context_baselines(result, split, 0)
    assert set(predictions) == {"no-change", "fold-mean"}
    assert predictions["no-change"].obs["perturbation"].tolist() == ["context-b::G1", "context-b::G2"]
    assert predictions["fold-mean"].shape == (2, 201)
