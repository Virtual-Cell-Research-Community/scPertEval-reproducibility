from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from models.baselines.context_baselines import predict_context_baselines
from models.baselines.perturbation_baselines import (
    predict_perturbation_baseline,
    validate_fold_conversion,
)
from models.biolord.context_config import (
    assign_biolord_splits,
    relabel_biolord_predictions,
    select_heldout_controls,
)
from models.cpa.context_config import assign_cpa_splits, build_counterfactual_queries
from models.data.prepare_context import _input_provenance, _verify_inputs_unchanged, prepare_context_view
from models.data.prepare_data import (
    canonical_annotations,
    finalize_model_annotations,
    prepare_canonical_raw,
    prepare_frozen_experiment_raw,
    smoke_subsample,
    validate_frozen_experiment_metadata,
)
from models.data.prepare_split import build_split_manifest, kfold_gene_splits, to_tabicl_custom_split
from models.scdisinfact.context_config import (
    build_prediction_adata,
    select_scdisinfact_training_data,
)
from models.scdisinfact.context_config import select_heldout_controls as select_scdisinfact_controls
from models.scdisinfact.joint_patch import choose_present_reference, coerce_reference_id
from models.state.context_config import (
    build_context_toml,
    model_conditions,
    observed_population_sizes,
    relabel_context_predictions,
    sample_heldout_controls,
)
from models.tabicl.split_compat import initialize_with_nonempty_validation
from models.trvae.context_config import build_prediction_adata as build_trvae_prediction_adata
from models.trvae.context_config import densify_expression as densify_trvae_expression
from models.trvae.context_config import select_heldout_controls as select_trvae_controls
from models.trvae.context_config import select_trvae_training_data
from models.trvae.scarchon_patch import eligible_mmd_groups
from workflows import data as data_workflow
from workflows.data import MANIFEST_ROOT, load_manifest, source_artifacts
from workflows.evaluate import (
    apply_prediction_evaluation_transform,
    compare_to_reference,
    evaluate_bundle,
    matched_draw,
    paired_perturbation_bootstrap,
    rank_models,
    summarize_evaluations,
    validate_matched_population_sizes,
)
from workflows.predictions import build_prediction_manifest, expected_test_conditions, validate_prediction_h5ad
from workflows.preparation import harmonize_scperturb, merge_disambiguated_gene_names
from workflows.preparation_utils import CELL_CYCLE_SHA256, assert_raw_counts, file_checksum
from workflows.run_contract import (
    begin_run,
    complete_run,
    completed_run_is_valid,
    resume_incomplete_run,
    run_directory,
)
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
        (ROOT / "configs" / "experiments" / "replogle-nadig-four-context-50-cell.json").read_text()
    )
    assert genetic["n_perturbations"] == len(genetic["perturbations"]) == 480
    assert genetic["perturbations"] == sorted(set(genetic["perturbations"]))
    assert genetic["n_test_populations"] == len(genetic["contexts"]) * genetic["n_perturbations"]
    assert len(genetic["folds"]) == len(genetic["contexts"]) == 4
    assert genetic["minimum_treated_cells_per_context"] == 50
    assert genetic["training_seeds"] == [42, 43, 44]
    assert genetic["prediction_population_policy"] == {
        "mode": "match-observed",
        "control_sampling": "with-replacement-if-needed",
    }
    assert len(genetic["validation_conditions"]) == 48
    assert set(genetic["validation_conditions"]) <= set(genetic["perturbations"])
    assert all(len(expected_test_conditions(genetic, fold["id"], "population")) == 480 for fold in genetic["folds"])

    chemical = json.loads(
        (ROOT / "configs" / "experiments" / "sciplex3-nine-drug-context-transfer.json").read_text()
    )
    assert chemical["n_drugs"] == len(chemical["drugs"]) == 9
    assert chemical["n_point_conditions"] == len(chemical["drugs"]) * len(chemical["doses"]) == 36
    assert chemical["n_population_conditions"] == 33
    assert chemical["population_minimum_cells_per_context"] == 50
    assert chemical["training_seeds"] == [42, 43, 44]
    assert chemical["prediction_population_policy"] == {
        "mode": "match-observed",
        "control_sampling": "with-replacement-if-needed",
    }
    assert len(chemical["population_exclusions"]) == 3
    assert len(chemical["folds"]) == len(chemical["contexts"]) == 3
    assert len(chemical["validation_conditions"]) == 4
    assert set(chemical["validation_conditions"]) <= set(model_conditions(chemical, capability="point"))
    for fold in chemical["folds"]:
        assert len(expected_test_conditions(chemical, fold["id"], "point")) == 36
        assert len(expected_test_conditions(chemical, fold["id"], "population")) == 33


def test_gwps_primary_experiment_is_frozen_and_reconstructable() -> None:
    experiment = json.loads(
        (
            ROOT
            / "configs"
            / "experiments"
            / "replogle-weissman-2022-k562-gwps-perturbation-generalization.json"
        ).read_text()
    )
    assert experiment["role"] == "primary"
    assert experiment["training_seeds"] == [42, 43, 44]
    assert experiment["gene_panel"] == {"strategy": "all-measured-genes", "n_genes": 8248}
    assert experiment["model_view_preparation"] == {
        "cell_selection": "all-cells-for-frozen-targets-plus-all-controls",
        "reapply_generic_qc_filters": False,
        "miller_mean_cell_downsampling": False,
        "gene_selection": "all-measured-genes",
        "normalization": "normalize-total-10000-log1p",
        "raw_counts_remain_immutable": True,
    }
    assert experiment["n_perturbations"] == len(experiment["perturbations"]) == 6935
    assert experiment["perturbations"] == sorted(set(experiment["perturbations"]))
    assert experiment["eligibility"]["n_targets_shared"] == 6935
    assert experiment["development_pool_policy"]["test_targets_never_available_to_models"]
    assert experiment["development_pool_policy"][
        "fixed_budget_and_training_free_methods_may_use_train_plus_validation"
    ]
    assert experiment["model_development_usage"]["tabicl"] == {
        "in_context_examples": "train-plus-validation",
        "selection": "none-equal-modality-fusion",
        "early_stopping": False,
        "gradient_training": False,
    }
    assert experiment["model_development_usage"]["gears"]["selection"] == (
        "best-validation-checkpoint"
    )

    folds = kfold_gene_splits(
        experiment["perturbations"],
        n_folds=experiment["split"]["n_folds"],
        seed=experiment["split"]["seed"],
    )
    payload = json.dumps(folds, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == experiment["split"]["folds_sha256"]
    assert [len(fold["train"]) for fold in folds] == experiment["split"][
        "training_targets_per_fold"
    ]
    assert [len(fold["val"]) for fold in folds] == experiment["split"][
        "validation_targets_per_fold"
    ]
    assert [len(fold["test"]) for fold in folds] == experiment["split"][
        "test_targets_per_fold"
    ]

    tabicl = to_tabicl_custom_split(folds[0])
    assert tabicl["val"] == []
    assert set(tabicl["train"]) == {*folds[0]["train"], *folds[0]["val"], "control"}
    assert tabicl["test"] == folds[0]["test"]
    assert set(tabicl["train"]).isdisjoint(tabicl["test"])

    validation = json.loads(
        (
            ROOT
            / "configs"
            / "experiments"
            / "replogle-weissman-2022-k562-gwps-metadata-validation.json"
        ).read_text()
    )
    assert validation["input_sha256"] == experiment["canonical_input"]["sha256"]
    assert validation["shape"] == [
        experiment["canonical_input"]["n_obs"],
        experiment["canonical_input"]["n_vars"],
    ]
    assert validation["n_frozen_targets"] == experiment["n_perturbations"]
    assert validation["n_targets_at_minimum"] == experiment["eligibility"][
        "n_targets_at_least_50_cells"
    ]
    assert validation["n_measured_targets_at_minimum"] == experiment["eligibility"][
        "n_targets_measured_at_least_50_cells"
    ]
    assert validation["status"] == "valid"


def test_perturbation_baselines_validate_native_fold_and_predict_test_conditions() -> None:
    fold = {
        "train": ["ctrl", "G1+ctrl", "G2+ctrl"],
        "val": ["G3+ctrl"],
        "test": ["G4+ctrl"],
    }
    validate_fold_conversion(
        fold,
        {"train": ["G1", "G2"], "val": ["G3"], "test": ["G4"]},
    )
    with pytest.raises(ValueError, match="test split differs"):
        validate_fold_conversion(
            fold,
            {"train": ["G1", "G2"], "val": ["G3"], "test": ["G5"]},
        )

    perturbations = [
        "control",
        "control",
        "G1",
        "G1",
        "G2",
        "G2",
        "G3",
        "G3",
        "G4",
        "G4",
    ]
    data = AnnData(
        X=np.arange(1, 41, dtype=np.float32).reshape(10, 4),
        obs=pd.DataFrame(
            {"perturbation": perturbations},
            index=[f"cell-{index}" for index in range(10)],
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    for model in ["no-change", "fold-mean", "linear"]:
        prediction = predict_perturbation_baseline(model, data, fold, pca_dim=2)
        assert prediction.shape == (1, 4)
        assert prediction.obs["perturbation"].tolist() == ["G4"]
        assert prediction.var_names.tolist() == data.var_names.tolist()
        assert np.isfinite(prediction.X).all()

    with pytest.raises(ValueError, match="halves are required"):
        predict_perturbation_baseline("interpolated-technical-duplicate", data, fold)


def test_tabicl_initializes_with_frozen_validation_before_using_merged_pool(
    tmp_path: Path,
) -> None:
    validation_split = tmp_path / "fold_presage.json"
    model_split = tmp_path / "fold_tabicl.json"
    validation_split.write_text(
        json.dumps({"train": ["G1", "control"], "val": ["G2"], "test": ["G3"]})
    )
    model_split.write_text(
        json.dumps({"train": ["G1", "G2", "control"], "val": [], "test": ["G3"]})
    )

    class FakeDataModule:
        split_json_path = ""
        initialized_from: Path | None = None

        def setup(self) -> None:
            path = Path(self.split_json_path)
            assert json.loads(path.read_text())["val"]
            self.initialized_from = path

    datamodule = FakeDataModule()
    initialize_with_nonempty_validation(
        datamodule,
        validation_split_path=validation_split,
        model_split_path=model_split,
    )

    assert datamodule.initialized_from == validation_split
    assert Path(datamodule.split_json_path) == model_split


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


def test_prepare_canonical_raw_accepts_explicit_source_annotation_fields() -> None:
    source = AnnData(
        X=sp.csr_matrix(
            np.array(
                [
                    [1, 1],
                    [2, 1],
                    [1, 3],
                    [4, 1],
                ],
                dtype=np.int32,
            )
        ),
        obs=pd.DataFrame(
            {
                "target_gene": ["non-targeting", "non-targeting", "G1", "G2"],
                "control_class": ["non-targeting", "non-targeting", "targeted", "targeted"],
            },
            index=["control-1", "control-2", "g1", "g2"],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    labels, controls = canonical_annotations(
        source.obs,
        perturbation_key="target_gene",
        control_key="control_class",
        control_value="non-targeting",
    )
    assert labels.tolist() == ["control", "control", "G1", "G2"]
    assert controls.tolist() == [True, True, False, False]

    result = prepare_canonical_raw(
        source,
        min_genes=1,
        min_cells=1,
        perturbation_key="target_gene",
        control_key="control_class",
        control_value="non-targeting",
    )
    assert result.obs["perturbation"].tolist() == ["control", "control", "G1", "G2"]
    assert result.obs["is_control"].tolist() == [True, True, False, False]
    assert result.uns["log1p"] == {}
    np.testing.assert_allclose(np.expm1(result.X.toarray()).sum(axis=1), 10_000, rtol=1e-5)


def test_metadata_only_validation_checks_frozen_cohort_without_reading_expression() -> None:
    source = AnnData(
        X=sp.csr_matrix(np.ones((6, 2), dtype=np.int32)),
        obs=pd.DataFrame(
            {
                "target_gene": ["non-targeting", "non-targeting", "G1", "G1", "G2", "G2"],
                "control_class": [
                    "non-targeting",
                    "non-targeting",
                    "targeted",
                    "targeted",
                    "targeted",
                    "targeted",
                ],
            },
            index=[f"cell-{index}" for index in range(6)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    experiment = {
        "dataset": "example",
        "canonical_input": {"n_obs": 6, "n_vars": 2},
        "minimum_treated_cells_per_perturbation": 2,
        "perturbations": ["G1", "G2"],
        "eligibility": {"observed_perturbations": 4},
    }
    report = validate_frozen_experiment_metadata(
        source,
        experiment,
        perturbation_key="target_gene",
        control_key="control_class",
        control_value="non-targeting",
    )
    assert report == {
        "dataset": "example",
        "shape": [6, 2],
        "source_fields": {
            "perturbation_key": "target_gene",
            "control_key": "control_class",
            "control_value": "non-targeting",
        },
        "n_control_cells": 2,
        "n_targeted_cells": 4,
        "n_source_targets": 2,
        "n_targets_at_minimum": 2,
        "n_measured_targets_at_minimum": 2,
        "n_frozen_targets": 2,
        "n_frozen_targeted_cells": 4,
        "frozen_cells_minimum": 2,
        "frozen_cells_median": 2.0,
        "frozen_cells_maximum": 2,
        "all_frozen_targets_measured": True,
        "minimum_cells_requirement": 2,
        "status": "valid",
    }

    materialized = prepare_frozen_experiment_raw(
        source,
        {
            **experiment,
            "gene_panel": {"strategy": "all-measured-genes", "n_genes": 2},
        },
        perturbation_key="target_gene",
        control_key="control_class",
        control_value="non-targeting",
    )
    assert materialized.shape == (6, 2)
    assert materialized.obs["perturbation"].tolist() == [
        "control",
        "control",
        "G1",
        "G1",
        "G2",
        "G2",
    ]
    assert materialized.obs["is_control"].tolist() == [True, True, False, False, False, False]
    assert materialized.uns["log1p"] == {}
    np.testing.assert_allclose(
        np.expm1(materialized.X.toarray()).sum(axis=1),
        10_000,
        rtol=1e-5,
    )


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


def test_smoke_subsample_can_retain_every_perturbation_with_bounded_cells() -> None:
    perturbations = ["control"] * 4 + ["G1"] * 3 + ["G2"] * 3
    source = AnnData(
        X=sp.csr_matrix(np.ones((len(perturbations), 2), dtype=np.float32)),
        obs=pd.DataFrame(
            {"perturbation": perturbations},
            index=[f"cell-{index}" for index in range(len(perturbations))],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    result = smoke_subsample(
        source,
        n_perts=2,
        cells_per_pert=2,
        control_cells=3,
        seed=42,
    )

    assert result.obs["perturbation"].value_counts().to_dict() == {
        "control": 3,
        "G1": 2,
        "G2": 2,
    }


def test_frozen_smoke_selects_backed_rows_before_materialization(tmp_path: Path) -> None:
    perturbations = ["non-targeting"] * 4 + ["G1"] * 3 + ["G2"] * 3
    source = AnnData(
        X=sp.csr_matrix(np.arange(1, 21, dtype=np.int32).reshape(10, 2)),
        obs=pd.DataFrame(
            {
                "target_gene": perturbations,
                "control_class": ["non-targeting"] * 4 + ["targeted"] * 6,
            },
            index=[f"cell-{index}" for index in range(10)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    experiment = {
        "perturbations": ["G1", "G2"],
        "gene_panel": {"n_genes": 2},
    }
    full = prepare_frozen_experiment_raw(
        source,
        experiment,
        perturbation_key="target_gene",
        control_key="control_class",
        control_value="non-targeting",
    )
    expected = smoke_subsample(full, n_perts=2, cells_per_pert=2, control_cells=3, seed=42)

    source_path = tmp_path / "source.h5ad"
    source.write_h5ad(source_path)
    backed = ad.read_h5ad(source_path, backed="r")
    try:
        actual = prepare_frozen_experiment_raw(
            backed,
            experiment,
            perturbation_key="target_gene",
            control_key="control_class",
            control_value="non-targeting",
            smoke=(2, 2, 3, 42),
        )
    finally:
        backed.file.close()

    assert actual.obs_names.tolist() == expected.obs_names.tolist()
    assert actual.obs["perturbation"].value_counts().to_dict() == {
        "control": 3,
        "G1": 2,
        "G2": 2,
    }
    np.testing.assert_allclose(actual.X.toarray(), expected.X.toarray())


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


def test_model_annotations_serialize_mixed_object_columns(tmp_path: Path) -> None:
    model_view = AnnData(
        X=sp.csr_matrix(np.eye(3, dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "perturbation": ["control", "G1", "G2"],
                "batch": ["batch-a", 2, None],
                "is_control": [True, False, False],
                "numeric_qc": [1.0, 2.0, 3.0],
            },
            index=["cell-1", "cell-2", "cell-3"],
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )
    finalized = finalize_model_annotations(model_view, "K562")
    output = tmp_path / "model-view.h5ad"
    finalized.write_h5ad(output)
    restored = ad.read_h5ad(output)

    assert restored.obs["batch"].tolist() == ["batch-a", "2", ""]
    assert restored.obs["condition"].tolist() == ["ctrl", "G1+ctrl", "G2+ctrl"]
    assert restored.obs["is_control"].tolist() == [True, False, False]
    assert restored.obs["numeric_qc"].tolist() == [1.0, 2.0, 3.0]


def test_stage_dataset_sources_verifies_and_symlinks_without_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_directory = tmp_path / "upstream"
    source_directory.mkdir()
    source = source_directory / "source.txt"
    source.write_bytes(b"verified source")
    manifest = {
        "source": {
            "filename": source.name,
            "url": "https://example.invalid/source.txt",
            "size_bytes": source.stat().st_size,
            "checksum_type": "md5",
            "checksum": data_workflow.file_checksum(source, "md5"),
        }
    }
    monkeypatch.setattr(data_workflow, "load_manifest", lambda _dataset_id: manifest)

    staged = data_workflow.stage_dataset_sources("example", source_directory, tmp_path / "data")

    assert len(staged) == 1
    assert staged[0].is_symlink()
    assert staged[0].resolve() == source.resolve()


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
    assert manifest["prediction_id"].endswith("-seed-42")
    assert manifest["expression"]["space"] == "log1p_expression"
    assert manifest["expression"]["evaluation_space"] == "log1p_expression"
    assert manifest["expression"]["evaluation_transform"] == "identity"
    assert manifest["provenance"]["training_seed"] == 42
    assert manifest["artifact"]["n_obs"] == 1
    assert len(manifest["artifact"]["sha256"]) == 64


def test_run_contract_skips_only_checksum_valid_completed_runs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text('{"frozen": true}\n')
    root = tmp_path / "runs"
    kwargs = {
        "experiment": "experiment",
        "model": "model",
        "fold": "holdout-a",
        "seed": 42,
        "configuration": {"epochs": 1},
        "inputs": {"input": input_path},
        "command": ["python", "runner.py"],
        "repo_root": ROOT,
    }

    first = begin_run(root, **kwargs)
    assert not first.skipped
    assert first.directory == run_directory(
        root,
        experiment="experiment",
        model="model",
        fold="holdout-a",
        seed=42,
    )
    prediction = first.directory / "predictions" / "predictions.h5ad"
    prediction.write_bytes(b"prediction")
    complete_run(first, {"prediction": prediction})
    assert completed_run_is_valid(first.directory, resolved_sha256=first.resolved_sha256)

    second = begin_run(root, **kwargs)
    assert second.skipped
    prediction.write_bytes(b"tampered")
    third = begin_run(root, **kwargs)
    assert not third.skipped


def test_run_contract_accepts_a_valid_precomputed_artifact_checksum(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text('{"frozen": true}\n')
    handle = begin_run(
        tmp_path / "runs",
        experiment="experiment",
        model="model",
        fold="holdout-a",
        seed=42,
        configuration={"epochs": 1},
        inputs={"input": input_path},
        command=["python", "runner.py"],
        repo_root=ROOT,
    )
    prediction = handle.directory / "predictions" / "predictions.h5ad"
    prediction.write_bytes(b"prediction")
    checksum = hashlib.sha256(b"prediction").hexdigest()

    completion = complete_run(
        handle,
        {"prediction": prediction},
        precomputed_checksums={"prediction": checksum},
    )

    record = json.loads(completion.read_text())
    assert record["artifacts"][0]["sha256"] == checksum
    assert completed_run_is_valid(handle.directory, resolved_sha256=handle.resolved_sha256)


def test_run_contract_resumes_incomplete_run_without_rewriting_provenance(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text('{"frozen": true}\n')
    root = tmp_path / "runs"
    first = begin_run(
        root,
        experiment="experiment",
        model="model",
        fold="holdout-a",
        seed=42,
        configuration={"epochs": 1},
        inputs={"input": input_path},
        command=["python", "runner.py"],
        repo_root=ROOT,
    )
    provenance = (first.directory / "resolved-run.json").read_bytes()

    resumed = resume_incomplete_run(
        root,
        experiment="experiment",
        model="model",
        fold="holdout-a",
        seed=42,
    )

    assert resumed.directory == first.directory
    assert resumed.resolved_sha256 == first.resolved_sha256
    assert (first.directory / "resolved-run.json").read_bytes() == provenance


def test_population_prediction_archive_rejects_more_than_100_cells(tmp_path: Path) -> None:
    truth = AnnData(
        X=sp.csr_matrix(np.ones((2, 2), dtype=np.float32)),
        obs=pd.DataFrame({"perturbation": ["control", "G1"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    prediction = AnnData(
        X=np.ones((101, 2), dtype=np.float32),
        obs=pd.DataFrame({"perturbation": ["G1"] * 101}, index=[f"prediction-{i}" for i in range(101)]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    truth_path = tmp_path / "truth.h5ad"
    prediction_path = tmp_path / "prediction.h5ad"
    truth.write_h5ad(truth_path)
    prediction.write_h5ad(prediction_path)
    split = {"folds": [{"test": ["G1"]}]}

    with pytest.raises(ValueError, match="exceeds archival cap"):
        validate_prediction_h5ad(
            prediction_path,
            truth_path,
            split,
            0,
            "population",
            minimum_cells_per_condition=50,
        )


def test_population_prediction_archive_matches_observed_counts(tmp_path: Path) -> None:
    truth = AnnData(
        X=np.ones((6, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"perturbation": ["control", "G1", "G1", "G2", "G2", "G2"]},
            index=[f"truth-{index}" for index in range(6)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    prediction = AnnData(
        X=np.ones((5, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"perturbation": ["G1", "G1", "G2", "G2", "G2"]},
            index=[f"prediction-{index}" for index in range(5)],
        ),
        var=truth.var.copy(),
    )
    truth_path = tmp_path / "truth.h5ad"
    prediction_path = tmp_path / "prediction.h5ad"
    truth.write_h5ad(truth_path)
    prediction.write_h5ad(prediction_path)
    split = {
        "prediction_population_policy": {"mode": "match-observed"},
        "folds": [{"test": ["G1", "G2"]}],
    }
    assert validate_prediction_h5ad(
        prediction_path,
        truth_path,
        split,
        0,
        "population",
    )[:2] == (5, 2)

    prediction[:-1].copy().write_h5ad(prediction_path)
    with pytest.raises(ValueError, match="do not match observed treated-cell counts"):
        validate_prediction_h5ad(
            prediction_path,
            truth_path,
            split,
            0,
            "population",
        )


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


def test_population_sizes_must_match_observed_treated_counts() -> None:
    truth = AnnData(
        X=np.ones((8, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"perturbation": ["control"] * 2 + ["G1"] * 2 + ["G2"] * 4},
            index=[f"truth-{index}" for index in range(8)],
        ),
        var=pd.DataFrame(index=["a", "b"]),
    )
    matching = AnnData(
        X=np.ones((6, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"perturbation": ["G1"] * 2 + ["G2"] * 4},
            index=[f"prediction-{index}" for index in range(6)],
        ),
        var=truth.var.copy(),
    )
    assert validate_matched_population_sizes(truth, matching, ["G1", "G2"]) == {
        "G1": 2,
        "G2": 4,
    }

    mismatched = matching[:-1].copy()
    with pytest.raises(ValueError, match="do not match observed treated-cell counts"):
        validate_matched_population_sizes(truth, mismatched, ["G1", "G2"])


def test_one_factor_rank_comparisons_and_paired_bootstrap_report_reversals() -> None:
    rows = []
    reference_scores = {
        "m1": [10.0, 0.0, 0.0, 0.0],
        "m2": [2.0, 2.0, 2.0, 2.0],
        "m3": [1.0, 1.0, 1.0, 1.0],
    }
    alternative_scores = {
        "m1": [0.0, 0.0, 0.0, 0.0],
        "m2": [1.0, 1.0, 1.0, 1.0],
        "m3": [2.0, 2.0, 2.0, 2.0],
    }
    for model, scores in reference_scores.items():
        for index, score in enumerate(scores):
            rows.append(
                {
                    "model": model,
                    "dataset": "d",
                    "split": "s",
                    "capability": "point",
                    "variant": "reference",
                    "factor": "reference",
                    "level": "reference",
                    "protocol": "pearson_ctrl",
                    "better": "higher",
                    "perturbation": f"G{index}",
                    "score": score,
                    "effect_strength": float(index + 1),
                }
            )
    for model, scores in alternative_scores.items():
        for index, score in enumerate(scores):
            rows.append(
                {
                    "model": model,
                    "dataset": "d",
                    "split": "s",
                    "capability": "point",
                    "variant": "metric-alternative",
                    "factor": "metric",
                    "level": "alternative",
                    "protocol": "r2_ctrl",
                    "better": "higher",
                    "perturbation": f"G{index}",
                    "score": score,
                    "effect_strength": float(index + 1),
                }
            )

    detail = pd.DataFrame(rows)
    summary = summarize_evaluations(detail)
    rank_changes, comparisons = compare_to_reference(summary)

    metric_comparison = comparisons.set_index("scenario").loc["metric-alternative"]
    assert bool(metric_comparison["winner_changed"])
    assert metric_comparison["alternative_winners"] == "m3"
    assert metric_comparison["pairwise_reversal_fraction"] > 0
    median_comparison = comparisons.set_index("scenario").loc["aggregation-median"]
    assert bool(median_comparison["winner_changed"])
    assert median_comparison["alternative_winners"] == "m2"
    assert not rank_changes.empty

    first_ranks, first_comparisons = paired_perturbation_bootstrap(detail, n_bootstrap=50, seed=7)
    second_ranks, second_comparisons = paired_perturbation_bootstrap(detail, n_bootstrap=50, seed=7)
    pd.testing.assert_frame_equal(first_ranks, second_ranks)
    pd.testing.assert_frame_equal(first_comparisons, second_comparisons)
    assert set(first_comparisons["scenario"]) == {
        "metric-alternative",
        "aggregation-effect-weighted-mean",
        "aggregation-median",
    }
    assert (first_ranks["n_bootstrap"] == 50).all()


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
    assert "counts" not in result.layers
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
    assert set(predictions) == {"no-change", "fold-mean", "linear"}
    assert predictions["no-change"].obs["perturbation"].tolist() == ["context-b::G1", "context-b::G2"]
    assert predictions["fold-mean"].shape == (2, 201)

    population_predictions = predict_context_baselines(
        result,
        split,
        0,
        capability="population",
        population_size=2,
    )
    for prediction in population_predictions.values():
        assert prediction.shape == (4, 201)
        assert prediction.obs["perturbation"].value_counts().to_dict() == {
            "context-b::G1": 2,
            "context-b::G2": 2,
        }
    np.testing.assert_allclose(
        population_predictions["no-change"].X[:2].toarray(),
        population_predictions["no-change"].X[2:].toarray(),
    )


def test_sciplex_context_view_applies_frozen_global_gene_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    genes = [f"gene-{index}" for index in range(220)]
    rows = []
    matrices = []
    for context in ["A", "B"]:
        for perturbation, dose, is_control in [
            ("DMSO", 0.0, True),
            ("drug-a", 10.0, False),
            ("drug-b", 10.0, False),
        ]:
            matrix = rng.poisson(5, size=(6, len(genes))).astype(np.float32)
            if perturbation == "drug-a":
                matrix[:, :4] += 15
            elif perturbation == "drug-b":
                matrix[:, 4:8] += 15
            matrices.append(matrix)
            rows.extend(
                {
                    "cell_line": context,
                    "perturbation": perturbation,
                    "dose": dose,
                    "is_control": is_control,
                }
                for _ in range(len(matrix))
            )
    canonical = AnnData(
        X=sp.csr_matrix(np.vstack(matrices)),
        obs=pd.DataFrame(rows, index=[f"cell-{index}" for index in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    source = tmp_path / "sciplex.h5ad"
    canonical.write_h5ad(source)
    source_sha256 = file_checksum(source)
    config = {
        "id": "chemical-test",
        "dataset": "chemical-test",
        "contexts": ["A", "B"],
        "drugs": [
            {"label": "drug-a", "canonical_label": "drug-a"},
            {"label": "drug-b", "canonical_label": "drug-b"},
        ],
        "doses": [10.0],
        "n_point_conditions": 2,
        "population_exclusions": [],
        "gene_panel": {
            "n_hvgs": 8,
            "hvg_flavor": "seurat",
            "n_de_genes_per_condition": 4,
            "de_method": "t-test",
        },
    }

    result = prepare_context_view(
        config,
        {"chemical-test": source},
        minimum_cells=2,
        max_cells=6,
        max_controls=6,
        seed=42,
    )

    assert 8 <= result.n_vars <= 16
    assert result.var["gene_panel_selected"].all()
    assert result.var["gene_panel_hvg"].sum() == 8
    assert result.uns["gene_panel"]["differential_expression"]["n_conditions"] == 2
    assert file_checksum(source) == source_sha256


def test_context_view_can_preserve_aligned_native_counts(tmp_path: Path) -> None:
    inputs = {}
    for context in ["context-a", "context-b"]:
        labels = ["non-targeting"] * 2 + ["G1"] * 2
        matrix = np.arange(4 * 201, dtype=np.int32).reshape(4, 201) % 7 + 1
        canonical = AnnData(
            X=sp.csr_matrix(matrix),
            obs=pd.DataFrame(
                {
                    "perturbation": labels,
                    "is_control": [True, True, False, False],
                },
                index=[f"{context}-cell-{index}" for index in range(4)],
            ),
            var=pd.DataFrame(index=[f"gene-{index}" for index in range(201)]),
        )
        path = tmp_path / f"{context}.h5ad"
        canonical.write_h5ad(path)
        inputs[context] = path

    result = prepare_context_view(
        {
            "id": "counts-test",
            "contexts": ["context-a", "context-b"],
            "perturbations": ["G1"],
        },
        inputs,
        minimum_cells=2,
        max_cells=2,
        max_controls=2,
        seed=42,
        retain_counts_layer=True,
    )

    assert result.shape == (8, 201)
    assert "counts" in result.layers
    assert_raw_counts(result.layers["counts"])
    assert np.issubdtype(result.layers["counts"].dtype, np.integer)
    assert np.max(result.X) < np.max(result.layers["counts"])


def test_context_preparation_reuses_validated_perturbationdata_checksum(tmp_path: Path) -> None:
    root = tmp_path / "PerturbationData"
    registry_dir = root / "registry"
    derived_dir = root / "derived" / "example"
    registry_dir.mkdir(parents=True)
    derived_dir.mkdir(parents=True)
    source = derived_dir / "adata.h5ad"
    source.write_bytes(b"canonical data")
    checksum = file_checksum(source)
    registry = registry_dir / "example.toml"
    registry.write_text(
        "\n".join(
            [
                'id = "example"',
                "[derived]",
                'path = "derived/example/adata.h5ad"',
                f"size_bytes = {source.stat().st_size}",
                'checksum_type = "sha256"',
                f'checksum = "{checksum}"',
                "",
            ]
        )
    )

    records, states = _input_provenance({"example": source}, {"example": registry})
    _verify_inputs_unchanged({"example": source}, records, states)

    assert records["example"]["sha256"] == checksum
    assert records["example"]["checksum_source"] == "PerturbationData-registry"
    assert records["example"]["verified_unchanged"] is True


def test_state_context_toml_retains_heldout_controls_and_relabels_predictions(tmp_path: Path) -> None:
    split = {
        "contexts": ["context-a", "context-b"],
        "perturbations": ["G1", "G2"],
        "validation_conditions": ["G1"],
        "folds": [
            {
                "test_context": "context-b",
                "train_contexts": ["context-a"],
            }
        ],
    }
    toml_path = build_context_toml(split, 0, tmp_path / "split.toml", "example", tmp_path / "data")
    toml = tomllib.loads(toml_path.read_text())
    assert toml["training"] == {"example": "train"}
    assert toml["fewshot"]["example.context-a"] == {"val": ["G1"], "test": []}
    assert toml["fewshot"]["example.context-b"] == {"val": [], "test": ["G1", "G2"]}

    prediction = AnnData(
        X=np.ones((7, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "cell_type": ["context-a"] * 2 + ["context-b"] * 5,
                "perturbation": ["G1", "G1", "control", "G1", "G1", "G2", "G2"],
            },
            index=[f"prediction-{index}" for index in range(7)],
        ),
    )
    relabeled = relabel_context_predictions(prediction, split, 0)
    assert relabeled.shape == (4, 3)
    assert relabeled.obs["perturbation"].value_counts().to_dict() == {
        "context-b::G1": 2,
        "context-b::G2": 2,
    }

    model_key_prediction = prediction.copy()
    model_key_prediction.obs["model_perturbation"] = model_key_prediction.obs.pop("perturbation")
    relabeled_model_key = relabel_context_predictions(
        model_key_prediction,
        split,
        0,
        perturbation_key="model_perturbation",
    )
    assert relabeled_model_key.obs["perturbation"].value_counts().to_dict() == {
        "context-b::G1": 2,
        "context-b::G2": 2,
    }


def test_cpa_context_helpers_assign_joint_fold_and_build_bounded_queries() -> None:
    split = {
        "perturbations": ["G1", "G2"],
        "validation_conditions": ["G2"],
        "folds": [
            {
                "id": "holdout-b",
                "test_context": "context-b",
                "train_contexts": ["context-a"],
            }
        ],
    }
    rows = []
    for context in ["context-a", "context-b"]:
        for perturbation in ["control", "G1", "G2"]:
            for index in range(4):
                rows.append(
                    {
                        "context": context,
                        "model_perturbation": perturbation,
                        "cell": f"{context}-{perturbation}-{index}",
                    }
                )
    obs = pd.DataFrame(rows).set_index("cell")
    data = AnnData(
        X=np.ones((len(obs), 3), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )

    prepared = assign_cpa_splits(data, split, 0)
    assert prepared.obs["cpa_split"].value_counts().to_dict() == {
        "train": 12,
        "test": 8,
        "valid": 4,
    }

    queries = build_counterfactual_queries(prepared, split, 0, population_size=3, seed=7)
    assert queries.shape == (6, 3)
    assert queries.obs["perturbation"].value_counts().to_dict() == {
        "context-b::G1": 3,
        "context-b::G2": 3,
    }
    assert set(queries.obs["context"].astype(str)) == {"context-b"}


def test_context_population_queries_match_observed_counts_and_resample_controls() -> None:
    split = {
        "perturbations": ["G1", "G2"],
        "folds": [
            {
                "id": "holdout-b",
                "test_context": "context-b",
                "train_contexts": ["context-a"],
            }
        ],
    }
    labels = ["control"] * 2 + ["G1"] * 3 + ["G2"] * 5
    data = AnnData(
        X=np.arange(30, dtype=np.float32).reshape(10, 3),
        obs=pd.DataFrame(
            {
                "context": ["context-b"] * 10,
                "model_perturbation": labels,
            },
            index=[f"cell-{index}" for index in range(10)],
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )
    assert observed_population_sizes(data, split, 0) == {"G1": 3, "G2": 5}
    controls = sample_heldout_controls(data, split, 0, 5, seed=7)
    assert controls.n_obs == 5
    assert controls.obs_names.is_unique
    assert controls.obs["source_obs_name"].nunique() <= 2

    queries = build_counterfactual_queries(data, split, 0, seed=7)
    assert queries.obs["perturbation"].value_counts().to_dict() == {
        "context-b::G2": 5,
        "context-b::G1": 3,
    }


def test_biolord_context_helpers_build_and_relabel_bounded_populations() -> None:
    split = {
        "perturbations": ["G1", "G2"],
        "validation_conditions": ["G2"],
        "folds": [
            {
                "id": "holdout-b",
                "test_context": "context-b",
                "train_contexts": ["context-a"],
            }
        ],
    }
    rows = []
    for context in ["context-a", "context-b"]:
        for perturbation in ["control", "G1", "G2"]:
            for index in range(4):
                rows.append(
                    {
                        "context": context,
                        "model_perturbation": perturbation,
                        "cell": f"{context}-{perturbation}-{index}",
                    }
                )
    obs = pd.DataFrame(rows).set_index("cell")
    data = AnnData(
        X=np.ones((len(obs), 3), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )

    prepared = assign_biolord_splits(data, split, 0)
    assert prepared.obs["biolord_split"].value_counts().to_dict() == {
        "train": 12,
        "test": 8,
        "valid": 4,
    }
    controls = select_heldout_controls(prepared, split, 0, population_size=3, seed=7)
    assert controls.shape == (3, 3)
    assert set(controls.obs["model_perturbation"].astype(str)) == {"control"}
    assert set(controls.obs["context"].astype(str)) == {"context-b"}

    prediction = ad.concat(
        [
            AnnData(
                X=np.ones((3, 3), dtype=np.float32),
                obs=pd.DataFrame(
                    {"model_perturbation": [condition] * 3},
                    index=[f"{condition}-{index}" for index in range(3)],
                ),
                var=data.var.copy(),
            )
            for condition in ["control", "G1", "G2"]
        ]
    )
    relabeled = relabel_biolord_predictions(
        prediction,
        split,
        0,
        population_size=3,
    )
    assert relabeled.shape == (6, 3)
    assert relabeled.obs["perturbation"].value_counts().to_dict() == {
        "context-b::G1": 3,
        "context-b::G2": 3,
    }


def test_scdisinfact_context_helpers_keep_native_counts_and_package_predictions() -> None:
    split = {
        "perturbations": ["G1", "G2"],
        "folds": [
            {
                "id": "holdout-b",
                "test_context": "context-b",
                "train_contexts": ["context-a"],
            }
        ],
    }
    rows = []
    for context in ["context-a", "context-b"]:
        for perturbation in ["control", "G1", "G2"]:
            for index in range(4):
                rows.append(
                    {
                        "context": context,
                        "model_perturbation": perturbation,
                        "cell": f"{context}-{perturbation}-{index}",
                    }
                )
    obs = pd.DataFrame(rows).set_index("cell")
    data = AnnData(
        X=sp.csr_matrix(np.arange(1, len(obs) * 3 + 1, dtype=np.int32).reshape(len(obs), 3)),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )

    training = select_scdisinfact_training_data(data, split, 0)
    assert training.shape == (16, 3)
    assert set(training.obs.loc[training.obs["context"] == "context-b", "model_perturbation"]) == {
        "control"
    }
    controls = select_scdisinfact_controls(data, split, 0, population_size=3, seed=7)
    assert controls.shape == (3, 3)
    assert set(controls.obs["model_perturbation"]) == {"control"}

    native = {
        "G1": np.asarray([[1, 3, 6], [2, 3, 5], [4, 4, 2]], dtype=np.float32),
        "G2": np.asarray([[6, 3, 1], [5, 3, 2], [2, 4, 4]], dtype=np.float32),
    }
    prediction = build_prediction_adata(
        native,
        data.var,
        test_context="context-b",
        population_size=3,
    )
    assert prediction.shape == (6, 3)
    assert prediction.obs["perturbation"].value_counts().to_dict() == {
        "context-b::G1": 3,
        "context-b::G2": 3,
    }
    np.testing.assert_array_equal(
        prediction[prediction.obs["model_perturbation"] == "G1"].X,
        native["G1"],
    )
    evaluated = apply_prediction_evaluation_transform(
        prediction,
        "normalize-total-10000-log1p",
    )
    np.testing.assert_allclose(np.expm1(evaluated.X).sum(axis=1), 10_000, rtol=1e-5)


def test_scdisinfact_joint_mmd_uses_a_reference_present_in_the_minibatch() -> None:
    assert coerce_reference_id(0) == 0
    assert coerce_reference_id(np.asarray(2)) == 2
    assert coerce_reference_id(None) is None
    assert choose_present_reference([1, 2], requested=0) == 1
    assert choose_present_reference([0, 2], requested=0) == 0
    assert choose_present_reference([2], requested=0) is None


def test_trvae_context_helpers_build_one_joint_leak_free_fold() -> None:
    split = {
        "perturbations": ["G1", "G2"],
        "folds": [
            {
                "id": "holdout-b",
                "test_context": "context-b",
                "train_contexts": ["context-a"],
            }
        ],
    }
    rows = []
    for context in ["context-a", "context-b"]:
        for perturbation in ["control", "G1", "G2"]:
            for index in range(4):
                rows.append(
                    {
                        "context": context,
                        "model_perturbation": perturbation,
                        "cell": f"{context}-{perturbation}-{index}",
                    }
                )
    obs = pd.DataFrame(rows).set_index("cell")
    data = AnnData(
        X=np.arange(1, len(obs) * 3 + 1, dtype=np.float32).reshape(len(obs), 3),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )

    training = select_trvae_training_data(data, split, 0)
    assert training.shape == (16, 3)
    assert set(training.obs.loc[training.obs["context"] == "context-b", "model_perturbation"]) == {
        "control"
    }
    controls = select_trvae_controls(data, split, 0, population_size=3, seed=7)
    assert controls.shape == (3, 3)

    sparse_training = training.copy()
    sparse_training.X = sp.csr_matrix(sparse_training.X)
    dense_training = densify_trvae_expression(sparse_training)
    assert isinstance(dense_training.X, np.ndarray)
    np.testing.assert_array_equal(dense_training.X, training.X)

    prediction = build_trvae_prediction_adata(
        {
            "G1": np.ones((3, 3), dtype=np.float32),
            "G2": np.full((3, 3), 2, dtype=np.float32),
        },
        data.var,
        test_context="context-b",
        population_size=3,
    )
    assert prediction.shape == (6, 3)
    assert prediction.obs["perturbation"].value_counts().to_dict() == {
        "context-b::G1": 3,
        "context-b::G2": 3,
    }


def test_trvae_mmd_discards_only_groups_that_cannot_contribute() -> None:
    groups = [
        np.empty((0, 2)),
        np.ones((1, 2)),
        np.full((2, 2), 2),
        np.full((3, 2), 3),
    ]
    eligible = eligible_mmd_groups(groups)
    assert [len(group) for group in eligible] == [2, 3]
    np.testing.assert_array_equal(eligible[0], groups[2])
    np.testing.assert_array_equal(eligible[1], groups[3])
