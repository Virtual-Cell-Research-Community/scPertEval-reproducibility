from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from models.data.prepare_data import prepare_canonical_raw
from models.data.prepare_split import build_split_manifest, kfold_gene_splits
from workflows.data import MANIFEST_ROOT, load_manifest, source_artifacts
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

    chemical = json.loads(
        (ROOT / "configs" / "experiments" / "sciplex3-nine-drug-context-transfer.json").read_text()
    )
    assert chemical["n_drugs"] == len(chemical["drugs"]) == 9
    assert chemical["n_point_conditions"] == len(chemical["drugs"]) * len(chemical["doses"]) == 36
    assert chemical["n_population_conditions"] == 33
    assert len(chemical["population_exclusions"]) == 3


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
