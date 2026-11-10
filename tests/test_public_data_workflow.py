from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from workflows.data import MANIFEST_ROOT, load_manifest, source_artifacts
from workflows.preparation import harmonize_scperturb, merge_disambiguated_gene_names
from workflows.preparation_utils import CELL_CYCLE_SHA256, assert_raw_counts, file_checksum

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
