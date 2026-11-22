from __future__ import annotations

import anndata as ad
import numpy as np
from scperteval.blocks import spaces
from scperteval.protocols import metrics
from scperteval.types import Protocol

from analyses.ranking_sensitivity.truth_coverage_runtime import REASON_MAP, evaluate_fold


def test_current_empty_gene_space_reason_maps_to_frozen_analysis_vocabulary():
    assert REASON_MAP["empty_gene_space"] == "empty_feature_set"


def test_evaluate_fold_emits_ground_truth_only_availability(monkeypatch):
    rng = np.random.default_rng(1)
    labels = ["control"] * 60
    parts = [rng.poisson(1, size=(60, 20)).astype(np.float32)]
    for index, perturbation in enumerate(["pertA", "pertB", "pertC", "pertD"]):
        values = rng.poisson(1, size=(20, 20)).astype(np.float32)
        values[:, index : index + 3] += 5
        parts.append(values)
        labels.extend([perturbation] * 20)
    dataset_adata = ad.AnnData(np.vstack(parts))
    dataset_adata.var_names = [f"g{i}" for i in range(20)]
    dataset_adata.obs["perturbation"] = labels
    protocols = [
        Protocol("atlas-a", metrics.mse, representation="centroid", better="lower", perfect=0.0),
        Protocol("atlas-b", metrics.pearson, representation="centroid", space=spaces.full()),
    ]
    monkeypatch.setattr(
        "analyses.ranking_sensitivity.truth_coverage_runtime.applicable_protocol_ids",
        lambda task, path=None: {protocol.name for protocol in protocols},
    )
    monkeypatch.setattr(
        "analyses.ranking_sensitivity.truth_coverage_runtime.build_protocols",
        lambda path=None: protocols,
    )
    result = evaluate_fold(
        dataset_adata,
        task="synthetic",
        fold_label="fold-0",
        expected_units=["pertA", "pertB", "pertC", "pertD"],
        minimum_cells=10,
        workers=1,
    )
    assert len(result) == 8
    assert result["ground_truth_defined"].all()
    assert set(result.columns) == {
        "task",
        "fold",
        "protocol_id",
        "evaluation_unit",
        "ground_truth_defined",
        "undefined_reason",
        "self_score",
    }
    assert not np.isnan(result["self_score"]).any()


def test_task_protocols_follow_the_task_manifest():
    from analyses.ranking_sensitivity.truth_coverage_runtime import task_protocols

    counts = {
        "essential_perturbation_generalization": (115, 13),
        "genetic_context_transfer": (115, 13),
        "chemical_context_transfer": (104, 7),
    }
    for task, (total, retrieval) in counts.items():
        protocols = task_protocols(task)
        assert len(protocols) == total
        assert sum(protocol.group == "atlas-retrieval" for protocol in protocols) == retrieval
