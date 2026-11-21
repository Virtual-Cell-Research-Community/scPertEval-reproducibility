from __future__ import annotations

import json
from pathlib import Path

from analyses.ranking_sensitivity.atlas_protocol_audit import (
    canonical_metric,
    de_method_audit,
    group_families,
    scope_decision,
    target_operation,
    task_protocol_records,
)

CONFIG = json.loads(Path("configs/evaluation/atlas-protocol-eligibility-v1.json").read_text())


def row(**overrides: str) -> dict[str, str]:
    base = {
        "protocol_id": "p1",
        "work_id": "w1",
        "paper": "paper-one",
        "population": "PBULK",
        "space": "GEX",
        "centering": "uncentered",
        "reference_population": "none",
        "feature_set": "All",
        "feature_ranking_basis": "",
        "feature_k_bin": "",
        "feature_universe_set": "",
        "feature_universe_k": "",
        "feature_universe_ranking": "",
        "target_gene_policy": "ordinary",
        "weighting": "uniform",
        "weight_source": "",
        "weight_reference": "",
        "weight_transform": "",
        "ranking_equivalence_family": "pearson",
        "wide_display_family": "pearson",
        "cross_perturbation_group": "",
        "evaluation_sampling": "",
        "de_definition": "",
        "feature_de_definition": "",
    }
    return base | overrides


def test_no_special_target_policies_collapse_without_duplicate_votes() -> None:
    rows = [
        row(protocol_id="ordinary", target_gene_policy="ordinary"),
        row(protocol_id="unspecified", target_gene_policy="unspecified"),
        row(protocol_id="na", target_gene_policy="not_applicable"),
    ]
    assert {target_operation(item, CONFIG) for item in rows} == {"no_target_operation"}
    families = group_families(rows, CONFIG)
    assert len(families) == 1
    assert set(families[0].protocol_ids) == {"ordinary", "unspecified", "na"}


def test_metric_equivalence_collapses_mse_and_rmse_family_labels() -> None:
    assert canonical_metric(row(wide_display_family="MSE"), CONFIG) == "squared-error"
    assert canonical_metric(row(wide_display_family="squared-error"), CONFIG) == "squared-error"


def test_scope_excludes_cross_perturbation_and_accepts_existing_population_metric() -> None:
    cross = group_families([row(cross_perturbation_group="identity_matching")], CONFIG)[0]
    assert scope_decision(cross, CONFIG) == ("excluded", "pooled_or_cross_perturbation")

    population = group_families(
        [row(population="SC", space="PCA", wide_display_family="mmd-rbf")], CONFIG
    )[0]
    assert scope_decision(population, CONFIG) == ("included", "")


def test_de_audit_aliases_wilcoxon_and_mann_whitney() -> None:
    rows = [
        row(
            work_id="w1",
            de_definition=json.dumps({"method": "wilcoxon_rank_sum"}),
        ),
        row(
            protocol_id="p2",
            work_id="w2",
            de_definition=json.dumps({"method": "mann_whitney_u"}),
        ),
    ]
    records = de_method_audit(rows, CONFIG, in_scope_protocol_ids={"p1", "p2"})
    record = next(
        item
        for item in records
        if item["definition_field"] == "de_definition" and item["canonical_method"] == "rank_sum"
    )
    assert record["independent_work_count"] == 2
    assert record["in_scope_independent_work_count"] == 2
    assert record["meets_two_work_threshold"] == "true"
    assert record["scperteval_support"] == "existing"


def test_de_audit_does_not_count_missing_definitions_as_unspecified_callers() -> None:
    records = de_method_audit([row()], CONFIG)
    assert records == []


def test_limma_only_family_is_postponed_but_remains_auditable() -> None:
    family = group_families(
        [
            row(
                population="DE",
                wide_display_family="mean-rowwise-rmse",
                de_definition=json.dumps({"method": "limma"}),
            )
        ],
        CONFIG,
    )[0]
    assert scope_decision(family, CONFIG) == ("excluded", "de_method_postponed")


def test_task_manifest_expands_applicability_without_duplicate_votes() -> None:
    records = task_protocol_records(
        [
            {
                "family_id": "family",
                "applicable_tasks": "genetic_context_transfer;essential_perturbation_generalization",
                "population": "PBULK",
                "space": "GEX",
                "metric_family": "pearson",
                "target_operation": "exclude_own_target",
                "implementation_status": "selector_or_composition",
                "required_extensions": "target_gene_policy",
            }
        ]
    )
    assert [record["task"] for record in records] == [
        "essential_perturbation_generalization",
        "genetic_context_transfer",
    ]
    assert {record["protocol_id"] for record in records} == {"family"}
