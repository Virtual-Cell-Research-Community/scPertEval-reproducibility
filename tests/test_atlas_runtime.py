from __future__ import annotations

from functools import partial

from scperteval.blocks import spaces
from scperteval.protocols import metrics

from analyses.ranking_sensitivity.atlas_runtime import (
    build_protocols,
    load_eligible_families,
    target_key_for_task,
)


def test_every_eligible_family_resolves_to_one_concrete_protocol():
    families = load_eligible_families()
    protocols = build_protocols()
    assert len(protocols) == 111
    assert [protocol.name for protocol in protocols] == families["family_id"].tolist()
    assert len({protocol.name for protocol in protocols}) == 111
    assert all(not protocol.parameterised for protocol in protocols)


def test_resolved_protocol_representations_match_the_frozen_manifest():
    families = load_eligible_families().set_index("family_id")
    expected = {"PBULK": "centroid", "SC": "population", "DE": "de"}
    for protocol in build_protocols():
        row = families.loc[protocol.name]
        assert protocol.representation == expected[row["population"]]
        if protocol.representation == "de":
            assert isinstance(protocol.metric, partial)


def test_target_policy_is_part_of_the_executable_panel():
    families = load_eligible_families().set_index("family_id")
    protocols = {protocol.name: protocol for protocol in build_protocols()}
    # Retrieval families resolve target policies on the shared gene axis (tests/test_retrieval.py).
    direct = families[families["cross_perturbation_group"] == ""]
    target_rows = direct[direct["target_operation"] != "no_target_operation"]
    assert len(target_rows) == 4
    for family_id, row in target_rows.iterrows():
        protocol = protocols[family_id]
        operation = row["target_operation"]
        cache_key = protocol.space.cache_key
        assert cache_key[0:2] == ("combine", "union" if operation == "force_include_target" else "difference")
        assert cache_key[-1] == spaces.own_target().cache_key


def test_de_metric_parameters_preserve_the_frozen_family_semantics():
    families = load_eligible_families().set_index("family_id")
    protocols = {protocol.name: protocol for protocol in build_protocols()}

    auroc = protocols["atlas-536de444734a"]
    assert auroc.metric.func is metrics.de_auroc
    assert auroc.metric.keywords == {"padj": 0.05, "top_k": 50, "rank_by": "significance"}
    # A classifier needs both positive and negative genes; the DEG selector therefore belongs to
    # the pure metric rather than pre-filtering its universe to positives only.
    assert auroc.space.cache_key == spaces.full().cache_key

    spearman = protocols["atlas-aa0897b93146"]
    assert spearman.metric.func is metrics.de_spearman
    assert spearman.metric.keywords["degs_only"] is True
    assert spearman.metric.keywords["top_k"] == 2000
    assert families.loc[spearman.name, "feature_set"] == "DEG"


def test_namespaced_genetic_context_uses_the_bare_model_target_column():
    assert (
        target_key_for_task(
            "genetic_context_transfer",
            ["perturbation", "evaluation_condition", "model_perturbation"],
        )
        == "model_perturbation"
    )
    assert target_key_for_task("essential_perturbation_generalization", ["perturbation"]) is None
