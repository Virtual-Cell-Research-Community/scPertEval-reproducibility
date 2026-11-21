from __future__ import annotations

from scperteval.protocols.de_metrics import DEComparison

from analyses.ranking_sensitivity.atlas_runtime import build_protocols, load_eligible_families


def test_every_eligible_family_resolves_to_one_concrete_protocol():
    families = load_eligible_families()
    protocols = build_protocols()
    assert len(protocols) == 99
    assert [protocol.name for protocol in protocols] == families["family_id"].tolist()
    assert len({protocol.name for protocol in protocols}) == 99
    assert all(not protocol.parameterised for protocol in protocols)


def test_resolved_protocol_representations_match_the_frozen_manifest():
    families = load_eligible_families().set_index("family_id")
    expected = {"PBULK": "centroid", "SC": "population", "DE": "de"}
    for protocol in build_protocols():
        row = families.loc[protocol.name]
        assert protocol.representation == expected[row["population"]]
        if protocol.representation == "de":
            assert isinstance(protocol.metric, DEComparison)


def test_target_policy_is_part_of_the_executable_panel():
    families = load_eligible_families().set_index("family_id")
    protocols = {protocol.name: protocol for protocol in build_protocols()}
    target_rows = families[families["target_operation"] != "no_target_operation"]
    assert len(target_rows) == 4
    for family_id, row in target_rows.iterrows():
        protocol = protocols[family_id]
        assert row["target_operation"].replace("no_target_operation", "no_special") in protocol.space.replace("-", "_")
