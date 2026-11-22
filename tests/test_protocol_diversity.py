import math

from analyses.protocol_diversity.diversity import (
    AtlasInputs,
    corpus_summary,
    intervention_label,
    pair_sharing,
    pinned_atlas_revision,
    protocol_prefix,
    rank_frequency,
    single_axis_agreement,
    task_label,
)

COLUMN = "PBULK__GEX__control-centered__All__target-unspecified__uniform__pearson"


def context(compositions, tcd="perturbation_responses_available", dose="", time=""):
    return {
        "intervention_compositions": compositions,
        "target_context_data": tcd,
        "dose_generalization": dose,
        "time_generalization": time,
    }


def test_pooled_singletons_and_combinations_fold_into_combination():
    assert intervention_label('[{"genetic":1},{"genetic":2}]') == "Genetic combination"
    assert intervention_label('[{"genetic":1}]') == "Genetic single"
    assert intervention_label('[{"chemical":1,"genetic":1}]') is None
    assert intervention_label('[{"chemical":1},{"biological":1}]') is None


def test_sparse_strata_are_excluded():
    assert task_label(context('[{"genetic":1}]')) == "Genetic single, Few-shot"
    assert task_label(context('[{"chemical":1}]', tcd="controls_only")) == "Chemical single, Zero-shot"
    assert task_label(context('[{"genetic":1}]', tcd="no_target_context_data")) is None
    assert task_label(context('[{"chemical":1}]', dose="extrapolation")) is None
    assert task_label(context('[{"chemical":1}]', time="interpolation")) is None


def test_protocol_prefix_keeps_cross_perturbation_suffix_at_full_level():
    assert protocol_prefix(COLUMN, 1) == "PBULK"
    assert protocol_prefix(COLUMN + "__hit_retrieval", 7) == COLUMN + "__hit_retrieval"


def test_pair_sharing():
    assert pair_sharing({"a": {"x"}, "b": {"x", "y"}, "c": {"z"}}) == 1 / 3
    assert math.isnan(pair_sharing({"a": {"x"}}))


def test_rank_frequency_is_fraction_of_group_papers():
    ranks = rank_frequency({"t": {"a": {"x", "y"}, "b": {"x"}}})
    assert ranks[["protocol", "rank", "paper_fraction"]].values.tolist() == [["x", 1, 1.0], ["y", 2, 0.5]]


def test_single_axis_agreement_compares_one_axis_and_keeps_metric_suffix():
    a = "pb__expr__none__all__keep__none__pearson"
    b = "pb__expr__none__hvg__keep__none__pearson__rank"
    groups = {"task": {"p1": {a}, "p2": {b}}}
    table = single_axis_agreement(groups).set_index("axis")["pair_sharing"]
    assert table["representation"] == 1.0
    assert table["feature_selection"] == 0.0
    assert table["metric"] == 0.0


def test_census_reads_the_atlas_revision_pinned_for_ranking_sensitivity():
    assert len(pinned_atlas_revision()) == 40


def test_corpus_summary_counts_only_works_with_protocol_records():
    atlas = AtlasInputs(
        papers=[
            {"work_id": "w1", "publication_role": "model_paper"},
            {"work_id": "w2", "publication_role": "benchmark_paper"},
            {"work_id": "w3", "publication_role": "model_paper"},
        ],
        protocols=[{"work_id": "w1", "matrix_column": "x"}, {"work_id": "w2", "matrix_column": "x"}],
        contexts=[],
        manifest={},
        dataset_names={},
        sha256={},
        revision="test",
    )
    summary = dict(corpus_summary(atlas).values.tolist())
    assert summary["works"] == 2
    assert summary["works_model_paper"] == 1
    assert summary["distinct_protocols"] == 1
