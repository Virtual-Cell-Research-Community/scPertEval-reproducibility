import math

from analyses.protocol_diversity.diversity import (
    AtlasInputs,
    corpus_summary,
    intervention_label,
    pair_sharing,
    pair_sharing_within,
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


def test_pair_sharing_within_compares_only_papers_with_the_same_attribute():
    groups = {"t": {"a": {"x"}, "b": {"x"}, "c": {"x"}, "d": {"y"}}}
    table = pair_sharing_within(groups, {"a": "2024", "b": "2024", "c": "2025", "d": "2025"}, "year")
    rows = table.set_index("year")
    assert rows.loc["2024", "pair_sharing"] == 1.0
    assert rows.loc["2025", "pair_sharing"] == 0.0
    assert rows.loc["2025", "papers"] == 2


def test_census_panels_mark_the_full_protocol_and_cumulative_rows():
    from analyses.protocol_diversity.figure import FULL_PROTOCOL, choice_rows, combined_labels

    rows = choice_rows()
    labels = combined_labels()
    choices = [key for key in rows if key != FULL_PROTOCOL]
    assert rows[FULL_PROTOCOL] < min(rows[c] for c in choices)
    assert [rows[c] for c in choices] == sorted((rows[c] for c in choices), reverse=True)
    assert not labels[choices[0]].startswith("+")
    assert all(labels[c].startswith("+ ") for c in choices[1:])
    assert labels[FULL_PROTOCOL].startswith("= ")


def test_census_figure_draws_one_legend_for_both_panels(tmp_path):
    import pandas as pd

    from analyses.protocol_diversity.figure import census_figure
    from analyses.vocabulary import ATLAS_AXES

    groups = ["Genetic single, Few-shot", "Chemical single, Zero-shot"]
    pd.DataFrame(
        [
            {
                "group": g,
                "level": i + 1,
                "axis": a,
                "papers": 25,
                "distinct_protocols": i + 2,
                "pair_sharing": 0.9 - i / 10,
            }
            for g in groups
            for i, a in enumerate(ATLAS_AXES)
        ]
    ).to_csv(tmp_path / "consensus-decay.csv", index=False)
    pd.DataFrame(
        [{"group": g, "axis": a, "distinct_values": 3, "pair_sharing": 0.6} for g in groups for a in ATLAS_AXES]
    ).to_csv(tmp_path / "single-choice-agreement.csv", index=False)

    figure = census_figure(tmp_path, dark=False)

    assert len(figure.legends) == 1
    assert [t.get_text() for t in figure.legends[0].get_texts()] == groups


def test_census_table_reports_pairs_sharing_no_protocol():
    import pandas as pd

    from analyses.protocol_diversity.figure import ALL_TASKS, to_latex

    table = pd.DataFrame(
        [
            {
                "task": ALL_TASKS,
                "dataset": "All",
                "papers": 10,
                "protocols": 20,
                "single_paper_protocol_fraction": 0.5,
                "most_common_protocol_paper_fraction": 0.3,
                "pair_sharing": 0.25,
            }
        ]
    )
    latex = to_latex(table)

    assert r"Pairs sharing\\no protocol" in latex
    assert r"All works, all tasks & 10 & 20 & 50\% & 30\% & 75\% \\" in latex
