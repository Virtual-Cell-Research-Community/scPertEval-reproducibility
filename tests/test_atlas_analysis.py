from __future__ import annotations

import pandas as pd

from analyses.ranking_sensitivity.atlas_analysis import (
    compare_rankings,
    factor_for_pair,
    rank_models,
    select_bump_examples,
)


def family(**updates: str) -> pd.Series:
    values = {
        "population": "PBULK",
        "space": "GEX",
        "metric_family": "pearson",
        "feature_set": "All",
        "feature_ranking_basis": "",
        "canonical_feature_k": "all",
        "centering": "uncentered",
        "reference_population": "none",
        "target_operation": "no_target_operation",
        "weighting": "uniform",
        "weight_source": "",
        "weight_reference": "",
        "weight_transform": "",
    }
    values.update(updates)
    return pd.Series(values)


def test_rank_models_uses_average_ranks_and_fractional_winner_sets() -> None:
    ranking = rank_models({"a": 0.8, "b": 0.8, "c": 0.2}, "higher")
    assert ranking.ranks == {"a": 1.5, "b": 1.5, "c": 3.0}
    assert ranking.winners == ("a", "b")


def test_compare_rankings_counts_strict_pairwise_reversals() -> None:
    first = rank_models({"a": 3.0, "b": 2.0, "c": 1.0}, "higher")
    second = rank_models({"a": 1.0, "b": 2.0, "c": 3.0}, "higher")
    comparison = compare_rankings(first, second)
    assert comparison["winner_changed"]
    assert comparison["n_reversed_model_pairs"] == 3
    assert comparison["pairwise_reversal_fraction"] == 1.0


def test_factor_pair_separates_evaluation_family_score_and_space() -> None:
    score_change = factor_for_pair(family(), family(metric_family="r2"))
    assert score_change == ("score_function", "pearson", "r2")
    family_change = factor_for_pair(family(), family(population="SC", metric_family="mmd-rbf"))
    assert family_change == ("evaluation_family", "PBULK", "SC")
    space_change = factor_for_pair(family(metric_family="mmd-rbf"), family(metric_family="mmd-rbf", space="PCA"))
    assert space_change == ("expression_space", "GEX", "PCA")
    assert factor_for_pair(family(), family(space="PCA", metric_family="mmd-rbf")) is None
    assert factor_for_pair(family(), family(metric_family="r2", feature_set="HVG")) is None


def test_select_bump_examples_covers_tasks_then_adds_a_factor() -> None:
    rows = []
    tasks = [
        "essential_perturbation_generalization",
        "genetic_context_transfer",
        "chemical_context_transfer",
    ]
    for index, task in enumerate(tasks):
        rows.append(
            {
                "comparison_id": f"{task}|score-function",
                "task": task,
                "analysis_eligible": True,
                "factor": "score_function",
                "winner_changed": True,
                "pairwise_reversal_fraction": 0.8 - index * 0.1,
                "minimum_fold_common_fraction": 1.0,
            }
        )
    rows.append(
        {
            "comparison_id": "essential_perturbation_generalization|centering",
            "task": "essential_perturbation_generalization",
            "analysis_eligible": True,
            "factor": "centering",
            "winner_changed": True,
            "pairwise_reversal_fraction": 0.4,
            "minimum_fold_common_fraction": 1.0,
        }
    )
    selected = select_bump_examples(pd.DataFrame(rows))
    chosen = selected[selected["selected"]]
    assert len(chosen) == 4
    assert set(chosen["task"]) == set(tasks)
    assert set(chosen["factor"]) == {"score_function", "centering"}
