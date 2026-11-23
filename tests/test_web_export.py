from __future__ import annotations

import pandas as pd
import pytest

from analyses.web_export.export import (
    _count,
    choice_label,
    metric_label,
    pair_ranking,
    reversed_model_pairs,
    verify_explorer,
)


def _ranking(ranks, winners):
    return {"scores": [float(r) for r in ranks], "ranks": [float(r) for r in ranks], "winners": winners}


def _task():
    protocols = [
        {"id": "p0", "mean": _ranking([1, 2, 3], [0]), "median": _ranking([2, 1, 3], [1])},
        {"id": "p1", "mean": _ranking([1, 3, 2], [0]), "median": _ranking([1, 3, 2], [0])},
        {"id": "p2", "mean": _ranking([3, 2, 1], [2]), "median": _ranking([3, 2, 1], [2])},
    ]
    # On the units p0 and p2 share, p2's winner is model 1.
    pair = {"a": 0, "b": 2, "ranking_a": _ranking([1, 2, 3], [0]), "ranking_b": _ranking([2, 1, 3], [1])}
    return {"id": "task", "protocols": protocols, "pair_rankings": [pair]}


def _comparison(a, b, changed, reversed_pairs, eligible=True):
    return {
        "comparison_id": f"task|{a}|{b}",
        "task": "task",
        "protocol_a": a,
        "protocol_b": b,
        "analysis_eligible": str(eligible),
        "winner_changed": str(changed),
        "n_reversed_model_pairs": str(float(reversed_pairs)),
    }


def test_pair_rankings_override_protocol_rankings_in_either_order():
    task = _task()
    first, second = pair_ranking(task, 2, 0)
    assert second["winners"] == [0] and first["winners"] == [1]
    first, second = pair_ranking(task, 0, 1)
    assert first is task["protocols"][0]["mean"] and second is task["protocols"][1]["mean"]


def test_reversed_model_pairs_ignores_ties():
    assert reversed_model_pairs(_ranking([1, 2, 3], []), _ranking([3, 2, 1], [])) == 3
    assert reversed_model_pairs(_ranking([1, 1, 3], []), _ranking([2, 1, 3], [])) == 0


def test_explorer_must_reproduce_every_analysed_comparison():
    explorer = {"tasks": [_task()]}
    comparisons = pd.DataFrame(
        [
            _comparison("p0", "p1", False, 1),
            _comparison("p0", "p2", True, 1),
            _comparison("p1", "p2", True, 2),
        ]
    )
    aggregation = pd.DataFrame(
        [{"comparison_id": "task|aggregation|p0", "task": "task", "protocol_a": "p0", "winner_changed": "True"}]
    )
    assert verify_explorer(explorer, comparisons, aggregation) == 4

    wrong = comparisons.assign(winner_changed=["False", "False", "True"])
    with pytest.raises(ValueError, match="task\\|p0\\|p2"):
        verify_explorer(explorer, wrong, aggregation)


def test_explorer_must_cover_every_eligible_comparison():
    explorer = {"tasks": [_task()]}
    comparisons = pd.DataFrame([_comparison("p0", "p1", False, 1), _comparison("p0", "missing", True, 1)])
    with pytest.raises(ValueError, match="covers 1 of 2"):
        verify_explorer(explorer, comparisons, pd.DataFrame(columns=["task", "protocol_a", "winner_changed"]))


def test_labels_read_as_evaluation_choices():
    assert metric_label("pearson") == "Pearson"
    assert metric_label("auprc") == "AUPRC"
    assert metric_label("transposed-retrieval-rank-MAE") == "Retrieval rank, Manhattan distance, prediction as query"
    assert metric_label("retrieval-rank-cosine") == "Retrieval rank, cosine distance, observation as query"
    family = pd.Series(
        {
            "population": "PBULK",
            "space": "GEX",
            "centering": "control-centered",
            "reference_population": "control",
            "feature_set": "DEG",
            "feature_ranking_basis": "significance",
            "canonical_feature_k": "50",
            "target_operation": "exclude_own_target",
            "weighting": "uniform",
            "weight_source": "",
            "weight_reference": "",
            "weight_transform": "",
            "metric_family": "squared-error",
        }
    )
    labels = {
        choice: choice_label(choice, family)
        for choice in ["representation", "effect_isolation", "feature_selection", "target_gene_policy", "metric"]
    }
    assert labels == {
        "representation": "Pseudobulk",
        "effect_isolation": "Control-centered",
        "feature_selection": "Top-50 DEGs",
        "target_gene_policy": "Own target excluded",
        "metric": "MSE",
    }


def test_counts_must_be_whole_numbers_of_pairs():
    assert _count(0.25, 8) == 2
    with pytest.raises(ValueError):
        _count(0.3, 8)
