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


def test_cross_perturbation_factor_requires_the_same_distance_and_all_other_choices() -> None:
    direct = family(metric_family="squared-error", centering="control-centered", reference_population="control")
    retrieval = family(
        metric_family="transposed-retrieval-rank-MSE", centering="control-centered", reference_population="control"
    )
    assert factor_for_pair(direct, retrieval) == (
        "cross_perturbation",
        "direct · squared-error",
        "retrieval_predicted_query · squared-error",
    )
    observed = family(metric_family="retrieval-rank-rmse", centering="control-centered", reference_population="control")
    assert factor_for_pair(retrieval, observed)[0] == "cross_perturbation"
    # A different distance, panel or representation makes it a multi-factor change.
    assert (
        factor_for_pair(family(metric_family="pearson"), family(metric_family="transposed-retrieval-rank-MSE")) is None
    )
    assert factor_for_pair(direct, retrieval.copy().replace({"All": "HVG"})) is None
    assert (
        factor_for_pair(family(population="SC", metric_family="mmd-rbf"), family(metric_family="retrieval-rank-cosine"))
        is None
    )
    # Distances compared within retrieval remain score-function contrasts.
    assert factor_for_pair(
        family(metric_family="transposed-retrieval-rank-MAE"), family(metric_family="transposed-retrieval-rank-cosine")
    ) == ("score_function", "transposed-retrieval-rank-MAE", "transposed-retrieval-rank-cosine")


def _bundle(root, protocols, *, prediction="p" * 64, model="m1"):
    import hashlib
    import json

    path = root / "task" / model / "fold-0" / "seed-42.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "task": "task",
            "fold": "fold-0",
            "held_out_context": "",
            "model": model,
            "training_seed": 42,
            "protocol_id": protocol,
            "evaluation_unit": "u1",
            "score": 0.5,
            "undefined_reason": "",
            "representation": "centroid",
            "better": "lower",
            "score_defined": True,
        }
        for protocol in protocols
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    record = {
        "status": "complete",
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "inputs": {"prediction_sha256": prediction},
    }
    path.with_suffix(".complete.json").write_text(json.dumps(record))


def test_collect_details_merges_disjoint_roots_bound_to_the_same_prediction(tmp_path) -> None:
    import pytest

    from analyses.ranking_sensitivity.atlas_analysis import OPERATIONAL_COLUMNS, collect_details

    families = pd.DataFrame(
        [{"family_id": protocol, **{column: "x" for column in OPERATIONAL_COLUMNS}} for protocol in ("a", "b", "c")]
    )
    manifest = pd.DataFrame({"task": ["task"] * 3, "protocol_id": ["a", "b", "c"]})
    _bundle(tmp_path / "v1", ["a", "b"])
    _bundle(tmp_path / "v2", ["c"])
    merged = collect_details([tmp_path / "v1", tmp_path / "v2"], families, manifest)
    assert sorted(merged["protocol_id"]) == ["a", "b", "c"]

    _bundle(tmp_path / "stale", ["c"], prediction="q" * 64)
    with pytest.raises(ValueError, match="different predictions"):
        collect_details([tmp_path / "v1", tmp_path / "stale"], families, manifest)
    _bundle(tmp_path / "overlap", ["b", "c"])
    with pytest.raises(ValueError, match="overlap"):
        collect_details([tmp_path / "v1", tmp_path / "overlap"], families, manifest)
    with pytest.raises(ValueError, match="task manifest"):
        collect_details([tmp_path / "v1"], families, manifest)
