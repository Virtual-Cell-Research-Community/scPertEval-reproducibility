from __future__ import annotations

from pathlib import Path

import pandas as pd

from analyses.manuscript_figures.evaluation_practice import (
    disagreement_data,
    evaluation_practice_figure,
    factor_data,
    workflow_numbers,
)
from analyses.vocabulary import AXIS_LABELS, BENCHMARK_TASKS, CENSUS_TASKS

TASKS = list(BENCHMARK_TASKS)


def test_factor_grid_keeps_every_choice_in_taxonomy_order_and_blanks_sparse_cells():
    factors = pd.DataFrame(
        [
            {
                "task": TASKS[0],
                "factor": "score_function",
                "n_matched_contrasts": 20,
                "n_winner_changes": 5,
                "winner_change_rate": 0.25,
            },
            {
                "task": TASKS[1],
                "factor": "score_function",
                "n_matched_contrasts": 9,
                "n_winner_changes": 3,
                "winner_change_rate": 1 / 3,
            },
        ]
    )
    rows = factor_data(factors)

    assert list(rows["row"].cat.categories)[-1] == AXIS_LABELS["representation"]
    assert len(rows) == 9 * len(TASKS)
    assert list(rows["row"].cat.categories)[1] == AXIS_LABELS["cross_perturbation"]
    metric = rows[rows["choice"] == "metric"].set_index("task")
    assert metric.loc[TASKS[0], ["label", "count"]].tolist() == ["25%", "5/20"]
    assert metric.loc[TASKS[1], "label"] == "–"
    assert pd.isna(metric.loc[TASKS[1], "rate"])


def test_disagreement_columns_follow_the_task_order_and_census_names():
    disagreement = pd.DataFrame(
        {
            "task": list(reversed(TASKS)),
            "n_protocol_pairs": [100, 100, 100],
            "n_winner_disagreements": [80, 50, 20],
            "winner_disagreement_rate": [0.8, 0.5, 0.2],
        }
    )
    rows = disagreement_data(disagreement)

    assert rows["task"].tolist() == TASKS
    assert rows["tick"].tolist() == [CENSUS_TASKS[t].replace(" ", "\n").replace(",", "") for t in TASKS]
    assert rows.loc[0, "label"] == "20%\n20/100"


def _source_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    census, ranking = tmp_path / "census", tmp_path / "ranking"
    census.mkdir()
    ranking.mkdir()
    pd.DataFrame(
        [
            ("works", 3),
            ("works_model_paper", 2),
            ("works_benchmark_paper", 1),
            ("protocol_records", 5),
            ("distinct_protocols", 4),
        ],
        columns=["quantity", "value"],
    ).to_csv(census / "corpus-summary.csv", index=False)
    pd.DataFrame({"group": ["Genetic single, Few-shot"], "papers": [25]}).to_csv(
        census / "task-summary.csv", index=False
    )
    axes = [
        "representation",
        "embedding",
        "effect_isolation",
        "feature_selection",
        "target_gene_policy",
        "feature_weighting",
        "metric",
    ]
    pd.DataFrame({"group": "Genetic single, Few-shot", "level": range(1, 8), "axis": axes, "pair_sharing": 0.5}).to_csv(
        census / "consensus-decay.csv", index=False
    )
    pd.DataFrame({"group": "Genetic single, Few-shot", "axis": axes, "pair_sharing": 0.5}).to_csv(
        census / "single-choice-agreement.csv", index=False
    )
    pd.DataFrame(
        [{"task": t, "model": m, "n_protocols": 10, "winner_share": 0.5} for t in TASKS for m in ("a", "b")]
    ).to_csv(ranking / "winner-shares.csv", index=False)
    pd.DataFrame(
        {"task": TASKS, "n_protocol_pairs": 45, "n_winner_disagreements": 9, "winner_disagreement_rate": 0.2}
    ).to_csv(ranking / "task-protocol-disagreement.csv", index=False)
    pd.DataFrame(
        {
            "task": TASKS,
            "factor": "aggregation",
            "n_matched_contrasts": 10,
            "n_winner_changes": 2,
            "winner_change_rate": 0.2,
        }
    ).to_csv(ranking / "factor-sensitivity.csv", index=False)
    scope = tmp_path / "scope.csv"
    pd.DataFrame({"scope_decision": ["included"] * 10 + ["excluded"]}).to_csv(scope, index=False)
    return census, ranking, scope


def test_workflow_numbers_come_from_the_source_data(tmp_path: Path):
    census, ranking, scope = _source_data(tmp_path)
    numbers = workflow_numbers(census, ranking, scope)

    assert numbers["works"] == 3
    assert numbers["census_tasks"] == 1
    assert numbers["recurring_protocols"] == 11
    assert numbers["implemented_protocols"] == 10
    assert numbers["models_per_task"] == [2]


def test_figure_draws_five_panels_from_source_data(tmp_path: Path, monkeypatch):
    census, ranking, scope = _source_data(tmp_path)
    figure = evaluation_practice_figure(census, ranking, scope_audit=scope)

    visible = [ax for ax in figure.axes if ax.get_visible()]
    assert len(visible) == 5


def test_workflow_numbers_reject_a_scope_audit_smaller_than_the_ranked_universe(tmp_path: Path):
    import pytest

    census, ranking, scope = _source_data(tmp_path)
    pd.DataFrame({"scope_decision": ["included"] * 9}).to_csv(scope, index=False)
    with pytest.raises(ValueError, match="different universes"):
        workflow_numbers(census, ranking, scope)
