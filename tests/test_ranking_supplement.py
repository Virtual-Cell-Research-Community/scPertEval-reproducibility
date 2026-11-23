from __future__ import annotations

import pandas as pd

from analyses.manuscript_figures.ranking_supplement import (
    MIN_POOLED_CONTRASTS,
    MIN_TASK_CONTRASTS,
    contrast_data,
    feature_label,
    level_label,
    winner_share_data,
)
from analyses.vocabulary import BENCHMARK_TASKS

TASKS = list(BENCHMARK_TASKS)


def test_feature_levels_read_as_gene_sets():
    assert feature_label("All · k=all") == "All genes"
    assert feature_label("HVG · k=2000") == "2,000 HVGs"
    assert feature_label("DEG/significance · k=50") == "Top-50 DEGs"
    assert feature_label("DEG/effect · k=50") == "Top-50 DEGs by effect size"
    assert feature_label("DEG/significance · k=all") == "All significant DEGs"
    assert level_label("evaluation_family", "SC") == "Single-cell"
    assert level_label("score_function", "squared-error") == "MSE"


def test_contrasts_pool_over_tasks_and_drop_thinly_supported_points():
    n = MIN_POOLED_CONTRASTS
    contrasts = pd.DataFrame(
        [
            {
                "task": TASKS[0],
                "factor": "feature_panel",
                "level_a": "All · k=all",
                "level_b": "HVG · k=2000",
                "n_backgrounds": n - 1,
                "n_winner_changes": "1",
                "winner_change_rate": 1 / (n - 1),
            },
            {
                "task": TASKS[1],
                "factor": "feature_panel",
                "level_a": "All · k=all",
                "level_b": "HVG · k=2000",
                "n_backgrounds": 1,
                "n_winner_changes": "True",
                "winner_change_rate": 1.0,
            },
            {
                "task": TASKS[0],
                "factor": "score_function",
                "level_a": "MAE",
                "level_b": "pearson",
                "n_backgrounds": n - 1,
                "n_winner_changes": "2",
                "winner_change_rate": 2 / (n - 1),
            },
        ]
    )

    pooled, points = contrast_data(contrasts)

    assert pooled["contrast"].tolist() == ["All genes ↔ 2,000 HVGs"]
    assert pooled[["n_backgrounds", "n_winner_changes"]].values.tolist() == [[n, 2]]
    assert pooled["label"].tolist() == [f"2/{n}"]
    assert (points["n_backgrounds"] >= MIN_TASK_CONTRASTS).all()
    assert points["task"].tolist() == [TASKS[0]]


def test_winner_shares_omit_models_that_never_win_and_list_baselines_last():
    shares = pd.DataFrame(
        [
            {"task": TASKS[0], "model": "fold-mean", "winner_share": 0.5},
            {"task": TASKS[0], "model": "tabicl", "winner_share": 0.5},
            {"task": TASKS[0], "model": "gears", "winner_share": 0.0},
        ]
    )

    rows = winner_share_data(shares)

    assert list(rows["model_label"].cat.categories) == ["TabICLv2", "Fold mean"]
