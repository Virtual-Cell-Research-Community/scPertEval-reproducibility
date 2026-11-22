from __future__ import annotations

from pathlib import Path

import pandas as pd

from analyses.ranking_sensitivity.atlas_figures import (
    TASK_ORDER,
    _contrast_data,
    _factor_data,
    _winner_share_data,
    global_plot,
)


def test_winner_share_data_omits_models_that_never_win() -> None:
    shares = pd.DataFrame(
        [
            {"task": task, "model": model, "winner_share": share}
            for task in TASK_ORDER
            for model, share in [("state", 0.75), ("no-change", 0.25), ("gears", 0.0)]
        ]
    )

    rows, models = _winner_share_data(shares)

    assert models == ["state", "no-change"]
    assert set(rows["model"]) == {"state", "no-change"}
    assert set(rows.loc[rows["winner_share"] == 0.75, "share_label"]) == {"75%"}


def test_factor_data_applies_the_display_support_threshold_per_task() -> None:
    factors = pd.DataFrame(
        [
            {
                "task": task,
                "factor": "score_function",
                "n_matched_contrasts": count,
                "n_winner_changes": changes,
                "winner_change_rate": changes / count,
            }
            for task, count, changes in zip(TASK_ORDER, [10, 9, 12], [3, 2, 6], strict=True)
        ]
    )

    rows = _factor_data(factors)

    essential = rows[rows["task"] == TASK_ORDER[0]].iloc[0]
    genetic = rows[rows["task"] == TASK_ORDER[1]].iloc[0]
    chemical = rows[rows["task"] == TASK_ORDER[2]].iloc[0]
    assert essential["annotation"] == "30.0%\n3/10"
    assert genetic["annotation"] == ""
    assert chemical["annotation"] == "50.0%\n6/12"


def test_contrast_labels_are_readable_and_keep_high_values_inside_panel() -> None:
    contrasts = pd.DataFrame(
        [
            {
                "task": TASK_ORDER[0],
                "factor": "feature_panel",
                "level_a": "All · k=all",
                "level_b": "DEG/significance · k=50",
                "n_backgrounds": 12,
                "n_winner_changes": 9,
                "winner_change_rate": 0.75,
                "maximum_pairwise_reversal_fraction": 0.5,
            }
        ]
    )

    row = _contrast_data(contrasts).iloc[0]

    assert "All genes ↔ Top-50 DE genes" in str(row["contrast_label"])
    assert row["annotation"] == "75%  (9/12)"
    assert row["annotation_ha"] == "right"
    assert row["annotation_x"] < row["winner_change_rate"]


def test_global_plot_draws_from_figure_specific_csvs(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "task": task,
                "n_protocol_pairs": 10,
                "n_winner_disagreements": 4,
                "winner_disagreement_rate": 0.4,
            }
            for task in TASK_ORDER
        ]
    ).to_csv(tmp_path / "task-protocol-disagreement.csv", index=False)
    pd.DataFrame(
        [
            {"task": task, "model": model, "winner_share": share}
            for task in TASK_ORDER
            for model, share in [("state", 0.75), ("no-change", 0.25)]
        ]
    ).to_csv(tmp_path / "winner-shares.csv", index=False)
    pd.DataFrame(
        [
            {
                "task": task,
                "factor": "score_function",
                "n_matched_contrasts": 10,
                "n_winner_changes": 4,
                "winner_change_rate": 0.4,
            }
            for task in TASK_ORDER
        ]
    ).to_csv(tmp_path / "factor-sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "task": task,
                "factor": "aggregation",
                "level_a": "mean",
                "level_b": "median",
                "n_backgrounds": 10,
                "n_winner_changes": 4,
                "winner_change_rate": 0.4,
                "maximum_pairwise_reversal_fraction": 0.5,
            }
            for task in TASK_ORDER
        ]
    ).to_csv(tmp_path / "concrete-factor-contrasts.csv", index=False)

    figure = global_plot(tmp_path, dark=False).draw()

    assert len(figure.axes) >= 4
