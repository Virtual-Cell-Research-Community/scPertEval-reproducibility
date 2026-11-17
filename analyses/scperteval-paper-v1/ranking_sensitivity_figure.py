"""Create the ranking-sensitivity main figure and complete supplementary views."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

TASKS = {
    "replogle-weissman-2022-k562-essential": ("Perturbation generalization", "#0072B2"),
    "replogle-nadig-four-context-50-cell": ("Genetic context transfer", "#D55E00"),
    "srivatsan-trapnell-2020-sciplex3": ("Chemical context transfer", "#009E73"),
}
DARK_TASKS = {
    "replogle-weissman-2022-k562-essential": ("Perturbation generalization", "#56B4E9"),
    "replogle-nadig-four-context-50-cell": ("Genetic context transfer", "#E69F00"),
    "srivatsan-trapnell-2020-sciplex3": ("Chemical context transfer", "#00CC96"),
}
SCENARIOS = [
    ("control-reference-all-perturbed", "Control reference"),
    ("representation-absolute", "Absolute expression"),
    ("genes-top50", "Top-50 effect genes"),
    ("de-method-mwu", "MWU DE genes"),
    ("metric-mse", "MSE"),
    ("metric-r2", "R²"),
    ("aggregation-median", "Median aggregation"),
    ("population-mmd-pca50", "MMD · PCA-50"),
]
BASELINES = {"fold-mean", "linear", "no-change"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read(results: Path, name: str) -> pd.DataFrame:
    path = results / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _summary_dots(
    ax: plt.Axes,
    comparisons: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    dark: bool = False,
) -> None:
    joined = comparisons.merge(
        bootstrap[
            [
                "dataset",
                "split",
                "scenario",
                "pairwise_reversal_fraction_median",
                "pairwise_reversal_fraction_ci_low",
                "pairwise_reversal_fraction_ci_high",
            ]
        ],
        on=["dataset", "split", "scenario"],
        how="left",
        validate="one_to_one",
    )
    task_styles = DARK_TASKS if dark else TASKS
    grid_color = "#343A46" if dark else "#D9D9D9"
    winner_color = "#E5E7EB" if dark else "#555555"
    y_positions = np.arange(len(SCENARIOS))[::-1]
    offsets = np.linspace(0.22, -0.22, len(task_styles))
    for task_index, (dataset, (label, color)) in enumerate(task_styles.items()):
        task = joined[joined["dataset"] == dataset].set_index("scenario")
        for y, (scenario, _) in zip(y_positions, SCENARIOS, strict=True):
            if scenario not in task.index:
                continue
            row = task.loc[scenario]
            point = row["pairwise_reversal_fraction_median"]
            low = row["pairwise_reversal_fraction_ci_low"]
            high = row["pairwise_reversal_fraction_ci_high"]
            if not np.isfinite(point):
                point = row["pairwise_reversal_fraction"]
                low = high = point
            marker = "D" if bool(row["winner_changed"]) else "o"
            ax.errorbar(
                point,
                y + offsets[task_index],
                xerr=np.array([[point - low], [high - point]]),
                fmt=marker,
                color=color,
                markersize=4.2,
                capsize=2,
                linewidth=0.9,
                markeredgewidth=0,
            )
    ax.set_yticks(y_positions, [label for _, label in SCENARIOS])
    ax.set_xlim(-0.015, 0.62)
    ax.set_xticks(np.arange(0, 0.61, 0.1), [f"{value:.0%}" for value in np.arange(0, 0.61, 0.1)])
    ax.set_xlabel("Pairwise model-order reversals (bootstrap median and 95% interval)")
    ax.grid(axis="x", color=grid_color, linewidth=0.6)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=5)
        for label, color in task_styles.values()
    ]
    handles.append(Line2D([0], [0], marker="D", linestyle="", color=winner_color, label="Winner changed", markersize=5))
    legend_location = "lower left" if dark else "upper right"
    legend_anchor = (0.0, 1.02) if dark else (1.0, 1.05)
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        loc=legend_location,
        bbox_to_anchor=legend_anchor,
        fontsize=6.8,
    )
    ax.set_title(
        "a  Evaluation choices reorder models in task-specific ways",
        loc="left",
        fontweight="bold",
        pad=34 if dark else None,
    )


def panel_a_dark(results: Path, output: Path) -> None:
    """Render the ranking-reversal summary as a standalone dark-mode figure."""
    comparisons = _read(results, "comparisons.csv")
    bootstrap = _read(results, "bootstrap-comparisons.csv")
    dark = "#0E1117"
    foreground = "#F3F4F6"
    with plt.rc_context(
        {
            "figure.facecolor": dark,
            "axes.facecolor": dark,
            "axes.edgecolor": foreground,
            "axes.labelcolor": foreground,
            "axes.titlecolor": foreground,
            "xtick.color": foreground,
            "ytick.color": foreground,
            "text.color": foreground,
            "savefig.facecolor": dark,
            "savefig.edgecolor": dark,
        }
    ):
        figure, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        _summary_dots(ax, comparisons, bootstrap, dark=True)
        output.mkdir(parents=True, exist_ok=True)
        figure.savefig(output / "ranking-sensitivity-panel-a-dark.png", dpi=300, bbox_inches="tight")
        figure.savefig(output / "ranking-sensitivity-panel-a-dark.pdf", bbox_inches="tight")
        plt.close(figure)


def _slope_plot(ax: plt.Axes, rows: pd.DataFrame, comparison: pd.Series, panel: str) -> None:
    rows = rows.sort_values("reference_rank")
    dataset = str(comparison["dataset"])
    title, color = TASKS[dataset]
    reference_winners = set(str(comparison["reference_winners"]).split(";"))
    alternative_winners = set(str(comparison["alternative_winners"]).split(";"))
    highlighted = reference_winners | alternative_winners
    maximum = int(np.ceil(max(rows["reference_rank"].max(), rows["alternative_rank"].max())))
    for row in rows.itertuples(index=False):
        if row.model in highlighted:
            line_color, width, alpha, style = color, 2.0, 1.0, "-"
        elif row.model in BASELINES:
            line_color, width, alpha, style = "#8A8A8A", 1.0, 0.85, "--"
        else:
            line_color, width, alpha, style = "#BDBDBD", 0.8, 0.75, "-"
        ax.plot(
            [0, 1],
            [row.reference_rank, row.alternative_rank],
            color=line_color,
            linewidth=width,
            alpha=alpha,
            linestyle=style,
            zorder=2 if row.model in highlighted else 1,
        )
        if row.model in highlighted or row.model in BASELINES:
            ax.text(-0.03, row.reference_rank, f"{row.model}  {row.reference_rank:g}", ha="right", va="center", fontsize=6.8)
            ax.text(1.03, row.alternative_rank, f"{row.alternative_rank:g}  {row.model}", ha="left", va="center", fontsize=6.8)
    ax.scatter(
        np.zeros(len(rows)),
        rows["reference_rank"],
        color=[color if model in highlighted else "#8A8A8A" for model in rows["model"]],
        s=10,
        zorder=3,
    )
    ax.scatter(
        np.ones(len(rows)),
        rows["alternative_rank"],
        color=[color if model in highlighted else "#8A8A8A" for model in rows["model"]],
        s=10,
        zorder=3,
    )
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(maximum + 0.5, 0.5)
    ax.set_xticks([0, 1], ["Reference", str(comparison["level"])])
    ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    reversal = float(comparison["pairwise_reversal_fraction"])
    winner = f"{comparison['reference_winners']} → {comparison['alternative_winners']}"
    ax.set_title(
        f"{panel}  {title}\n{reversal:.1%} reversals; winner: {winner}",
        loc="left",
        fontsize=8,
        fontweight="bold",
    )


def _selected_examples(
    comparisons: pd.DataFrame,
    showcase: pd.DataFrame,
) -> list[pd.Series]:
    examples: list[pd.Series] = []
    for dataset in TASKS:
        selected = showcase[(showcase["dataset"] == dataset) & (showcase["stratum"] == "profile")]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one selected profile scenario for {dataset}")
        examples.append(selected.iloc[0])
    chemical_population = showcase[
        (showcase["dataset"] == "srivatsan-trapnell-2020-sciplex3")
        & (showcase["stratum"] == "population")
    ]
    if len(chemical_population) != 1:
        raise ValueError("expected exactly one selected chemical population scenario")
    examples.append(chemical_population.iloc[0])
    return examples


def main_figure(results: Path, output: Path) -> None:
    comparisons = _read(results, "comparisons.csv")
    bootstrap = _read(results, "bootstrap-comparisons.csv")
    rank_changes = _read(results, "rank-changes.csv")
    showcase = _read(results, "showcase-scenarios.csv")
    examples = _selected_examples(comparisons, showcase)

    figure = plt.figure(figsize=(7.2, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[1.4, 1, 1])
    _summary_dots(figure.add_subplot(grid[0, :]), comparisons, bootstrap)
    for index, (comparison, panel) in enumerate(zip(examples, ["b", "c", "d", "e"], strict=True)):
        rows = rank_changes[
            (rank_changes["dataset"] == comparison["dataset"])
            & (rank_changes["scenario"] == comparison["scenario"])
        ]
        _slope_plot(figure.add_subplot(grid[1 + index // 2, index % 2]), rows, comparison, panel)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "ranking-sensitivity-main.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "ranking-sensitivity-main.pdf", bbox_inches="tight")
    plt.close(figure)


def sensitivity_map(results: Path, output: Path) -> None:
    comparisons = _read(results, "comparisons.csv")
    matrix = np.full((len(SCENARIOS), len(TASKS)), np.nan)
    changed = np.zeros_like(matrix, dtype=bool)
    for column, dataset in enumerate(TASKS):
        task = comparisons[comparisons["dataset"] == dataset].set_index("scenario")
        for row, (scenario, _) in enumerate(SCENARIOS):
            if scenario in task.index:
                matrix[row, column] = task.loc[scenario, "pairwise_reversal_fraction"]
                changed[row, column] = bool(task.loc[scenario, "winner_changed"])
    figure, ax = plt.subplots(figsize=(5.3, 4.1), constrained_layout=True)
    image = ax.imshow(matrix, cmap="magma_r", vmin=0, vmax=0.6, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                suffix = " ◆" if changed[row, column] else ""
                text_color = "white" if matrix[row, column] >= 0.38 else "black"
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1%}{suffix}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )
    ax.set_xticks(range(len(TASKS)), [label.replace(" ", "\n") for label, _ in TASKS.values()])
    ax.set_yticks(range(len(SCENARIOS)), [label for _, label in SCENARIOS])
    ax.set_title("Ranking sensitivity is task-dependent", loc="left", fontweight="bold")
    colorbar = figure.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("Pairwise reversals")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "ranking-sensitivity-map.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "ranking-sensitivity-map.pdf", bbox_inches="tight")
    plt.close(figure)


def all_slopes(results: Path, output: Path) -> None:
    comparisons = _read(results, "comparisons.csv")
    rank_changes = _read(results, "rank-changes.csv")
    output.mkdir(parents=True, exist_ok=True)
    with PdfPages(output / "ranking-sensitivity-all-slopes.pdf") as pdf:
        for comparison in comparisons.itertuples(index=False):
            rows = rank_changes[
                (rank_changes["dataset"] == comparison.dataset)
                & (rank_changes["scenario"] == comparison.scenario)
            ]
            figure, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
            _slope_plot(ax, rows, pd.Series(comparison._asdict()), "")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--panel-a-only",
        action="store_true",
        help="render only the standalone dark-mode summary panel",
    )
    args = parser.parse_args()
    _style()
    if args.panel_a_only:
        panel_a_dark(args.results, args.output)
        return 0
    main_figure(args.results, args.output)
    panel_a_dark(args.results, args.output)
    sensitivity_map(args.results, args.output)
    all_slopes(args.results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
