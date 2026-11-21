"""Render light- and dark-mode figures for the Atlas ranking-sensitivity analysis."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

TASK_ORDER = [
    "essential_perturbation_generalization",
    "genetic_context_transfer",
    "chemical_context_transfer",
]
TASK_LABELS = {
    "essential_perturbation_generalization": "Perturbation\ngeneralization",
    "genetic_context_transfer": "Genetic context\ntransfer",
    "chemical_context_transfer": "Chemical context\ntransfer",
}
FACTOR_LABELS = {
    "metric": "Metric and output representation",
    "feature_panel": "Gene selection",
    "centering": "Centering and reference",
    "target_policy": "Target-gene policy",
    "weighting": "Gene weighting",
    "aggregation": "Perturbation aggregation",
}
BASELINES = {"no-change", "fold-mean", "linear"}
MODEL_COLORS = {
    "gears": "#0072B2",
    "presage": "#E69F00",
    "scgpt": "#56B4E9",
    "sclambda": "#009E73",
    "tabicl": "#CC79A7",
    "txpert-public": "#D55E00",
    "genepert": "#F0E442",
    "fmlp-genept": "#332288",
    "fmlp-esm2": "#88CCEE",
    "cellflow": "#44AA99",
    "state": "#117733",
    "cpa": "#999933",
    "biolord": "#DDCC77",
    "scdisinfact": "#AA4499",
    "trvae": "#882255",
    "sams-vae-s": "#661100",
    "no-change": "#9CA3AF",
    "fold-mean": "#6B7280",
    "linear": "#374151",
}


def _read(source: Path, filename: str) -> pd.DataFrame:
    path = source / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


@contextmanager
def theme(dark: bool):
    background = "#0E1117" if dark else "#FFFFFF"
    foreground = "#F3F4F6" if dark else "#171717"
    grid = "#343A46" if dark else "#D9D9D9"
    muted = "#9CA3AF" if dark else "#6B7280"
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": background,
            "axes.edgecolor": foreground,
            "axes.labelcolor": foreground,
            "axes.titlecolor": foreground,
            "figure.facecolor": background,
            "savefig.facecolor": background,
            "savefig.edgecolor": background,
            "text.color": foreground,
            "xtick.color": foreground,
            "ytick.color": foreground,
            "grid.color": grid,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        yield {"background": background, "foreground": foreground, "grid": grid, "muted": muted}


def _save(figure: plt.Figure, output: Path, stem: str, mode: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 300} if suffix == "png" else {}
        figure.savefig(output / f"{stem}-{mode}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def _contrast_text_color(color: str | tuple[float, float, float, float]) -> str:
    red, green, blue = to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#171717" if luminance > 0.62 else "#FFFFFF"


def _panel_label(ax: plt.Axes, panel: str | None) -> None:
    if panel is not None:
        ax.text(0, 1.02, panel, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")


def _winner_share_panel(
    ax: plt.Axes,
    shares: pd.DataFrame,
    colors: dict[str, str],
    panel: str | None = None,
) -> None:
    models = sorted(shares["model"].unique(), key=lambda model: (model in BASELINES, model))
    for y, task in enumerate(TASK_ORDER):
        task_rows = shares[shares["task"] == task].set_index("model")
        left = 0.0
        for model in models:
            value = float(task_rows.loc[model, "winner_share"]) if model in task_rows.index else 0.0
            if value <= 0:
                continue
            ax.barh(
                y,
                value,
                left=left,
                color=MODEL_COLORS.get(model, colors["muted"]),
                edgecolor=colors["background"],
                linewidth=0.7,
                height=0.62,
            )
            if value >= 0.075:
                segment = MODEL_COLORS.get(model, colors["muted"])
                ax.text(
                    left + value / 2,
                    y,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=_contrast_text_color(segment),
                )
            left += value
    ax.set_yticks(range(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2), [f"{value:.0%}" for value in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("Share of literature-eligible protocols won")
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.6)
    _panel_label(ax, panel)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=MODEL_COLORS.get(model, colors["muted"]), label=model)
        for model in models
        if (shares.loc[shares["model"] == model, "winner_share"] > 0).any()
    ]
    ax.legend(handles=handles, frameon=False, ncol=4, fontsize=6.1, loc="upper center", bbox_to_anchor=(0.5, -0.24))


def _disagreement_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    colors: dict[str, str],
    panel: str | None = None,
) -> None:
    task_colors = ["#56B4E9", "#E69F00", "#00CC96"]
    rows = summary.set_index("task").reindex(TASK_ORDER)
    values = rows["winner_disagreement_rate"].to_numpy(float)
    bars = ax.bar(range(3), values, color=task_colors, width=0.62)
    for bar, value, n_pairs in zip(bars, values, rows["n_protocol_pairs"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.1%}\n({int(n_pairs):,} pairs)",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    upper = max(0.1, float(values.max()) * 1.28)
    ax.set_ylim(0, min(1, upper))
    ax.set_xticks(range(3), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylabel("Protocol pairs selecting different winners")
    ax.grid(axis="y", linewidth=0.6)
    _panel_label(ax, panel)
    ax.spines["left"].set_color(colors["foreground"])


def _factor_panel(
    ax: plt.Axes,
    factors: pd.DataFrame,
    colors: dict[str, str],
    panel: str | None = None,
) -> None:
    factor_order = [factor for factor in FACTOR_LABELS if factor in set(factors["factor"])]
    matrix = np.full((len(factor_order), len(TASK_ORDER)), np.nan)
    counts = np.zeros_like(matrix)
    for row, factor in enumerate(factor_order):
        for column, task in enumerate(TASK_ORDER):
            match = factors[(factors["factor"] == factor) & (factors["task"] == task)]
            if len(match) == 1:
                matrix[row, column] = float(match.iloc[0]["winner_change_rate"])
                counts[row, column] = int(match.iloc[0]["n_matched_contrasts"])
    maximum = max(0.5, float(np.nanmax(matrix)))
    color_map = plt.get_cmap("magma").copy()
    color_map.set_bad(colors["background"])
    image = ax.imshow(matrix, cmap=color_map, vmin=0, vmax=maximum, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                color = _contrast_text_color(color_map(matrix[row, column] / maximum))
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1%}\nn={int(counts[row, column])}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=6.5,
                )
    ax.set_xticks(range(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_yticks(range(len(factor_order)), [FACTOR_LABELS[factor] for factor in factor_order])
    _panel_label(ax, panel)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Winner-change rate")
    colorbar.ax.yaxis.set_tick_params(color=colors["foreground"])


def _contrast_panel(
    ax: plt.Axes,
    contrasts: pd.DataFrame,
    colors: dict[str, str],
    panel: str | None = None,
) -> None:
    plotted = contrasts[contrasts["n_backgrounds"] >= 2].copy()
    plotted = plotted.sort_values(
        ["n_backgrounds", "winner_change_rate", "maximum_pairwise_reversal_fraction"],
        ascending=False,
    ).head(12)
    plotted = plotted.iloc[::-1]
    labels = [
        f"{row.factor}: {row.level_a} ↔ {row.level_b}\n{TASK_LABELS[row.task].replace(chr(10), ' ')}"
        for row in plotted.itertuples(index=False)
    ]
    y = np.arange(len(plotted))
    task_color = dict(zip(TASK_ORDER, ["#56B4E9", "#E69F00", "#00CC96"], strict=True))
    ax.scatter(
        plotted["winner_change_rate"],
        y,
        s=np.clip(plotted["n_backgrounds"].to_numpy(float) * 5, 20, 180),
        color=[task_color[task] for task in plotted["task"]],
        edgecolor=colors["background"],
        linewidth=0.5,
    )
    ax.set_yticks(y, labels, fontsize=6.1)
    ax.set_xlim(-0.02, min(1.0, max(0.35, float(plotted["winner_change_rate"].max()) * 1.12)))
    ax.set_xlabel("Winner-change rate across matched backgrounds")
    ax.grid(axis="x", linewidth=0.6)
    _panel_label(ax, panel)


def global_figure(source: Path, output: Path, *, dark: bool) -> None:
    shares = _read(source, "winner-shares.csv")
    disagreement = _read(source, "task-protocol-disagreement.csv")
    factors = _read(source, "factor-sensitivity.csv")
    contrasts = _read(source, "concrete-factor-contrasts.csv")
    mode = "dark" if dark else "light"
    with theme(dark) as colors:
        figure = plt.figure(figsize=(11.2, 9.2), constrained_layout=True)
        rows = figure.add_gridspec(2, 1, height_ratios=[1, 1.2])
        top = rows[0].subgridspec(1, 2, width_ratios=[1, 1.25])
        bottom = rows[1].subgridspec(1, 2, width_ratios=[1.25, 1])
        _disagreement_panel(figure.add_subplot(top[0, 0]), disagreement, colors, "a")
        _winner_share_panel(figure.add_subplot(top[0, 1]), shares, colors, "b")
        _factor_panel(figure.add_subplot(bottom[0, 0]), factors, colors, "c")
        _contrast_panel(figure.add_subplot(bottom[0, 1]), contrasts, colors, "d")
        _save(figure, output, "ranking-sensitivity-global", mode)


def _spread_labels(values: list[float], *, minimum: float, maximum: float, gap: float = 0.48) -> list[float]:
    order = np.argsort(values)
    placed = np.asarray(values, dtype=float)[order]
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)
    overflow = placed[-1] - maximum if len(placed) else 0
    if overflow > 0:
        placed -= overflow
    for index in range(len(placed) - 2, -1, -1):
        placed[index] = min(placed[index], placed[index + 1] - gap)
    underflow = minimum - placed[0] if len(placed) else 0
    if underflow > 0:
        placed += underflow
    restored = np.empty_like(placed)
    restored[order] = placed
    return restored.tolist()


def _bump_panel(ax: plt.Axes, rows: pd.DataFrame, colors: dict[str, str], panel: str) -> None:
    rows = rows.sort_values("rank_a")
    first = rows.iloc[0]
    winners = set(rows.loc[rows["winner_a"].astype(bool), "model"]) | set(
        rows.loc[rows["winner_b"].astype(bool), "model"]
    )
    maximum = int(np.ceil(max(rows["rank_a"].max(), rows["rank_b"].max())))
    labelled = rows[rows["model"].isin(winners | BASELINES)].copy()
    left_positions = dict(
        zip(
            labelled["model"],
            _spread_labels(labelled["rank_a"].tolist(), minimum=1, maximum=maximum),
            strict=True,
        )
    )
    right_positions = dict(
        zip(
            labelled["model"],
            _spread_labels(labelled["rank_b"].tolist(), minimum=1, maximum=maximum),
            strict=True,
        )
    )
    for row in rows.itertuples(index=False):
        baseline = row.model in BASELINES
        highlighted = row.model in winners
        color = MODEL_COLORS.get(row.model, colors["muted"]) if highlighted else colors["muted"]
        width = 2.2 if highlighted else 1.0 if baseline else 0.7
        alpha = 1.0 if highlighted else 0.8 if baseline else 0.42
        ax.plot([0, 1], [row.rank_a, row.rank_b], color=color, linewidth=width, alpha=alpha)
        if highlighted or baseline:
            ax.annotate(
                f"{row.model}  {row.rank_a:g}",
                xy=(0, row.rank_a),
                xytext=(-0.04, left_positions[row.model]),
                ha="right",
                va="center",
                fontsize=6.2,
                arrowprops={"arrowstyle": "-", "color": colors["muted"], "linewidth": 0.45},
            )
            ax.annotate(
                f"{row.rank_b:g}  {row.model}",
                xy=(1, row.rank_b),
                xytext=(1.04, right_positions[row.model]),
                ha="left",
                va="center",
                fontsize=6.2,
                arrowprops={"arrowstyle": "-", "color": colors["muted"], "linewidth": 0.45},
            )
    ax.scatter(np.zeros(len(rows)), rows["rank_a"], color=colors["muted"], s=10, zorder=3)
    ax.scatter(np.ones(len(rows)), rows["rank_b"], color=colors["muted"], s=10, zorder=3)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(maximum + 0.5, 0.5)
    ax.set_xticks([0, 1], [str(first["level_a"]), str(first["level_b"])], fontsize=6.2)
    ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    _panel_label(ax, panel)


def bump_figure(source: Path, output: Path, *, dark: bool) -> None:
    rows = _read(source, "bump-plot-source.csv")
    mode = "dark" if dark else "light"
    selected = sorted(rows["selection_order"].astype(int).unique())
    with theme(dark) as colors:
        figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
        for ax, selection, panel in zip(axes.flat, selected, ["a", "b", "c", "d"], strict=False):
            _bump_panel(ax, rows[rows["selection_order"].astype(int) == selection], colors, panel)
        for ax in axes.flat[len(selected) :]:
            ax.set_visible(False)
        _save(figure, output, "ranking-sensitivity-selected-bumps", mode)


def standalone_figures(source: Path, output: Path, *, dark: bool) -> None:
    shares = _read(source, "winner-shares.csv")
    disagreement = _read(source, "task-protocol-disagreement.csv")
    factors = _read(source, "factor-sensitivity.csv")
    contrasts = _read(source, "concrete-factor-contrasts.csv")
    mode = "dark" if dark else "light"
    panels = [
        ("winner-shares", shares, _winner_share_panel, (8.2, 3.4)),
        ("protocol-pair-disagreement", disagreement, _disagreement_panel, (6.2, 3.8)),
        ("factor-sensitivity", factors, _factor_panel, (7.5, 4.6)),
        ("concrete-contrasts", contrasts, _contrast_panel, (8.6, 6.2)),
    ]
    with theme(dark) as colors:
        for stem, frame, renderer, size in panels:
            figure, ax = plt.subplots(figsize=size, constrained_layout=True)
            renderer(ax, frame, colors)
            _save(figure, output, f"ranking-sensitivity-{stem}", mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for dark in [False, True]:
        global_figure(args.source_data, args.output, dark=dark)
        standalone_figures(args.source_data, args.output, dark=dark)
        bump_figure(args.source_data, args.output, dark=dark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
