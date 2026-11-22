"""Render publication and presentation figures for the Atlas ranking analysis.

The plotting layer consumes only the frozen, figure-specific CSV exports. All
scientific calculations stay in :mod:`atlas_analysis`; this module is limited
to display-oriented transformations and Plotnine rendering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_col,
    geom_point,
    geom_segment,
    geom_text,
    geom_tile,
    ggplot,
    guide_colorbar,
    guide_legend,
    guides,
    labs,
    position_stack,
    scale_alpha_identity,
    scale_color_identity,
    scale_color_manual,
    scale_fill_gradientn,
    scale_fill_manual,
    scale_size_identity,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_discrete,
    theme,
    theme_minimal,
)

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
SHORT_TASK_LABELS = {
    "essential_perturbation_generalization": "Perturbation",
    "genetic_context_transfer": "Genetic",
    "chemical_context_transfer": "Chemical",
}
TASK_COLORS = {
    "essential_perturbation_generalization": "#4EA8DE",
    "genetic_context_transfer": "#E69F00",
    "chemical_context_transfer": "#00B894",
}
FACTOR_LABELS = {
    "evaluation_family": "Evaluation family",
    "score_function": "Score function",
    "expression_space": "Expression space",
    "feature_panel": "Gene selection",
    "centering": "Centering and reference",
    "target_policy": "Target-gene policy",
    "weighting": "Gene weighting",
    "aggregation": "Perturbation aggregation",
}
MIN_FACTOR_CONTRASTS = 10
BASELINES = {"no-change", "fold-mean", "linear"}
MODEL_LABELS = {
    "biolord": "bioLORD",
    "cellflow": "CellFlow",
    "cpa": "CPA",
    "fmlp-esm2": "fMLP-ESM2",
    "fmlp-genept": "fMLP-GenePT",
    "fold-mean": "Fold mean",
    "gears": "GEARS",
    "genepert": "GenePert",
    "linear": "Linear",
    "no-change": "No change",
    "presage": "PRESAGE",
    "sams-vae-s": "SAMS-VAE(S)",
    "scdisinfact": "scDisInFact",
    "scgpt": "scGPT",
    "sclambda": "scLAMBDA",
    "state": "STATE",
    "tabicl": "TabICLv2",
    "trvae": "trVAE",
    "txpert-public": "TxPert-public",
}
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
    "no-change": "#A0A7B2",
    "fold-mean": "#707784",
    "linear": "#3F4858",
}
LEVEL_LABELS = {
    "PBULK": "Pseudobulk",
    "SC": "Single-cell",
    "mean": "Mean",
    "median": "Median",
    "control-centered · reference=control": "Control-centered",
    "uncentered · reference=none": "Uncentered",
    "All · k=all": "All genes",
    "DEG/significance · k=50": "Top-50 DE genes",
    "HVG · k=2000": "2,000 HVGs",
}
LIGHT = {
    "background": "#FFFFFF",
    "foreground": "#171717",
    "grid": "#D7DBE0",
    "muted": "#68707D",
}
DARK = {
    "background": "#0E1117",
    "foreground": "#F3F4F6",
    "grid": "#343A46",
    "muted": "#AAB1BC",
}
HEATMAP_COLORS = ["#10002B", "#3C096C", "#9D4EDD", "#FF6D5A", "#FFF3B0"]


def _read(source: Path, filename: str) -> pd.DataFrame:
    path = source / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def _percent(values: list[float] | np.ndarray) -> list[str]:
    return [f"{value:.0%}" for value in values]


def _contrast_text_color(color: str) -> str:
    red, green, blue = (int(color[index : index + 2], 16) / 255 for index in (1, 3, 5))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#171717" if luminance > 0.62 else "#FFFFFF"


def _colors(dark: bool) -> dict[str, str]:
    return DARK if dark else LIGHT


def _plot_theme(dark: bool, *, base_size: float = 8) -> Any:
    colors = _colors(dark)
    return theme_minimal(base_family="DejaVu Sans", base_size=base_size) + theme(
        text=element_text(color=colors["foreground"]),
        axis_text=element_text(color=colors["foreground"], size=base_size - 0.5),
        axis_title=element_text(color=colors["foreground"], size=base_size),
        axis_ticks=element_blank(),
        panel_grid_major=element_line(color=colors["grid"], size=0.35),
        panel_grid_minor=element_blank(),
        panel_background=element_rect(fill=colors["background"], color=colors["background"]),
        plot_background=element_rect(fill=colors["background"], color=colors["background"]),
        plot_tag=element_text(size=10, weight="bold", color=colors["foreground"]),
        plot_tag_location="panel",
        plot_tag_position=(0.01, 0.99),
        legend_background=element_rect(fill=colors["background"], color=colors["background"]),
        legend_key=element_rect(fill=colors["background"], color=colors["background"]),
        legend_text=element_text(color=colors["foreground"], size=6.5),
        legend_title=element_text(color=colors["foreground"], size=7),
        svg_usefonts=True,
    )


def _task_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["task_label"] = prepared["task"].map(TASK_LABELS)
    return prepared


def _disagreement_plot(summary: pd.DataFrame, *, dark: bool, tag: str | None = None) -> Any:
    colors = _colors(dark)
    rows = _task_frame(summary.set_index("task").reindex(TASK_ORDER).reset_index())
    rows["task_label"] = pd.Categorical(rows["task_label"], [TASK_LABELS[task] for task in TASK_ORDER[::-1]])
    rows["annotation"] = rows.apply(
        lambda row: f"{row.winner_disagreement_rate:.1%}\n{int(row.n_winner_disagreements):,}/{int(row.n_protocol_pairs):,}",
        axis=1,
    )
    maximum = float(rows["winner_disagreement_rate"].max())
    upper = min(1.0, max(0.25, maximum + 0.16))
    breaks = np.arange(0, upper + 0.001, 0.2)
    return (
        ggplot(rows, aes("task_label", "winner_disagreement_rate", fill="task"))
        + geom_col(width=0.62, show_legend=False)
        + geom_text(aes(label="annotation"), nudge_y=0.035, size=6.6, lineheight=0.9, color=colors["foreground"])
        + coord_flip()
        + scale_fill_manual(values=TASK_COLORS)
        + scale_y_continuous(limits=(0, upper), breaks=breaks, labels=_percent(breaks), expand=(0, 0))
        + labs(x="", y="Protocol pairs selecting different winners", tag=tag)
        + _plot_theme(dark)
        + theme(panel_grid_major_y=element_blank(), axis_title_y=element_blank())
    )


def _winner_share_data(shares: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = _task_frame(shares)
    rows = rows[rows["winner_share"] > 0].copy()
    models = sorted(rows["model"].unique(), key=lambda model: (model in BASELINES, model))
    model_labels = [MODEL_LABELS.get(model, model) for model in models]
    rows["model_label"] = rows["model"].map(MODEL_LABELS).fillna(rows["model"])
    rows["model_label"] = pd.Categorical(rows["model_label"], model_labels, ordered=True)
    rows["task_label"] = pd.Categorical(
        rows["task_label"], [TASK_LABELS[task] for task in TASK_ORDER[::-1]], ordered=True
    )
    rows["share_label"] = rows["winner_share"].map(lambda value: f"{value:.0%}" if value >= 0.075 else "")
    rows["text_color"] = rows["model"].map(lambda model: _contrast_text_color(MODEL_COLORS[model]))
    return rows, models


def _winner_share_plot(shares: pd.DataFrame, *, dark: bool, tag: str | None = None) -> Any:
    rows, models = _winner_share_data(shares)
    model_labels = [MODEL_LABELS.get(model, model) for model in models]
    palette = {MODEL_LABELS.get(model, model): MODEL_COLORS[model] for model in models}
    return (
        ggplot(rows, aes("task_label", "winner_share", fill="model_label", group="model_label"))
        + geom_col(
            width=0.62,
            position=position_stack(reverse=True),
            color=_colors(dark)["background"],
            size=0.35,
        )
        + geom_text(
            aes(label="share_label", color="text_color"),
            position=position_stack(vjust=0.5, reverse=True),
            size=6.3,
            show_legend=False,
        )
        + coord_flip()
        + scale_fill_manual(values=palette, breaks=model_labels)
        + scale_color_identity()
        + scale_y_continuous(
            limits=(0, 1),
            breaks=np.arange(0, 1.01, 0.2),
            labels=_percent(np.arange(0, 1.01, 0.2)),
            expand=(0, 0),
        )
        + labs(x="", y="Share of literature-eligible protocols won", fill="", tag=tag)
        + guides(fill=guide_legend(ncol=2, byrow=True))
        + _plot_theme(dark)
        + theme(
            legend_position="bottom",
            legend_box_spacing=0.01,
            legend_key_width=8,
            legend_key_height=8,
            panel_grid_major_y=element_blank(),
            axis_title_y=element_blank(),
        )
    )


def _factor_data(factors: pd.DataFrame) -> pd.DataFrame:
    displayed = factors[factors["n_matched_contrasts"] >= MIN_FACTOR_CONTRASTS].copy()
    factor_order = [factor for factor in FACTOR_LABELS if factor in set(displayed["factor"])]
    complete = pd.MultiIndex.from_product([factor_order, TASK_ORDER], names=["factor", "task"]).to_frame(index=False)
    rows = complete.merge(displayed, on=["factor", "task"], how="left", validate="one_to_one")
    rows["factor_label"] = rows["factor"].map(FACTOR_LABELS)
    rows["task_label"] = rows["task"].map(TASK_LABELS)
    rows["factor_label"] = pd.Categorical(
        rows["factor_label"], [FACTOR_LABELS[factor] for factor in factor_order[::-1]], ordered=True
    )
    rows["task_label"] = rows["task"].map(SHORT_TASK_LABELS)
    rows["task_label"] = pd.Categorical(
        rows["task_label"], [SHORT_TASK_LABELS[task] for task in TASK_ORDER], ordered=True
    )
    rows["annotation"] = rows.apply(
        lambda row: ""
        if pd.isna(row.winner_change_rate)
        else f"{row.winner_change_rate:.1%}\n{int(row.n_winner_changes)}/{int(row.n_matched_contrasts)}",
        axis=1,
    )
    return rows


def _factor_plot(factors: pd.DataFrame, *, dark: bool, tag: str | None = None) -> Any:
    colors = _colors(dark)
    rows = _factor_data(factors)
    maximum = max(0.5, float(rows["winner_change_rate"].max(skipna=True)))
    rows["text_color"] = rows["winner_change_rate"].map(
        lambda value: colors["foreground"] if pd.isna(value) else "#171717" if value >= maximum * 0.68 else "#FFFFFF"
    )
    return (
        ggplot(rows, aes("task_label", "factor_label", fill="winner_change_rate"))
        + geom_tile(width=0.99, height=0.99, color=colors["background"], size=0.35)
        + geom_text(aes(label="annotation", color="text_color"), size=6.2, lineheight=0.9, show_legend=False)
        + scale_fill_gradientn(
            colors=HEATMAP_COLORS,
            limits=(0, maximum),
            na_value=colors["background"],
            labels=lambda values: [f"{value:.0%}" for value in values],
        )
        + scale_color_identity()
        + labs(x="", y="", fill="Winner-change\nrate", tag=tag)
        + guides(fill=guide_colorbar())
        + _plot_theme(dark)
        + theme(
            panel_grid=element_blank(),
            legend_position="right",
            legend_key_height=50,
            legend_key_width=5,
            axis_text_x=element_text(angle=30, ha="right", size=6.5),
            axis_title=element_blank(),
        )
    )


def _short_level(level: str) -> str:
    return LEVEL_LABELS.get(level, level.replace(" · ", ", "))


def _contrast_data(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = contrasts[contrasts["n_backgrounds"] >= 2].copy()
    rows = rows.sort_values(
        ["n_backgrounds", "winner_change_rate", "maximum_pairwise_reversal_fraction"],
        ascending=False,
    ).head(12)
    rows["contrast_label"] = rows.apply(
        lambda row: (
            f"{FACTOR_LABELS.get(row.factor, row.factor.replace('_', ' '))}: "
            f"{_short_level(str(row.level_a))} ↔ {_short_level(str(row.level_b))}\n"
            f"{SHORT_TASK_LABELS[row.task].replace(chr(10), ' ')}"
        ),
        axis=1,
    )
    rows["contrast_label"] = pd.Categorical(rows["contrast_label"], rows["contrast_label"].iloc[::-1], ordered=True)
    rows["annotation"] = rows.apply(
        lambda row: f"{row.winner_change_rate:.0%}  ({int(row.n_winner_changes)}/{int(row.n_backgrounds)})",
        axis=1,
    )
    rows["annotation_x"] = rows["winner_change_rate"].map(lambda value: value - 0.025 if value > 0.65 else value + 0.025)
    rows["annotation_ha"] = rows["winner_change_rate"].map(lambda value: "right" if value > 0.65 else "left")
    return rows


def _contrast_plot(contrasts: pd.DataFrame, *, dark: bool, tag: str | None = None) -> Any:
    colors = _colors(dark)
    rows = _contrast_data(contrasts)
    maximum = 1.0
    return (
        ggplot(rows, aes("winner_change_rate", "contrast_label", color="task"))
        + geom_segment(
            aes(x=0, xend="winner_change_rate", yend="contrast_label"),
            color=colors["grid"],
            size=0.55,
        )
        + geom_point(size=3.0)
        + geom_text(
            aes(x="annotation_x", label="annotation", ha="annotation_ha"),
            size=5.8,
            show_legend=False,
        )
        + scale_color_manual(values=TASK_COLORS, guide=None)
        + scale_x_continuous(
            limits=(0, maximum),
            labels=lambda values: [f"{value:.0%}" for value in values],
            expand=(0, 0),
        )
        + scale_y_discrete(drop=False)
        + labs(x="Winner-change rate", y="", tag=tag)
        + _plot_theme(dark, base_size=7.5)
        + theme(
            panel_grid_major_y=element_blank(),
            axis_text_y=element_text(size=5.8),
            axis_title_y=element_blank(),
            plot_margin_right=0.025,
        )
    )


def global_plot(source: Path, *, dark: bool) -> Any:
    disagreement = _read(source, "task-protocol-disagreement.csv")
    shares = _read(source, "winner-shares.csv")
    factors = _read(source, "factor-sensitivity.csv")
    contrasts = _read(source, "concrete-factor-contrasts.csv")
    top = _disagreement_plot(disagreement, dark=dark, tag="a") | _winner_share_plot(shares, dark=dark, tag="b")
    bottom = _factor_plot(factors, dark=dark, tag="c") | _contrast_plot(contrasts, dark=dark, tag="d")
    return (top / bottom) + theme(figure_size=(7.2, 8.8), dpi=300)


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


def _bump_plot(rows: pd.DataFrame, *, dark: bool, tag: str) -> Any:
    colors = _colors(dark)
    rows = rows.sort_values("rank_a").copy()
    first = rows.iloc[0]
    winners = set(rows.loc[rows["winner_a"].astype(bool), "model"]) | set(
        rows.loc[rows["winner_b"].astype(bool), "model"]
    )
    maximum = int(np.ceil(max(rows["rank_a"].max(), rows["rank_b"].max())))
    rows["line_color"] = rows["model"].map(
        lambda model: MODEL_COLORS.get(model, colors["muted"]) if model in winners else colors["muted"]
    )
    rows["line_size"] = rows["model"].map(
        lambda model: 1.15 if model in winners else 0.55 if model in BASELINES else 0.35
    )
    rows["line_alpha"] = rows["model"].map(
        lambda model: 1.0 if model in winners else 0.8 if model in BASELINES else 0.4
    )
    segments = rows.assign(x=0.0, xend=1.0)
    points = pd.concat(
        [
            rows.assign(x=0.0, rank=rows["rank_a"]),
            rows.assign(x=1.0, rank=rows["rank_b"]),
        ],
        ignore_index=True,
    )
    labelled = rows[rows["model"].isin(winners | BASELINES)].copy()
    labelled["left_y"] = _spread_labels(labelled["rank_a"].tolist(), minimum=1, maximum=maximum)
    labelled["right_y"] = _spread_labels(labelled["rank_b"].tolist(), minimum=1, maximum=maximum)
    left_labels = labelled.assign(
        x=-0.05,
        label_y=labelled["left_y"],
        label=labelled.apply(lambda row: f"{MODEL_LABELS.get(row.model, row.model)}  {row.rank_a:g}", axis=1),
    )
    right_labels = labelled.assign(
        x=1.05,
        label_y=labelled["right_y"],
        label=labelled.apply(lambda row: f"{row.rank_b:g}  {MODEL_LABELS.get(row.model, row.model)}", axis=1),
    )
    return (
        ggplot()
        + geom_segment(
            segments,
            aes("x", "rank_a", xend="xend", yend="rank_b", color="line_color", size="line_size", alpha="line_alpha"),
        )
        + geom_point(points, aes("x", "rank", color="line_color", alpha="line_alpha"), size=1.8)
        + geom_text(left_labels, aes("x", "label_y", label="label"), ha="right", size=5.5, color=colors["foreground"])
        + geom_text(right_labels, aes("x", "label_y", label="label"), ha="left", size=5.5, color=colors["foreground"])
        + scale_color_identity()
        + scale_size_identity()
        + scale_alpha_identity()
        + scale_x_continuous(
            limits=(-0.62, 1.62),
            breaks=[0.0, 1.0],
            labels=[str(first["level_a"]), str(first["level_b"])],
        )
        + scale_y_continuous(limits=(maximum + 0.5, 0.5), breaks=[])
        + labs(x=None, y=None, tag=tag)
        + _plot_theme(dark)
        + theme(panel_grid=element_blank(), axis_text_x=element_text(size=6.2), axis_line_x=element_blank())
    )


def bump_plot(source: Path, *, dark: bool) -> Any:
    rows = _read(source, "bump-plot-source.csv")
    selected = sorted(rows["selection_order"].astype(int).unique())
    plots = [
        _bump_plot(rows[rows["selection_order"].astype(int) == selection], dark=dark, tag=tag)
        for selection, tag in zip(selected, ["a", "b", "c", "d"], strict=False)
    ]
    if not plots:
        raise ValueError("bump-plot source contains no selected examples")
    if len(plots) == 1:
        return plots[0] + theme(figure_size=(7.2, 3.6), dpi=300)
    if len(plots) == 2:
        return (plots[0] | plots[1]) + theme(figure_size=(7.2, 3.6), dpi=300)
    composition = (plots[0] | plots[1]) / (plots[2] | plots[3])
    return composition + theme(figure_size=(7.2, 7.0), dpi=300)


def _save(plot: Any, output: Path, stem: str, mode: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        plot.save(output / f"{stem}-{mode}.{suffix}", dpi=300, verbose=False)


def render(source: Path, output: Path) -> None:
    for dark in (False, True):
        mode = "dark" if dark else "light"
        disagreement = _read(source, "task-protocol-disagreement.csv")
        shares = _read(source, "winner-shares.csv")
        factors = _read(source, "factor-sensitivity.csv")
        contrasts = _read(source, "concrete-factor-contrasts.csv")
        _save(global_plot(source, dark=dark), output, "ranking-sensitivity-global", mode)
        _save(
            _disagreement_plot(disagreement, dark=dark),
            output,
            "ranking-sensitivity-protocol-pair-disagreement",
            mode,
        )
        _save(_winner_share_plot(shares, dark=dark), output, "ranking-sensitivity-winner-shares", mode)
        _save(_factor_plot(factors, dark=dark), output, "ranking-sensitivity-factor-sensitivity", mode)
        _save(_contrast_plot(contrasts, dark=dark), output, "ranking-sensitivity-concrete-contrasts", mode)
        _save(bump_plot(source, dark=dark), output, "ranking-sensitivity-selected-bumps", mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source_data, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
