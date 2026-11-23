"""Supplementary figures for the ranking-sensitivity analysis, in the style of Fig. 2.

Reads only the exported ranking source data and, for the bump-plot labels, the frozen protocol
universe. Tasks are named by the census task they belong to, as in Fig. 2 D and E.

    winner-shares      which models win, per benchmark task (fractional credit for ties)
    level-contrasts    winner-change rate for specific switches of one evaluation choice
    bump-examples      how model ranks move between two protocols that differ in one choice

From the repository root:

    uv run python -m analyses.manuscript_figures.ranking_supplement \\
      --ranking /path/to/figures/ranking-sensitivity-atlas/source-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_line,
    element_text,
    facet_grid,
    geom_col,
    geom_point,
    geom_segment,
    geom_text,
    ggplot,
    guide_legend,
    guides,
    labs,
    position_stack,
    scale_color_identity,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_reverse,
    theme,
)

from analyses.figure_style import BENCHMARK_TASK_COLORS, SMALL_SIZE, TEXT_WIDTH_IN, colors, paper_theme, percent
from analyses.ranking_sensitivity.atlas_figures import BASELINES, MODEL_COLORS, MODEL_LABELS, _bump_plot
from analyses.vocabulary import BENCHMARK_TASKS, CENSUS_TASKS, axis_label

REPO_ROOT = Path(__file__).resolve().parents[2]
ELIGIBILITY = REPO_ROOT / "configs/evaluation/atlas-protocol-eligibility.csv"
DEFAULT_OUTPUT = Path("figures/ranking-supplement")
TASK_ORDER = list(BENCHMARK_TASKS)
MIN_POOLED_CONTRASTS = 15
"""A level switch is shown if at least this many matched protocol pairs exist over all tasks."""
MIN_TASK_CONTRASTS = 3
"""A task's point is drawn only if the task has at least this many matched pairs for the switch."""
FACTOR_ORDER = ["evaluation_family", "centering", "feature_panel", "score_function", "aggregation"]
"""Factors in taxonomy order, as in Fig. 2 E (representation, effect isolation, features, metric, aggregation)."""

METRIC_LABELS = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "squared-error": "MSE",
    "MAE": "MAE",
    "r2": "R²",
    "squared-pearson": "Squared Pearson",
    "energy-distance": "Energy distance",
    "mmd-rbf": "MMD",
    "sinkhorn-divergence": "Sinkhorn divergence",
    "cosine": "Cosine",
    "explained-variance": "Explained variance",
}
POPULATION_LABELS = {"PBULK": "Pseudobulk", "SC": "Single-cell", "DE": "DE statistics"}
CENTERING_LABELS = {
    "control-centered · reference=control": "Control-centered",
    "uncentered · reference=none": "Uncentered",
    "uncentered · reference=control": "Uncentered, control reference",
}
TARGET_LABELS = {
    "exclude_own_target": "Own target excluded",
    "force_include_target": "Own target included",
    "no_target_operation": "No target handling",
}


def task_label(task: str, sep: str = ", ") -> str:
    return CENSUS_TASKS[task].replace(", ", sep)


def feature_label(level: str) -> str:
    """'DEG/significance · k=50' → 'Top-50 DEGs'; effect-ranked DEG sets say so."""

    selection, _, k = level.partition(" · k=")
    feature_set, _, basis = selection.partition("/")
    if feature_set == "All":
        return "All genes"
    if feature_set in {"HVG", "HEG"}:
        return f"{int(k):,} {feature_set}s" if k.isdigit() else f"{feature_set}s"
    if feature_set == "DEG":
        if k == "all":
            label = "All significant DEGs"
        elif k == "truth_set_size":
            label = "DEGs, size of true set"
        else:
            label = f"Top-{k} DEGs"
        return f"{label} by effect size" if basis == "effect" else label
    return level


def level_label(factor: str, level: str) -> str:
    if factor == "evaluation_family":
        return POPULATION_LABELS.get(level, level)
    if factor == "score_function":
        return METRIC_LABELS.get(level, level.replace("-", " ").capitalize())
    if factor == "feature_panel":
        return feature_label(level)
    if factor == "centering":
        return CENTERING_LABELS.get(level, level)
    if factor == "target_policy":
        return TARGET_LABELS.get(level, level)
    return level.capitalize()


def _read(source: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(source / name, keep_default_na=False)


def _counts(series: pd.Series) -> pd.Series:
    """Winner-change counts; exports before the dtype fix store single-pair counts as True/False."""

    return series.replace({"True": 1, "False": 0, True: 1, False: 0}).astype(int)


# Winner shares ---------------------------------------------------------------------------------------


def winner_share_data(shares: pd.DataFrame) -> pd.DataFrame:
    rows = shares[shares["winner_share"] > 0].copy()
    models = sorted(
        rows["model"].unique(), key=lambda model: (model in BASELINES, MODEL_LABELS.get(model, model).casefold())
    )
    rows["model_label"] = pd.Categorical(
        rows["model"].map(lambda m: MODEL_LABELS.get(m, m)), [MODEL_LABELS.get(m, m) for m in models]
    )
    rows["task_label"] = pd.Categorical(
        rows["task"].map(task_label), [task_label(t) for t in reversed(TASK_ORDER)], ordered=True
    )
    rows["share_label"] = [f"{v:.0%}" if v >= 0.08 else "" for v in rows["winner_share"]]
    rows["ink"] = rows["model"].map(lambda m: _ink(MODEL_COLORS[m]))
    return rows


def _ink(color: str) -> str:
    red, green, blue = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return "#171717" if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.6 else "#FFFFFF"


def winner_share_plot(shares: pd.DataFrame, *, dark: bool = False, tag: str | None = None) -> Any:
    c = colors(dark)
    rows = winner_share_data(shares)
    palette = {MODEL_LABELS.get(m, m): MODEL_COLORS[m] for m in rows["model"].unique()}
    return (
        ggplot(rows, aes("task_label", "winner_share", fill="model_label", group="model_label"))
        + geom_col(width=0.6, position=position_stack(reverse=True), color=c["background"], size=0.3)
        + geom_text(
            aes(label="share_label", color="ink"),
            position=position_stack(vjust=0.5, reverse=True),
            size=SMALL_SIZE,
        )
        + scale_fill_manual(values=palette, breaks=list(rows["model_label"].cat.categories), name=None)
        + scale_color_identity()
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1], labels=percent, expand=(0, 0))
        + labs(x="", y="Share of protocols won by each model (tied winners share credit)", tag=tag)
        + guides(fill=guide_legend(nrow=2))
        + paper_theme(dark)
        + theme(
            legend_position="bottom",
            legend_title=element_blank(),
            legend_key_size=7,
            legend_text=element_text(size=SMALL_SIZE),
            panel_grid_major_y=element_blank(),
        )
        + coord_flip()
    )


# Level contrasts -----------------------------------------------------------------------------------


def group_label(factor: str) -> str:
    """Facet label of an evaluation choice; long names wrap after the first word."""

    label = axis_label(factor)
    return label.replace(" ", "\n", 1) if len(label) > 14 else label


def contrast_data(contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled rows (one per level switch) and task points for the switches with enough support."""

    frame = contrasts[contrasts["factor"].isin(FACTOR_ORDER)].copy()
    frame["n_winner_changes"] = _counts(frame["n_winner_changes"])
    frame["n_backgrounds"] = frame["n_backgrounds"].astype(int)
    keys = ["factor", "level_a", "level_b"]
    pooled = frame.groupby(keys, as_index=False)[["n_backgrounds", "n_winner_changes"]].sum()
    pooled = pooled[pooled["n_backgrounds"] >= MIN_POOLED_CONTRASTS].copy()
    pooled["rate"] = pooled["n_winner_changes"] / pooled["n_backgrounds"]
    pooled["contrast"] = [
        f"{level_label(f, a)} ↔ {level_label(f, b)}"
        for f, a, b in zip(pooled["factor"], pooled["level_a"], pooled["level_b"], strict=True)
    ]
    pooled["group"] = pd.Categorical(
        pooled["factor"].map(group_label), [group_label(f) for f in FACTOR_ORDER if f in set(pooled["factor"])]
    )
    pooled["order"] = pooled["factor"].map(FACTOR_ORDER.index)
    pooled = pooled.sort_values(["order", "rate"], ascending=[True, True])
    pooled["contrast"] = pd.Categorical(pooled["contrast"], list(dict.fromkeys(pooled["contrast"])))
    pooled["label"] = [f"{k}/{n}" for k, n in zip(pooled["n_winner_changes"], pooled["n_backgrounds"], strict=True)]
    points = frame.merge(pooled[[*keys, "contrast", "group"]], on=keys)
    points = points[points["n_backgrounds"] >= MIN_TASK_CONTRASTS].copy()
    points["task_label"] = pd.Categorical(points["task"].map(task_label), [task_label(t) for t in TASK_ORDER])
    return pooled, points


def contrast_plot(contrasts: pd.DataFrame, *, dark: bool = False, tag: str | None = None) -> Any:
    c = colors(dark)
    pooled, points = contrast_data(contrasts)
    task_colors = {task_label(t): BENCHMARK_TASK_COLORS[t] for t in TASK_ORDER}
    return (
        ggplot()
        + geom_segment(pooled, aes(x=0, xend="rate", y="contrast", yend="contrast"), size=3.2, color=c["grid"])
        + geom_point(points, aes("winner_change_rate", "contrast", color="task_label"), size=1.4)
        + geom_text(pooled, aes(x=1.04, y="contrast", label="label"), ha="left", size=SMALL_SIZE, color=c["muted"])
        + facet_grid("group ~ .", scales="free_y", space="free_y")
        + scale_color_manual(values=task_colors, name=None)
        + scale_x_continuous(limits=(0, 1.16), breaks=[0, 0.25, 0.5, 0.75, 1], labels=percent, expand=(0, 0.01))
        + labs(
            x="Matched protocol pairs selecting a different winner\n"
            "(grey bar: all tasks; points: tasks with ≥ 3 pairs; numbers: winner changes / pairs)",
            y="",
            tag=tag,
        )
        + paper_theme(dark)
        + theme(
            panel_grid_major_y=element_blank(),
            panel_grid_major_x=element_line(color=c["grid"], size=0.35),
            strip_text_y=element_text(angle=0, ha="left", size=SMALL_SIZE + 0.5, color=c["foreground"]),
            strip_background=element_blank(),
            legend_position="bottom",
            legend_title=element_blank(),
            legend_key_size=7,
            legend_text=element_text(size=SMALL_SIZE),
            axis_text_y=element_text(size=SMALL_SIZE),
        )
    )


# Bump examples -------------------------------------------------------------------------------------


def protocol_description(family: pd.Series) -> str:
    """Two-line description of a protocol: metric and representation, then the features."""

    metric = METRIC_LABELS.get(family["metric_family"], family["metric_family"])
    population = POPULATION_LABELS.get(family["population"], family["population"]).lower()
    centering = ", control-centered" if family["centering"] == "control-centered" else ""
    ranking = f"/{family['feature_ranking_basis']}" if family["feature_ranking_basis"] else ""
    features = feature_label(f"{family['feature_set']}{ranking} · k={family['canonical_feature_k']}")
    return f"{metric}, {population}{centering}\n{features}"


def bump_plots(source: Path, *, dark: bool = False, eligibility: Path = ELIGIBILITY) -> Any:
    rows = _read(source, "bump-plot-source.csv")
    families = pd.read_csv(eligibility, keep_default_na=False).set_index("family_id")
    plots = []
    for tag, (_, example) in zip("ABCD", rows.groupby("selection_order", sort=True), strict=False):
        first = example.iloc[0]
        if first["factor"] == "aggregation":
            base = protocol_description(families.loc[first["protocol_a"]]).replace("\n", ", ")
            names = (f"Mean over units\n({base})", f"Median over units\n({base})")
        else:
            names = (
                protocol_description(families.loc[first["protocol_a"]]),
                protocol_description(families.loc[first["protocol_b"]]),
            )
        relabelled = example.assign(level_a=names[0], level_b=names[1])
        # Leading spaces keep the title clear of the panel tag, which Plotnine draws at the same corner.
        title = f"      {task_label(first['task'])}: {axis_label(first['factor']).lower()} changes"
        plots.append(
            # Only the winners are coloured and labelled: one rule that needs no explanation.
            _bump_plot(relabelled, dark=dark, tag=tag, highlight_baselines=False)
            # Rank 1 (best) at the top; the base plot's reversed limits do not reverse the axis in Plotnine.
            + scale_y_reverse(breaks=[], expand=(0, 0.6))
            + labs(title=title)
            + theme(
                axis_title=element_blank(),
                axis_text_x=element_text(size=SMALL_SIZE, color=colors(dark)["foreground"]),
                plot_title=element_text(size=SMALL_SIZE + 0.5, ha="left"),
            )
        )
    if len(plots) != 4:
        raise ValueError(f"expected four bump examples, found {len(plots)}")
    return (plots[0] | plots[1]) / (plots[2] | plots[3])


# Rendering -----------------------------------------------------------------------------------------


FIGURES = {
    # Panels A and B of one supplementary figure; LaTeX stacks the two files, so that each keeps its own
    # height (a Plotnine composition would give both the same panel height).
    "winner-shares": (
        lambda source, dark: winner_share_plot(_read(source, "winner-shares.csv"), dark=dark, tag="A"),
        2.3,
    ),
    "level-contrasts": (
        lambda source, dark: contrast_plot(_read(source, "concrete-factor-contrasts.csv"), dark=dark, tag="B"),
        5.0,
    ),
    "bump-examples": (lambda source, dark: bump_plots(source, dark=dark), 4.2),
}


def render(ranking: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, (build, height) in FIGURES.items():
        for dark in (False, True):
            plot = build(ranking, dark)
            size = theme(figure_size=(TEXT_WIDTH_IN, height), dpi=300)
            sized = plot + size if isinstance(plot, ggplot) else plot & size
            mode = "dark" if dark else "light"
            for suffix in ("png", "pdf", "svg"):
                sized.save(output / f"ranking-{name}-{mode}.{suffix}", dpi=300, verbose=False)
    inputs = [ranking / n for n in ("winner-shares.csv", "concrete-factor-contrasts.csv", "bump-plot-source.csv")]
    (output / "figure-manifest.json").write_text(
        json.dumps(
            {
                "figures": list(FIGURES),
                "min_pooled_contrasts": MIN_POOLED_CONTRASTS,
                "min_task_contrasts": MIN_TASK_CONTRASTS,
                "inputs_sha256": {
                    **{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    ELIGIBILITY.name: hashlib.sha256(ELIGIBILITY.read_bytes()).hexdigest(),
                },
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.ranking, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
