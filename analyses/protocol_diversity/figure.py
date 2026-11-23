"""Render the protocol-diversity census panels, source data and supplementary table.

From the repository root:

    uv run python -m analyses.protocol_diversity.figure --atlas-repo /path/to/PerturbationEvaluationAtlas

The Atlas is read at the revision pinned for the ranking-sensitivity analysis
unless ``--revision`` is given (``--revision working-tree`` reads uncommitted files).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    geom_hline,
    geom_line,
    geom_path,
    geom_point,
    geom_segment,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    theme,
)

from analyses.figure_style import (
    SMALL_SIZE,
    TARGET_CONTEXT_LINETYPES,
    TASK_COLORS,
    TEXT_WIDTH_IN,
    colors,
    font_family,
    paper_theme,
    percent,
)
from analyses.protocol_diversity.diversity import (
    all_papers,
    consensus_decay,
    corpus_summary,
    pair_sharing_within,
    papers_by_task,
    pinned_atlas_revision,
    rank_frequency,
    read_atlas,
    single_axis_agreement,
    task_summary,
)
from analyses.vocabulary import ATLAS_AXES, AXIS_LABELS

MIN_TASK_PAPERS = 20
MIN_TABLE_DATASET_PAPERS = 10
ALL_TASKS = "All papers"
DEFAULT_OUTPUT = Path("figures/protocol-diversity")


def task_order(groups) -> list[str]:
    """Census tasks in palette order (modality, then few- before zero-shot)."""

    order = list(TASK_COLORS)
    return sorted(set(groups), key=order.index)


def with_task_columns(frame: pd.DataFrame) -> pd.DataFrame:
    parts = frame["group"].str.split(", ", n=1, expand=True)
    tasks = task_order(frame["group"])
    return frame.assign(
        task=pd.Categorical(frame["group"], tasks, ordered=True),
        target_context=parts[1],
    )


def task_scales(tasks: list[str]) -> list:
    """One legend: colour and line type both keyed by task."""

    linetypes = {t: TARGET_CONTEXT_LINETYPES[t.split(", ")[1]] for t in tasks}
    return [
        scale_color_manual(values=TASK_COLORS, breaks=tasks, name=None),
        scale_linetype_manual(values=linetypes, breaks=tasks, name=None),
    ]


FULL_PROTOCOL = "full_protocol"
"""Row key for protocols compared on every evaluation choice (the last cumulative level)."""
SINGLE_CHOICE_TITLE = "Pairs of works that disagree\non this choice alone"
COMBINED_TITLE = "Pairs of works that disagree on\nany of the choices combined so far"
LEGEND_BAND_PT = 24.0
"""Height reserved below the census panels for the shared task legend."""
FULL_PROTOCOL_GAP = 0.4
"""Extra vertical space that separates the full-protocol row from the evaluation-choice rows."""


def choice_rows() -> dict[str, float]:
    """Row key → y position: evaluation choices in taxonomy order from top to bottom, then the full protocol."""

    n = len(ATLAS_AXES)
    return {**{axis: n - i + FULL_PROTOCOL_GAP for i, axis in enumerate(ATLAS_AXES)}, FULL_PROTOCOL: 0.0}


def choice_scale(labels: dict[str, str]) -> Any:
    rows = choice_rows()
    return scale_y_continuous(
        breaks=list(rows.values()),
        labels=[labels[key] for key in rows],
        name="",
        limits=(min(rows.values()) - 0.5, max(rows.values()) + 0.5),
        expand=(0, 0),
    )


def choice_labels() -> dict[str, str]:
    """Row labels of the single-choice panel."""

    return {**{axis: AXIS_LABELS[axis] for axis in ATLAS_AXES}, FULL_PROTOCOL: "All choices\n(full protocol)"}


def combined_labels() -> dict[str, str]:
    """Row labels of the combined panel: each row adds one choice to the rows above it."""

    short = {**AXIS_LABELS, "effect_isolation": "Effect isolation"}
    labels = {axis: ("" if i == 0 else "+ ") + short[axis] for i, axis in enumerate(ATLAS_AXES)}
    return {**labels, FULL_PROTOCOL: "= Full protocol"}


def task_point_scales(tasks: list[str], dark: bool, legend: bool) -> list:
    """Colour keyed by task; few-shot tasks are filled points and zero-shot tasks open points."""

    background = colors(dark)["background"]
    fills = {t: (TASK_COLORS[t] if t.endswith("Few-shot") else background) for t in tasks}
    guide = "legend" if legend else None
    return [
        scale_color_manual(values=TASK_COLORS, breaks=tasks, name=None, guide=guide),
        scale_fill_manual(values=fills, breaks=tasks, name=None, guide=guide),
    ]


def disagreement_axis(name: str) -> Any:
    return scale_x_continuous(labels=percent, limits=(0, 1), breaks=[0, 0.5, 1], expand=(0, 0.02), name=name)


def full_protocol_rule(dark: bool) -> Any:
    return geom_hline(yintercept=FULL_PROTOCOL_GAP + 0.5, color=colors(dark)["muted"], size=0.3, linetype="dotted")


def single_choice_panel(
    agreement: pd.DataFrame, decay: pd.DataFrame, *, dark: bool = False, tag: str | None = None
) -> Any:
    """Pairs of works that disagree on each evaluation choice alone, and on the full protocol.

    Points are tasks, the grey bar is the median over tasks. Unlike the combined panel, a row does not
    depend on the order of the choices.
    """

    c = colors(dark)
    full = decay[decay["level"] == decay["level"].max()].assign(axis=FULL_PROTOCOL)
    frame = with_task_columns(pd.concat([agreement[["group", "axis", "pair_sharing"]], full]))
    frame["disagreement"] = 1 - frame["pair_sharing"]
    tasks = list(frame["task"].cat.categories)
    offsets = {t: 0.25 - 0.5 * i / max(len(tasks) - 1, 1) for i, t in enumerate(tasks)}
    rows = choice_rows()
    frame["y"] = frame["axis"].map(rows) + frame["group"].map(offsets)
    median = frame.groupby("axis", as_index=False)["disagreement"].median()
    median["y"] = median["axis"].map(rows)
    return (
        ggplot()
        + geom_segment(median, aes(x=0, xend="disagreement", y="y", yend="y"), size=5, color=c["grid"])
        + geom_point(frame, aes("disagreement", "y", color="task", fill="task"), size=1.2, stroke=0.5)
        + task_point_scales(tasks, dark, legend=False)
        + disagreement_axis(SINGLE_CHOICE_TITLE)
        + choice_scale(choice_labels())
        + full_protocol_rule(dark)
        + labs(tag=tag)
        + paper_theme(dark)
        + theme(panel_grid_major_y=element_blank(), panel_grid_major_x=element_line(color=c["grid"], size=0.35))
    )


def combined_panel(decay: pd.DataFrame, *, dark: bool = False, tag: str | None = None) -> Any:
    """Pairs of works that disagree on any of the evaluation choices combined so far, per task.

    Each row adds one choice in taxonomy order, so the steps depend on that order; the last row is the
    full protocol and is repeated below the rule, joined by a dotted line.
    """

    c = colors(dark)
    frame = with_task_columns(decay).sort_values(["group", "level"])
    frame["disagreement"] = 1 - frame["pair_sharing"]
    frame["y"] = frame["axis"].map(choice_rows())
    tasks = list(frame["task"].cat.categories)
    last = frame[frame["level"] == frame["level"].max()]
    full = last.assign(y=choice_rows()[FULL_PROTOCOL])
    return (
        ggplot(frame, aes("disagreement", "y", group="task", color="task", fill="task"))
        + geom_path(size=0.5)
        + geom_segment(last.assign(yend=full["y"]), aes(yend="yend", xend="disagreement"), size=0.4, linetype="dotted")
        + geom_point(size=1.2, stroke=0.5)
        + geom_point(full, size=1.2, stroke=0.5)
        + task_point_scales(tasks, dark, legend=False)
        + disagreement_axis(COMBINED_TITLE)
        + choice_scale(combined_labels())
        + full_protocol_rule(dark)
        + labs(tag=tag)
        + paper_theme(dark)
        + theme(panel_grid_major_y=element_blank(), panel_grid_major_x=element_line(color=c["grid"], size=0.35))
    )


def legend_band() -> Any:
    """Bottom margin that keeps room for :func:`add_task_legend` (plots must have the text-width figure size)."""

    # Plotnine 0.15 measures plot margins as a fraction of the figure width.
    return theme(plot_margin_bottom=LEGEND_BAND_PT / (TEXT_WIDTH_IN * 72))


def add_task_legend(
    figure: Any, tasks: list[str], *, dark: bool, top: float | None = None, frame: bool = False, gap_pt: float = 2.0
) -> None:
    """One legend for both census panels, centred under them in the band kept by :func:`legend_band`.

    Plotnine would centre a bottom legend under one panel, where it does not fit, and a legend beside
    the panels would narrow every panel aligned with them.
    """

    from matplotlib.lines import Line2D

    if top is None:
        figure.savefig(io.BytesIO(), dpi=figure.dpi)  # run the Plotnine layout so that the axis titles are placed
        titles = [t for t in figure.texts if t.get_text() in (SINGLE_CHOICE_TITLE, COMBINED_TITLE)]
        if len(titles) != 2:
            raise ValueError("the census axis titles were not found in the figure")
        top = min(t.get_window_extent().y0 for t in titles) / (figure.get_figheight() * figure.dpi) - gap_pt / (
            figure.get_figheight() * 72
        )
    background = colors(dark)["background"]
    handles = [
        Line2D(
            [],
            [],
            color=TASK_COLORS[t],
            marker="o",
            markersize=3.2,
            markeredgewidth=0.7,
            markerfacecolor=TASK_COLORS[t] if t.endswith("Few-shot") else background,
            linewidth=0.8,
        )
        for t in tasks
    ]
    legend = figure.legend(
        handles,
        tasks,
        loc="upper center",
        bbox_to_anchor=(0.5, top),
        ncol=3,
        frameon=frame,
        fancybox=False,
        edgecolor=colors(dark)["muted"],
        facecolor=background,
        framealpha=1,
        prop={"family": font_family(), "size": SMALL_SIZE},
        handlelength=1.6,
        columnspacing=1.4,
        labelspacing=0.3,
        borderaxespad=0,
        borderpad=0.6 if frame else 0,
    )
    legend.get_frame().set_linewidth(0.5)
    for text in legend.get_texts():
        text.set_color(colors(dark)["foreground"])


def rank_panel(ranks: pd.DataFrame, *, dark: bool = False, tag: str | None = None) -> Any:
    """Share of a task's papers using each protocol, by protocol rank (supplementary)."""

    frame = with_task_columns(ranks)
    tasks = list(frame["task"].cat.categories)
    return (
        ggplot(frame, aes("rank", "paper_fraction", group="task", color="task", linetype="task"))
        + geom_line(size=0.6)
        + task_scales(tasks)
        + scale_x_log10(breaks=[1, 3, 10, 30, 100, 300])
        + scale_y_continuous(labels=percent, limits=(0, None))
        + labs(x="Protocol rank within task (log scale)", y="Papers using protocol", tag=tag)
        + paper_theme(dark)
        + theme(legend_position="right", legend_key_size=7, legend_title=element_blank())
    )


def supplementary_table(
    tasks: pd.DataFrame, datasets: pd.DataFrame, overall: pd.DataFrame, names: dict
) -> pd.DataFrame:
    """One row per task, followed by that task's datasets with enough papers."""

    rows = [overall.assign(task=ALL_TASKS, dataset="All")]
    for task in tasks["group"]:
        rows.append(tasks[tasks["group"] == task].assign(task=task, dataset="All"))
        within = datasets[datasets["group"].str.startswith(f"{task} | ")]
        within = within[within["papers"] >= MIN_TABLE_DATASET_PAPERS]
        rows.append(within.assign(task=task, dataset=within["group"].str.split(" | ").str[-1].map(names)))
    table = pd.concat(rows, ignore_index=True)
    return table[
        [
            "task",
            "dataset",
            "papers",
            "protocols",
            "single_paper_protocol_fraction",
            "most_common_protocol_paper_fraction",
            "pair_sharing",
        ]
    ]


def to_latex(table: pd.DataFrame) -> str:
    """Compact LaTeX table: one header row per task, then its datasets (fits the manuscript text width).

    The last column reports disagreement (pairs sharing no protocol), as the census panels of Fig. 2 do.
    """

    def cells(row) -> str:
        values = (
            f"{row.papers} & {row.protocols} & {row.single_paper_protocol_fraction:.0%} & "
            f"{row.most_common_protocol_paper_fraction:.0%} & {1 - row.pair_sharing:.0%} \\\\"
        )
        return values.replace("%", r"\%")

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        (
            r"Task and dataset & Works & Protocols & \makecell{Single-work\\protocols} & "
            r"\makecell{Most common\\protocol} & \makecell{Pairs sharing\\no protocol} \\"
        ),
        r"\midrule",
    ]
    for row in table.itertuples():
        if row.task == ALL_TASKS:
            lines.append(f"All works, all tasks & {cells(row)}")
            continue
        if row.dataset == "All":
            lines += [r"\addlinespace", rf"\multicolumn{{6}}{{l}}{{\textit{{{row.task}}}}} \\"]
            lines.append(rf"\quad All datasets & {cells(row)}")
        else:
            lines.append(rf"\quad {row.dataset} & {cells(row)}")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def census_figure(source: Path, *, dark: bool) -> Any:
    """Stand-alone census figure (single choices | choices combined) at text width."""

    decay = pd.read_csv(source / "consensus-decay.csv")
    agreement = pd.read_csv(source / "single-choice-agreement.csv")
    band = legend_band()
    panels = (single_choice_panel(agreement, decay, dark=dark, tag="a") + band) | (
        combined_panel(decay, dark=dark, tag="b") + band
    )
    figure = (panels & theme(figure_size=(TEXT_WIDTH_IN, 3.0), dpi=300)).draw()
    figure.set_facecolor(colors(dark)["background"])
    add_task_legend(figure, task_order(decay["group"]), dark=dark)
    return figure


def rank_figure(source: Path, *, dark: bool) -> Any:
    ranks = pd.read_csv(source / "rank-frequency.csv")
    return rank_panel(ranks, dark=dark) + theme(figure_size=(TEXT_WIDTH_IN * 0.75, 2.6), dpi=300)


def write_source_data(atlas_repo: Path, revision: str | None, output: Path) -> None:
    atlas = read_atlas(atlas_repo, revision)
    tasks = papers_by_task(atlas)
    summary = task_summary(tasks)
    shown = summary.loc[summary["papers"] >= MIN_TASK_PAPERS, "group"].tolist()
    groups = {t: tasks[t] for t in shown}
    table = supplementary_table(
        summary[summary["group"].isin(shown)],
        task_summary(papers_by_task(atlas, by_dataset=True)),
        task_summary({ALL_TASKS: all_papers(atlas)}),
        atlas.dataset_names,
    )

    source = output / "source-data"
    source.mkdir(parents=True, exist_ok=True)
    outputs = {
        "corpus-summary.csv": corpus_summary(atlas),
        "task-summary.csv": summary,
        "consensus-decay.csv": consensus_decay(groups),
        "single-choice-agreement.csv": single_axis_agreement(groups),
        "rank-frequency.csv": rank_frequency(groups),
        "pair-sharing-by-year.csv": pair_sharing_within(
            groups, {p["work_id"]: p["year"] for p in atlas.papers}, "year"
        ),
        "pair-sharing-by-role.csv": pair_sharing_within(
            groups, {p["work_id"]: p["publication_role"] for p in atlas.papers}, "publication_role"
        ),
        "supplementary-table.csv": table,
    }
    for name, frame in outputs.items():
        frame.to_csv(source / name, index=False)
    (output / "supplementary-table.tex").write_text(to_latex(table))
    (output / "analysis-manifest.json").write_text(
        json.dumps(
            {
                "atlas_revision": atlas.revision,
                "atlas_source_commit": atlas.manifest.get("source_commit"),
                "atlas_source_dirty": atlas.manifest.get("source_dirty"),
                "atlas_input_sha256": atlas.sha256,
                "tasks_shown": shown,
                "min_task_papers": MIN_TASK_PAPERS,
                "min_table_dataset_papers": MIN_TABLE_DATASET_PAPERS,
            },
            indent=2,
        )
        + "\n"
    )


def render(output: Path) -> None:
    source = output / "source-data"
    for dark in (False, True):
        mode = "dark" if dark else "light"
        for stem, plot in [
            ("protocol-diversity-census", census_figure(source, dark=dark)),
            ("protocol-diversity-rank-frequency", rank_figure(source, dark=dark)),
        ]:
            for suffix in ("png", "pdf", "svg"):
                if hasattr(plot, "savefig"):
                    plot.savefig(output / f"{stem}-{mode}.{suffix}", dpi=300, facecolor=plot.get_facecolor())
                else:
                    plot.save(output / f"{stem}-{mode}.{suffix}", dpi=300, verbose=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas-repo", type=Path, required=True)
    parser.add_argument("--revision", default=None, help="Atlas git revision (default: pinned; 'working-tree')")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    revision = args.revision or pinned_atlas_revision()
    write_source_data(args.atlas_repo, None if revision == "working-tree" else revision, args.output)
    render(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
