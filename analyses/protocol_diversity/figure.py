"""Render the protocol-diversity census panels, source data and supplementary table.

From the repository root:

    uv run python -m analyses.protocol_diversity.figure --atlas-repo /path/to/PerturbationEvaluationAtlas

The Atlas is read at the revision pinned for the ranking-sensitivity analysis
unless ``--revision`` is given (``--revision working-tree`` reads uncommitted files).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
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
    paper_theme,
    percent,
)
from analyses.protocol_diversity.diversity import (
    all_papers,
    consensus_decay,
    corpus_summary,
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
DEFAULT_OUTPUT = Path("figures/protocol-diversity-v1")


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


def choice_rows() -> dict[str, int]:
    """Evaluation choice → y position, taxonomy order from top to bottom (shared by panels b and c)."""

    return {axis: len(ATLAS_AXES) - i for i, axis in enumerate(ATLAS_AXES)}


def choice_scale(show_labels: bool = True) -> Any:
    rows = choice_rows()
    return scale_y_continuous(
        breaks=list(rows.values()),
        labels=[AXIS_LABELS[a] for a in rows] if show_labels else [""] * len(rows),
        name="",
        limits=(0.5, len(rows) + 0.5),
        expand=(0, 0),
    )


def decay_panel(decay: pd.DataFrame, *, dark: bool = False, tag: str | None = None) -> Any:
    """Paper pairs sharing ≥ 1 protocol as evaluation choices are added from top to bottom."""

    c = colors(dark)
    frame = with_task_columns(decay).sort_values(["group", "level"])
    frame["y"] = frame["axis"].map(choice_rows())
    tasks = list(frame["task"].cat.categories)
    return (
        ggplot(frame, aes("pair_sharing", "y", group="task", color="task", linetype="task"))
        + geom_path(size=0.6)
        + geom_point(size=1.0)
        + task_scales(tasks)
        + scale_x_continuous(labels=percent, limits=(0, 1), breaks=[0, 0.5, 1], expand=(0, 0.02))
        + choice_scale()
        + labs(x="Paper pairs sharing ≥ 1 protocol,\nchoices added top to bottom", tag=tag)
        + paper_theme(dark)
        + theme(
            panel_grid_major_y=element_line(color=c["grid"], size=0.35),
            legend_position="inside",
            legend_position_inside=(1, 0),
            legend_justification_inside=(1, 0),
            legend_key_size=7,
            legend_title=element_blank(),
            legend_text=element_text(size=SMALL_SIZE),
        )
    )


def axis_panel(agreement: pd.DataFrame, *, dark: bool = False, tag: str | None = None, show_labels: bool = True) -> Any:
    """Pair sharing per evaluation choice alone: grey bar = median over tasks, points = tasks."""

    c = colors(dark)
    frame = with_task_columns(agreement)
    tasks = list(frame["task"].cat.categories)
    offsets = {t: 0.25 - 0.5 * i / max(len(tasks) - 1, 1) for i, t in enumerate(tasks)}
    rows = choice_rows()
    frame["y"] = frame["axis"].map(rows) + frame["group"].map(offsets)
    median = frame.groupby("axis", as_index=False)["pair_sharing"].median()
    median["y"] = median["axis"].map(rows)
    few, zero = frame[frame["target_context"] == "Few-shot"], frame[frame["target_context"] == "Zero-shot"]
    return (
        ggplot()
        + geom_segment(median, aes(x=0, xend="pair_sharing", y="y", yend="y"), size=5, color=c["grid"])
        + geom_point(few, aes("pair_sharing", "y", color="task", fill="task"), size=1.2)
        + geom_point(zero, aes("pair_sharing", "y", color="task"), fill=c["background"], size=1.2, stroke=0.5)
        + scale_color_manual(values=TASK_COLORS, guide=None)
        + scale_fill_manual(values=TASK_COLORS, guide=None)
        + scale_x_continuous(
            labels=percent,
            limits=(0, 1),
            breaks=[0, 0.5, 1],
            expand=(0, 0.02),
            name="Paper pairs agreeing on\nthis choice alone",
        )
        + choice_scale(show_labels)
        + labs(tag=tag)
        + paper_theme(dark)
        + theme(panel_grid_major_y=element_blank(), panel_grid_major_x=element_line(color=c["grid"], size=0.35))
    )


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
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        (
            r"Task & Dataset & Papers & Protocols & \makecell{Single-paper\\protocols} & "
            r"\makecell{Most common\\protocol} & \makecell{Paper pairs\\sharing $\geq 1$} \\"
        ),
        r"\midrule",
    ]
    previous = None
    for row in table.itertuples():
        if row.dataset == "All" and previous is not None:
            lines.append(r"\midrule")
        task = row.task if row.dataset == "All" else ""
        lines.append(
            f"{task} & {row.dataset} & {row.papers} & {row.protocols} & "
            f"{row.single_paper_protocol_fraction:.0%} & {row.most_common_protocol_paper_fraction:.0%} & "
            f"{row.pair_sharing:.0%} \\\\".replace("%", r"\%")
        )
        previous = row
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def census_figure(source: Path, *, dark: bool) -> Any:
    """Stand-alone census figure (decay | single-choice agreement) at text width."""

    decay = pd.read_csv(source / "consensus-decay.csv")
    agreement = pd.read_csv(source / "single-choice-agreement.csv")
    figure = decay_panel(decay, dark=dark, tag="a") | axis_panel(agreement, dark=dark, tag="b", show_labels=False)
    return figure + theme(figure_size=(TEXT_WIDTH_IN, 2.8), dpi=300)


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
