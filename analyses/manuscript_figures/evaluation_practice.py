"""Fig. 2: evaluation practice differs across papers, and the differences change model rankings.

Composes the Atlas census (protocol diversity) and the ranking-sensitivity
analysis into one manuscript figure. Reads only the two analyses' exported
source data; nothing is recomputed here.

    a  workflow: corpus → taxonomy annotation → census (b, c) and scPertEval → ranking sensitivity (d, e)
    b  census: paper pairs that disagree on each evaluation choice alone, and on the full protocol
    c  census: paper pairs that disagree on any of the evaluation choices combined so far
    d  ranking: protocol pairs selecting different winning models, per benchmark task
    e  ranking: winner-change rate when one evaluation choice changes

From the repository root:

    uv run python -m analyses.manuscript_figures.evaluation_practice \\
      --ranking /path/to/figures/ranking-sensitivity-atlas/source-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.textpath import text_to_path
from plotnine import (
    aes,
    element_blank,
    geom_col,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_identity,
    scale_fill_gradient,
    scale_fill_manual,
    scale_x_discrete,
    scale_y_continuous,
    scale_y_discrete,
    theme,
)

from analyses.figure_style import (
    BENCHMARK_TASK_COLORS,
    SMALL_SIZE,
    TAG_SIZE,
    TEXT_WIDTH_IN,
    colors,
    font_family,
    paper_theme,
    percent,
)
from analyses.protocol_diversity.figure import (
    add_task_legend,
    combined_panel,
    legend_band,
    single_choice_panel,
    task_order,
)
from analyses.vocabulary import AXIS_LABELS, BENCHMARK_TASKS, CENSUS_TASKS, RANKING_FACTORS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPE_AUDIT = REPO_ROOT / "configs/evaluation/atlas-protocol-scope-audit.csv"
MIN_FACTOR_CONTRASTS = 10
MIN_CENSUS_TASK_WORKS = 20
DEFAULT_OUTPUT = Path("figures/evaluation-practice")
FIGURE_HEIGHT_IN = 6.4
WORKFLOW_TOP_PT = 2.0
"""Figure margin above the workflow (panel a)."""
WORKFLOW_MARGIN_PT = 3.0
"""Space above and below the workflow boxes inside panel a."""
WORKFLOW_GAP_PT = 6.0
"""Space between panel a and the tags of panels b and c."""
SEQUENTIAL = ("#F2F0F7", "#54278F")
"""Single-hue sequential ramp (purple) for winner-change rates; distinct from every task hue."""

TASK_TICKS = {task: CENSUS_TASKS[task].replace(" ", "\n").replace(",", "") for task in BENCHMARK_TASKS}
"""Benchmark task → three-line census-task label (e.g. Genetic / single / Few-shot), so that d and e name
tasks as b and c do while fitting one heatmap column."""


# Workflow panel (a) ---------------------------------------------------------------------------------


def workflow_numbers(census: Path, ranking: Path, scope_audit: Path = SCOPE_AUDIT) -> dict[str, Any]:
    corpus = dict(pd.read_csv(census / "corpus-summary.csv").values.tolist())
    tasks = pd.read_csv(census / "task-summary.csv")
    shares = pd.read_csv(ranking / "winner-shares.csv")
    scope = pd.read_csv(scope_audit)
    per_task = shares.groupby("task").agg(models=("model", "nunique"), protocols=("n_protocols", "max"))
    implemented = int((scope["scope_decision"] == "included").sum())
    if int(per_task["protocols"].max()) > implemented:
        raise ValueError(
            f"ranking source data rank up to {int(per_task['protocols'].max())} protocols per task but the scope "
            f"audit {scope_audit.name} implements only {implemented}; the inputs come from different universes"
        )
    return {
        "works": int(corpus["works"]),
        "model_papers": int(corpus.get("works_model_paper", 0)),
        "benchmarks": int(corpus.get("works_benchmark_paper", 0)),
        "competitions": int(corpus.get("works_competition", 0)),
        "protocol_records": int(corpus["protocol_records"]),
        "distinct_protocols": int(corpus["distinct_protocols"]),
        "census_tasks": int((tasks["papers"] >= MIN_CENSUS_TASK_WORKS).sum()),
        "recurring_protocols": len(scope),
        "implemented_protocols": implemented,
        "protocols_per_task": [int(v) for v in sorted(per_task["protocols"].unique())],
        "models_per_task": [int(v) for v in sorted(per_task["models"].unique())],
        "benchmark_tasks": len(per_task),
    }


def _span(values: list[int]) -> str:
    return f"{values[0]}" if values[0] == values[-1] else f"{values[0]}–{values[-1]}"


WorkflowBox = tuple[int, str, str, list[tuple[str, str]]]
"""(column, lane, title, rows): column 0–3 from left to right, ``main``/``upper``/``lower`` lane,
bold title, and (number, label) rows."""


def workflow_boxes(n: dict[str, Any]) -> list[WorkflowBox]:
    """The five workflow steps; the annotation step branches into an upper and a lower lane."""

    return [
        (
            0,
            "main",
            "Literature corpus",
            [
                (f"{n['works']}", "works"),
                (f"{n['model_papers']}", "model papers"),
                (f"{n['benchmarks']}", "benchmarks"),
                (f"{n['competitions']}", "competitions"),
            ],
        ),
        (
            1,
            "main",
            "Taxonomy annotation",
            [
                (f"{n['protocol_records']:,}", "protocol records"),
                (f"{n['distinct_protocols']:,}", "distinct protocols"),
            ],
        ),
        (
            2,
            "upper",
            "Census (B, C)",
            [(f"{n['census_tasks']}", "prediction tasks"), (f"≥ {MIN_CENSUS_TASK_WORKS}", "works per task")],
        ),
        (
            2,
            "lower",
            "scPertEval",
            [
                (f"{n['recurring_protocols']}", "protocols used in ≥ 2 works"),
                (f"{n['implemented_protocols']}", "in scope, implemented"),
            ],
        ),
        (
            3,
            "lower",
            "Ranking sensitivity (D, E)",
            [
                (f"{n['benchmark_tasks']}", "benchmark tasks"),
                (_span(n["models_per_task"]), "models per task"),
                (_span(n["protocols_per_task"]), "protocols per task"),
            ],
        ),
    ]


WORKFLOW_PAD_PT = 4.0
"""Space between a box edge and its title (top) or last row (bottom)."""
WORKFLOW_TITLE_PT = SMALL_SIZE + 0.5
WORKFLOW_TITLE_GAP_PT = 3.0
WORKFLOW_LINE_PT = SMALL_SIZE * 1.35
WORKFLOW_LANE_GAP_PT = 10.0
"""Vertical space between the upper and the lower lane."""
WORKFLOW_HALF_WIDTH = 0.1025
"""Half box width in axes coordinates (the axes span the full text width)."""
WORKFLOW_COLUMN_GAP = 0.05
"""Horizontal space between neighbouring columns, which every arrow spans; axes coordinates."""
WORKFLOW_LEFT = 0.015
WORKFLOW_SIDE_PAD_PT = 3.0
"""Minimum space between a box edge and its widest line."""


def workflow_column_x(column: int) -> float:
    return WORKFLOW_LEFT + WORKFLOW_HALF_WIDTH + column * (2 * WORKFLOW_HALF_WIDTH + WORKFLOW_COLUMN_GAP)


def workflow_box_height_pt(n_rows: int) -> float:
    return 2 * WORKFLOW_PAD_PT + WORKFLOW_TITLE_PT + WORKFLOW_TITLE_GAP_PT + n_rows * WORKFLOW_LINE_PT


def workflow_height_pt(boxes: list[WorkflowBox]) -> float:
    """Height the workflow needs: the two lanes stacked, or the tallest main box if that is taller."""

    def tallest(lane: str) -> float:
        return max(workflow_box_height_pt(len(rows)) for _, box_lane, _, rows in boxes if box_lane == lane)

    return max(tallest("upper") + WORKFLOW_LANE_GAP_PT + tallest("lower"), tallest("main"))


def draw_workflow(figure: Any, rect: tuple[float, float, float, float], n: dict[str, Any], *, dark: bool) -> None:
    """Draw the workflow with Matplotlib into ``rect`` (figure coordinates), spanning the full text width.

    Each box is as tall as its content. Its rows form a table centred in the box: numbers are
    right-aligned in one column so that their digits line up, labels are left-aligned beside them.
    """

    c = colors(dark)
    ax = figure.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    width_pt, height_pt = (size * 72 for size in ax.get_window_extent().size / figure.dpi)
    font = FontProperties(family=font_family(), size=SMALL_SIZE)
    space_pt = text_to_path.get_text_width_height_descent("0 0", font, ismath=False)[0] - (
        2 * text_to_path.get_text_width_height_descent("0", font, ismath=False)[0]
    )

    def width(s: str) -> float:
        return text_to_path.get_text_width_height_descent(s, font, ismath=False)[0] / width_pt

    boxes = workflow_boxes(n)
    if workflow_height_pt(boxes) > height_pt:
        raise ValueError(
            f"the workflow needs {workflow_height_pt(boxes):.0f} pt but its row is {height_pt:.0f} pt tall"
        )
    lanes = {
        lane: max(workflow_box_height_pt(len(r)) for _, b, _, r in boxes if b == lane) for lane in ("upper", "lower")
    }
    stack = (lanes["upper"] + WORKFLOW_LANE_GAP_PT + lanes["lower"]) / height_pt
    lane_y = {
        "main": 0.5,
        "upper": 0.5 + stack / 2 - lanes["upper"] / 2 / height_pt,
        "lower": 0.5 - stack / 2 + lanes["lower"] / 2 / height_pt,
    }
    text = {"family": font_family(), "color": c["foreground"]}
    edges = []
    title_font = FontProperties(family=font_family(), size=WORKFLOW_TITLE_PT, weight="bold")
    for column_index, lane, title, rows in boxes:
        x, y = workflow_column_x(column_index), lane_y[lane]
        half_h = workflow_box_height_pt(len(rows)) / 2 / height_pt
        edges.append((x - WORKFLOW_HALF_WIDTH, x + WORKFLOW_HALF_WIDTH, y))
        ax.add_patch(
            FancyBboxPatch(
                (x - WORKFLOW_HALF_WIDTH, y - half_h),
                2 * WORKFLOW_HALF_WIDTH,
                2 * half_h,
                boxstyle="round,pad=0,rounding_size=0.012",
                linewidth=0.6,
                edgecolor=c["muted"],
                facecolor=c["background"],
            )
        )
        top = y + half_h - WORKFLOW_PAD_PT / height_pt
        ax.text(x, top, title, ha="center", va="top", fontsize=WORKFLOW_TITLE_PT, fontweight="bold", **text)
        number_w = max(width(number) for number, _ in rows)
        label_w = max(width(label) for _, label in rows)
        table_w = number_w + space_pt / width_pt + label_w
        title_w = text_to_path.get_text_width_height_descent(title, title_font, ismath=False)[0] / width_pt
        if max(table_w, title_w) + 2 * WORKFLOW_SIDE_PAD_PT / width_pt > 2 * WORKFLOW_HALF_WIDTH:
            raise ValueError(f"the text of workflow box {title!r} is wider than the box")
        column = x - table_w / 2 + number_w
        first = top - (WORKFLOW_TITLE_PT + WORKFLOW_TITLE_GAP_PT) / height_pt
        for i, (number, label) in enumerate(rows):
            row_y = first - (i + 0.5) * WORKFLOW_LINE_PT / height_pt
            ax.text(column, row_y, number, ha="right", va="center_baseline", fontsize=SMALL_SIZE, **text)
            ax.text(
                column + space_pt / width_pt, row_y, label, ha="left", va="center_baseline", fontsize=SMALL_SIZE, **text
            )

    corpus, annotation, census, implemented, ranking = edges
    branch = 0.3 * (lane_y["upper"] - lane_y["main"])
    for (_, start_x, start_y), (end_x, _, end_y) in [
        (corpus, annotation),
        ((None, annotation[1], annotation[2] + branch), census),
        ((None, annotation[1], annotation[2] - branch), implemented),
        (implemented, ranking),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                (start_x, start_y), (end_x, end_y), arrowstyle="-|>", mutation_scale=6, linewidth=0.6, color=c["muted"]
            )
        )
    figure.text(
        0.005,
        rect[1] + rect[3],
        "A",
        fontsize=TAG_SIZE,
        fontweight="bold",
        va="top",
        family=font_family(),
        color=c["foreground"],
    )


# Ranking panels (d, e) ------------------------------------------------------------------------------


def disagreement_data(disagreement: pd.DataFrame) -> pd.DataFrame:
    rows = disagreement.set_index("task").reindex(list(BENCHMARK_TASKS)).reset_index()
    rows["tick"] = pd.Categorical(rows["task"].map(TASK_TICKS), list(TASK_TICKS.values()))
    rows["label"] = [
        f"{rate:.0%}\n{int(k):,}/{int(n):,}"
        for rate, k, n in zip(
            rows["winner_disagreement_rate"], rows["n_winner_disagreements"], rows["n_protocol_pairs"], strict=True
        )
    ]
    rows["label_y"] = rows["winner_disagreement_rate"] + 0.03
    return rows


def disagreement_panel(disagreement: pd.DataFrame, *, dark: bool = False, tag: str | None = None) -> Any:
    """Fraction of all eligible protocol pairs that select different winning models, one column per task.

    Columns follow the task order of panel e, so each task's overall rate sits above its breakdown.
    """

    c = colors(dark)
    rows = disagreement_data(disagreement)
    return (
        ggplot(rows, aes("tick", "winner_disagreement_rate", fill="task"))
        + geom_col(width=0.6, show_legend=False)
        + geom_text(aes(y="label_y", label="label"), va="bottom", size=SMALL_SIZE, color=c["foreground"])
        + scale_fill_manual(values=BENCHMARK_TASK_COLORS)
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.5, 1], labels=percent, expand=(0, 0))
        # The title sits under the columns, like e's: a y-axis title would sit at the far left, because
        # Plotnine aligns this panel with the long row labels of b.
        + scale_x_discrete(name="Protocol pairs selecting\ndifferent winning models")
        + labs(y="", tag=tag)
        + paper_theme(dark)
        + theme(panel_grid_major_x=element_blank())
    )


def factor_data(factors: pd.DataFrame) -> pd.DataFrame:
    """Task × evaluation choice grid in taxonomy order; cells below the support threshold stay empty."""

    factors = factors.assign(choice=factors["factor"].map(RANKING_FACTORS))
    choices = list(dict.fromkeys(RANKING_FACTORS.values()))
    grid = pd.MultiIndex.from_product([choices, list(BENCHMARK_TASKS)], names=["choice", "task"]).to_frame(index=False)
    rows = grid.merge(factors, on=["choice", "task"], how="left", validate="one_to_one")
    shown = (rows["n_matched_contrasts"] >= MIN_FACTOR_CONTRASTS).to_numpy()
    rows["rate"] = rows["winner_change_rate"].where(shown)
    rows["label"] = [f"{r:.0%}" if s else "–" for r, s in zip(rows["winner_change_rate"], shown, strict=True)]
    rows["count"] = [
        f"{int(k)}/{int(n)}" if s else ""
        for k, n, s in zip(rows["n_winner_changes"], rows["n_matched_contrasts"], shown, strict=True)
    ]
    rows["row"] = pd.Categorical(rows["choice"].map(AXIS_LABELS), [AXIS_LABELS[ch] for ch in reversed(choices)])
    rows["tick"] = pd.Categorical(rows["task"].map(TASK_TICKS), list(TASK_TICKS.values()))
    return rows


def factor_panel(factors: pd.DataFrame, *, dark: bool = False, tag: str | None = None) -> Any:
    """Winner-change rate when one evaluation choice changes; values are labelled, so no colour bar."""

    c = colors(dark)
    rows = factor_data(factors)
    rows["ink"] = np.where(rows["rate"].fillna(0) > 0.5, "#FFFFFF", c["foreground"])
    header_y = rows["row"].cat.categories.size + 0.62
    plot = (
        ggplot(rows, aes("tick", "row"))
        + geom_tile(aes(fill="rate"), width=0.95, height=0.9, color=c["background"], show_legend=False)
        + geom_text(aes(label="label", color="ink"), size=SMALL_SIZE, nudge_y=0.15)
        + geom_text(aes(label="count", color="ink"), size=SMALL_SIZE - 1, nudge_y=-0.2)
    )
    for task, tick in TASK_TICKS.items():
        header = pd.DataFrame({"tick": pd.Categorical([tick], list(TASK_TICKS.values())), "y": [header_y]})
        plot += geom_tile(
            header, aes("tick", "y"), height=0.16, width=0.95, fill=BENCHMARK_TASK_COLORS[task], inherit_aes=False
        )
    return (
        plot
        + scale_fill_gradient(low=SEQUENTIAL[0], high=SEQUENTIAL[1], limits=(0, 1), na_value=c["grid"])
        + scale_color_identity()
        + scale_y_discrete(name="")
        + scale_x_discrete(name="Winner-change rate when one\nevaluation choice changes")
        + labs(tag=tag)
        + paper_theme(dark)
        + theme(panel_grid=element_blank())
    )


# Composition ------------------------------------------------------------------------------------------


def evaluation_practice_figure(
    census: Path, ranking: Path, *, dark: bool = False, scope_audit: Path = SCOPE_AUDIT
) -> Any:
    """Draw the figure and return the Matplotlib figure.

    Plotnine gives every row of a composition the same panel height, so panel a cannot be a
    (shorter) spacer row. Instead, panels b and c reserve a top margin exactly as tall as the
    workflow, and the workflow is drawn into it after the Plotnine layout has run.
    """

    decay = pd.read_csv(census / "consensus-decay.csv")
    agreement = pd.read_csv(census / "single-choice-agreement.csv")
    disagreement = pd.read_csv(ranking / "task-protocol-disagreement.csv")
    factors = pd.read_csv(ranking / "factor-sensitivity.csv")
    numbers = workflow_numbers(census, ranking, scope_audit)
    workflow_pt = workflow_height_pt(workflow_boxes(numbers)) + 2 * WORKFLOW_MARGIN_PT
    # Plotnine 0.15 measures plot margins as a fraction of the figure width.
    reserve = theme(plot_margin_top=(WORKFLOW_TOP_PT + workflow_pt + WORKFLOW_GAP_PT) / (TEXT_WIDTH_IN * 72))
    reserve += legend_band()
    census_row = (single_choice_panel(agreement, decay, dark=dark, tag="B") + reserve) | (
        combined_panel(decay, dark=dark, tag="C") + reserve
    )
    ranking_row = disagreement_panel(disagreement, dark=dark, tag="D") | factor_panel(factors, dark=dark, tag="E")
    # Every plot gets the figure size: Plotnine converts plot margins with each plot's own size.
    composition = (census_row / ranking_row) & theme(figure_size=(TEXT_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=300)
    figure = composition.draw()
    figure.set_facecolor(colors(dark)["background"])
    height_pt = FIGURE_HEIGHT_IN * 72
    bottom = 1 - (WORKFLOW_TOP_PT + workflow_pt) / height_pt
    draw_workflow(figure, (0.0, bottom, 1.0, workflow_pt / height_pt), numbers, dark=dark)
    add_task_legend(figure, task_order(decay["group"]), dark=dark)
    return figure


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(census: Path, ranking: Path, output: Path, scope_audit: Path = SCOPE_AUDIT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for dark in (False, True):
            figure = evaluation_practice_figure(census, ranking, dark=dark, scope_audit=scope_audit)
            for suffix in ("png", "pdf", "svg"):
                figure.savefig(
                    output / f"evaluation-practice-{'dark' if dark else 'light'}.{suffix}",
                    dpi=300,
                    facecolor=figure.get_facecolor(),
                )
    inputs = [
        *(
            census / name
            for name in ("corpus-summary.csv", "task-summary.csv", "consensus-decay.csv", "single-choice-agreement.csv")
        ),
        *(ranking / name for name in ("winner-shares.csv", "task-protocol-disagreement.csv", "factor-sensitivity.csv")),
        scope_audit,
    ]
    (output / "figure-manifest.json").write_text(
        json.dumps(
            {
                "figure": "evaluation-practice",
                "numbers": workflow_numbers(census, ranking, scope_audit),
                "min_factor_contrasts": MIN_FACTOR_CONTRASTS,
                "inputs_sha256": {path.name: _sha256(path) for path in inputs},
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--census", type=Path, default=Path("figures/protocol-diversity/source-data"))
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.census, args.ranking, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
