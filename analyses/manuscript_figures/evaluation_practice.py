"""Fig. 2: evaluation practice differs across papers, and the differences change model rankings.

Composes the Atlas census (protocol diversity) and the ranking-sensitivity
analysis into one manuscript figure. Reads only the two analyses' exported
source data; nothing is recomputed here.

    a  workflow: corpus → taxonomy annotation → census (b, c) and scPertEval → ranking sensitivity (d, e)
    b  census: paper pairs sharing a protocol as evaluation choices are added
    c  census: paper pairs agreeing on each evaluation choice alone
    d  ranking: protocol pairs selecting different winning models, per benchmark task
    e  ranking: winner-change rate when one evaluation choice changes

From the repository root:

    uv run python -m analyses.manuscript_figures.evaluation_practice \\
      --ranking /path/to/figures/ranking-sensitivity-atlas/source-data
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from plotnine import (
    aes,
    coord_flip,
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
from plotnine.composition import plot_spacer

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
from analyses.protocol_diversity.figure import axis_panel, decay_panel
from analyses.vocabulary import AXIS_LABELS, BENCHMARK_TASKS, RANKING_FACTORS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPE_AUDIT = REPO_ROOT / "configs/evaluation/atlas-protocol-scope-audit.csv"
MIN_FACTOR_CONTRASTS = 10
MIN_CENSUS_TASK_WORKS = 20
DEFAULT_OUTPUT = Path("figures/evaluation-practice")
FIGURE_HEIGHT_IN = 6.3
SEQUENTIAL = ("#F2F0F7", "#54278F")
"""Single-hue sequential ramp (purple) for winner-change rates; distinct from every task hue."""

TASK_TICKS = {task: "\n".join(label.rsplit(" ", 1)) for task, label in BENCHMARK_TASKS.items()}
SHORT_TASK_TICKS = dict(zip(BENCHMARK_TASKS, ["Perturbation", "Genetic", "Chemical"], strict=True))


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


def workflow_boxes(n: dict[str, Any]) -> list[tuple[float, float, str, str]]:
    """(x, y, title, body) in axes coordinates; two branches leave the annotation step."""

    return [
        (
            0.115,
            0.5,
            "Literature corpus",
            (
                f"{n['works']} works\n{n['model_papers']} model papers\n{n['benchmarks']} benchmarks\n"
                f"{n['competitions']} competitions"
            ),
        ),
        (
            0.34,
            0.5,
            "Taxonomy annotation",
            (
                f"Controlled vocabulary\n{n['protocol_records']:,} protocol records\n"
                f"{n['distinct_protocols']} distinct protocols"
            ),
        ),
        (0.605, 0.76, "Census (B, C)", f"Protocol sharing within\n{n['census_tasks']} prediction tasks"),
        (
            0.605,
            0.24,
            "scPertEval",
            f"{n['recurring_protocols']} protocols used in ≥ 2 works\n{n['implemented_protocols']} in scope, implemented",
        ),
        (
            0.87,
            0.24,
            "Ranking sensitivity (D, E)",
            (
                f"{n['benchmark_tasks']} benchmark tasks\n{_span(n['models_per_task'])} models per task\n"
                f"{_span(n['protocols_per_task'])} protocols per task"
            ),
        ),
    ]


def draw_workflow(figure: Any, rect: tuple[float, float, float, float], n: dict[str, Any], *, dark: bool) -> None:
    """Draw the workflow with Matplotlib into ``rect`` (figure coordinates), spanning the full text width."""

    c = colors(dark)
    ax = figure.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    half_w, half_h = 0.105, 0.2
    boxes = workflow_boxes(n)
    text = {"family": font_family(), "color": c["foreground"], "ha": "center"}
    for x, y, title, body in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x - half_w, y - half_h),
                2 * half_w,
                2 * half_h,
                boxstyle="round,pad=0,rounding_size=0.012",
                linewidth=0.6,
                edgecolor=c["muted"],
                facecolor=c["background"],
            )
        )
        ax.text(x, y + half_h - 0.045, title, va="top", fontsize=SMALL_SIZE + 0.5, fontweight="bold", **text)
        ax.text(x, y + half_h - 0.14, body, va="top", fontsize=SMALL_SIZE, linespacing=1.3, **text)
    (
        (corpus_x, corpus_y, *_),
        (annot_x, annot_y, *_),
        (census_x, census_y, *_),
        (impl_x, impl_y, *_),
        (
            rank_x,
            rank_y,
            *_,
        ),
    ) = boxes
    for start, end in [
        ((corpus_x + half_w, corpus_y), (annot_x - half_w, annot_y)),
        ((annot_x + half_w, annot_y + 0.06), (census_x - half_w, census_y)),
        ((annot_x + half_w, annot_y - 0.06), (impl_x - half_w, impl_y)),
        ((impl_x + half_w, impl_y), (rank_x - half_w, rank_y)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6, linewidth=0.6, color=c["muted"]))
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
    rows["tick"] = pd.Categorical(rows["task"].map(TASK_TICKS), [TASK_TICKS[t] for t in reversed(BENCHMARK_TASKS)])
    rows["label"] = [
        f"{rate:.0%} ({int(k):,}/{int(n):,})"
        for rate, k, n in zip(
            rows["winner_disagreement_rate"], rows["n_winner_disagreements"], rows["n_protocol_pairs"], strict=True
        )
    ]
    inside = rows["winner_disagreement_rate"] > 0.55
    rate = rows["winner_disagreement_rate"]
    rows["label_y"] = np.where(inside, rate - 0.02, rate + 0.02)
    rows["label_ha"] = np.where(inside, "right", "left")
    return rows


def disagreement_panel(disagreement: pd.DataFrame, *, dark: bool = False, tag: str | None = None) -> Any:
    """Fraction of all eligible protocol pairs that select different winning models, per task."""

    c = colors(dark)
    rows = disagreement_data(disagreement)
    return (
        ggplot(rows, aes("tick", "winner_disagreement_rate", fill="task"))
        + geom_col(width=0.55, show_legend=False)
        + geom_text(aes(y="label_y", label="label", ha="label_ha"), size=SMALL_SIZE, color=c["foreground"])
        + scale_fill_manual(values=BENCHMARK_TASK_COLORS)
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.5, 1], labels=percent, expand=(0, 0))
        + scale_x_discrete(name="")
        + labs(y="Protocol pairs selecting\ndifferent winning models", tag=tag)
        + coord_flip()
        + paper_theme(dark)
        + theme(panel_grid_major_y=element_blank())
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
    rows["tick"] = pd.Categorical(rows["task"].map(SHORT_TASK_TICKS), list(SHORT_TASK_TICKS.values()))
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
    for task, tick in SHORT_TASK_TICKS.items():
        header = pd.DataFrame({"tick": pd.Categorical([tick], list(SHORT_TASK_TICKS.values())), "y": [header_y]})
        plot += geom_tile(
            header, aes("tick", "y"), height=0.16, width=0.95, fill=BENCHMARK_TASK_COLORS[task], inherit_aes=False
        )
    return (
        plot
        + scale_fill_gradient(low=SEQUENTIAL[0], high=SEQUENTIAL[1], limits=(0, 1), na_value=c["grid"])
        + scale_color_identity()
        + scale_y_discrete(name="")
        + scale_x_discrete(name="Winner-change rate\nper benchmark task")
        + labs(tag=tag)
        + paper_theme(dark)
        + theme(panel_grid=element_blank())
    )


# Composition ------------------------------------------------------------------------------------------


def evaluation_practice_figure(
    census: Path, ranking: Path, *, dark: bool = False, scope_audit: Path = SCOPE_AUDIT
) -> Any:
    """Draw the figure and return the Matplotlib figure (panel a is drawn onto a spacer row)."""

    decay = pd.read_csv(census / "consensus-decay.csv")
    agreement = pd.read_csv(census / "single-choice-agreement.csv")
    disagreement = pd.read_csv(ranking / "task-protocol-disagreement.csv")
    factors = pd.read_csv(ranking / "factor-sensitivity.csv")
    census_row = decay_panel(decay, dark=dark, tag="B") | axis_panel(agreement, dark=dark, tag="C", show_labels=False)
    ranking_row = disagreement_panel(disagreement, dark=dark, tag="D") | factor_panel(factors, dark=dark, tag="E")
    composition = (plot_spacer() / census_row / ranking_row) + theme(
        figure_size=(TEXT_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=300
    )
    figure = composition.draw()
    figure.set_facecolor(colors(dark)["background"])
    spacer, *panels = figure.axes
    spacer.set_visible(False)
    figure.savefig(io.BytesIO(), dpi=300)  # run the Plotnine layout at output resolution to read panel positions
    top = max(ax.get_position().y1 for ax in panels) + 0.035  # leave room for the panel tags of b and c
    numbers = workflow_numbers(census, ranking, scope_audit)
    draw_workflow(figure, (0.0, top, 1.0, 0.995 - top), numbers, dark=dark)
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
    parser.add_argument("--census", type=Path, default=Path("figures/protocol-diversity-v1/source-data"))
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.census, args.ranking, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
