"""Shared Plotnine style and colours for every manuscript figure.

All figures are authored at the manuscript text width so that LaTeX never
rescales them; font sizes below are therefore final printed sizes in points.

Colour encodes the prediction task consistently across figures: hue is the
intervention modality and shade is the target-context data (dark few-shot,
light zero-shot). The palette is Okabe–Ito; the light shades fall below 3:1
contrast on white, so they are always paired with a second encoding (line type
or a direct label) and text is never drawn in a series colour.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from plotnine import element_blank, element_line, element_rect, element_text, theme, theme_minimal

from analyses.vocabulary import BENCHMARK_TASKS, CENSUS_TASKS

TEXT_WIDTH_IN = 16.5 / 2.54
"""MetricHub ``main.tex`` text width (16.5 cm)."""

BASE_SIZE = 7.0
TAG_SIZE = 9.0
SMALL_SIZE = 6.0

LIGHT = {"background": "#FFFFFF", "foreground": "#171717", "grid": "#E3E5E8", "muted": "#68707D"}
DARK = {"background": "#0E1117", "foreground": "#F3F4F6", "grid": "#343A46", "muted": "#AAB1BC"}

TASK_COLORS = {
    "Genetic single, Few-shot": "#0072B2",
    "Genetic single, Zero-shot": "#56B4E9",
    "Genetic combination, Few-shot": "#CC79A7",
    "Genetic combination, Zero-shot": "#CC79A7",
    "Chemical single, Few-shot": "#D55E00",
    "Chemical single, Zero-shot": "#E69F00",
    "Biological single, Few-shot": "#009E73",
    "Biological single, Zero-shot": "#009E73",
}
"""Census task → colour. Combination and biological tasks occur with one target context only."""

TARGET_CONTEXT_LINETYPES = {"Few-shot": "solid", "Zero-shot": "dashed"}

BENCHMARK_TASK_COLORS = {task: TASK_COLORS[CENSUS_TASKS[task]] for task in BENCHMARK_TASKS}
"""Benchmark task id → the colour of the census task it belongs to."""


def colors(dark: bool = False) -> dict[str, str]:
    return DARK if dark else LIGHT


@cache
def font_family() -> str:
    """Arial for print; DejaVu Sans where Arial is not installed (e.g. Linux servers)."""

    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    return "Arial" if "Arial" in installed else "DejaVu Sans"


def paper_theme(dark: bool = False, base_size: float = BASE_SIZE) -> Any:
    """Base theme: no axis lines or ticks, recessive major grid, bold lower-case panel tags."""

    c = colors(dark)
    return theme_minimal(base_family=font_family(), base_size=base_size) + theme(
        text=element_text(color=c["foreground"]),
        axis_text=element_text(color=c["foreground"], size=base_size - 0.5),
        axis_title=element_text(color=c["foreground"], size=base_size),
        axis_ticks=element_blank(),
        panel_grid_major=element_line(color=c["grid"], size=0.35),
        panel_grid_minor=element_blank(),
        panel_background=element_rect(fill=c["background"], color=c["background"]),
        plot_background=element_rect(fill=c["background"], color=c["background"]),
        plot_tag=element_text(size=TAG_SIZE, weight="bold", color=c["foreground"]),
        plot_tag_location="plot",
        plot_tag_position="topleft",
        legend_background=element_rect(fill=c["background"], color=c["background"]),
        legend_key=element_rect(fill=c["background"], color=c["background"]),
        legend_text=element_text(color=c["foreground"], size=SMALL_SIZE),
        legend_title=element_text(color=c["foreground"], size=base_size - 0.5),
        svg_usefonts=True,
    )


def percent(values) -> list[str]:
    return [f"{value:.0%}" for value in values]
