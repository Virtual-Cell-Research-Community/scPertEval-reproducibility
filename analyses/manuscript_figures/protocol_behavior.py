"""Manuscript figures for the protocol-behavior analysis (production v2).

Renders publication versions of the DRF table (main text), the DEG-method
comparison (supplement) and the runtime table (supplement) from the v2 source
CSVs written by ``analyses/protocol_behavior`` (``figures/`` of the production-v2
result directory). Nothing is recomputed here.

The configuration columns of the DRF table are derived from the concrete
scPertEval protocols and scPertEval's default calibration controls, so they cannot
drift from what was evaluated.

From the repository root:

    uv run python -m analyses.manuscript_figures.protocol_behavior \\
      --source /path/to/production-v2/figures --output figures/protocol-behavior-manuscript
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_cartesian,
    element_blank,
    element_rect,
    element_text,
    facet_wrap,
    geom_line,
    geom_ribbon,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_identity,
    scale_fill_gradient,
    scale_fill_gradient2,
    scale_fill_manual,
    scale_x_discrete,
    scale_x_log10,
    scale_y_continuous,
    scale_y_discrete,
    theme,
    theme_void,
)

from analyses.figure_style import SMALL_SIZE, TASK_COLORS, TEXT_WIDTH_IN, colors, font_family, paper_theme

UNITS = [
    "Perturbation K562 essential",
    "Genetic K562 essential",
    "Genetic RPE1",
    "Genetic HepG2",
    "Genetic Jurkat",
    "SciPlex3 A549",
    "SciPlex3 K562",
    "SciPlex3 MCF7",
]
UNIT_LABELS = {
    "Perturbation K562 essential": "K562",
    "Genetic K562 essential": "K562 ",
    "Genetic RPE1": "RPE1",
    "Genetic HepG2": "HepG2",
    "Genetic Jurkat": "Jurkat",
    "SciPlex3 A549": "A549",
    "SciPlex3 K562": "K562  ",
    "SciPlex3 MCF7": "MCF7",
}
"""Cell line per analysis unit; the colour strip above each column gives its benchmark task (Fig. 2)."""
FACET_LABELS = {
    "Perturbation K562 essential": "K562 (perturbation)",
    "Genetic K562 essential": "K562 (genetic context)",
    "Genetic RPE1": "RPE1 (genetic context)",
    "Genetic HepG2": "HepG2 (genetic context)",
    "Genetic Jurkat": "Jurkat (genetic context)",
    "SciPlex3 A549": "A549 (chemical context)",
    "SciPlex3 K562": "K562 (chemical context)",
    "SciPlex3 MCF7": "MCF7 (chemical context)",
}
UNIT_COLORS = {
    unit: TASK_COLORS[
        "Genetic single, Few-shot"
        if unit.startswith("Perturbation")
        else "Genetic single, Zero-shot"
        if unit.startswith("Genetic")
        else "Chemical single, Zero-shot"
    ]
    for unit in UNITS
}
"""Each analysis unit takes the colour of the benchmark task it comes from (Fig. 2)."""

PROTOCOLS = [
    "pearson",
    "pearson_ctrl",
    "pearson_pert",
    "pearson_pert_degs_padj=0.05",
    "pearson_pert_top_k=50",
    "mse",
    "mse_degs_padj=0.05",
    "mse_top_k=50",
    "wmse_exp1",
    "wmse_exp2",
    "wmse_exp4",
    "transpose_rank",
    "rank",
    "de_overlap_k=50",
    "de_auprc",
    "de_auroc",
    "unbiased_mmd_median_top_k=50",
    "unbiased_mmd_median_pca_k=50",
    "energy_distance_top_k=50",
    "energy_distance_pca_k=50",
    "sinkhorn_w2_top_k=50",
    "sinkhorn_w2_pca_k=50",
]
BASE_METRICS = {
    "pearson": "Pearson",
    "mse": "MSE",
    "wmse": "WMSE",
    "transpose_rank": "Transposed rank",
    "rank": "Rank",
    "de_overlap": "DE overlap@50",
    "de_auprc": "DE AUPRC",
    "de_auroc": "DE AUROC",
    "unbiased_mmd_median": "MMD",
    "energy_distance": "Energy distance",
    "sinkhorn_w2": "$W_2$ (Sinkhorn)",
}
REPRESENTATIONS = {"centroid": "Pseudobulk", "population": "Single-cell", "de": "DE"}
CENTERING = {None: "–", "control_mean": "Ctrl", "all_perturbed_mean": "Pert."}
CONTROLS = {
    "interpolated": "Interp. TD",
    "tech_dup": "TD",
    "all_perturbed_mean": "Pert. mean",
    "control_mean": "Ctrl mean",
    "global_mean": "Global mean",
    "all_perturbed": "Pert. cells",
}
"""Abbreviated control names; the caption expands TD (technical duplicate), Ctrl and Pert."""
SPACES = {"full": "All", "degs_0.05": "DEGs", "top_50": "Top-50", "pca_50": "PCA-50"}
CONFIG_COLUMNS = [
    ("metric", "Metric", 1.25),
    ("centering", "Cen-\ntering", 0.6),
    ("representation", "Represen-\ntation", 0.95),
    ("positive", "Positive\nreference", 0.85),
    ("negative", "Negative\nreference", 0.9),
    ("space", "Feature\nspace", 0.7),
    ("weight", "Weight\nexp.", 0.5),
]
DIVERGING = ("#D55E00", "#F2F2F2", "#0072B2")
SEQUENTIAL = ("#F2F0F7", "#54278F")


# Protocol configuration -------------------------------------------------------------------------------


def protocol_configuration(names: list[str] = PROTOCOLS) -> pd.DataFrame:
    """Reader-facing configuration of each protocol, from scPertEval's concrete protocols and controls."""

    from scperteval.protocols import PROTOCOLS as REGISTRY
    from scperteval.runner import GENERIC_CONTROLS

    rows = []
    for name in names:
        if "=" in name:
            family, value = name.split("=", 1)
            protocol = REGISTRY.get(family).resolve(REGISTRY.get(family).param.cast(value))
        else:
            protocol = REGISTRY.get(name)
            if protocol.parameterised:
                protocol = protocol.resolve(protocol.param.default)
        positive, negative = GENERIC_CONTROLS[protocol.representation]
        base = next(key for key in sorted(BASE_METRICS, key=len, reverse=True) if name.startswith(key))
        rows.append(
            {
                "protocol": name,
                "metric": BASE_METRICS[base],
                "centering": CENTERING[protocol.centering],
                "representation": REPRESENTATIONS[protocol.representation],
                "positive": CONTROLS[protocol.default_positive or positive],
                "negative": CONTROLS[protocol.default_negative or negative],
                "space": SPACES.get(protocol.space, protocol.space),
                "weight": name.rsplit("_exp", 1)[1] if name.startswith("wmse_exp") else "–",
            }
        )
    return pd.DataFrame(rows)


# Generic table renderer -------------------------------------------------------------------------------


def _table_layout(config: pd.DataFrame, value_columns: list[str], value_width: float = 0.62):
    """x centres and widths for configuration columns followed by value columns."""

    widths = [w for _, _, w in CONFIG_COLUMNS] + [value_width] * len(value_columns)
    edges = np.concatenate([[0], np.cumsum(widths)])
    centres = (edges[:-1] + edges[1:]) / 2
    keys = [k for k, _, _ in CONFIG_COLUMNS] + value_columns
    return dict(zip(keys, zip(centres, widths, strict=True), strict=True)), edges[-1]


def table_plot(
    config: pd.DataFrame,
    values: pd.DataFrame,
    *,
    value_labels: dict[str, str],
    fill_scale: Any,
    label: Any,
    ink_threshold: float,
    fill_transform=lambda v: v,
    dark: bool = False,
) -> Any:
    """A table with text configuration columns and coloured, labelled value cells."""

    c = colors(dark)
    value_columns = [col for col in value_labels if col in values.columns]
    layout, total_width = _table_layout(config, value_columns)
    n = len(config)
    y_of = {p: n - i for i, p in enumerate(config["protocol"])}

    text_cells = [
        {"x": layout[key][0], "y": y_of[row.protocol], "label": getattr(row, key)}
        for row in config.itertuples()
        for key, _, _ in CONFIG_COLUMNS
    ]
    value_frame = values[values["protocol"].isin(y_of)].melt(
        id_vars="protocol", value_vars=value_columns, var_name="column", value_name="value"
    )
    value_frame["x"] = value_frame["column"].map(lambda col: layout[col][0])
    value_frame["w"] = value_frame["column"].map(lambda col: layout[col][1])
    value_frame["y"] = value_frame["protocol"].map(y_of)
    value_frame["fill"] = fill_transform(value_frame["value"].astype(float))
    value_frame["label"] = value_frame["value"].map(label)
    value_frame["ink"] = np.where(np.abs(value_frame["fill"]) > ink_threshold, "#FFFFFF", c["foreground"])
    headers = pd.DataFrame(
        [{"x": layout[k][0], "label": title} for k, title, _ in CONFIG_COLUMNS]
        + [{"x": layout[col][0], "label": value_labels[col]} for col in value_columns]
    )
    stripes = pd.DataFrame({"y": [y for y in range(1, n + 1) if y % 2 == 0]})
    strip_cells = [
        {"x": layout[col][0], "w": layout[col][1], "color": UNIT_COLORS[col]}
        for col in value_columns
        if col in UNIT_COLORS
    ]

    return (
        ggplot()
        + geom_tile(stripes, aes(x=total_width / 2, y="y"), width=total_width, height=1, fill=c["grid"], alpha=0.35)
        + geom_tile(value_frame, aes("x", "y", width="w", fill="fill"), height=0.92, color=c["background"])
        + geom_text(value_frame, aes("x", "y", label="label", color="ink"), size=SMALL_SIZE - 0.5)
        + geom_text(
            pd.DataFrame(text_cells), aes("x", "y", label="label"), size=SMALL_SIZE - 0.5, color=c["foreground"]
        )
        + geom_text(
            headers,
            aes("x", n + 0.95, label="label"),
            size=SMALL_SIZE - 0.5,
            fontweight="bold",
            va="bottom",
            lineheight=1.0,
            color=c["foreground"],
        )
        + [
            geom_tile(pd.DataFrame([cell]), aes("x", n + 0.72), width=cell["w"] * 0.96, height=0.14, fill=cell["color"])
            for cell in strip_cells
        ]
        + fill_scale
        + scale_color_identity()
        + coord_cartesian(xlim=(0, total_width), ylim=(0.4, n + 2.4), expand=False)
        + theme_void(base_family=font_family())
        + theme(
            legend_position="none",
            plot_background=element_rect(fill=c["background"], color=c["background"]),
        )
    )


def drf_table(source: Path, *, statistic: str = "mean", dark: bool = False) -> Any:
    values = pd.read_csv(source / f"drf_table_{statistic}.csv")
    return table_plot(
        protocol_configuration(),
        values,
        value_labels={u: UNIT_LABELS[u] for u in UNITS},
        fill_scale=scale_fill_gradient2(
            low=DIVERGING[0], mid=DIVERGING[1], high=DIVERGING[2], midpoint=0, limits=(-1, 1)
        ),
        label=lambda v: f"{v:+.2f}",
        ink_threshold=0.55,
        dark=dark,
    ) + theme(figure_size=(TEXT_WIDTH_IN, 4.6))


def timing_table(source: Path, *, dark: bool = False) -> Any:
    values = pd.read_csv(source / "metric_timing_table.csv")
    labels = {u: UNIT_LABELS[u] for u in UNITS} | {"TOTAL": "Total"}
    return table_plot(
        protocol_configuration(),
        values,
        value_labels=labels,
        fill_scale=scale_fill_gradient(low="#FFFFFF", high="#3F4858", limits=(-1, 4)),
        label=lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 100 else f"{v:,.0f}",
        ink_threshold=2.4,
        fill_transform=lambda v: np.log10(v.clip(lower=0.1)),
        dark=dark,
    ) + theme(figure_size=(TEXT_WIDTH_IN, 4.6))


# DEG-method comparison --------------------------------------------------------------------------------


def criterion_label(criterion: str) -> str:
    return (
        criterion.replace("Welch t-test", "t-test")
        .replace("|stat|>", "|t| > ")
        .replace("|delta|>", "|δ| > ")
        .replace("padj<", "padj < ")
        .replace(" |", ", |")
        .replace(" padj", ", padj")
    )


def concordance_panel(source: Path, *, dark: bool = False, tag: str | None = None) -> Any:
    frame = pd.read_csv(source / "deg_concordance.csv")
    frame["unit"] = pd.Categorical(frame["dataset"].map(FACET_LABELS), [FACET_LABELS[u] for u in UNITS])
    frame["color"] = frame["dataset"].map(UNIT_COLORS)
    aucc = frame.groupby("unit", observed=True, as_index=False)["median_aucc"].first()
    aucc["label"] = aucc["median_aucc"].map(lambda v: f"AUCC {v:.2f}")
    return (
        ggplot(frame, aes("k", "median"))
        + geom_ribbon(aes(ymin="p25", ymax="p75", fill="color"), alpha=0.25)
        + geom_line(aes(color="color"), size=0.6)
        + geom_text(
            aucc, aes(x=1.2, y=0.92, label="label"), ha="left", size=SMALL_SIZE - 0.5, color=colors(dark)["foreground"]
        )
        + facet_wrap("unit", nrow=2)
        + scale_color_identity()
        + scale_fill_manual(values={c: c for c in UNIT_COLORS.values()}, guide=None)
        + scale_x_log10(breaks=[1, 10, 100])
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.5, 1])
        + labs(x="Top-k genes", y="Concordance@k\n(t-test vs MWU)", tag=tag)
        + paper_theme(dark)
        + theme(strip_text=element_text(size=SMALL_SIZE), panel_spacing=0.02)
    )


def jaccard_panel(source: Path, *, dark: bool = False, tag: str | None = None) -> Any:
    frame = pd.read_csv(source / "deg_jaccard_with_counts.csv")
    frame = frame[frame["record_type"] == "jaccard"].copy()
    order = list(dict.fromkeys(frame["left_criterion"]))
    labels = [criterion_label(c) for c in order]
    frame["left"] = pd.Categorical(frame["left_criterion"].map(criterion_label), labels[::-1])
    frame["right"] = pd.Categorical(frame["right_criterion"].map(criterion_label), labels)
    frame["unit"] = pd.Categorical(frame["dataset"].map(FACET_LABELS), [FACET_LABELS[u] for u in UNITS])
    return (
        ggplot(frame, aes("right", "left", fill="mean_jaccard"))
        + geom_tile()
        + facet_wrap("unit", nrow=2)
        + scale_fill_gradient(
            low=SEQUENTIAL[0], high=SEQUENTIAL[1], limits=(0, 1), breaks=[0, 0.5, 1], name="Mean\nJaccard"
        )
        + scale_x_discrete(name="")
        + scale_y_discrete(name="")
        + labs(tag=tag)
        + paper_theme(dark)
        + theme(
            panel_grid=element_blank(),
            axis_text_x=element_text(rotation=90, size=SMALL_SIZE - 1.5),
            axis_text_y=element_text(size=SMALL_SIZE - 1.5),
            strip_text=element_text(size=SMALL_SIZE),
            legend_position="right",
            legend_key_height=30,
            legend_key_width=5,
            panel_spacing=0.01,
        )
    )


def deg_figure(source: Path, *, dark: bool = False) -> Any:
    composition = concordance_panel(source, dark=dark, tag="A") / jaccard_panel(source, dark=dark, tag="B")
    return composition + theme(figure_size=(TEXT_WIDTH_IN, 6.3), dpi=300)


def criterion_latex(criterion: str) -> str:
    """LaTeX label for one DEG selection criterion, e.g. ``$t$-test, $|t| > 2$``."""

    method, rule = criterion.replace("Welch t-test", "$t$-test").rsplit(" ", 1)
    rule = (
        rule.replace("|stat|>", r"$|t| > ")
        .replace("|delta|>", r"$|\delta| > ")
        .replace("padj<", r"$p_{\mathrm{adj}} < ")
        + "$"
    )
    return f"{method}, {rule}"


def deg_set_size_latex(source: Path) -> str:
    """Median number of genes selected per criterion and analysis unit, as a LaTeX tabular."""

    frame = pd.read_csv(source / "deg_jaccard_with_counts.csv")
    sizes = frame[frame["record_type"] == "set_size"].pivot(
        index="criterion", columns="dataset", values="median_set_size"
    )
    order = list(dict.fromkeys(frame.loc[frame["record_type"] == "set_size", "criterion"]))
    sizes = sizes.loc[order, UNITS]
    lines = [
        r"\begin{tabular}{l" + "r" * len(UNITS) + "}",
        r"\toprule",
        (
            r" & \multicolumn{1}{c}{Perturbation} & \multicolumn{4}{c}{Genetic context} & "
            r"\multicolumn{3}{c}{Chemical context} \\"
        ),
        r"\cmidrule(lr){2-2} \cmidrule(lr){3-6} \cmidrule(lr){7-9}",
        "Criterion & " + " & ".join(UNIT_LABELS[u].strip() for u in UNITS) + r" \\",
        r"\midrule",
    ]
    for criterion, row in sizes.iterrows():
        cells = " & ".join(f"{v:,.0f}" if v >= 10 else f"{v:g}" for v in row)
        lines.append(rf"{criterion_latex(criterion)} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# Rendering --------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for dark in (False, True):
        mode = "dark" if dark else "light"
        for stem, plot in [
            ("drf-table-mean", drf_table(source, statistic="mean", dark=dark)),
            ("drf-table-median", drf_table(source, statistic="median", dark=dark)),
            ("runtime-table", timing_table(source, dark=dark)),
            ("deg-method-comparison", deg_figure(source, dark=dark)),
        ]:
            for suffix in ("png", "pdf"):
                plot.save(output / f"{stem}-{mode}.{suffix}", dpi=300, verbose=False)
    (output / "deg-set-sizes.tex").write_text(deg_set_size_latex(source))
    protocol_configuration().to_csv(output / "protocol-configuration.csv", index=False)
    inputs = sorted(source.glob("*.csv"))
    (output / "figure-manifest.json").write_text(
        json.dumps({"inputs_sha256": {p.name: _sha256(p) for p in inputs}}, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures/protocol-behavior-manuscript"))
    args = parser.parse_args()
    render(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
