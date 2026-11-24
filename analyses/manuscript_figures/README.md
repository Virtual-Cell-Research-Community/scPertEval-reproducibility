# Manuscript figures

Composite figures for the Metrics Paper, built only from the exported source
data of the underlying analyses. Shared style, colours and labels come from
`analyses/figure_style.py` and `analyses/vocabulary.py`.

## Fig. 2: evaluation practice (`evaluation_practice.py`)

Evaluation practice differs across papers, and the differences change model
rankings.

| Panel | Content | Source data |
|---|---|---|
| a | Workflow: literature corpus → taxonomy annotation → census, and → scPertEval implementation → ranking sensitivity | `corpus-summary.csv`, `task-summary.csv`, `winner-shares.csv`, `configs/evaluation/atlas-protocol-scope-audit.csv` |
| b | Paper pairs that disagree on each evaluation choice alone, and on the full protocol | `single-choice-agreement.csv`, `consensus-decay.csv` |
| c | Paper pairs that disagree on any of the evaluation choices combined so far (taxonomy order; last row = full protocol) | `consensus-decay.csv` |
| d | Protocol pairs selecting different winning models, one column per benchmark task (named by its census task, in the column order of e) | `task-protocol-disagreement.csv` |
| e | Winner-change rate when one evaluation choice changes (cells with < 10 matched contrasts left empty) | `factor-sensitivity.csv` |

Regenerate the census source data first (`analyses/protocol_diversity`), then,
from the repository root:

```bash
uv run python -m analyses.manuscript_figures.evaluation_practice \
  --ranking /path/to/figures/ranking-sensitivity-atlas/source-data
```

Writes `figures/evaluation-practice/evaluation-practice-{light,dark}.{png,pdf,svg}`
and `figure-manifest.json` (the workflow numbers and SHA-256 of every input).
The light PDF is the manuscript asset.

Panel a is drawn with Matplotlib after the Plotnine layout has run, so that the
workflow can use the full text width instead of being aligned to the panels
below. Plotnine gives every row of a composition the same panel height, so panel
a is not a spacer row: panels b and c reserve a top margin as tall as the
workflow needs. Each workflow box is as tall as its rows, and its numbers are
right-aligned in one column. Panels b and c share one task legend, drawn with
Matplotlib under both panels in a band they keep free at their bottom.

## Supplementary ranking figures (`ranking_supplement.py`)

| Figure | Content | Source data |
|---|---|---|
| `ranking-winner-shares` | Share of protocols won by each model, per benchmark task (tied winners share credit) | `winner-shares.csv` |
| `ranking-level-contrasts` | Winner-change rate for specific switches of one evaluation choice (e.g. 2,000 HVGs ↔ top-50 DEGs): grey bar pooled over tasks, points per task with ≥ 3 matched pairs; switches with ≥ 15 matched pairs over all tasks | `concrete-factor-contrasts.csv` |
| `ranking-bump-examples` | Model ranks under two protocols that differ in one choice, for the four selected examples | `bump-plot-source.csv`, `configs/evaluation/atlas-protocol-eligibility.csv` |

```bash
uv run python -m analyses.manuscript_figures.ranking_supplement \
  --ranking /path/to/figures/ranking-sensitivity-atlas/source-data
```

Writes `figures/ranking-supplement/ranking-{name}-{light,dark}.{png,pdf,svg}` and
`figure-manifest.json`. Benchmark tasks are named by their census task, as in Fig. 2.
`ranking-winner-shares` (tag A) and `ranking-level-contrasts` (tag B) are the two panels of one
supplementary figure; the manuscript stacks the two files so that each keeps its own height.

## Protocol-behavior tables and figures (`protocol_behavior.py`)

| Figure | Manuscript | Source data |
|---|---|---|
| `drf-table-mean` | Main-text table of dynamic range fraction (DRF) per protocol and analysis unit (`tab:drf-empirical`) | `drf_table_mean.csv` |
| `drf-table-median` | The same table with the median over perturbations | `drf_table_median.csv` |
| `deg-method-comparison` | Supplementary figure: (A) concordance at k between Welch t-test and MWU rankings, (B) Jaccard index between DEG sets (`fig:deg-comparison`) | `deg_concordance.csv`, `deg_jaccard_with_counts.csv` |
| `runtime-table` | Supplementary runtime table | `metric_timing_table.csv` |
| `deg-set-sizes.tex` | Supplementary table of median DEG-set sizes (`tab:deg-set-sizes`) | `deg_jaccard_with_counts.csv` |

The source CSVs are the `figures/` folder of the production-v2 result of
`analyses/protocol_behavior`. The configuration columns of the DRF table come from
scPertEval's concrete protocols and default calibration controls, and are also
written to `protocol-configuration.csv`.

```bash
uv run python -m analyses.manuscript_figures.protocol_behavior \
  --source /path/to/production-v2/figures
```

Writes `figures/protocol-behavior-manuscript/{name}-{light,dark}.{png,pdf}`,
`deg-set-sizes.tex`, `protocol-configuration.csv` and `figure-manifest.json`
(SHA-256 of every input). The light PDFs are the manuscript assets.
