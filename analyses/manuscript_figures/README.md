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
| d | Protocol pairs selecting different winning models, per benchmark task | `task-protocol-disagreement.csv` |
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
