# Manuscript figures

Composite figures for the Metrics Paper, built only from the exported source
data of the underlying analyses. Shared style, colours and labels come from
`analyses/figure_style.py` and `analyses/vocabulary.py`.

## Fig. 2: evaluation practice (`evaluation_practice.py`)

Evaluation practice differs across papers, and the differences change model
rankings.

| Panel | Content | Source data |
|---|---|---|
| a | Workflow: literature corpus → taxonomy annotation → census, and → scPertEval implementation → ranking sensitivity | `corpus-summary.csv`, `task-summary.csv`, `winner-shares.csv`, `configs/evaluation/atlas-protocol-scope-audit-v2.csv` (`--scope-audit` selects v1) |
| b | Paper pairs sharing ≥ 1 protocol as evaluation choices are added (top to bottom) | `consensus-decay.csv` |
| c | Paper pairs agreeing on each evaluation choice alone (same rows as b) | `single-choice-agreement.csv` |
| d | Protocol pairs selecting different winning models, per benchmark task | `task-protocol-disagreement.csv` |
| e | Winner-change rate when one evaluation choice changes (cells with < 10 matched contrasts left empty) | `factor-sensitivity.csv` |

Regenerate the census source data first (`analyses/protocol_diversity`), then,
from the repository root:

```bash
uv run python -m analyses.manuscript_figures.evaluation_practice \
  --ranking /path/to/figures/ranking-sensitivity-atlas-v2/source-data
```

v2 (111 protocols, including retrieval ranks) is the primary ranking analysis. For the v1
supplement pass `--ranking …/ranking-sensitivity-atlas-v1/source-data --scope-audit
configs/evaluation/atlas-protocol-scope-audit-v1.csv`; a mismatched pair is rejected.

Writes `figures/evaluation-practice-v1/evaluation-practice-{light,dark}.{png,pdf,svg}`
and `figure-manifest.json` (the workflow numbers and SHA-256 of every input).
The light PDF is the manuscript asset.

Panel a is drawn with Matplotlib into the empty top row after the Plotnine
layout has run, so that the workflow can use the full text width instead of
being aligned to the panels below.
