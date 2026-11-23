# Web export for the Atlas website

This module writes the census and ranking-sensitivity results as compact JSON for the page
`#/analysis/evaluation-diversity` of the Perturbation Evaluation Atlas website. The website draws
its own charts from these files.

Nothing is recomputed. Every number comes from the exported source data of
`analyses.protocol_diversity` and `analyses.ranking_sensitivity.atlas_analysis`, so the page is a
frozen snapshot that matches the manuscript. To update the page after the analyses change, rerun
those analyses and then this export.

```bash
uv run python -m analyses.web_export \
  --census figures/protocol-diversity \
  --ranking figures/ranking-sensitivity-atlas \
  --out ../PerturbationEvaluationAtlas/public/data/analyses/evaluation-diversity
```

## Files

| File | Contents |
|---|---|
| `evaluation-diversity.json` | **Census:** works per task, and the disagreement between works per evaluation choice, alone and cumulatively in taxonomy order. **Ranking:** per-task winner disagreement, winner-change rate per evaluation choice, winner shares, and specific level switches. The display rules are the manuscript's: at least 20 works per census task, at least 10 matched pairs per choice, and at least 15 (pooled) or 3 (per task) pairs per switch. |
| `protocol-explorer.json` | For every headline protocol: readable values on each evaluation choice, plus each model's score, rank and winner flag under mean and median aggregation. It also lists the matched one-choice pairs with the evaluation choice the analysis attributes them to, and the pair rankings described below. Models carry their manuscript label and colour. |
| `manifest.json` | Atlas and scPertEval pins, this repository's commit, checksums of the census inputs and of the ranking source data, and checksums of the two JSON files. |

## The explorer reproduces the analysis

The analysis compares two protocols on the evaluation units both of them define. For most pairs,
those shared units leave each protocol's ranks and winners unchanged. The explorer can then pair any
two protocols from their own rankings.

For the remaining pairs (1,131 of 18,363), the export writes the ranking computed on the shared
units. Before writing, `verify_explorer` checks every analysed comparison:

- for every protocol pair: whether the winner changes, and how many model pairs swap order;
- for every mean-versus-median comparison: whether the winner changes.

The export fails if any of these differs from `protocol-pair-comparisons.csv` or
`aggregation-comparisons.csv`.

Only headline protocols are exported: 115, 115 and 103 per task. The one chemical protocol below
the coverage rule is left out, together with its 103 excluded pairs.
