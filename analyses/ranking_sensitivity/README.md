# Ranking-sensitivity analysis

This package renders the model-ranking sensitivity figures from versioned tidy
results produced by the production benchmark. It does not train models, load
checkpoints, or recompute prediction bundles.

From the repository root:

```bash
python -m analyses.ranking_sensitivity.figure \
  --results results/ranking-sensitivity-v1 \
  --output results/ranking-sensitivity-v1/figures
```

The frozen original-paper reproduction remains under
`analyses/scperteval-paper-v1/`; new benchmark figures must not be added there.
# Ranking-sensitivity figures

This package renders the benchmark ranking-sensitivity analysis from retained
predictions and tidy evaluation results. It is intentionally separate from the
frozen original-paper reproduction in `analyses/scperteval-paper-v1/` and from
the DRF/DE protocol-behavior workflow.

From the repository root:

```bash
python -m analyses.ranking_sensitivity.figure \
  --results results/ranking-sensitivity-v1 \
  --output figures/ranking-sensitivity-v1
```

The main figure combines cluster-bootstrap reversal intervals with examples
selected by the rule frozen in `configs/evaluation/ranking-sensitivity-v1.json`.
The supplement contains every task and evaluation-choice slope plot.
