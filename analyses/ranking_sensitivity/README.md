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

## Atlas-derived protocol universe

The literature-eligible protocol universe is generated independently of the
older hand-written pilot grid:

```bash
python -m analyses.ranking_sensitivity.atlas_protocol_audit \
  --atlas-repo /path/to/PerturbationEvaluationAtlas
```

The generator reads the exact Atlas revision pinned in
`configs/evaluation/atlas-protocol-eligibility-v1.json` with `git show`, so a
newer Atlas working tree cannot silently change the audit. It writes the
included universe, the complete scope audit, the separate DE-caller audit, and
the human-readable summary. Reporting aggregation and DE caller are deliberately
not part of atomic-family identity.
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
