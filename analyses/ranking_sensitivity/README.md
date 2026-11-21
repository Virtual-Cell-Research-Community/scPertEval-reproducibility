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

The current analysis policy is
`configs/evaluation/ranking-sensitivity-atlas-v1.json`. Before any model ranks
are inspected, produce a complete ground-truth availability table with columns
`task`, `fold`, `protocol_id`, `evaluation_unit`, `ground_truth_defined`, and
`undefined_reason`. Each fold is truth-self-evaluated through the exact concrete
protocols built by `atlas_runtime.py`; no model predictions are read. For example:

```bash
python -m analyses.ranking_sensitivity.truth_coverage_runtime \
  --task essential_perturbation_generalization \
  --truth models/data/replogle-weissman-2022-k562-essential/raw.h5ad \
  --split-manifest models/data/replogle-weissman-2022-k562-essential/split_manifest.json \
  --fold 0 --fold-label fold-0 \
  --out results/ground-truth-coverage/fold-0.csv
```

Concatenate the fold CSVs without changing rows, then run:

```bash
python -m analyses.ranking_sensitivity.ground_truth_coverage \
  --availability results/ground-truth-coverage/availability.csv \
  --output-dir results/ground-truth-coverage
```

The command rejects model or score columns. It reports protocol coverage and
the matched common set for every unordered protocol pair. The candidate 90%
headline threshold remains explicitly unfrozen until these results are reviewed.
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
