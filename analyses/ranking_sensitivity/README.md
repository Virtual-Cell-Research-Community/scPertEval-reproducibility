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
`configs/evaluation/atlas-protocol-eligibility.json` with `git show`, so a
newer Atlas working tree cannot silently change the audit. It writes the
included universe, the complete scope audit, the separate DE-caller audit, and
the human-readable summary. Reporting aggregation and DE caller are deliberately
not part of atomic-family identity.

The universe has 111 families (111, 111 and 104 per task). It includes ten
retrieval-rank families (Atlas `cross_perturbation_group == identity_matching`),
which were added post hoc to the prespecified 101 families; see
`docs/ranking-sensitivity-atlas-scope-change.md` for the disclosure.
Retrieval ranks are dataset-scope (the candidate set is every evaluation unit of
the fold) and are implemented in `retrieval.py`.

The current analysis policy is
`configs/evaluation/ranking-sensitivity-atlas.json`. Before any model ranks
are inspected, produce a complete ground-truth availability table with columns
`task`, `fold`, `protocol_id`, `evaluation_unit`, `ground_truth_defined`, and
`undefined_reason`. Each fold is truth-self-evaluated through the exact concrete
protocols built by `atlas_runtime.py`; no model predictions are read. For example:

The live builder is pinned to the exact scPertEval revision recorded under
`evaluation_engine`. See `docs/scperteval-main-ranking-migration.md` for the API migration and
the boundary between retained predictions, stale pre-migration scores, and recomputed outputs.

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
the matched common set for every unordered protocol pair. The headline rule is
frozen at 75% shared coverage in every fold, selected from the ground-truth-only
audit before model ranks were inspected.
The completed audit and its checksummed SDS result bundle are summarized in
`docs/ground-truth-coverage-audit.md` and
`configs/evaluation/atlas-ground-truth-coverage-audit.json`.

## Retained-prediction scoring

Production scoring is fold-scoped so truth-derived DE results, feature panels,
and PCA bases are prepared once and reused across all models in the fold. For
example:

```bash
python scripts/evaluate_atlas_production_benchmark.py \
  --project-root /path/to/scPertEval-reproducibility \
  --model-data-root /fast/local/cache/models/data \
  --runs-root /shared/sds/scPertEval-reproducibility/runs \
  --results-root /path/to/results/ranking-sensitivity-atlas \
  --track chemical-context \
  --fold holdout-a549 \
  --workers 16
```

The evaluator reads only completed retained prediction bundles. It prefers a
frozen control-residual population lift when present and otherwise uses the
native population prediction. Centroid protocols therefore preserve the exact
point prediction while population and DE protocols receive matched cell
populations. Each model/fold output has a checksum-bound completion record and
is skipped safely on restart.
The optional data and runs roots keep large truth reads on a workstation-local
cache while predictions and authoritative results remain on SDS.

## Atlas production analysis

After all checksum-bound detail bundles and the independent ground-truth
coverage audit are complete, assemble the canonical score table and calculate
the reference-free headline summaries:

```bash
python -m analyses.ranking_sensitivity.atlas_analysis \
  --details-root results/ranking-sensitivity-atlas-v1/details \
  --details-root results/ranking-sensitivity-atlas-v2/details \
  --availability results/ground-truth-coverage-v2/availability.csv \
  --pair-coverage results/ground-truth-coverage-v2/pairwise-coverage-summary.csv \
  --output results/ranking-sensitivity-atlas-v2/analysis-v2
```

The SDS directory names are historical. The 101 prespecified protocols were
scored into `ranking-sensitivity-atlas-v1/details` and the ten retrieval
protocols into `ranking-sensitivity-atlas-v2/details`; `--details-root` is
repeatable and merges disjoint protocol sets only when every bundle binds the
same prediction checksum. A fresh run of the scorer writes all 111 protocols
into one root.

The analysis keeps the three tasks separate. Winner shares use the standard
mean reduction, equal protocol votes, and fractional credit for ties. Every
unordered protocol pair is compared on the ground-truth-defined evaluation-unit
intersection within each fold and must meet the frozen 75% minimum-fold
coverage rule. Factor sensitivity separates evaluation family (profile,
population, or DE), score function within a fixed family and space, and
expression space. Cross-perturbation evaluation compares direct comparison with
retrieval under the same distance and every other choice fixed. An evaluation-family transition may necessarily change to a
compatible score function; all other factors are strict one-factor contrasts.
Mean-versus-median aggregation is evaluated separately. The main figure shows
only task-level factor estimates supported by at least 10 matched contrasts;
the complete source table retains sparser comparisons for audit.

Render all global and selected bump figures in both light and dark mode:

```bash
python -m analyses.ranking_sensitivity.atlas_figures \
  --source-data results/ranking-sensitivity-atlas-v2/analysis-v2/source-data \
  --output results/ranking-sensitivity-atlas-v2/analysis-v2/figures
```

The rendering layer uses Plotnine and fixes the global figure at a 7.2-inch
manuscript width. The light PDF is the publication asset; the dark variant is
intended for presentations. Panel tags are rendered in the figure, whereas
descriptive panel titles and all methodological detail belong in the caption.
The plotting code performs display-only transformations and never recomputes
scores, ranks, winners or protocol eligibility.

Every figure reads an exported CSV in `source-data/`. The compressed
`canonical-scores.csv.gz` preserves the complete validated long-form analysis
input without duplicating it across figure-specific tables.
The four post-hoc illustrative examples are frozen with explicit rationales in
`configs/evaluation/ranking-sensitivity-atlas-examples.json`; the complete
ranked candidate table remains available as `bump-plot-candidates.csv`.

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
