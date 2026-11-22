# Protocol diversity

Census of evaluation protocols in the Perturbation Evaluation Atlas: how often
papers evaluating the same prediction task share an evaluation protocol, and on
which evaluation choices they disagree. Panels use the shared manuscript style
(`analyses/figure_style.py`) and vocabulary (`analyses/vocabulary.py`).

From the repository root:

```bash
uv run python -m analyses.protocol_diversity.figure \
  --atlas-repo /path/to/PerturbationEvaluationAtlas
```

The Atlas is read with `git show` at the revision pinned in
`configs/evaluation/atlas-protocol-eligibility-v1.json`, the same revision the
ranking-sensitivity analysis uses. `--revision <rev>` overrides it, and
`--revision working-tree` reads uncommitted files (not for publication).

## Definitions

- **Protocol**: one Atlas matrix column (population, feature space, centering,
  feature selection, target-gene policy, weighting, metric, and any
  cross-perturbation suffix).
- **Task**: intervention modality and order × target-context data, assigned per
  protocol-context pair, so a paper counts in every task it evaluates.
  Contexts pooling singletons and combinations of one kind count as
  combinations. Excluded as too sparse: no target-context data, dose or time
  generalization, and contexts mixing intervention kinds. The figure shows tasks
  with at least 20 papers.
- **Pair sharing**: fraction of paper pairs within a task that share at least
  one protocol, recomputed as protocol identity is specified one taxonomy axis
  at a time (census panel a).
- **Single-axis agreement**: pair sharing when papers are compared on one
  evaluation choice alone (census panel b). Unlike the cumulative panel, it does not depend on the
  order in which axes are added.

The task definition is deliberately coarse: split design (e.g. scaffold versus
random holdouts) is recorded only as free text in the Atlas. Finer tasks need
not raise agreement: restricted to one dataset, pair sharing stays at 2–29%
(supplementary table).

## Outputs

Written to `figures/protocol-diversity-v1/` (override with `--output`):

- `source-data/`: `task-summary.csv`, `consensus-decay.csv`,
  `single-choice-agreement.csv`, `rank-frequency.csv`, `supplementary-table.csv`
- `supplementary-table.tex` (tasks and their datasets with ≥ 10 papers)
- `analysis-manifest.json` (Atlas revision and SHA-256 of every Atlas input)
- `protocol-diversity-census-{light,dark}.{png,pdf,svg}` and
  `protocol-diversity-rank-frequency-{light,dark}.{png,pdf,svg}`
