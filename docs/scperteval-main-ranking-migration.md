# scPertEval main migration for ranking sensitivity

The live ranking-sensitivity integration targets scPertEval commit
`58ccc91095ccb30bbe8f4470dff8b398d2eb4525`, which contains the merged concrete-protocol,
composable-space, signed-DE, configurable-DE and cache changes from PRs 40–43.

The scientific design remains frozen. The migration changes only how the existing 101 Atlas
families are represented and executed:

- `GenePanel` becomes a composable `spaces` expression;
- target inclusion and exclusion use union and difference with `spaces.own_target()`;
- subset-to-PCA uses `spaces.pca(50).on(subset)`;
- `DEComparison` becomes a pure DE metric with an explicit gene universe and bound metric
  parameters;
- namespaced genetic-context conditions resolve their target through `model_perturbation`;
- completion records bind both the exact scPertEval revision and ranking evaluator source hash.

No model checkpoint or prediction artifact depends on these interfaces. Existing predictions
remain the inputs to the migrated evaluator.

## Pre-production validation

The repository suite passes against the pinned revision. A deterministic synthetic fold was also
scored once with the pre-migration evaluator and once with the migrated evaluator. All 101
protocols completed, and all 404 protocol-by-perturbation rows matched exactly in numerical score
and structured undefined reason. This is a migration check only; authoritative benchmark scores
must still be recomputed from the retained production predictions.

The production-data staging audit was then repeated for all 12 benchmark folds at repository
commit `7c43be7`. It produced the expected 340,075 protocol-by-evaluation-unit rows. The old and
new task-level matched-coverage summaries are identical, including the number of protocol pairs
passing the frozen 75% common-coverage threshold.

The row-level comparison identified one intentional semantic correction. For 19 genetic-context
evaluation units with no genes in the selected truth-derived DEG set, nine set-agreement protocols
now define empty-versus-empty agreement instead of reporting `empty_feature_set`. This changes 171
of 340,075 availability rows from undefined to defined. It implements the frozen convention that
two explicitly empty sets agree, while leaving recall, AUPRC and one-class discrimination undefined
when their ground-truth positives or classes do not exist. No other defined/undefined decisions
changed.

A retained CellFlow production bundle was evaluated as a canary for the essential-screen fold 0.
The migrated evaluator produced the same 27,371 keys, metadata values, defined/undefined decisions
and reasons as the pre-migration evaluator. On Beauty, all 27,340 finite scores agreed within
`4.55e-13`; no score differed by more than `1e-12`. Running the same current code on Beast exposed
host-dependent floating-point behavior in Sinkhorn divergence (maximum absolute difference
`0.00620`, concentrated in PCA/HVG Sinkhorn scores around 7–9). The full production re-evaluation
should therefore run on Beauty only, or otherwise use one homogeneous execution host, so that
near-tied model rankings cannot depend on CPU-specific Sinkhorn numerics.

## Result-generation boundary

The checked-in ground-truth coverage manifest and the current authoritative ranking score tree
were generated with scPertEval commit `3950557`. They remain intact as the pre-migration evidence
until the migrated workflow has been recomputed into staging, compared, and explicitly promoted.
They must not be described as results from the new evaluator.

The maintained DRF/DEG protocol-behavior workflow is a separate workstream. The frozen
`analyses/scperteval-paper-v1` and protocol-behavior `production-v1` environments are historical
reproductions and retain their original dependency pins.
