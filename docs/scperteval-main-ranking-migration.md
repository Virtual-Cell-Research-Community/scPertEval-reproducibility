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

## Result-generation boundary

The checked-in ground-truth coverage manifest and the current authoritative ranking score tree
were generated with scPertEval commit `3950557`. They remain intact as the pre-migration evidence
until the migrated workflow has been recomputed into staging, compared, and explicitly promoted.
They must not be described as results from the new evaluator.

The maintained DRF/DEG protocol-behavior workflow is a separate workstream. The frozen
`analyses/scperteval-paper-v1` and protocol-behavior `production-v1` environments are historical
reproductions and retain their original dependency pins.
