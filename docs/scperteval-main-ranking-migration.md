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
and structured undefined reason. This was a migration check only; the authoritative benchmark
scores were subsequently recomputed from the retained production predictions as described below.

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

## Full production evaluation and promotion

The complete migrated evaluation was run on Beauty on 2026-11-22. After scientific review and a
second provenance audit, it was promoted to the canonical SDS tree
`results/ranking-sensitivity-atlas-v1`. All 120 expected bundles are present and current:

- 65 perturbation-generalization bundles;
- 28 genetic-context-transfer bundles;
- 27 chemical-context-transfer bundles.

Together they contain 3,219,043 score rows. Every schema-v2 completion record matches the retained
prediction checksum, manifests, exact scPertEval revision and evaluator source hash. The validation
found no missing or invalid bundles, duplicate score keys, temporary files or logged execution
errors.

The reference-free analysis was regenerated from these details and the migrated ground-truth
coverage audit. The winner identities for all 598 protocol/aggregation combinations, winner shares,
23 factor-sensitivity estimates, 403 concrete contrasts and four selected bump examples are
unchanged from the pre-migration analysis. The only headline-table numerical change is the genetic
task's mean pairwise model-order reversal fraction (`0.325818` to `0.325903`); its winner-disagreement
count remains 922 of 5,050 protocol pairs. The 36 light- and dark-mode figure files and their five
direct source CSVs are mirrored under `figures/ranking-sensitivity-atlas-v1/`.

The migrated ground-truth availability and matched-coverage audit was promoted alongside the score
tree to `results/ground-truth-coverage-v1`. Its canonical audit contains 340,075 availability rows
from all 12 folds. The four root-level coverage summaries were regenerated after promotion so their
recorded input path and checksum refer to the canonical availability file.

The promotion revalidated every result against its retained prediction and manifest, confirmed all
3,219,043 score rows and all 12 analysis source-data checksums, and found no temporary files. At the
user's request, the displaced pre-migration ranking and coverage trees were permanently deleted
after validation rather than archived.

## Result-generation boundary

The ranking score and ground-truth coverage trees of the 111-family universe derive from scPertEval commit
`58ccc91095ccb30bbe8f4470dff8b398d2eb4525`. Completion records, analysis manifests and coverage
summaries bind the relevant input checksums and evaluator provenance. The retained model
predictions were not regenerated or modified during this evaluator migration.

The maintained DRF/DEG protocol-behavior workflow is a separate workstream. The frozen
`analyses/scperteval-paper-v1` and protocol-behavior `production-v1` environments are historical
reproductions and retain their original dependency pins.

## Update to scPertEval `774e600` (2026-11-22)

All analyses now pin scPertEval `774e6001b09f889905c353a8fd4022a36482eb80` (PRs 45–47 on top of
`58ccc91`). Two changes matter for the ranking evaluator:

- `Protocol` now rejects `reference`/`neg_reference` on centroid and population protocols. In
  `58ccc91` that field was read only by the DE view, so it was removed from the profile,
  population and retrieval protocols without changing any score. DE protocols keep
  `reference="control"` and `neg_reference="control"`; every DEG selector keeps an explicit
  `rank_by`, so the new `de_*` default (`significance`) does not apply.
- Significance ranking inside `de_*` metrics now keeps exact ties as dense ranks instead of
  breaking them by gene index, and DE-set metrics whose universe is all positive return `NaN`
  with `single_ground_truth_class`.

Canary on Beauty (essential fold 0, linear baseline): of the 102 families shared with the
previous universe, 100 gave bitwise-identical scores and identical defined/undefined decisions
over 27,642 rows. The two significance-ranked DE AUROC families (`atlas-536de444734a`,
`atlas-e446cff6c7e8`) differed by at most 1.2 × 10⁻⁶ because of the tie-preserving ranks.

The universe was re-derived from Atlas `0b0fca9` at the same time
(`docs/ranking-sensitivity-atlas-scope-change.md`), and every protocol of every bundle was
rescored into one new results root. The protocol-behavior production-v2 generation moved to the
same revision.
