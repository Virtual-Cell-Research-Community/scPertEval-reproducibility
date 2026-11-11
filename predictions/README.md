# Prediction bundles

Evaluation consumes prediction bundles, not model checkpoints. A bundle contains
one H5AD and one JSON manifest conforming to
`schemas/prediction-manifest.schema.json`.

## Common H5AD contract

- `X` contains predictions in the model-native output space declared by the
  manifest. The manifest also declares the deterministic conversion into the
  shared evaluation space.
- `var_names` are unique gene identifiers and the set matches the prepared
  truth dataset; evaluation may reorder columns but never guesses mappings.
- `obs["perturbation"]` contains the canonical evaluation-population identifier.
- Every perturbation required by the split is present exactly once for point
  predictions or at least `minimum_cells_per_condition` times for population
  predictions.
- Perturbations outside the declared test split are rejected from scored bundles.
- NaN and infinite values are rejected.

## Point capability

One row per evaluated perturbation. The row is a predicted expression profile. Point
bundles participate only in metrics whose interface consumes profiles.

## Population capability

One row per genuine predicted cell or sample. Context-transfer artifacts retain
at most 100 rows per evaluation population; distributional evaluation draws 50
matched observed and predicted cells using frozen draw seeds. Point profiles are
never duplicated to manufacture a population.

## Archival

The manifest records dataset and split checksums, model and adapter revisions,
environment identity, training seed, command, native expression space,
evaluation conversion, shape, and artifact checksum. Exact baseline outputs
follow the same contract and are archived with model outputs. Fold/seed
predictions remain separate rather than being averaged into an ensemble.

