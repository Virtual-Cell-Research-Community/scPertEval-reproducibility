# Prediction bundles

Evaluation consumes prediction bundles, not model checkpoints. A bundle contains
one H5AD and one JSON manifest conforming to
`schemas/prediction-manifest.schema.json`.

## Common H5AD contract

- `X` contains predictions in the expression space declared by the manifest.
- `var_names` are unique gene identifiers and the set matches the prepared
  truth dataset; evaluation may reorder columns but never guesses mappings.
- `obs["perturbation"]` contains the canonical test-condition identifier.
- Every condition required by the split is present exactly once for point
  predictions or at least `minimum_cells_per_condition` times for population
  predictions.
- Conditions outside the declared test split are rejected from scored bundles.
- NaN and infinite values are rejected.

## Point capability

One row per test condition. The row is a predicted expression profile. Point
bundles participate only in metrics whose interface consumes profiles.

## Population capability

One row per predicted cell or sample. For the initial context-transfer study,
the bundle must provide at least 60 rows per test condition. Distributional
evaluation draws matched observed and predicted samples using frozen seeds.

## Archival

The manifest records dataset and split checksums, model and adapter revisions,
environment identity, random seed, command, shape, and artifact checksum. Exact
baseline outputs follow the same contract and are archived with model outputs.

