# Workflows

The eventual top-level workflow exposes a small user-facing interface:

1. prepare and validate a named dataset;
2. freeze or verify a named split;
3. train and predict a named model/fold;
4. validate and package predictions;
5. evaluate a named post-training protocol; and
6. rebuild named figures from versioned results.

Model-specific commands remain implementation details under `models/`.
Figure-specific scripts remain under `analyses/`. Do not add a pass-through
wrapper for every existing script: the workflow should earn its seam by
validating manifests, enforcing dependencies, and producing complete artifacts.

Before population evaluation, `scripts/lift_production_predictions.py` materializes a
centroid-preserving population for every point model by adding its predicted effect to a shared,
deterministic sample of held-out controls. The same construction is applied to TxPert's released
batch-average-control workflow. Generated `population-lifted.h5ad` bundles sit beside the native
prediction and are preferred automatically by the production evaluation driver.

`workflows.evaluate aggregate` keeps the one-factor-at-a-time design explicit:
protocol variants use the reference mean aggregation, while median and
effect-weighted aggregation use the reference protocol. In addition to tidy
scores and ranks it can write model-level rank changes, Spearman/Kendall rank
agreement, pairwise and winner reversals, and paired perturbation-bootstrap
intervals. Population metrics are compared against reference ranks recomputed
on the same population-capable model set, so changing the competitor set cannot
masquerade as evaluation sensitivity.

