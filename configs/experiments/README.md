# Experiment declarations

An experiment declaration freezes training-affecting choices:

- dataset and dataset-manifest checksum;
- split and split-manifest checksum;
- participating model declarations;
- required prediction capability and population-size policy;
- training seeds; and
- the native-output-to-evaluation-space conversion contract.

The post-training evaluation grid is separately versioned. It may be extended
from saved predictions without retraining, but the complete grid must be
reported; variants are not split into privileged primary and inferior
exploratory groups. A named reference variant is retained only as an anchor for
describing rank changes.

The frozen population-size policy is `match-observed`: for each held-out
condition, population-capable models archive the same number of predictions as
there are observed treated cells. Control-transforming models may sample
held-out control cells with replacement to reach that condition-specific size.

Large canonical inputs may have source-specific annotation names even when the
generated model view follows the standard contract. A sibling
`*-metadata-validation.json` records the source-field mapping and the result of
the backed obs/var-only validation performed before expensive materialization.

