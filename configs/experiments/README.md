# Experiment declarations

An experiment declaration freezes training-affecting choices:

- dataset and dataset-manifest checksum;
- split and split-manifest checksum;
- participating model declarations;
- required prediction capability and population-size policy;
- training seeds; and
- the native-output-to-evaluation-space conversion contract.

The post-training evaluation policy is separately versioned. The current
ranking-sensitivity study derives eligible atomic protocols from the frozen
PerturbationEvaluationAtlas snapshot and compares protocol pairs without a
privileged reference protocol. Saved predictions permit additional evaluations
without retraining, but any extension receives a new versioned declaration and
its complete source table is reported.

The frozen population-size policy is `match-observed`: for each held-out
condition, population-capable models archive the same number of predictions as
there are observed treated cells. Control-transforming models may sample
held-out control cells with replacement to reach that condition-specific size.

Large canonical inputs may have source-specific annotation names even when the
generated model view follows the standard contract. A sibling
`*-metadata-validation.json` records the source-field mapping and the result of
the backed obs/var-only validation performed before expensive materialization.

