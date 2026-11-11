# Experiment declarations

An experiment declaration freezes training-affecting choices:

- dataset and dataset-manifest checksum;
- split and split-manifest checksum;
- participating model declarations;
- required prediction capability and archive size;
- training seeds; and
- the native-output-to-evaluation-space conversion contract.

The post-training evaluation grid is separately versioned. It may be extended
from saved predictions without retraining, but the complete grid must be
reported; variants are not split into privileged primary and inferior
exploratory groups. A named reference variant is retained only as an anchor for
describing rank changes.

