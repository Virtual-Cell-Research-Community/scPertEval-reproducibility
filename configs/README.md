# Configuration

Versioned declarations belong in four groups:

- `datasets/`: source, preparation, eligibility, and expected output metadata;
- `models/`: implementation revision, capability, environment, assets, and
  resource requirements;
- `experiments/`: dataset, split, participating models, prediction contract,
  and evaluation reference pipeline; and
- `evaluation/`: literature-derived protocol eligibility, undefined-score
  semantics, matched-coverage policy, and post-training analysis choices.

Generated paths and machine-specific cluster settings do not belong in these
declarations. Every declaration receives a stable ID and is copied into the
resulting artifact manifest.

