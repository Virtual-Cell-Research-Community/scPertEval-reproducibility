# Configuration

Versioned declarations belong in three groups:

- `datasets/`: source, preparation, eligibility, and expected output metadata;
- `models/`: implementation revision, capability, environment, assets, and
  resource requirements;
- `experiments/`: dataset, split, participating models, prediction contract,
  and evaluation reference pipeline.

Generated paths and machine-specific cluster settings do not belong in these
declarations. Every declaration receives a stable ID and is copied into the
resulting artifact manifest.

