# Protocol-behavior analyses

This directory contains the maintainable successor to the original paper analyses in
`analyses/scperteval-paper-v1/`. Run or resume the complete production workflow from the repository
root with:

```bash
uv run prepare-protocol-data
uv run reproduce-protocol-behavior
```

The first command acquires and validates the six public source datasets, constructs their canonical
raw-count objects, and materializes the three frozen analysis views. It writes one checksummed
preparation record under `models/data/`. Use `uv run prepare-protocol-data --plan` to inspect every
resolved command without downloading or processing data.

The production manifest is `configs/protocol_behavior/production-v1.json`. It declares eight
analysis units: GWPS K562, four genetic contexts, and three SciPlex3 contexts. For every unit the
workflow computes the six full-cell DE combinations (three methods × two references), 20 CPU DRF
protocols, two GPU Sinkhorn protocols, and per-protocol timings. It then renders the mean/median DRF tables,
timing table, standard and conservative DEG-set agreement figures, and corresponding concordance
figures. The Jaccard source tables retain every heatmap cell, median selected-gene count, and
per-perturbation adjusted-p ranking correlation. Concordance@k is top-k set overlap divided by k;
it does not require matching effect directions or identical within-set ranks. The curves use the
legacy fractional-anchor grid and report median per-perturbation AUCC.

Every expensive stage has a completion record tied to the manifest checksum and output checksums.
A repeated command validates and skips completed stages before loading the dataset. The final
`completion.json` records checksums for all rendered artifacts.

On Helix, use the chunked and resumable Slurm submission wrapper:

```bash
bash scripts/submit_protocol_behavior_helix.sh
```

It first creates the locked environment once, then submits 56 CPU and eight GPU compute stages as
bounded arrays and a dependent render/validation job. The array tasks call the same command through its internal stage selector;
there is no separate scientific implementation for cluster runs.

An isolated checkout can reuse an existing prepared-data tree without copying it by setting
`SCPERTEVAL_DATA_ROOT` to the repository root that contains `models/data/`. The Helix launcher
defaults this to the shared primary checkout.
