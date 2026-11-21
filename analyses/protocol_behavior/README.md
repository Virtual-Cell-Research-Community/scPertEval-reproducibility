# Protocol-behavior analyses

This directory is the maintained successor to the frozen original-paper analysis
in `analyses/scperteval-paper-v1/`. It is deliberately a separate UV project:
the protocol code needs target-versus-reference DE exports from the review-pending
scPertEval `codex/de-extension` revision, while model training and the rest of the
benchmark remain on the stable scPertEval release.

The lock pins the exact scPertEval commit recorded by each production manifest.
A scientific stage refuses to run if the installed revision differs. This lets
Zach's review proceed independently: accepted changes can be adopted later by
updating the v2 manifest and lock before v2 is run, without changing completed v1
results or silently altering the benchmark environment.

From the repository root, inspect or materialize the current v2 data inputs with:

```bash
uv run prepare-protocol-data
uv run reproduce-protocol-behavior --list-stages
```

The first command acquires and validates the five public source datasets,
constructs their canonical raw-count objects, and materializes the current
Essential K562, four-context genetic, and three-context chemical views. It writes
one checksummed preparation record under `models/data/`. Use
`uv run prepare-protocol-data --plan` to inspect every resolved command without
downloading or processing data.

Two production generations are explicit:

- `production-v1.json` is the completed GWPS K562 analysis. Its manifest remains
  byte-identical to the manifest used for the checksummed result tree at
  `results/protocol_behavior/production-v1`; treat both as immutable.
- `production-v2.json` replaces GWPS with the current K562 Essential benchmark
  and retains the four genetic and three SciPlex3 context units. It is planned,
  not completed, and remains pending Zach's feedback on the scPertEval DEG branch.

For every analysis unit the workflow computes six full-cell DE combinations
(three methods × two references), 20 CPU DRF protocols, two GPU Sinkhorn
protocols, and per-protocol timings. It then renders the mean/median DRF tables,
timing table, standard and conservative DEG-set agreement figures, and matching
concordance figures. The Jaccard source tables retain every heatmap cell, median
selected-gene count, and per-perturbation adjusted-p ranking correlation.
Concordance@k is top-k set overlap divided by k; it does not require matching
effect directions or identical within-set ranks. The curves use the legacy
fractional-anchor grid and report median per-perturbation AUCC.

Every expensive stage has a completion record tied to the manifest checksum and output checksums.
A repeated command validates and skips completed stages before loading the dataset. The final
`completion.json` records checksums for all rendered artifacts.

To validate or render the completed v1 result tree explicitly:

```bash
uv run reproduce-protocol-behavior \
  --manifest configs/protocol_behavior/production-v1.json \
  --output-dir results/protocol_behavior/production-v1 \
  --finalize
```

After v2's scPertEval revision is accepted and its inputs are materialized, use
the chunked and resumable Slurm submission wrapper from an OTP-authenticated
Helix session:

```bash
bash scripts/submit_protocol_behavior_helix.sh
```

It first creates the isolated locked environment once, then submits 56 CPU and
eight GPU compute stages as bounded arrays and a dependent render/validation
job. The array tasks call the same implementation through its internal stage
selector. Set `SCPERTEVAL_PROTOCOL_MANIFEST` and
`SCPERTEVAL_PROTOCOL_OUTPUT` to reproduce a non-default generation.

An isolated checkout can reuse an existing prepared-data tree without copying it by setting
`SCPERTEVAL_DATA_ROOT` to the repository root that contains `models/data/`. The Helix launcher
defaults this to the shared primary checkout.
