# Protocol-behavior analyses

This directory is the maintained successor to the frozen original-paper analysis
in `analyses/scperteval-paper-v1/`. It is deliberately a separate UV project and
pins scPertEval commit `774e6001b09f889905c353a8fd4022a36482eb80`. The workflow
uses current concrete `Protocol` objects, current feature spaces, pure DE metrics,
and the current `prepare()` / `calibrate()` / `de()` interfaces.

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
  and retains the four genetic and three SciPlex3 context units. The current
  result tree, at scPertEval `774e600`, is
  `results/protocol_behavior/production-v2-scperteval-774e600` (completed on Beauty,
  2026-11-22). The earlier staging tree at `58ccc91`
  (`production-v2-scperteval-main-staging`) has identical DEG agreement and
  concordance tables; its DRF values differ only for `de_auprc`, `de_auroc` and
  `de_overlap_k=50` (by at most 0.011), because scPertEval `774e600` ranks DE
  genes by significance by default and keeps exact ties.

For every analysis unit the workflow computes six full-cell DE combinations
(three methods × two references), 20 CPU DRF protocols, two GPU Sinkhorn
protocols, and per-protocol timings. It then renders the mean/median DRF tables,
timing table, standard and conservative DEG-set agreement figures, and matching
concordance figures. The Jaccard source tables retain every heatmap cell, median
selected-gene count, and per-perturbation adjusted-p ranking correlation.
Concordance@k is top-k set overlap divided by k; it does not require matching
effect directions or identical within-set ranks. The curves use the legacy
fractional-anchor grid and report median per-perturbation AUCC.

Before any stage runs, `--preflight` checks the existing prepared H5ADs against
their data manifests and writes `inputs.json`. Every stage is then bound to the
production/cohort/data manifests, isolated project and lock, workflow source,
repository revision, scPertEval revision, and output checksums. A repeated command
validates and skips completed stages before loading the dataset. The final
`completion.json` inventories every stage and rendered artifact.

```bash
uv run reproduce-protocol-behavior --preflight
uv run reproduce-protocol-behavior --status
uv run reproduce-protocol-behavior --stage drf.cpu.chemical-a549
```

To validate or render the completed v1 result tree explicitly:

```bash
uv run reproduce-protocol-behavior \
  --manifest configs/protocol_behavior/production-v1.json \
  --output-dir results/protocol_behavior/production-v1 \
  --finalize
```

If workstation execution is insufficient, the chunked Slurm wrapper remains
available from an OTP-authenticated Helix session:

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
