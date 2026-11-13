# Protocol-behavior analyses

This directory contains the maintainable successor to the original paper analyses in
`analyses/scperteval-paper-v1/`. Run or resume the complete production workflow from the repository
root with:

```bash
uv run reproduce-protocol-behavior
```

The production manifest is `configs/protocol_behavior/production-v1.json`. It declares eight
analysis units: GWPS K562, four genetic contexts, and three SciPlex3 contexts. For every unit the
workflow computes the six full-cell DE combinations (three methods × two references), 20 CPU DRF
protocols, two GPU Sinkhorn protocols, and per-protocol timings. It then renders the mean/median DRF tables,
timing table, standard and conservative DEG-set agreement figures, and corresponding concordance
figures.

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

The bounded GWPS cost pilot remains available explicitly:

```bash
uv run reproduce-protocol-behavior --pilot gwps-de
```

It deterministically selects 16 perturbations from each treated-cell-count quartile (64 total),
uses all 8,248 genes, and evaluates the full shared-DE grid: all observed target cells against the
all-perturbed and non-targeting-control references using Welch, conservative t-test, and MWU. It
also repeats MWU with split-half target cells for both references to measure the incremental cost
of using all observed target cells. References are capped at 8,192 cells with seed 42.

Its `summary.json` records load time, per-stage runtime, peak memory, output size, and selected targets.
