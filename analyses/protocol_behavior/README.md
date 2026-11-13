# Protocol-behavior analyses

This directory contains the maintainable successor to the original paper analyses in
`analyses/scperteval-paper-v1/`. Run every currently implemented protocol-behavior stage from the
repository root with:

```bash
uv run reproduce-protocol-behavior
```

The command is resumable: it validates and reuses completed artifacts. At present, its only
implemented stage is the GWPS differential-expression cost pilot. Adding the production DE,
DRF, and figure stages will extend this same command rather than introduce new user-facing
entry points. `--dataset`, `--output-dir`, and the pilot tuning flags remain available for
nonstandard paths and development runs.

It deterministically selects 16 perturbations from each treated-cell-count quartile (64 total),
uses all 8,248 genes, and evaluates the full shared-DE grid: all observed target cells against the
all-perturbed and non-targeting-control references using Welch, conservative t-test, and MWU. It
also repeats MWU with split-half target cells for both references to measure the incremental cost
of using all observed target cells. References are capped at 8,192 cells with seed 42.

Each completed HDF5 is validated and reused on a later invocation. `summary.json` records loading
time, per-stage runtime, peak resident memory, output size, and the exact selected perturbations.
