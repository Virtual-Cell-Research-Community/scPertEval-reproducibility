# Data provenance

## Contract

The repository does not redistribute raw or processed expression matrices.
Each declaration under `configs/datasets/` pins:

- the authoritative public URL, release, byte size, and checksum;
- the public preparation adapter and dataset-specific options;
- the required observation fields and raw-count `X` contract; and
- the dimensions, byte size, and SHA-256 of a validated reference output.

`workflows/data.py` downloads to an immutable dataset-specific raw directory,
verifies every artifact before use, writes one canonical processed H5AD, and
checks public regeneration against the reference artifact. The checksums are
validation targets, not redistributed data.

## CRISPRi screens

The Replogle and Nadig adapters start from scPerturb v1.4 raw-count H5ADs. They
standardize perturbation fields, apply technical cell and gene QC, reproduce
the historical CRISPRi residual-expression rule, score cell cycle using the
pinned Regev gene list, and calculate the historical VST annotations. Nadig
duplicate gene-symbol columns are merged; RPE1 deliberately records but does
not filter on residual target expression. `X` is restored to integral raw
counts before writing.

## sci-Plex3

The sci-Plex3 adapter combines the corrected scPerturb v1.4 count matrix with
compound metadata from the original publication repository at immutable commit
`6bab132d9f7efe677d4b02cdc1c56890ec946812`. It retains assigned observations
and vehicle controls and derives canonical parent SMILES and InChIKeys without
discarding the published strings. The upstream records do not specify a data
license, so users must evaluate their intended reuse independently.

## Frozen context-transfer cohorts

The genetic cohort is the strict intersection of perturbations with at least
60 treated cells in each of K562, RPE1, HepG2, and Jurkat. It contains 322
targets and 1,288 held-out context–target populations.

The chemical cohort follows the scPerturBench sci-Plex3 task definition: nine
drugs, four doses, and three cell lines. All 36 drug–dose conditions are valid
for point evaluations. Three conditions have fewer than 60 observations in at
least one cell line and are excluded from the population-evaluation stratum,
leaving 33 fixed drug–dose conditions. This keeps the competitor and condition
sets fixed within each evaluation stratum.
