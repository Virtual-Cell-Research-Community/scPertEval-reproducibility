# scPertEval reproducibility

End-to-end reproduction of the scPertEval metrics paper: acquire the original
data, prepare benchmark datasets, train prediction models, store their outputs,
evaluate the same predictions under alternative post-training choices, and
rebuild the paper figures.

This repository does **not** redistribute raw datasets. Dataset manifests point
to authoritative sources and the workflows verify downloads before processing.
Exact model and baseline predictions will be released separately as versioned
research artifacts.

## Repository roles

| Repository | Responsibility |
|---|---|
| `scPertEval` | Public evaluation package and metric implementations |
| `scPertEval-reproducibility` | Data preparation, splits, model execution, prediction artifacts, paper analyses and figures |

## Layout

- `models/` — isolated model environments, data preparation, training and
  prediction adapters, baselines, and cluster jobs.
- `analyses/scperteval-paper-v1/` — scripts, notebook, and reference outputs for
  reproducing the original paper figures.
- `configs/` — versioned dataset, model, and experiment declarations.
- `data/` — download and raw-to-analysis workflow documentation; generated data
  are ignored.
- `predictions/` — the public prediction-bundle contract and local artifact
  layout; generated predictions are ignored.
- `workflows/` — orchestration entry points that connect preparation, model
  execution, validation, evaluation, and figures.
- `schemas/` — machine-readable artifact manifests.
- `docs/` — study design and architecture decisions.

## Data policy

- Raw downloads are immutable and checksum-verified.
- Processed datasets and splits have versioned manifests and checksums.
- Public processing code must be self-contained and must not require private repositories.
- Files copied or reimplemented from internal workflows record their provenance.
- Generated datasets, checkpoints, predictions, caches, and cluster logs are not
  committed.

