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
- `runs/` — generated model × fold × seed directories containing resolved
  configurations, checkpoints, predictions, checksums, and completion records.
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

## Reproduction commands

The stable user-facing interface is intentionally small:

```bash
uv run reproduce-protocol-behavior
uv run reproduce-ranking-sensitivity
uv run reproduce-all
```

The first command is implemented and currently reproduces the resumable GWPS DE
cost pilot. The ranking-sensitivity and combined commands remain planned until
their production prediction inputs and end-to-end artifact tests are complete.
See `docs/study-design.md` for the command contract and current scope.

## Licensing

This repository is not covered by a single repository-wide license. The pinned
original-paper analysis in `analyses/scperteval-paper-v1/` is available under
the MIT License included in that directory. Files outside explicitly licensed
subdirectories do not currently carry a license grant. Third-party model
implementations, pretrained weights, datasets, and other external artifacts
remain subject to their own upstream terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before running or
redistributing model workflows.

