# Architecture

## Principle

The repository exposes a small number of artifact interfaces. Dataset-specific
and model-specific complexity stays behind those interfaces so a new adapter
does not require changes throughout the evaluation and figure code.

## Modules and seams

### Dataset preparation module

**Interface:** a dataset manifest, a frozen split manifest, and a prepared H5AD.

**Implementation:** authoritative download, checksum verification, dataset-
specific metadata harmonization, technical QC, and split generation.

The dataset ID is stable. A changed raw source, preparation revision, output
checksum, or eligibility rule produces a changed manifest rather than silently
reusing an old artifact.

### Model execution module

**Interface:** a model declaration plus a dataset and split manifest in;
validated prediction bundles out.

**Implementation:** each model's native environment, training command,
checkpoint selection, prediction conversion, and cluster resources. The model
adapters under `models/` are real seams because the model implementations have
genuinely different environments and execution paths.

### Prediction artifact module

**Interface:** `schemas/prediction-manifest.schema.json` and the H5AD conventions
in `predictions/README.md`.

This is the central seam. Evaluation code consumes prediction bundles without
knowing how a model was trained. Point and population predictions are two
explicit capabilities of the same artifact interface.

### Evaluation module

**Interface:** prediction bundle, prepared truth dataset, split manifest, and
evaluation configuration in; tidy per-condition metric results out.

**Implementation:** `scPertEval` supplies metric behavior. This repository owns
the post-training choices used in the paper: gene set, representation, control
construction, weighting, and aggregation across perturbations.

### Analysis module

**Interface:** versioned tidy result tables in; figures and paper tables out.

Analysis code must not load model checkpoints or recreate predictions. That
keeps figure reproduction cheap and makes the archived prediction bundle
sufficient for reviewers.

## Dependency direction

```text
authoritative raw sources
        ↓
dataset preparation → dataset + split manifests
        ↓                         ↓
model adapters ─────────────→ prediction bundles
                                  ↓
                       scPertEval evaluation
                                  ↓
                         tidy result tables
                                  ↓
                       paper analyses/figures
```

Internal evidence and data-management systems may inform development but sit
outside this runtime graph.

## Migration rule

The existing `models/` and `analyses/scperteval-paper-v1/` workflows remain
operational while they are refactored. Replace old entry points only after an
end-to-end vertical-slice test exists at the new artifact interfaces. The
interface is the test surface; tests should validate observable artifacts rather
than internal model state.

