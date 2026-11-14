# Ranking-sensitivity study design

This document is the source of truth for the study plan. It separates choices
that affect training from the post-training evaluation grid. Training inputs,
splits, model configurations, and prediction artifacts are frozen before model
ranks are inspected; the evaluation grid is versioned and may be extended
without retraining, provided the complete grid is reported.

## Scientific question

Can reasonable post-training evaluation choices change model rankings when the
trained models and their saved predictions are held fixed?

Models are trained once per prespecified split at the single training seed 42. Gene subsets
for scoring, representation, controls, metrics, weighting, and aggregation are
varied only after predictions have been saved.

## Benchmark tracks

The study contains both generalization axes; neither is optional.

### Perturbation generalization

- Replogle K562 essential is the sole perturbation-generalization benchmark in
  the ranking-sensitivity study.
- Replogle K562 GWPS remains a registered dataset with retained exploratory
  outputs, but it is retired from the revised benchmark.
- Nadig HepG2 participates in the four-context genetic track rather than a
  separate perturbation-generalization arm.
- Perturbed genes are divided into seeded, full-coverage five-fold splits.
- Every eligible model and baseline uses the same cells, genes, folds, and test
  perturbations.

The initial K562-essential eligibility audit yields 1,574 common targets at the 50-cell minimum
after requiring a measured target gene and compatibility with the GEARS and
scLAMBDA references. Five folds therefore contain 314 or 315 test targets each.
The 190 targets absent from the measured gene space are deliberately excluded;
this study will not add an alias-recovery branch.

Before the exact cohort and five folds are frozen, that target list must also
pass the public TxPert graph and ESM2 feature-coverage audits required by the
expanded model panel. This is a factual compatibility check, not a model-selection
branch. The model view retains every canonical cell belonging to the final
targets plus every canonical control cell and all 8,563 measured genes. The
generated view changes only annotations and the model expression representation
(`normalize_total(10,000)` then `log1p`); canonical raw counts remain immutable.

### Context generalization

- Genetic: Replogle–Nadig K562, RPE1, HepG2, and Jurkat, using four
  leave-one-context-out folds.
- Chemical: sci-Plex3 A549, K562, and MCF7, using three
  leave-one-cell-line-out folds.
- A held-out context contributes control cells but no treated cells to training.
- One jointly perturbation-conditioned model is fit per fold; no model is fit
  separately for each perturbation.
- An evaluated perturbation must be represented in every training context.

At a common minimum of 50 treated cells per context, the genetic track contains
480 shared perturbations (1,920 held-out context–perturbation evaluation
populations over four folds). The former 60-cell cohort is obsolete for final
training.

For the selected sci-Plex3 panel, the 50-cell rule retains 33 treated drug–dose
evaluation populations in every context. Three low-count drug–dose combinations
remain available to point-profile analyses but are excluded from population
metrics.

## Cell-count, prediction, and seed policy

- Every benchmark uses a minimum of 50 observed treated cells per evaluated
  perturbation and context.
- Population models archive exactly as many predicted cells as there are
  observed treated cells for that evaluation population.
- Models that transform held-out controls may sample those controls with
  replacement when a treated population is larger than the available unique
  control pool. These are genuine cell-level model outputs, not duplicated
  point predictions.
- Distributional metrics compare each complete observed population with its
  equally sized predicted population. No fixed-size evaluation subsampling is
  applied.
- Point predictions remain eligible only for profile metrics.
- Every trainable model uses the single training seed 42. The several held-out
  folds provide the replication needed for this compact evaluation-sensitivity
  study; the study does not claim to estimate initialization variance.
- Deterministic and training-free methods produce one prediction artifact per
  fold at archival seed 42.
- Each prediction artifact is retained separately by model, fold, and seed.
  Predictions are never averaged into an ensemble.

Metrics are calculated at the following grain:

`dataset × track × evaluation_variant × metric × perturbation × model × fold × training_seed × held_out_context`

For a headline summary, perturbations are averaged within each fold and folds
are then averaged. All underlying perturbation- and fold-level results remain
available.

## Prediction capabilities

- **Point-capable:** at least one predicted expression profile per evaluated
  perturbation; eligible for profile and centroid metrics.
- **Population-capable:** genuine predicted cells for each evaluated
  perturbation; eligible for both profile and distributional metrics.

The common profile leaderboard aggregates genuine population predictions to a
mean profile. Population metrics are reported in a separate capability stratum.
Unsupported results remain missing rather than being imputed.

## Model panels

### Perturbation generalization

The substantive panel is GEARS, PRESAGE, scGPT, scLAMBDA, TabICLv2,
TxPert-public, GenePert, fMLP-GenePT, fMLP-ESM2, and CellFlow. The TxPert name
refers specifically to the reproducible public STRING+GO configuration, not the
paper's proprietary-graph headline model. GenePert and fMLP-GenePT use the same
pinned GenePT table to isolate linear versus nonlinear prediction, while
fMLP-ESM2 supplies the matched protein-sequence representation comparison.
CellFlow supplies genuine population predictions for this track.
No-change, training-fold mean, and the bilinear PCA linear model are ranked
baselines. Technical-duplicate and interpolated technical-duplicate predictors
are oracle controls reported separately rather than ranked as deployable models.
STATE is excluded: its categorical perturbation coordinate is untrained for a
completely held-out perturbation, so it is valid for context transfer but not
zero-shot perturbation generalization.

### Context generalization

The substantive panel for both context tracks is:

- STATE;
- CPA, using the maintained `theislab/cpa` implementation;
- bioLORD;
- scDisInFact.

The chemical track additionally includes trVAE and the simplified SAMS-VAE(S)
variant from PerturBench. SAMS-VAE(S), rather than vanilla SAMS-VAE, is used
because the published benchmark found the simplified variant less prone to mode
collapse. Its joint chemical adapter must pass the same population-prediction
contract as the other models. trVAE's joint chemical smoke test
completed readily, while the 480-perturbation genetic fold took about 70 minutes
for one epoch before an arithmetic-preserving MMD-loop optimization and still
more than 20 minutes afterward. At the established 200 epochs, that would exceed
66 GPU-hours per genetic fold. trVAE is therefore excluded from the genetic
track under the prespecified scaling rule rather than being fit separately per
perturbation.

Every included model must represent all perturbations jointly within one fit per
context fold. scPRAM, scGen, and scVIDR are excluded because
their reference workflows require perturbation-specific or compound-specific
fits. chemCPA is unnecessary because the chemical task holds out contexts, not
unseen compounds. No-change, training-fold mean, and a context-aware linear
baseline accompany both context tracks.

## Gene-space policy

### Shared principle

Every saved prediction covers the complete prespecified output panel needed by
all downstream evaluation spaces. A model may use a smaller native internal
representation while decoding to the shared output panel. Gene identities used
for evaluation never trigger retraining.

### Replogle K562

Retain all 8,563 measured genes as the shared output panel. This avoids an
otherwise unnecessary selection rule at little storage cost.

## Development-pool and validation policy

Every fold first defines an 80% development pool and a 20% test set. Test
perturbations or held-out treated contexts are unavailable to every model.
Inside the common development pool, each model follows its documented native
selection procedure:

- models with validation-based early stopping or checkpoint selection retain
  the frozen validation subset for that purpose;
- fixed-budget models that do not select on validation data may train on the
  full development pool; and
- TabICLv2 receives the full development pool as in-context examples and never
  receives test perturbations.

This keeps the test information identical while avoiding an artificial data
penalty for methods that do not use early stopping. Each run manifest records
which development rows were used for fitting and which, if any, were used for
selection. In the context tracks, STATE, CPA, and bioLORD retain validation
conditions for checkpoint selection, whereas fixed-budget scDisInFact and
trVAE fit all treated cells from the training contexts. In perturbation
generalization, GEARS and scLAMBDA run fixed epoch budgets but still select a
validation checkpoint; PRESAGE and scGPT use validation-based early
stopping; TabICLv2 uses equal modality fusion with no validation-based component
selection and receives the full development pool as in-context examples.
The expanded models follow their pinned published selection policies. GenePert
and the fMLP probes use fixed budgets and may fit the full development pool;
TxPert-public and CellFlow must record their exact native validation usage in
their resolved run manifests before full launch.

scLAMBDA uses all training cells. Its adapter vectorizes the upstream
condition-by-cell embedding construction and omits target-coordinate arrays that
the selected `use_tg_coord = false` model never reads; upstream training and
prediction remain unchanged. Validation and mean prediction use scLAMBDA's
native `ctrl_size = 300` option, matching the fixed validation population in its
context-model implementation. The frozen run seed is passed into the model
constructor so the declared seed is reproducible.

### sci-Plex3

Reduce the approximately 27,000 measured genes to an approximately 9,000-gene
shared output panel: the global union of the top 8,192 HVGs and the top 100 DE
genes for every retained drug–dose combination across all contexts. Compound
target genes are not added. This intentionally uses outcome information from
all contexts for a fixed measurement panel; it is disclosed as a simplification
for an evaluation-sensitivity study, not a claim of deployment-time feature
selection.

If the union is materially larger than 9,000 genes, its exact size is reported
rather than silently truncating the panel.

### Perturbation target genes

GEARS requires the perturbed gene to be measured and covered by its GO
reference. scGPT's perturbation adapter inherits the measured-target requirement
through GEARS' `PertData` machinery. scLAMBDA, GenePert, and fMLP-GenePT require
a GenePT embedding for each target; fMLP-ESM2 and CellFlow require an ESM2
feature; and TxPert-public requires its public graph features. PRESAGE and
TabICLv2 do not inherently require the target to be an expression output.

The shared perturbation-generalization cohort therefore uses only targets
supported by every participating model. The 190 unmeasured targets are dropped
without symbol-alias recovery so that no model silently evaluates a different
test cohort.

## Native inputs and shared evaluation space

Canonical PerturbationData objects retain integral raw counts in `X`. Each
model receives the input representation required by its documented reference
workflow: counts, normalized expression, or log1p expression. Model-native
latent transformations remain allowed.

Predictions are archived in the model's declared native output space. Evaluation
maps them deterministically into a shared `normalize_total(target_sum=10,000)`
then `log1p` view. Log-scale outputs are not exponentiated to fabricate counts.
If a model predicts an effect, that effect is combined with the appropriate
held-out-context control in its declared native space before evaluation. The
conversion rule is recorded in each prediction manifest.

## Environment policy

- PerturbationData and evaluation/orchestration code use `uv`.
- Every model runner uses its own Pixi environment.
- Hugo's validated environments remain unchanged; STATE is reused only for
  context transfer.
- TxPert-public, GenePert, the shared fMLP runner, CellFlow, and SAMS-VAE(S)
  receive separate pinned Pixi environments.
- For scArchon/scPerturBench-derived models, pin the repository revision and
  container digest, reconstruct a Pixi environment from the reference
  container, and require a small container-versus-Pixi equivalence test.
- The SDS has enough capacity for these environments and retained predictions.

## Sensitivity panel

The versioned grid varies:

- gene set;
- absolute versus control-centered representation;
- control construction;
- metric;
- uniform versus observed-effect-strength perturbation weighting; and
- mean, median, or effect-strength-weighted aggregation across perturbations.

Metrics remain separate; no cross-metric composite or global leaderboard is
constructed. The reference variant is an interpretive anchor, not a privileged
primary analysis. Additional reasonable evaluation variants may be added from
the saved predictions without retraining, and the complete versioned grid is
reported rather than dividing results into primary and exploratory classes.

For each variant report model ranks, rank changes relative to the reference,
Spearman and Kendall agreement, pairwise-order or winner reversals, and paired
perturbation-bootstrap uncertainty.

## Hyperparameter and compute policy

- Use published or established benchmark configurations; perform no
  dataset-specific hyperparameter optimization.
- Compatibility or stability changes such as reducing batch size after an OOM
  are permitted, recorded, and made without consulting model ranks.
- Beast is the preferred first workstation when its GPUs are idle. Beauty may
  be used only after checking current GPU ownership and compatibility.
- Helix is used for high-memory preprocessing, overflow, and final tracked runs
  when workstation resources are insufficient.
- Run cheap truncated smoke tests before the 34 full context fits
  (6 models × 3 chemical folds plus 4 models × 4 genetic folds), and before the
  50 perturbation-generalization fits (nine trainable models × five folds plus
  deterministic TabICLv2 across five folds).
- Completed runs retain resolved configuration, environment lock and code
  revision, checkpoint, predictions, checksums, and a completion record; reruns
  skip validated completed artifacts.

The first full-data scLAMBDA scaling trial was run on Beauty GPU 0 only after
checking both workstations. Beast's two GPUs were occupied, Beauty GPU 1 was in
active use by another process, and Beauty GPU 0 had sufficient free capacity.
The patched initialization completed with about 46 GiB GPU memory and the
10-epoch end-to-end run took approximately 17 minutes. The resulting
1,387-by-8,248 point-prediction artifact passed the frozen-fold prediction
contract and all completion-record checksums. Production workstation sweeps are
serialized on one GPU so they neither compete with other users nor issue
concurrent 48 GiB reads from SDS.

The retired GWPS scLAMBDA production sweep completed on Beauty GPU 0 on 2026-11-13
using repository revision `0eb13a2`. All five folds were run at training seeds
42, 43, and 44, giving 15 independently archived exploratory runs. Every run produced one
point prediction for each of its 1,387 held-out perturbations across the frozen
8,248-gene universe. A final sweep-wide audit verified all 15 completion
records and declared artifact checksums, and re-ran the prediction contract for
every fold and seed. The resumable workstation launcher exited successfully
after the final run; completed runs remain safe to skip on restart.

## Reproducibility and storage policy

- Raw datasets are downloaded from authoritative sources and are not
  redistributed.
- Public scripts reproduce raw-to-analysis processing without private
  repositories.
- Full predictions and derived tidy metrics are retained. Population predictions
  match the number of observed treated cells per evaluation population.
- Storage is re-estimated after the final essential cohort and the CellFlow and
  SAMS-VAE(S) smoke outputs are materialized. All population outputs continue to
  match observed treated-cell counts, while point-model outputs remain compact.
- Exact model and baseline predictions may be published in a versioned Zenodo
  record, partitioned by dataset and model if needed.
- Figures consume versioned predictions or tidy result tables, never private
  data paths.

## Checks required before full training

1. Materialize and record the exact sci-Plex3 gene panel.
2. Regenerate the genetic context model view at the 50-cell rule.
3. Audit TxPert-public graph coverage and ESM2 coverage, then freeze the exact
   essential target list, folds, checksums, and metadata-validation record.
4. Implement and pin TxPert-public, GenePert, fMLP, CellFlow, and SAMS-VAE(S)
   adapters and Pixi environments.
5. Pin CPA, bioLORD, scDisInFact, and trVAE reference revisions and container
   digests, then reconstruct their Pixi environments.
6. Smoke-test every context adapter on one genetic and one chemical fold;
   explicitly verify that trVAE is jointly perturbation-conditioned.
7. Verify the declared native input and output conversion for every model.

## Current launch readiness

Smoke tests below mean one held-out-context fold with one training epoch. They
verify the complete adapter contract and final frozen input shapes; they are not
benchmark results and do not authorize the full launch.

| Gate | Chemical | Genetic | Environment/reference | Status |
| --- | --- | --- | --- | --- |
| Frozen input views | 27,940 cells × 8,811 genes | 201,043 cells × 6,640 genes | manifests and checksums validated | ready |
| STATE | passed | passed | pinned Pixi environment | ready |
| CPA 0.8.8 | passed | passed | matched to pinned scArchon image | ready |
| bioLORD | passed | passed | pinned official revision | ready |
| scDisInFact | passed | passed | source tree and core dependencies match the pinned scArchon image | ready |
| trVAE | passed | excluded by the prespecified scaling rule | matched to pinned scArchon image, using its corresponding stable pandas release | ready |
| SAMS-VAE(S) | pending | not included | pinned PerturBench revision; adapter not implemented | pending |
| No-change, fold-mean, and linear baselines | passed | passed | completion records validated | ready |
| Helix launch scripts | not submitted | not submitted | `sbatch --test-only` still required | pending |

The full benchmark remains deliberately unlaunched until every row is ready.
