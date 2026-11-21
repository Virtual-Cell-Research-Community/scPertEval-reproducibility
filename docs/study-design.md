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

## Related protocol-behavior analysis

The DRF/DE protocol-behavior study is versioned separately from model training.
Its completed `production-v1` result uses GWPS K562 and is retained unchanged as
historical production evidence. Planned `production-v2` replaces that unit with
the current K562 Essential benchmark while retaining the genetic and chemical
context units. V2 is not launched until the pinned scPertEval DE revision has
completed review; this review gate does not block model prediction generation or
the ranking-sensitivity evaluation.

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

The initial K562-essential eligibility audit yielded 1,574 common targets at the 50-cell minimum
after requiring a measured target gene and compatibility with the GEARS and
scLAMBDA references.
The 190 targets absent from the measured gene space are deliberately excluded;
this study will not add an alias-recovery branch.

The expanded-panel audit found TxPert-public support for all 1,574 preliminary
targets. The pinned ESM2-15B checkpoint has 1,026 positional embeddings, leaving
1,024 residues after its two special tokens; reproducing Miller's longest-canonical-
protein selection therefore supports 1,352 targets (85.90%). The 222 targets
without an eligible canonical protein sequence or whose selected protein exceeds
that limit are excluded. This remains inside the pre-agreed 20% coverage-loss
gate. The final five-fold cohort contains 1,352 targets, with 271 test targets in
the first two folds and 270 in each remaining fold. The exact target list,
support-key hashes, source revisions, and fold hash are frozen in the
experiment declarations. The model view retains every canonical cell belonging
to the final targets plus every canonical control cell and all 8,563 measured
genes. The generated view changes only annotations and the model expression
representation
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

Point-capable chemical models may retain predictions for all 36 drug–dose
conditions, but the shared profile leaderboard is evaluated on the same 33
population-compatible conditions used by every model. The three lower-coverage
conditions therefore remain in point prediction artifacts but are excluded from
cross-model ranking. Aggregation rejects any model/fold/variant whose
perturbation cohort differs from its competitors.

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
- Point predictions are expanded for population metrics with a shared control-residual lift:
  sample held-out-context controls, center their residuals, and add the model's predicted
  centroid. The construction preserves every nonnegative point prediction as the lifted
  population mean; invalid negative log-expression values are clipped only in the lifted
  population. This makes the inherited-versus-learned heterogeneity limitation measurable
  without altering the original artifact used for profile metrics.
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
mean profile. For population metrics, centroid-only outputs use the same deterministic,
batch-matched-when-available control-residual lift. The released TxPert workflow uses
batch-average controls, so its batch-centroid outputs use this lift as well. This is a declared
evaluation construction, not a claim that centroid models learn perturbation-dependent
single-cell heterogeneity.

## Model panels

### Perturbation generalization

The substantive panel is GEARS, PRESAGE, scGPT, scLAMBDA, TabICLv2,
TxPert-public, GenePert, fMLP-GenePT, fMLP-ESM2, and CellFlow. The TxPert name
refers specifically to the reproducible public STRING+GO configuration, not the
paper's proprietary-graph headline model. GenePert and fMLP-GenePT use the same
pinned GenePT table to isolate linear versus nonlinear prediction, while
fMLP-ESM2 supplies the matched protein-sequence representation comparison.
CellFlow supplies genuine population predictions for this track. Its mean
prediction participates in the common profile leaderboard and its complete
cell population also participates in the population-only leaderboard.
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
uses the upstream L2-normalized GenePT-to-mean-expression ridge formulation with
`alpha = 10` frozen from its K562 example before our fold outcomes are inspected.
Its adapter passed a full synthetic prediction/checkpoint/manifest/resumption
smoke test. The shared fMLP adapter reproduces CellSimBench's encoder-decoder,
cell-level MSE objective, Adam/StepLR schedule, and validation-checkpoint rule;
both embedding variants passed the same synthetic archival contract. GenePert
may fit the full development pool, while fMLP retains the frozen validation
perturbations for checkpoint selection;
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

The executable protocol universe is frozen from PerturbationEvaluationAtlas
commit `0be116b2a871132c2b3f0c9e896e2fb544a41fe5`. An operational family is
eligible when it occurs in at least two independent works. Literature frequency
selects families but does not weight them. Ranking-equivalent variants receive
one vote, reporting aggregation is separated from atomic protocol identity, and
the approved scope excludes retrieval, cross-perturbation metrics, new
population-distance implementations, curated composite panels, and
effect-weighted perturbation aggregation. Limma-dependent evaluation is
postponed. The resulting primary universe contains 101 families, normalized by
task in `configs/evaluation/atlas-task-protocol-manifest-v1.csv`.

Metrics remain separate; no cross-metric composite or global leaderboard is
constructed. Tasks also remain separate. The global analysis is reference-free:
it reports fractional winner share, disagreement of winners across all eligible
protocol pairs, and winner-change rates for evaluation factors and concrete
factor contrasts. Selected bump plots illustrate especially clear, reasonable
ranking changes but do not define the global conclusion. Their selection is
disclosed and the complete candidate table is exported.

The primary perturbation reduction is the equal mean. Median aggregation is a
separate literature-supported sensitivity factor. Effect-strength weighting is
excluded. Exact score ties receive average ranks and tied winners split winner
credit fractionally.

Undefined scores follow explicit semantic rules. A constant prediction against
varying truth receives zero for Pearson or cosine, because it contains no
association or directional information. Constant truth remains undefined for
those metrics. When truth contains positives and prediction contains none,
recall, precision, and F1 use zero. Recall and precision-recall metrics remain
undefined when truth contains no positives, and AUROC remains undefined when
truth contains only one class. Explicit set comparisons assign one to two empty
sets and zero when exactly one set is empty. Invalid model output is a
model-protocol failure, not a missing value to discard; insufficient observed
cells are a ground-truth eligibility failure.

Every protocol pair is compared within task and fold on the intersection of
evaluation units for which both protocols are ground-truth-defined. Both
protocols and every model use that identical set. A model-specific failure may
not shrink the set for its competitors. The audit records the total, individual
defined counts, common count, common fraction, and structured undefined reasons.
The headline threshold is frozen at 75% common coverage in every fold. The
ground-truth-only v1 audit was completed before inspecting model ranks and is
recorded in `docs/ground-truth-coverage-audit-v1.md`. Every essential and
genetic pair passes, including the four explicit target-policy protocols, and
97.9% of chemical pairs pass. The only excluded chemical protocol is
uncentered Pearson on the top-50 significance-ranked DEGs, whose weakest fold
contains 21/33 defined conditions. Winner-share summaries include only
protocols passing this frozen rule and display their coverage.

Bootstrap analysis is supplementary. It uses 1,000 paired perturbation-identity
draws and applies the same draw to every model and protocol. Genetic transfer
resamples 480 target genes and retains all four contexts; chemical transfer
resamples 33 drug-dose conditions and retains all three contexts; the essential
screen resamples target genes directly. Illustrative bump plots are selected
after primary scores exist; the figure legend states that they are selected
examples, and the complete candidate ranking is exported.

## Hyperparameter and compute policy

- Use published or established benchmark configurations; perform no
  dataset-specific hyperparameter optimization.
- Compatibility or stability changes such as reducing batch size after an OOM
  are permitted, recorded, and made without consulting model ranks.
- A new model remains in the panel unless a representative full fold is
  projected to exceed 24 GPU-hours, cannot fit available hardware, produces
  invalid predictions, or cannot satisfy the shared prediction contract after
  reasonable adapter corrections. Crossing one of these gates triggers an
  explicit inclusion decision rather than silent removal.
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

1. **Complete:** the exact sci-Plex3 gene panel is materialized and recorded.
2. **Complete:** the genetic context model view uses the 50-cell rule.
3. **Complete:** the 1,352-target essential model view contains 198,346 cells ×
   8,563 genes. Its SHA-256 is
   `673ed91b5ad7bd04b16422a1273ed2e095d5670acefa130c414df9efdc97b70a`;
   the regenerated fold assignment exactly matches the frozen fold hash.
4. **Complete:** representative full-fold scaling gates passed for
   TxPert-public, CellFlow, SAMS-VAE(S), GenePert, and both fMLP variants. The
   validated 1,352-target, 5,120-dimensional ESM2 asset is materialized on SDS.
5. **Complete:** CPA, bioLORD, scDisInFact, and trVAE use pinned reference
   revisions and isolated Pixi environments reconstructed from their references.
6. **Complete:** every context adapter passed a representative genetic and/or
   chemical smoke appropriate to its declared panel; trVAE was explicitly
   verified as jointly perturbation-conditioned before its genetic exclusion by
   the scaling rule.
7. **Complete:** every model declaration records its native input, output, and
   deterministic evaluation-space conversion.

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
| SAMS-VAE(S) | 26-second one-epoch training probe; 3.76-hour conservative 500-epoch projection | not included | pinned PerturBench revision and Pixi environment | ready |
| No-change, fold-mean, and linear baselines | passed | passed | completion records validated | ready |
| Helix launch scripts | staged production submitted | staged production submitted | `sbatch --test-only` passed; family-local canary dependencies | queued across planned cluster maintenance |

The staged production launch began on 2026-11-14. Each Helix model family has a
full-budget canary plus an `afterok` remainder array; a failed canary cancels
only its own dependent jobs. Deterministic CPU baselines were submitted without
canary dependencies. Helix was drained for planned maintenance at submission,
so queued state does not indicate a benchmark failure. Workstation model
families use the equivalent persistent canary guards and checksum-validating
atomic SDS promotion.

The added perturbation-generalization models also passed their representative
fold-0 scaling gates on the frozen 198,346-cell × 8,563-gene K562-essential
view. These are engineering gates, not benchmark scores.

| Model | Representative evidence | Conservative full-budget projection | Status |
| --- | --- | --- | --- |
| GenePert | complete fold in 113 seconds | complete measurement | ready |
| fMLP-GenePT | one-epoch run in 292 seconds | 8.12 GPU-hours for 100 epochs | ready |
| fMLP-ESM2 | one-epoch run in 292 seconds | 8.12 GPU-hours for 100 epochs | ready |
| TxPert-public | one-epoch training in 122 seconds; 295 seconds end to end | 1.75 GPU-hours for 50 epochs | ready |
| CellFlow | 200 iterations plus all 271 populations in 1,475 seconds | at most 1.25 GPU-hours for 200,000 iterations | ready on Beauty's 98 GB GPU |

TxPert-public and CellFlow each archived 36,972 genuine cell predictions,
exactly matching the 51–475 observed treated cells for every test perturbation.
CellFlow used the complete published 4,096-wide architecture and peaked at
73,994 MiB of GPU memory, so it is assigned to Beauty rather than the 48 GB
Beast GPUs. Its measured model checkpoint, PCA reference, and prediction total
11.04 GB per fold, or about 55.2 GB over five folds. TxPert-public projects to
about 7.5 GB over five folds; GenePert and both fMLPs remain below 1 GB
combined. Together with SAMS-VAE(S), the added panel projects to roughly 68 GB,
well below the study's 500 GB prediction-and-checkpoint budget.

SAMS-VAE(S) passed a representative frozen sci-Plex3 chemical fold using the
published hyperparameters: validation-selected checkpointing, fresh checkpoint
reload before prediction, 3,285 exactly size-matched cells across the 33 held-out
populations, checksums, and restart-safe completion. One epoch took 26 seconds;
including the fixed I/O cost, a deliberately conservative 500-epoch projection
is 3.76 GPU-hours. Its measured checkpoint is 1.41 GB and its compressed
prediction artifact is 105 MB, projecting to about 4.6 GB across three folds.
