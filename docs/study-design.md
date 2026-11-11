# Ranking-sensitivity study design

This document is the source of truth for the study plan. It separates choices
that affect training from the post-training evaluation grid. Training inputs,
splits, model configurations, and prediction artifacts are frozen before model
ranks are inspected; the evaluation grid is versioned and may be extended
without retraining, provided the complete grid is reported.

## Scientific question

Can reasonable post-training evaluation choices change model rankings when the
trained models and their saved predictions are held fixed?

Models are trained once per prespecified split and training seed. Gene subsets
for scoring, representation, controls, metrics, weighting, and aggregation are
varied only after predictions have been saved.

## Benchmark tracks

The study contains both generalization axes; neither is optional.

### Perturbation generalization

- Replogle K562 essential is the implementation pilot.
- Replogle K562 GWPS is the final large genetic benchmark.
- Nadig HepG2 is an independent genetic benchmark.
- Perturbed genes are divided into seeded, full-coverage five-fold splits.
- Every eligible model and baseline uses the same cells, genes, folds, and test
  perturbations.

The K562-essential pilot yields 1,574 common targets at the 50-cell minimum
after requiring a measured target gene and compatibility with the GEARS and
scLAMBDA references. Five folds therefore contain 314 or 315 test targets each.
The 190 targets absent from the measured gene space are deliberately excluded;
this study will not add an alias-recovery branch.

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
- Population models archive at most 100 genuine predicted cells per evaluation
  population. They are never padded by duplicating point predictions.
- Distributional metrics compare matched draws of exactly 50 observed and 50
  predicted cells using ten prespecified draw seeds.
- Point predictions remain eligible only for profile metrics.
- Context models use training seeds 42, 43, and 44.
- Each prediction artifact is retained separately by model, fold, and seed.
  Predictions are never averaged into an ensemble.

Metrics are calculated at the following grain:

`dataset × track × evaluation_variant × metric × perturbation × model × fold × training_seed × held_out_context`

For a headline summary, perturbations are averaged within a fold, folds within
a training seed, and then the three training seeds are averaged. All underlying
perturbation-, fold-, and seed-level results remain available.

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

The substantive panel is GEARS, PRESAGE, scGPT, scLAMBDA, STATE, and TabICLv2.
No-change, training-fold mean, and the bilinear PCA linear model are ranked
baselines. The interpolated technical-duplicate predictor is an oracle control
and is reported separately rather than ranked as a deployable model.

### Context generalization

The substantive panel for both context tracks is:

- STATE;
- CPA, using the maintained `theislab/cpa` implementation;
- bioLORD;
- scDisInFact.

The chemical track additionally includes trVAE. Its joint chemical smoke test
completed readily, while the 480-perturbation genetic fold took about 70 minutes
for one epoch before an arithmetic-preserving MMD-loop optimization and still
more than 16 minutes afterward. At the established 200 epochs, that would exceed
53 GPU-hours per genetic fold. trVAE is therefore excluded from the genetic
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
through GEARS' `PertData` machinery. scLAMBDA requires a GenePT embedding for
each target, while PRESAGE, TabICLv2, and STATE do not inherently require the
target to be an expression output.

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
- Hugo's six validated environments remain unchanged.
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
- Run cheap truncated smoke tests before the 93 full context fits
  (5 models × 3 chemical folds × 3 seeds plus 4 models × 4 genetic folds × 3
  seeds).
- Completed runs retain resolved configuration, environment lock and code
  revision, checkpoint, predictions, checksums, and a completion record; reruns
  skip validated completed artifacts.

## Reproducibility and storage policy

- Raw datasets are downloaded from authoritative sources and are not
  redistributed.
- Public scripts reproduce raw-to-analysis processing without private
  repositories.
- Full predictions and derived tidy metrics are retained. Population predictions
  are capped at 100 cells per evaluation population.
- The pessimistic whole-study storage bound is below approximately 500 GB; the
  expected archive is smaller because several models produce point profiles.
- Exact model and baseline predictions may be published in a versioned Zenodo
  record, partitioned by dataset and model if needed.
- Figures consume versioned predictions or tidy result tables, never private
  data paths.

## Checks required before full training

1. Materialize and record the exact sci-Plex3 gene panel.
2. Regenerate the genetic context model view at the 50-cell rule.
3. Pin CPA, bioLORD, scDisInFact, and trVAE reference revisions and container
   digests, then reconstruct their Pixi environments.
4. Smoke-test every context adapter on one genetic and one chemical fold;
   explicitly verify that trVAE is jointly perturbation-conditioned.
5. Verify the declared native input and output conversion for every model.

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
| No-change, fold-mean, and linear baselines | passed | passed | completion records validated | ready |
| Helix launch scripts | not submitted | not submitted | `sbatch --test-only` still required | pending |

The full benchmark remains deliberately unlaunched until every row is ready.
