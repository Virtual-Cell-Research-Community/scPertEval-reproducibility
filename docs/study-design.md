# Ranking-sensitivity study design

This document is the source of truth for the study plan. It records confirmed
decisions separately from open questions so that design choices are frozen
before model ranks are inspected.

## Scientific question

Can reasonable post-training evaluation choices change model rankings when the
trained models and their saved predictions are held fixed?

Models are trained once per prespecified split. Gene selection for scoring,
representation, controls, metrics, weighting, and aggregation may change only
after predictions have been frozen.

## Benchmark tracks

The study contains both generalization axes; neither is optional.

### Perturbation generalization

- Replogle K562 essential is the implementation pilot.
- Replogle K562 GWPS is the final large genetic benchmark.
- Nadig HepG2 is an independent genetic benchmark.
- Perturbed genes are divided into seeded, full-coverage five-fold splits.
- Every eligible model and baseline uses the same cells, genes, folds, and test
  perturbations.

The K562-essential pilot currently yields 1,574 common targets at the confirmed
50-cell minimum after requiring a measured target gene and compatibility with
the GEARS and scLAMBDA references. Five folds therefore contain 314 or 315 test
targets each. These counts must be asserted by the preparation workflow before
training.

### Context generalization

- Genetic: Replogle–Nadig K562, RPE1, HepG2, and Jurkat, using leave-one-context-out
  folds.
- Chemical: sci-Plex3 A549, K562, and MCF7, using leave-one-cell-line-out folds.
- A held-out context contributes control cells but no treated cells to training.
- An evaluated perturbation must be represented in every training context.

At a common minimum of 50 treated cells per context, the genetic track contains
480 shared perturbations (1,920 held-out context–perturbation populations over
four folds). The existing 60-cell manifest and derived model view are obsolete
for final training and must be regenerated at 50 cells.

For the selected sci-Plex3 panel, the 50-cell rule retains the same 33 treated
drug–dose conditions as the former 60-cell rule. Three low-count conditions
remain excluded from population evaluation.

## Cell-count and population policy

- Every benchmark uses a minimum of 50 observed treated cells per evaluated
  condition.
- Distributional metrics compare matched draws of exactly 50 observed and 50
  predicted cells.
- Draws use ten prespecified seeds and are averaged within each condition before
  aggregation across conditions.
- Point predictions remain eligible only for profile metrics. They are never
  duplicated to manufacture a predicted population.

## Prediction capabilities

- **Point-capable:** at least one predicted expression profile per test
  condition; eligible for profile and centroid metrics.
- **Population-capable:** at least 50 genuine predicted cells per test condition;
  eligible for both profile and distributional metrics.

The common profile leaderboard aggregates genuine population predictions to a
mean profile. Population metrics are reported in a separate capability stratum.
Unsupported results remain missing rather than being imputed.

## Model panels

### Perturbation generalization — confirmed

The substantive panel is:

- GEARS;
- PRESAGE;
- scGPT;
- scLAMBDA;
- STATE; and
- TabICLv2.

No-change, training-fold mean, and the bilinear PCA linear model are ranked
baselines. The interpolated technical-duplicate predictor is an oracle control
and is reported separately rather than ranked as a deployable model.

### Context generalization — under feasibility review

STATE is confirmed. scArchon provides reusable population-prediction
implementations and containers for trVAE, scGen, scPRAM, and scVIDR. The current
leading candidate panel is STATE, trVAE, scGen, and scPRAM, with scVIDR as an
additional candidate.

This panel is not yet frozen. scArchon's published wrappers fit one binary
perturbation and held-out context per run. Used literally, the 480-perturbation
genetic track would require 1,920 trainings per model, whereas the 33-condition
sci-Plex3 track would require 99 per model. Before adoption, a smoke experiment
must establish whether the adapters can be made jointly perturbation-conditioned
without changing the scientific method. Otherwise the context panels may differ
between the genetic and chemical tracks.

## Gene-space policy

### Shared principle

Every saved prediction must cover the complete prespecified output panel needed
by every downstream evaluation space. A model may use a smaller native internal
representation (for example, STATE may encode 2,000 HVGs) while decoding to the
shared output panel.

Gene identities used for evaluation never trigger model retraining.

### Replogle K562

Retain all 8,563 measured genes as the shared output panel. The former panel of
8,192 HVGs union perturbation targets already retained 8,367 genes, so keeping
the remaining 196 genes avoids a selection rule at little additional cost.

### sci-Plex3

Reduce the approximately 27,000 measured genes to an approximately 9,000-gene
shared output panel. The proposed deterministic panel is the union of:

- the top 8,192 HVGs;
- the top 100 DE genes for every selected drug–dose condition; and
- measured genes corresponding to known compound targets.

The exact union size and the handling of any fold-information leakage must be
reported before training. If the union is materially larger than 9,000 genes,
the panel definition must be revisited rather than silently truncated.

### Perturbation target genes

Target-gene presence is a model-specific requirement:

- GEARS requires the perturbed gene to be a measured feature and to be covered
  by its GO reference; otherwise its data path crashes or mislabels the example
  as unperturbed.
- scGPT's perturbation adapter reuses GEARS' `PertData` machinery and inherits
  the measured-target requirement.
- scLAMBDA requires an external GenePT embedding for every target, but the
  target does not inherently have to be an expression output.
- PRESAGE and TabICLv2 represent perturbations with external gene embeddings and
  do not inherently require the target to be an output feature.
- STATE conditions on the perturbation identity and can use a smaller internal
  expression representation while predicting the shared output space.

Because the primary leaderboard includes GEARS and scGPT, its common cohort must
currently require the target gene in the measured expression space. The 190
K562 targets failing this check must be audited for unambiguous symbol aliases
before they are declared truly unavailable. No ambiguous alias is remapped.

## Environment policy

- PerturbationData and evaluation/orchestration code use `uv`.
- Hugo's six already validated model environments remain in Pixi.
- The study does not spend time migrating known-working GPU environments from
  Pixi to `uv` before model training.
- scArchon models may initially use their published containers. Container
  revisions and hashes must be recorded in the model declarations.

## Sensitivity panel

Start from one reference pipeline and vary one factor family at a time:

- gene set;
- absolute versus control-centered representation;
- control construction;
- metric;
- uniform versus observed-effect-strength perturbation weighting; and
- mean, median, or effect-strength-weighted aggregation across perturbations.

Metrics remain separate. The initial study does not construct a cross-metric
composite or global leaderboard.

For each alternative report model ranks, rank changes, Spearman and Kendall
agreement, pairwise-order or winner reversals, and paired perturbation-bootstrap
uncertainty.

## Compute policy

- Beast is the preferred first workstation for smoke tests and model training;
  both RTX A6000 GPUs were idle at the last check.
- Beauty GPU 1 may be used after a compatibility smoke test. GPU 0 was occupied
  at the last check and must not be touched without rechecking ownership.
- Helix is used for high-memory preprocessing, overflow, and final tracked runs
  when workstation resources are insufficient.
- No model training is submitted until the corresponding experiment declaration
  and cohort manifest have been frozen.

## Reproducibility policy

- Raw datasets are downloaded from authoritative sources and are not
  redistributed.
- Public scripts reproduce raw-to-analysis processing without requiring private
  repositories.
- Exact model and baseline predictions are archived in a versioned Zenodo record,
  partitioned by dataset and model if needed.
- Figures consume versioned predictions or tidy result tables, never private
  data paths.

## Open decisions and required checks

1. Audit the 190 K562 target labels absent from the measured feature space and
   verify the scLAMBDA exclusions against the pinned GenePT artifact.
2. Materialize and report the exact sci-Plex3 approximately 9,000-gene panel.
3. Regenerate the genetic context cohort and model view at the 50-cell rule.
4. Smoke-test the leading scArchon candidates on one genetic and one chemical
   context fold, including whether a scientifically faithful joint-condition
   adapter is possible.
5. Freeze the final context-generalization model panels only after that
   feasibility check.
