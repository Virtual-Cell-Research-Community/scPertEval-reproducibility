# Initial study design

## Question

Can reasonable post-training evaluation choices change model rankings when the
trained models and their predictions are held fixed?

No model is retrained when gene selection, representation, controls, metrics,
weighting, or aggregation changes.

## Benchmark tracks

### Unseen perturbations

- Replogle K562 essential: implementation pilot.
- Replogle K562 GWPS: final large genetic benchmark.
- Nadig HepG2: independent genetic benchmark.
- Seeded perturbation-level five-fold cross-validation with explicit split
  files shared by every eligible model and baseline.

### Context transfer

- Genetic: Replogle–Nadig K562, RPE1, HepG2, and Jurkat.
- Chemical: sci-Plex3.
- Leave one context out while retaining its controls and requiring evaluated
  perturbations to be represented in the training contexts.

The canonical Replogle–Nadig objects contain 1,918 targets shared across all
four contexts. A fixed minimum of 60 treated cells in every context retains 322
perturbations, or 1,288 held-out context–perturbation populations over four
folds. The same 322 perturbations are used for profile and population metrics.

## Eligibility strata

- **Point-capable:** one predicted expression profile per test condition;
  eligible for profile and centroid metrics.
- **Population-capable:** at least 60 predicted cells per test condition;
  eligible for both profile and distributional metrics.

Neither stratum is treated as intrinsically primary. Unsupported metrics remain
missing rather than being imputed.

For distributional metrics, compare matched draws of exactly 60 observed and 60
predicted cells. Repeat the draw with ten prespecified seeds and average within
each context–perturbation population before aggregation across perturbations.

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

## Reproducibility policy

- Raw datasets are downloaded from authoritative sources and are not
  redistributed.
- Public scripts reproduce raw-to-analysis processing without requiring private
  repositories.
- Exact model and baseline predictions are archived in a versioned Zenodo
  record, partitioned by dataset and model if needed.
- Figures consume versioned predictions or tidy result tables, never private
  data paths.

