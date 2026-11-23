# Ranking sensitivity: post-hoc changes to the protocol universe

## Decisions and disclosure

The prespecified ranking-sensitivity universe held 101 literature-eligible families from Atlas
`0be116b` (repository commit `aadbe94`). Two changes followed, both decided on 2026-11-22 after
model ranks over the prespecified universe had been inspected. **Both are post hoc and must be
disclosed in the Methods.**

1. **Retrieval ranks.** The recurring retrieval-rank families (Atlas
   `cross_perturbation_group == identity_matching`) were admitted. Over Atlas `0be116b` this gave
   111 families (repository commit `33a48f3`).
2. **Atlas and scPertEval update.** The universe was re-derived from the curated Atlas `0b0fca9`,
   and every protocol of every model bundle was rescored with scPertEval `774e600`. No score from an
   earlier universe is reused. The analysed universe has **115 families**.

The other recurring cross-perturbation families stay excluded, each with its own reason in
`configs/evaluation/atlas-protocol-scope-audit.csv`:

| Atlas group | Families | Reason |
|---|---:|---|
| `effect_size` | 4 | `cross_perturbation_effect_size_out_of_scope` |
| `pooled_response` | 2 | `pooled_response_out_of_scope` |

Effect-size families correlate effect magnitudes across perturbations. Pooled-response families
score one response pooled over perturbations. Neither scores one evaluation unit against its own
truth, which the matched-unit pair comparison requires.

Every analysis rule is unchanged: tasks and folds, the 75% minimum common coverage rule frozen on
2026-11-21, the undefined-score rules, equal protocol votes, fractional winner ties, mean
aggregation as primary, separate tasks, and the threshold of at least 10 matched contrasts for
display. Eligibility (at least two independent works) and the scope rules are unchanged too.

## Universe at Atlas `0b0fca9`

`atlas_protocol_audit.py` with `configs/evaluation/atlas-protocol-eligibility.json` reads Atlas
`0b0fca9` (`data/protocols.csv` SHA-256 `f05050d5…`). It yields 150 literature-eligible families, of
which 115 are in scope (`docs/atlas-protocol-eligibility.md`).

| Task | Prespecified (Atlas `0be116b`) | With retrieval (Atlas `0be116b`) | Analysed (Atlas `0b0fca9`) |
|---|---:|---:|---:|
| Perturbation generalization | 101 | 111 | 115 |
| Genetic context transfer | 101 | 111 | 115 |
| Chemical context transfer | 97 | 104 | 104 |

Relative to the 111 families, 102 are unchanged. The others differ because the Atlas curation
changed, not because any rule changed:

- **Target-gene policies resolved.** Most records with an unspecified policy now record one, mostly
  `ordinary`. Records that move to `force_include_target` or `exclude_own_target` leave their old
  family. This splits some families below two works and creates others: an uncentred all-gene R²
  family with forced target inclusion, and three Virtual Cell Challenge retrieval families that
  exclude the query target on the 2,000-HVG panel (MSE, MAE and cosine).
- **Metric labels reconciled.** `average-precision` and `deg-identification-auprc` became `auprc`;
  `class-precision` and the predicted-size DEG precision became `deg-set-precision`;
  `deg-set-intersection` became `topk-overlap`; `set-overlap-at-n` became
  `set-overlap-at-truth-size`. The labels are part of the family identity, so the families get new
  IDs. `auprc` and `deg-set-precision` were added to the DE scope as aliases of the metrics
  scPertEval already provides (`de_auprc`, `de_precision`). A DE AUPRC family reported by 9 works
  stays in the universe only because of these aliases.
- **New and regrouped families.** Control-centred Pearson on HEG panels (251–1,000 and over 1,000
  genes), on the top 1–50 control DEGs, and uncentred squared Pearson on HVGs reach two works.
- **Families that fell below two works.** Among them are the population MMD family in PCA-50
  space and the uncentred clipped-R² top-50 DEG family. The PCA-50 MMD family was part of a frozen
  bump example; it was replaced by the same Pearson-versus-population-MMD contrast on all genes
  (`configs/evaluation/ranking-sensitivity-atlas-examples.json`).

Newly out of scope, as before for metrics scPertEval does not provide: non-debiased
`sinkhorn-distance` (scPertEval implements the debiased divergence only) and
`symmetric-gaussian-kl`.

Two operational decisions were needed for the new families, both reusing existing rules:

- An unknown DEG ranking basis (Atlas `feature_ranking_basis = other`) takes the canonical
  ranking (`absolute_effect`), as an unspecified basis already did. This affects one family
  (`atlas-640989623469`, top-20/40 control DEGs).
- Own-target exclusion in retrieval now also works on a dataset-wide panel: each query's target is
  mapped from the full gene axis onto the panel and removed there. A measured target outside the
  panel has nothing to remove and the query stays defined; an unmeasured target stays undefined
  (`target_gene_not_measured`), as on the full axis.

## Retrieval operationalization

The universe has 13 retrieval families. All score control-centred or uncentred pseudobulk
profiles in gene space. The candidate set is every evaluation unit of the task fold. The score is
the zero-based rank of the matching candidate divided by `n − 1` (0 is a perfect top-1 retrieval,
lower is better). Exact ties are broken with the same seeded `[0, 1e-12)` noise that scPertEval's
`rank_retrieval` uses.

| Family | Query | Distance | Centring | Feature set | Target |
|---|---|---|---|---|---|
| `atlas-bc27e0942cc7` | prediction | squared Euclidean (MSE) | none | all genes | – |
| `atlas-fbb25436e8b0` | prediction | squared Euclidean (MSE) | control | all genes | – |
| `atlas-e6c567dcbbbd` | prediction | squared Euclidean (MSE) | control | HVG 2000 | – |
| `atlas-2715e9a4439c` | prediction | squared Euclidean (MSE) | control | HVG 2000 | query target excluded |
| `atlas-33a326484940` | prediction | L1 (MAE) | control | all genes | – |
| `atlas-fd5de292e067` | prediction | L1 (MAE) | control | HVG 2000 | – |
| `atlas-e1e926ba033b` | prediction | L1 (MAE) | control | all genes | query target excluded |
| `atlas-b94b1c70ad98` | prediction | L1 (MAE) | control | HVG 2000 | query target excluded |
| `atlas-08fea847a3af` | prediction | cosine | control | all genes | – |
| `atlas-e3787108346b` | prediction | cosine | control | HVG 2000 | – |
| `atlas-d81fdc541750` | prediction | cosine | control | HVG 2000 | query target excluded |
| `atlas-7ec3fb1dd670` | observation | cosine | control | union panel | all fold targets included |
| `atlas-ff0373c812cd` | observation | squared Euclidean (RMSE) | none | union panel | all fold targets included |

- *Query direction.* "Prediction" (Atlas *transposed* / `predicted_query_rank`, e.g. PDS, NIR)
  ranks each prediction's matching observed profile among all observed profiles
  (`rank_retrieval(transpose=True)`). "Observation" (PerturBench `rank`, `observed_query_rank`)
  ranks each observed profile's own prediction among all predictions (`transpose=False`).
- *Distance.* MSE, RMSE and Euclidean distance order a fixed-dimensional candidate set
  identically, so they all use squared Euclidean distance, computed exactly as in `rank_retrieval`
  (unit test: identical values). L1 and cosine are implemented in
  `analyses/ranking_sensitivity/retrieval.py` with the same orientation, tie-breaking and
  normalization. A zero-norm prediction has cosine similarity 0 to every candidate (the frozen
  constant-prediction rule).
- *Shared gene axis.* Retrieval stacks every unit into one matrix, so a protocol must not vary its
  genes by perturbation. "Target included" therefore adds every target gene of the fold
  (`spaces.perturbed_genes()`). Target exclusion (the Virtual Cell Challenge convention) removes the
  query's own target from every distance that query computes, on the full axis or on the HVG panel.
  A query whose target is not measured is undefined (`target_gene_not_measured`) but remains a
  candidate for the other queries. Target-including and target-excluding families apply only to
  the two genetic tasks.
- *Union panel.* Atlas `feature_set = Other` is the PerturBench panel: 4,000 HVGs, plus the union
  over the fold's evaluation units of each unit's top-25 truth-derived DEGs (adjusted p < 0.05,
  ranked by absolute effect, against controls), plus the targets. This feature set is admitted only
  for retrieval families.

## Cross-perturbation factor

Factor sensitivity includes *Cross-perturbation evaluation* (`cross_perturbation`), placed after
*Metric* in taxonomy order (`analyses/vocabulary.py`). A matched contrast changes only the strategy
— direct comparison, predicted-query retrieval or observed-query retrieval — while keeping the
distance (squared error, MAE or cosine) and every other choice. Retrieval protocols with different
distances remain score-function contrasts. Evaluation-family contrasts compare only direct
protocols.

## Execution

All steps ran on Beauty at repository commit `021b41c` (scPertEval `774e600`, Atlas `0b0fca9`).
New SDS directories, named by their pins, hold every output; earlier results are untouched.

- Ground-truth coverage: all 115 families truth-self-evaluated on all 12 folds in one pass
  (`results/ground-truth-coverage-atlas-0b0fca9-scperteval-774e600`;
  `docs/ground-truth-coverage-audit.md`).
- Scoring: the retained predictions of all model/fold bundles were scored with all applicable
  protocols into `results/ranking-sensitivity-atlas-0b0fca9-scperteval-774e600/details`.
  Completion records bind the prediction checksum, eligibility and manifest checksums, the
  scPertEval revision and the evaluator source hash.
- CellFlow: perturbation-generalization CellFlow was retrained on the full development pool (train
  plus validation perturbations, like the other models without validation-based selection) into
  `runs-cellflow-development-pool` and scored into the same root. The original train-only CellFlow
  runs were scored into `results/ranking-sensitivity-atlas-0b0fca9-scperteval-774e600-cellflow-trainonly`
  for comparison.
- Canary: before the full run, one bundle was scored at the new pins and compared with the previous
  scores. Of 102 shared families, 100 were bitwise identical; the two significance-ranked DE AUROC
  families differed by at most 1.2 × 10⁻⁶ (scPertEval's tie-preserving ranks).

## Results (2026-11-23)

The analysis (`results/ranking-sensitivity-atlas-0b0fca9-scperteval-774e600/analysis`) read one
details root with 120 bundles (65 perturbation-generalization, 28 genetic, 27 chemical). Every
completion record binds scPertEval `774e600`, the same eligibility checksum (`fdf58408…`), task
manifest checksum (`79c07799…`) and evaluator hash (`0f59263e…`). The five perturbation-generalization
CellFlow bundles are the development-pool runs.

Ground-truth coverage: 386,576 truth-self-evaluation rows, 4,997 undefined (3,132
`target_gene_not_measured` from the 9 target-policy protocols on the genetic tasks, 678
`constant_truth`, 587 `empty_feature_set`, 543 `no_ground_truth_positives`, 57
`single_ground_truth_class`). One chemical protocol fails the 75% rule in every comparison, as
before, so 103 protocols enter the chemical analyses (115 / 115 / 103).

| Task | Prespecified (101 families) | With retrieval (111 families) | Analysed (115 families) |
|---|---:|---:|---:|
| Perturbation generalization | 48% | 53.2% | 53.9% (3,530 / 6,555) |
| Genetic context transfer | 18% | 16.7% | 16.1% (1,057 / 6,555) |
| Chemical context transfer | 73% | 72.2% | 71.8% (3,772 / 5,253) |

The first two columns were computed at Atlas `0be116b` and scPertEval `58ccc91` and are kept for the
disclosure; they are not recomputed.

Winner credit (mean aggregation):

- Perturbation generalization: TabICLv2 75 of 115, scGPT 19, PRESAGE 17, fMLP-GenePT 3, CellFlow 1.
- Genetic context transfer: training mean 105 of 115, CPA 8, scDisInFact 1, STATE 1.
- Chemical context transfer: SAMS-VAE 46 of 103, no change 22, trVAE 17, linear 12, training mean 3,
  scDisInFact 2, STATE 1.

Factor sensitivity (winner changes / matched contrasts; perturbation generalization, genetic,
chemical): representation 16/64, 16/64, 48/62; perturbation effect isolation 3/20, 3/20, 3/19;
feature selection 57/177, 8/177, 71/171; metric 42/145, 32/145, 55/139; aggregation 23/115, 22/115,
37/103. Cross-perturbation evaluation has 7 matched contrasts per task (7, 2 and 2 winner changes),
target-gene policy 9 (genetic tasks only, no winner change), and expression space and weighting 1
each; all are below the 10-contrast display rule.

**CellFlow retraining.** The development-pool CellFlow and the train-only CellFlow (scored at the
same pins into `…-cellflow-trainonly`) give identical winners for every protocol, so every count
above is unchanged. CellFlow ranks better under 19 protocols, worse under 1 and equally under 95
(mean rank 8.74 → 8.58 of 13). Mean pairwise reversal fractions change by less than 0.002.
