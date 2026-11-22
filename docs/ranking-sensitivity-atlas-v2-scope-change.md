# Ranking-sensitivity v2: retrieval-rank scope extension

## Decision and disclosure

On 2026-11-22 the protocol universe of the Atlas ranking-sensitivity analysis was extended from
101 to 111 literature-eligible families by admitting the ten recurring retrieval-rank families
(Atlas `cross_perturbation_group == identity_matching`). **This change is post hoc.** It was
decided after the v1 model ranks had been inspected, so v2 is not a prespecified analysis.
The Methods must state this. v2 is the primary result in Fig. 2D–E; v1 is kept unchanged
(`configs/evaluation/*-v1.*`, SDS `results/ranking-sensitivity-atlas-v1`) and is reported in the
supplement.

The other recurring cross-perturbation families stay excluded, each with its own reason in
`configs/evaluation/atlas-protocol-scope-audit-v2.csv`:

| Atlas group | Families | v2 reason |
|---|---:|---|
| `effect_size` | 3 | `cross_perturbation_effect_size_out_of_scope` |
| `pooled_response` | 3 | `pooled_response_out_of_scope` |

Effect-size families correlate effect magnitudes across perturbations. Pooled-response families
score one response pooled over perturbations. Neither scores one evaluation unit against its own
truth, which the matched-unit pair comparison requires. Retrieval ranks do produce one score per
evaluation unit, so they fit the existing analysis unchanged.

Nothing else changed: the scPertEval revision (`58ccc91`), tasks and folds, the frozen 75%
minimum common coverage rule, the undefined-score rules, equal protocol votes, fractional winner
ties, mean aggregation as primary, separate tasks, the frozen bump examples, and the threshold of
at least 10 matched contrasts for display.

## Universe

`atlas_protocol_audit.py` with `configs/evaluation/atlas-protocol-eligibility-v2.json` reads the
same Atlas revision (`0be116b`, `data/protocols.csv` SHA-256 `cbd08885…`). It yields 142
literature-eligible families, of which 111 are in scope (101 + 10). The 101 v1 rows are identical
in the v2 table, and the generator still reproduces every v1 file byte-for-byte from the v1 config.

| Task | v1 protocols | v2 protocols |
|---|---:|---:|
| Perturbation generalization | 101 | 111 |
| Genetic context transfer | 101 | 111 |
| Chemical context transfer | 97 | 104 |

Three retrieval families include or exclude the target gene, so under the unchanged v1 rule they
apply only to the two genetic tasks.

## Operationalization

All ten families score control-centred or uncentred pseudobulk profiles in gene space. The
candidate set is every evaluation unit of the task fold. The score is the zero-based rank of the
matching candidate divided by `n − 1` (0 is a perfect top-1 retrieval, lower is better). Exact
ties are broken with the same seeded `[0, 1e-12)` noise that scPertEval's `rank_retrieval` uses.

| Family | Query | Distance | Centring | Feature set | Target |
|---|---|---|---|---|---|
| `atlas-bc27e0942cc7` | prediction | squared Euclidean (MSE) | none | all genes | – |
| `atlas-fbb25436e8b0` | prediction | squared Euclidean (MSE) | control | all genes | – |
| `atlas-e6c567dcbbbd` | prediction | squared Euclidean (MSE) | control | HVG 2000 | – |
| `atlas-33a326484940` | prediction | L1 (MAE) | control | all genes | – |
| `atlas-fd5de292e067` | prediction | L1 (MAE) | control | HVG 2000 | – |
| `atlas-e1e926ba033b` | prediction | L1 (MAE) | control | all genes | query target excluded |
| `atlas-08fea847a3af` | prediction | cosine | control | all genes | – |
| `atlas-e3787108346b` | prediction | cosine | control | HVG 2000 | – |
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
  normalization, so the scPertEval pin does not change. A zero-norm prediction has cosine
  similarity 0 to every candidate (the frozen constant-prediction rule).
- *Shared gene axis.* Retrieval stacks every unit into one matrix, so a protocol must not vary its
  genes by perturbation. "Target included" therefore adds every target gene of the fold
  (`spaces.perturbed_genes()`). The Virtual Cell Challenge target exclusion removes the query's own
  target from every distance that query computes. A query whose target is not measured is
  undefined (`target_gene_not_measured`) but remains a candidate for the other queries.
- *Union panel.* Atlas `feature_set = Other` is the PerturBench panel: 4,000 HVGs, plus the union
  over the fold's evaluation units of each unit's top-25 truth-derived DEGs (adjusted p < 0.05,
  ranked by absolute effect, against controls, i.e. the canonical v1 DEG selector), plus the
  targets. This feature set is admitted only for retrieval families; the five direct-comparison
  families with `Other` panels stay excluded.

## Cross-perturbation factor

Factor sensitivity gains one factor, *Cross-perturbation evaluation* (`cross_perturbation`),
placed after *Metric* in taxonomy order (`analyses/vocabulary.py`). A matched contrast changes only
the strategy — direct comparison, predicted-query retrieval or observed-query retrieval — while
keeping the distance (squared error, MAE or cosine) and every other choice. Retrieval protocols
with different distances remain score-function contrasts. Evaluation-family contrasts still
compare only direct protocols. Every v1 protocol pair therefore keeps its v1 factor classification.

## Execution

- Ground-truth coverage: the 10 added protocols were truth-self-evaluated on all 12 folds with
  `truth_coverage_runtime --eligibility-version v2 --added-since v1`. Their rows were appended to
  the unchanged v1 availability table and summarized with
  `configs/evaluation/ranking-sensitivity-atlas-v2.json` (SDS `results/ground-truth-coverage-v2`).
- Scoring: only the added protocols were scored, from the retained predictions of all 120
  model/fold bundles, on Beauty only (`evaluate_atlas_production_benchmark.py
  --eligibility-version v2 --added-since v1`, SDS `results/ranking-sensitivity-atlas-v2/details`).
  Schema-3 completion records bind the prediction checksum, v2 eligibility and manifest checksums,
  the v1 baseline manifest, the exact scored protocol list, the scPertEval revision and the
  evaluator source hash. v1 scores for the other 101 protocols are reused unchanged.
- Analysis: `atlas_analysis --details-root <v1 details> --details-root <v2 details>` merges the
  two disjoint protocol subsets. It refuses bundles that differ between roots or that bind
  different prediction checksums.

## Results (2026-11-22)

All runs used repository commit `caee425` on Beauty.

**Coverage.** 33,413 availability rows were added to the 340,075 v1 rows; the provenance record
is `configs/evaluation/atlas-ground-truth-coverage-audit-v2.json`. 348 added rows are undefined,
all `target_gene_not_measured`: the 87 genetic targets per fold without a measured gene, under the
target-excluding MAE family. Its weakest fold covers 81.9%. Against the frozen 75% rule:
essential 6,105/6,105 pairs pass, genetic 6,105/6,105 and chemical 5,253/5,356. The 103 failing
chemical pairs all involve the same uncentred top-50 DEG Pearson protocol excluded in v1.

**Scoring.** All 120 bundles completed (65 essential, 28 genetic, 27 chemical) with 316,397 new
score rows and no errors. The combined analysis table has 3,535,440 rows. Sanity check: the
no-change baseline scores exactly 0.5 under predicted-query MSE and MAE retrieval and about 0.5
under cosine. The uncentred and control-centred all-gene MSE families (`atlas-bc27e0942cc7`,
`atlas-fbb25436e8b0`) give identical scores. Prediction and truth are centred on the same
observed control mean, and squared distances do not change when both are shifted by the same
vector. The Atlas records some works that centre predictions on predicted controls instead; v1
does not model that variant. Both families still keep their own vote, as decided.

**Headline (mean aggregation, eligible pairs).**

| Task | Protocols (headline) | Winner disagreement v1 | Winner disagreement v2 |
|---|---:|---:|---:|
| Perturbation generalization | 111 (111) | 47.8% (2,413/5,050) | 53.2% (3,247/6,105) |
| Genetic context transfer | 111 (111) | 18.3% (922/5,050) | 16.7% (1,021/6,105) |
| Chemical context transfer | 104 (103) | 72.7% (3,315/4,560) | 72.2% (3,793/5,253) |

Winner shares in v2 (v1 in parentheses): perturbation generalization TabICL 65.8% (70.3%), scGPT
18.0% (19.8%), PRESAGE 13.5% (7.9%), fMLP-GenePT 1.8% (1.0%), CellFlow 0.9% (1.0%). Genetic
context transfer: fold-mean 91.0% (90.1%), CPA 7.2% (7.9%), scDisInFact 0.9%, STATE 0.9%.
Chemical context transfer: SAMS-VAE(S) 43.7% (41.7%), no-change 23.3% (25.0%), trVAE 15.5%
(16.7%), linear 10.7% (11.5%), fold-mean 3.9% (2.1%), scDisInFact 1.9%, STATE 1.0%. Retrieval
winners: PRESAGE in 7 of 8 predicted-query families and TabICL in both observed-query families
(essential); fold-mean in all 10 (genetic); SAMS-VAE(S) for MSE and MAE and fold-mean for cosine
(chemical).

**Cross-perturbation factor.** Each task has 7 matched direct-versus-retrieval contrasts (3
squared error, 2 MAE, 2 cosine). The target-excluding MAE and the two observed-query families
have no direct partner. Winner changes: essential 7/7, genetic 2/7, chemical 2/7. With fewer than
10 contrasts per task, the frozen display rule leaves this row empty in Fig. 2E; the complete
values remain in `factor-sensitivity.csv` and `concrete-factor-contrasts.csv`. The other
factor-sensitivity rows changed only marginally relative to v1.
