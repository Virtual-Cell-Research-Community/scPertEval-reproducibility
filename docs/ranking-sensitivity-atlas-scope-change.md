# Ranking sensitivity: retrieval-rank scope extension

## Decision and disclosure

On 2026-11-22 the protocol universe of the Atlas ranking-sensitivity analysis was extended from
the prespecified 101 literature-eligible families to 111. The extension admits the ten recurring
retrieval-rank families (Atlas `cross_perturbation_group == identity_matching`). **This change is
post hoc**: it was decided after model ranks over the prespecified universe had been inspected.
The Methods must say so. The analysis reported in Fig. 2D–E uses all 111 families. The
prespecified universe, its configuration and its results are frozen in repository commit
`aadbe94`; the repository no longer keeps a parallel copy.

The other recurring cross-perturbation families stay excluded, each with its own reason in
`configs/evaluation/atlas-protocol-scope-audit.csv`:

| Atlas group | Families | Reason |
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

`atlas_protocol_audit.py` with `configs/evaluation/atlas-protocol-eligibility.json` reads the
same Atlas revision (`0be116b`, `data/protocols.csv` SHA-256 `cbd08885…`). It yields 142
literature-eligible families, of which 111 are in scope (101 prespecified + 10 retrieval). The
101 prespecified rows are identical to those in commit `aadbe94`.

| Task | Prespecified protocols | Protocols analysed |
|---|---:|---:|
| Perturbation generalization | 101 | 111 |
| Genetic context transfer | 101 | 111 |
| Chemical context transfer | 97 | 104 |

Three retrieval families include or exclude the target gene, so under the unchanged task rule
they apply only to the two genetic tasks.

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
  ranked by absolute effect, against controls, i.e. the canonical DEG selector), plus the
  targets. This feature set is admitted only for retrieval families; the five direct-comparison
  families with `Other` panels stay excluded.

## Cross-perturbation factor

Factor sensitivity gains one factor, *Cross-perturbation evaluation* (`cross_perturbation`),
placed after *Metric* in taxonomy order (`analyses/vocabulary.py`). A matched contrast changes only
the strategy — direct comparison, predicted-query retrieval or observed-query retrieval — while
keeping the distance (squared error, MAE or cosine) and every other choice. Retrieval protocols
with different distances remain score-function contrasts. Evaluation-family contrasts still
compare only direct protocols, so every pair of prespecified protocols keeps its original factor
classification.

## Execution

The 101 prespecified protocols were not rescored; their existing scores were reused. Only the ten
retrieval protocols were evaluated, at repository commit `caee425` on Beauty, with the
incremental options that commit provided (`--eligibility-version v2 --added-since v1`). Those
options have since been removed: a fresh run scores all 111 protocols in one pass.

- Ground-truth coverage: the ten retrieval protocols were truth-self-evaluated on all 12 folds,
  and their rows were appended to the unchanged prespecified availability table (SDS
  `results/ground-truth-coverage-v2`).
- Scoring: retained predictions of all 120 model/fold bundles were scored into SDS
  `results/ranking-sensitivity-atlas-v2/details`, next to the prespecified scores in
  `results/ranking-sensitivity-atlas-v1/details`. The SDS directory names are historical.
  Completion records bind the prediction checksum, eligibility and manifest checksums, the scored
  protocol list, the scPertEval revision and the evaluator source hash.
- Analysis: `atlas_analysis` with both `--details-root` arguments merges the two disjoint protocol
  subsets. It refuses bundles that differ between roots or that bind different prediction
  checksums. Output: SDS `results/ranking-sensitivity-atlas-v2/analysis-v2`.

## Results (2026-11-22)

**Coverage.** 33,413 availability rows were added to the 340,075 prespecified rows; provenance is
in `configs/evaluation/atlas-ground-truth-coverage-audit.json`. 348 added rows are undefined, all
`target_gene_not_measured`: the 87 genetic targets per fold without a measured gene, under the
target-excluding MAE family. Its weakest fold covers 81.9%, above the 75% rule. Against the rule:
essential 6,105/6,105 pairs pass, genetic 6,105/6,105 and chemical 5,253/5,356. The 103 failing
chemical pairs all involve the same uncentred top-50 DEG Pearson protocol that was already
excluded in the prespecified analysis.

**Scoring.** All 120 bundles completed (65 essential, 28 genetic, 27 chemical) with 316,397 new
score rows and no errors. The combined analysis table has 3,535,440 rows. Sanity check: the
no-change baseline scores exactly 0.5 under predicted-query MSE and MAE retrieval and about 0.5
under cosine. The uncentred and control-centred all-gene MSE families (`atlas-bc27e0942cc7`,
`atlas-fbb25436e8b0`) give identical scores. Prediction and truth are centred on the same
observed control mean, and squared distances do not change when both are shifted by the same
vector. The Atlas records some works that centre predictions on predicted controls instead; the
analysis does not model that variant. Both families still keep their own vote, as decided.

**Headline (mean aggregation, eligible pairs).**

| Task | Protocols (headline) | Winner disagreement, prespecified | Winner disagreement, analysed |
|---|---:|---:|---:|
| Perturbation generalization | 111 (111) | 47.8% (2,413/5,050) | 53.2% (3,247/6,105) |
| Genetic context transfer | 111 (111) | 18.3% (922/5,050) | 16.7% (1,021/6,105) |
| Chemical context transfer | 104 (103) | 72.7% (3,315/4,560) | 72.2% (3,793/5,253) |

Winner shares over 111 families (prespecified universe in parentheses): perturbation
generalization TabICL 65.8% (70.3%), scGPT 18.0% (19.8%), PRESAGE 13.5% (7.9%), fMLP-GenePT 1.8%
(1.0%), CellFlow 0.9% (1.0%). Genetic context transfer: fold-mean 91.0% (90.1%), CPA 7.2% (7.9%),
scDisInFact 0.9%, STATE 0.9%. Chemical context transfer: SAMS-VAE(S) 43.7% (41.7%), no-change
23.3% (25.0%), trVAE 15.5% (16.7%), linear 10.7% (11.5%), fold-mean 3.9% (2.1%), scDisInFact 1.9%,
STATE 1.0%. Retrieval winners: PRESAGE in 7 of 8 predicted-query families and TabICL in both
observed-query families (essential); fold-mean in all 10 (genetic); SAMS-VAE(S) for MSE and MAE
and fold-mean for cosine (chemical).

**Cross-perturbation factor.** Each task has 7 matched direct-versus-retrieval contrasts (3
squared error, 2 MAE, 2 cosine). The target-excluding MAE and the two observed-query families
have no direct partner. Winner changes: essential 7/7, genetic 2/7, chemical 2/7. With fewer than
10 contrasts per task, the frozen display rule leaves this row empty in Fig. 2E; the complete
values remain in `factor-sensitivity.csv` and `concrete-factor-contrasts.csv`. The other
factor-sensitivity rows changed only marginally relative to the prespecified analysis.
