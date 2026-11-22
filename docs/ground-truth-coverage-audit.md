# Ground-truth protocol coverage audit

The current audit covers the 115-family universe derived from Atlas `0b0fca9`, truth-self-evaluated
at scPertEval `774e600` on all 12 folds in one pass (2026-11-22). The frozen 75% rule was not
changed; this audit was computed after model ranks over earlier universes had been inspected
(`docs/ranking-sensitivity-atlas-scope-change.md`). Machine-readable provenance, checksums and
summary values are in `configs/evaluation/atlas-ground-truth-coverage-audit.json`; the result
bundle is at:

```text
/home/pschaefer/sds-hd/sd22b002/projects/scPertEval-reproducibility/results/ground-truth-coverage-atlas-0b0fca9-scperteval-774e600
```

## Current audit (115 families)

The availability table has 386,576 protocol/evaluation-unit rows. 381,579 rows are defined and
4,997 are undefined: 3,132 unmeasured target genes, 678 constant truth vectors, 587 empty
truth-derived feature sets, 543 cases with no ground-truth positives, and 57 single-class truth
sets. Unmeasured targets arise only in genetic context transfer (87 of 480 targets per fold), under
nine target-including or target-excluding protocols (348 rows each).

| Task | Applicable protocols | Protocol pairs | Pairs at frozen 75% | Fraction | Minimum | Median |
|---|---:|---:|---:|---:|---:|---:|
| Essential perturbation generalization | 115 | 6,555 | 6,555 | 100.0% | 96.7% | 100.0% |
| Genetic context transfer | 115 | 6,555 | 6,555 | 100.0% | 77.5% | 98.1% |
| Chemical context transfer | 104 | 5,356 | 5,253 | 98.1% | 63.6% | 100.0% |

Minimum and median are taken over each pair's worst fold. As in every earlier audit, the 103
failing chemical pairs are exactly the comparisons involving the uncentred top-50 DEG Pearson
protocol (`atlas-dc86e02d8006`, 63.6% coverage in its weakest fold), so 103 chemical protocols
enter the headline analyses. The weakest genetic protocols cover 79–82% of units in their worst
fold, including the new target-excluding HVG retrieval families (81.9%), and pass the rule.

## Previous universe (111 families)

Recorded before the Atlas update, at Atlas `0be116b` and scPertEval `58ccc91`
(`results/ground-truth-coverage-v2` on SDS):

Its combined availability table has 373,488 protocol/evaluation-unit rows (the 340,075 rows of the
prespecified audit unchanged, plus 33,413 retrieval rows). 369,883 rows are defined and 3,605 are
undefined: 1,740 unmeasured target genes, 678 constant truth vectors, 587 empty truth-derived
feature sets, 543 cases with no ground-truth positives, and 57 single-class truth sets. All 348
undefined retrieval rows are unmeasured targets of the target-excluding MAE retrieval family.

| Task | Applicable protocols | Protocol pairs | Pairs at frozen 75% | Fraction | Minimum | Median |
|---|---:|---:|---:|---:|---:|---:|
| Essential perturbation generalization | 111 | 6,105 | 6,105 | 100.0% | 96.7% | 100.0% |
| Genetic context transfer | 111 | 6,105 | 6,105 | 100.0% | 77.5% | 98.1% |
| Chemical context transfer | 104 | 5,356 | 5,253 | 98.1% | 63.6% | 100.0% |

Minimum and median are taken over each pair's worst fold. The 103 failing chemical pairs are
exactly the comparisons involving the same weakest chemical protocol as before. The
target-excluding MAE retrieval family covers 393/480 genetic targets (81.9%) in every fold, the same
limit as the other target-policy protocols, and passes the rule.

## Prespecified audit (101 families)

The audit below is reproduced as recorded before the retrieval families were added. It was run
at scPertEval commit `3950557`. The 111-family table above uses the same 340,075 rows re-evaluated at
scPertEval `58ccc91` (`docs/scperteval-main-ranking-migration.md`). That re-evaluation has 3,257
undefined rows rather than the 3,428 listed below, and its pair counts are unchanged.

The ground-truth-only audit completed on 2026-11-21 before model ranks were
inspected. It truth-self-evaluated every task-applicable protocol built from the
101 frozen Atlas semantic families using scPertEval commit `3950557`. The 75%
coverage decision was frozen in reproducibility repository commit `b6d6170`.

Its authoritative 72 MB result bundle is retained at:

```text
/home/pschaefer/sds-hd/sd22b002/projects/scPertEval-reproducibility/results/ground-truth-coverage-v1
```

Its checksums are recorded in the history of
`configs/evaluation/atlas-ground-truth-coverage-audit.json` (commit `aadbe94`).

### Completeness

The audit contains 340,075 protocol/evaluation-unit rows across all 12 folds:

| Task | Folds | Applicable protocols | Evaluation units |
|---|---:|---:|---:|
| Essential perturbation generalization | 5 | 101 | 1,352 |
| Genetic context transfer | 4 | 101 | 1,920 (480 per fold) |
| Chemical context transfer | 3 | 97 | 99 (33 per fold) |

Of these rows, 336,647 are defined and 3,428 are undefined for an approved,
structured reason: 1,392 unmeasured target genes, 1,005 empty truth-derived
feature sets, 678 constant truth vectors, 334 cases with no ground-truth
positives, and 19 single-class truth sets.

### Matched protocol-pair coverage

The table reports each pair's worst fold, because the proposed rule requires
the threshold in every fold.

| Task | Protocol pairs | Pairs at frozen 75% | Fraction | Minimum | Median |
|---|---:|---:|---:|---:|---:|
| Essential perturbation generalization | 5,050 | 5,050 | 100.0% | 96.7% | 100.0% |
| Genetic context transfer | 5,050 | 5,050 | 100.0% | 77.5% | 98.1% |
| Chemical context transfer | 4,656 | 4,560 | 97.9% | 63.6% | 100.0% |

The audit exposed two scientifically interpretable coverage limitations rather
than runtime failures:

- Genetic target-policy protocols cover 393/480 targets (81.9%) in every fold,
  because 87 retained perturbation targets are absent from the shared measured
  gene universe. The most restrictive exclude-target DEG/Pearson protocol has
  79.2% coverage in its weakest fold after also accounting for empty or
  constant truth panels.
- Four chemical DEG/Pearson or squared-Pearson protocols fall below 90% in at
  least one context because some truth-derived 50- or 100-gene panels are
  constant. Only the weakest falls below the frozen 75% rule, covering 21/33
  conditions (63.6%) in A549.

The minimum shared coverage is frozen at 75% in every fold. This retains every
essential and genetic protocol pair, including the prespecified target-policy
factor, and 97.9% of chemical pairs. The 96 failing chemical pairs are exactly
the comparisons involving the weakest chemical protocol. That protocol is
excluded from headline winner shares and pairwise analyses, while its results
and structured missingness remain in the source tables. The threshold was
selected from ground-truth coverage before model ranks were inspected.

### Runtime corrections discovered by the audit

Two issues were found and fixed before the final tables were produced:

1. A DEG selector now defines positives while classification metrics retain
   non-DE genes as negatives; otherwise DEG AUROC was incorrectly single-class.
2. Target-gene operations now prefer canonical `obs["target_gene"]` metadata,
   allowing namespaced context-transfer labels such as `dataset::GENE` to map
   correctly.

The complete availability table and both per-fold and cross-fold pair tables
are retained in the authoritative result bundle, so every value above is
recomputable and auditable.
