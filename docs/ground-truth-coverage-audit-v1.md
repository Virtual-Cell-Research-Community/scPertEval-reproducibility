# Ground-truth protocol coverage audit v1

The ground-truth-only audit completed on 2026-11-21 before model ranks were
inspected. It truth-self-evaluated every task-applicable protocol built from the
101 frozen Atlas semantic families using scPertEval commit `3950557`. The 75%
coverage decision was frozen in reproducibility repository commit `b6d6170`.

The authoritative 72 MB result bundle is retained at:

```text
/home/pschaefer/sds-hd/sd22b002/projects/scPertEval-reproducibility/results/ground-truth-coverage-v1
```

Its machine-readable provenance, file checksums, and exact summary values are
recorded in
`configs/evaluation/atlas-ground-truth-coverage-audit-v1.json`.

## Completeness

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

## Matched protocol-pair coverage

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

## Runtime corrections discovered by the audit

Two issues were found and fixed before the final tables were produced:

1. A DEG selector now defines positives while classification metrics retain
   non-DE genes as negatives; otherwise DEG AUROC was incorrectly single-class.
2. Target-gene operations now prefer canonical `obs["target_gene"]` metadata,
   allowing namespaced context-transfer labels such as `dataset::GENE` to map
   correctly.

The complete availability table and both per-fold and cross-fold pair tables
are retained in the authoritative result bundle, so every value above is
recomputable and auditable.
