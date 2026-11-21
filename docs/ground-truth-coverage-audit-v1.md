# Ground-truth protocol coverage audit v1

The ground-truth-only audit completed on 2026-11-21 before model ranks were
inspected. It truth-self-evaluated every task-applicable protocol built from the
99 frozen Atlas semantic families using scPertEval commit `3950557` and
reproducibility repository commit `c54d791`.

The authoritative 71 MB result bundle is retained at:

```text
/home/pschaefer/sds-hd/sd22b002/projects/scPertEval-reproducibility/results/ground-truth-coverage-v1
```

Its machine-readable provenance, file checksums, and exact summary values are
recorded in
`configs/evaluation/atlas-ground-truth-coverage-audit-v1.json`.

## Completeness

The audit contains 333,333 protocol/evaluation-unit rows across all 12 folds:

| Task | Folds | Applicable protocols | Evaluation units |
|---|---:|---:|---:|
| Essential perturbation generalization | 5 | 99 | 1,352 |
| Genetic context transfer | 4 | 99 | 1,920 (480 per fold) |
| Chemical context transfer | 3 | 95 | 99 (33 per fold) |

Of these rows, 329,972 are defined and 3,361 are undefined for an approved,
structured reason: 1,392 unmeasured target genes, 967 empty truth-derived
feature sets, 649 constant truth vectors, 334 cases with no ground-truth
positives, and 19 single-class truth sets.

## Matched protocol-pair coverage

The table reports each pair's worst fold, because the proposed rule requires
the threshold in every fold.

| Task | Protocol pairs | Pairs at candidate 90% | Fraction | Minimum | Median |
|---|---:|---:|---:|---:|---:|
| Essential perturbation generalization | 4,851 | 4,851 | 100.0% | 96.7% | 100.0% |
| Genetic context transfer | 4,851 | 4,465 | 92.0% | 77.5% | 98.1% |
| Chemical context transfer | 4,465 | 4,095 | 91.7% | 63.6% | 100.0% |

The essential-screen result supports the proposed 90% guardrail directly. The
other two tasks expose two scientifically interpretable exceptions rather than
runtime failures:

- Genetic target-policy protocols cover 393/480 targets (81.9%) in every fold,
  because 87 retained perturbation targets are absent from the shared measured
  gene universe. The most restrictive exclude-target DEG/Pearson protocol has
  79.2% coverage in its weakest fold after also accounting for empty or
  constant truth panels. At 90%, all pairwise comparisons involving the four
  target-policy protocols are excluded.
- Four chemical DEG/Pearson or squared-Pearson protocols fall below 90% in at
  least one context because some truth-derived 50- or 100-gene panels are
  constant. The weakest protocol covers 21/33 conditions (63.6%) in A549.

The 90% value therefore remains marked as a candidate rather than silently
frozen. Freezing it would be simple and conservative, but it would remove the
explicit genetic target-policy factor from the headline pairwise analysis. A
lower 75% guardrail would retain every genetic pair and 97.9% of chemical pairs,
while still excluding the weakest chemical protocol. This is a scientific
design decision, not an implementation decision, and must be made before model
ranks are inspected.

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
