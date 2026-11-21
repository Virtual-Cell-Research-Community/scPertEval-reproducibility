# Atlas protocol eligibility audit v1

This audit freezes the ranking-sensitivity protocol universe against PerturbationEvaluationAtlas
commit `6f6e858097c0d8135d27c609040906d73d35ce55` and `data/protocols.csv` SHA-256 `3ae18cd1f7ea336c359cbbbfd9c575e4c88469ac046b8ecf02733e0a3779eedd`.

Eligibility means that an operationally distinct family occurs in at least
2 independent works. Reporting aggregation and DE caller are
audited separately from atomic protocol identity. `ordinary`, `unspecified`, and `not_applicable`
target policies all execute as `no_target_operation`; their raw Atlas values remain in the CSV.

## Result

- Literature-eligible operational families before scope exclusions: 137
- Families in the approved ranking-sensitivity scope: 100
- Families excluded by the approved scope: 37

### Included by representation

- `DE`: 34
- `PBULK`: 56
- `SC`: 10

### Included by target operation

- `exclude_own_target`: 2
- `force_include_target`: 2
- `no_target_operation`: 96

### Implementation status

- `de_recovery_module`: 23
- `direct_after_public_api`: 28
- `multiple_extensions`: 17
- `new_scalar_metric`: 14
- `selector_or_composition`: 18

### Scope exclusions

- `centering_out_of_scope`: 1
- `feature_set_out_of_scope`: 5
- `metric_out_of_scope`: 8
- `pooled_or_cross_perturbation`: 16
- `population_out_of_scope`: 3
- `reference_population_out_of_scope`: 4

## DE caller audit

The DE caller does not define atomic-family eligibility in this version. Existing scPertEval callers remain the
canonical implementations unless the literature audit below reveals a missing, sufficiently common
pipeline worth discussing before implementation.

- `limma`: 3 works globally, 2 supporting included families; scPertEval `not_implemented`
- `other`: 6 works globally, 5 supporting included families; scPertEval `not_actionable`
- `rank_sum`: 41 works globally, 28 supporting included families; scPertEval `existing`
- `t_test`: 24 works globally, 21 supporting included families; scPertEval `existing`
- `unspecified`: 107 works globally, 82 supporting included families; scPertEval `use_family_canonical_backend`

DE-based metrics are applicable to all three benchmark tasks because each has observed and predicted
cell populations (including the already-defined control-lifted population view for point models).
Explicit target operations apply only to genetic tasks and require an unambiguous measured target gene.

## Files

- `atlas-protocol-eligibility-v1.csv`: included families only.
- `atlas-protocol-scope-audit-v1.csv`: every literature-eligible family and its inclusion/exclusion reason.
- `atlas-de-method-audit-v1.csv`: DE caller usage and current scPertEval support.
