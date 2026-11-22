# Atlas protocol eligibility audit

This audit freezes the ranking-sensitivity protocol universe against PerturbationEvaluationAtlas
commit `0b0fca9dc997fd531c686fedeb95aa66c68c4d24` and `data/protocols.csv` SHA-256 `f05050d5e270870faad74843ad421eec3d86785d243837f76ae1411a3ab21514`.

Eligibility means that an operationally distinct family occurs in at least
2 independent works. Reporting aggregation and DE caller are
audited separately from atomic protocol identity. `ordinary`, `unspecified`, and `not_applicable`
target policies all execute as `no_target_operation`; their raw Atlas values remain in the CSV.

## Result

- Literature-eligible operational families before scope exclusions: 150
- Families in the approved ranking-sensitivity scope: 115
- Families excluded by the approved scope: 35

### Included by representation

- `DE`: 32
- `PBULK`: 73
- `SC`: 10

### Included by target operation

- `exclude_own_target`: 6
- `force_include_target`: 5
- `no_target_operation`: 104

### Implementation status

- `de_recovery_module`: 20
- `direct_after_public_api`: 29
- `multiple_extensions`: 25
- `new_scalar_metric`: 14
- `retrieval_module`: 7
- `selector_or_composition`: 20

### Scope exclusions

- `centering_out_of_scope`: 1
- `cross_perturbation_effect_size_out_of_scope`: 4
- `de_method_postponed`: 1
- `feature_set_out_of_scope`: 5
- `metric_out_of_scope`: 13
- `pooled_response_out_of_scope`: 2
- `population_out_of_scope`: 3
- `reference_population_out_of_scope`: 6

## Cross-perturbation scope

Admitted the recurring identity-matching (retrieval-rank) cross-perturbation families; effect-size and pooled-response cross-perturbation families stay excluded with distinct reasons. Decided after model ranks over the prespecified 101-family universe had been inspected. That prespecified universe is frozen in commit `aadbe94`.

Included cross-perturbation groups: `identity_matching`. Additional feature sets admitted only for
these groups: `identity_matching`: `Other`. Other recurring cross-perturbation groups remain excluded with
a group-specific reason:

- `effect_size` → `cross_perturbation_effect_size_out_of_scope`
- `pooled_response` → `pooled_response_out_of_scope`

## DE caller audit

The DE caller does not define atomic-family eligibility. Existing scPertEval callers remain the
canonical implementations unless the literature audit below reveals a missing, sufficiently common
pipeline worth discussing before implementation.

Limma-dependent families remain visible in the scope audit but are postponed from the executable primary
universe under the frozen DE-method policy.

- `limma`: 2 works globally, 0 supporting included families; scPertEval `not_implemented`
- `other`: 6 works globally, 3 supporting included families; scPertEval `not_actionable`
- `rank_sum`: 47 works globally, 33 supporting included families; scPertEval `existing`
- `t_test`: 27 works globally, 22 supporting included families; scPertEval `existing`
- `unspecified`: 100 works globally, 72 supporting included families; scPertEval `use_family_canonical_backend`

DE-based metrics are applicable to all three benchmark tasks because each has observed and predicted
cell populations (including the already-defined control-lifted population view for point models).
Explicit target operations apply only to genetic tasks and require an unambiguous measured target gene.

## Files

- `atlas-protocol-eligibility.csv`: included families only.
- `atlas-task-protocol-manifest.csv`: one row per applicable task and included family.
- `atlas-protocol-scope-audit.csv`: every literature-eligible family and its inclusion/exclusion reason.
- `atlas-de-method-audit.csv`: DE caller usage and current scPertEval support.
