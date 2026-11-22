# Atlas protocol eligibility audit

This audit freezes the ranking-sensitivity protocol universe against PerturbationEvaluationAtlas
commit `0be116b2a871132c2b3f0c9e896e2fb544a41fe5` and `data/protocols.csv` SHA-256 `cbd088852e213efc92c6af3c633a46fb0eef31efd14667b3d200da6e039f2280`.

Eligibility means that an operationally distinct family occurs in at least
2 independent works. Reporting aggregation and DE caller are
audited separately from atomic protocol identity. `ordinary`, `unspecified`, and `not_applicable`
target policies all execute as `no_target_operation`; their raw Atlas values remain in the CSV.

## Result

- Literature-eligible operational families before scope exclusions: 142
- Families in the approved ranking-sensitivity scope: 111
- Families excluded by the approved scope: 31

### Included by representation

- `DE`: 33
- `PBULK`: 67
- `SC`: 11

### Included by target operation

- `exclude_own_target`: 3
- `force_include_target`: 4
- `no_target_operation`: 104

### Implementation status

- `de_recovery_module`: 22
- `direct_after_public_api`: 28
- `multiple_extensions`: 21
- `new_scalar_metric`: 14
- `retrieval_module`: 7
- `selector_or_composition`: 19

### Scope exclusions

- `centering_out_of_scope`: 1
- `cross_perturbation_effect_size_out_of_scope`: 3
- `de_method_postponed`: 1
- `feature_set_out_of_scope`: 5
- `metric_out_of_scope`: 9
- `pooled_response_out_of_scope`: 3
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
- `other`: 6 works globally, 4 supporting included families; scPertEval `not_actionable`
- `rank_sum`: 43 works globally, 29 supporting included families; scPertEval `existing`
- `t_test`: 25 works globally, 21 supporting included families; scPertEval `existing`
- `unspecified`: 106 works globally, 80 supporting included families; scPertEval `use_family_canonical_backend`

DE-based metrics are applicable to all three benchmark tasks because each has observed and predicted
cell populations (including the already-defined control-lifted population view for point models).
Explicit target operations apply only to genetic tasks and require an unambiguous measured target gene.

## Files

- `atlas-protocol-eligibility.csv`: included families only.
- `atlas-task-protocol-manifest.csv`: one row per applicable task and included family.
- `atlas-protocol-scope-audit.csv`: every literature-eligible family and its inclusion/exclusion reason.
- `atlas-de-method-audit.csv`: DE caller usage and current scPertEval support.
