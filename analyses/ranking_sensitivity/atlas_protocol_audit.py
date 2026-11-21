"""Build the frozen Atlas-derived protocol eligibility audit.

The audit intentionally separates three questions:

1. Is an operational protocol family reported by at least two independent works?
2. Is that family inside the deliberately limited ranking-sensitivity scope?
3. Which scPertEval extension, if any, is required to execute it?

Differential-expression callers are audited separately. Their literature frequency
does not silently expand the executable protocol universe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/evaluation/atlas-protocol-eligibility-v1.json")


@dataclass(frozen=True)
class Family:
    """One operationally distinct protocol family and its literature evidence."""

    family_id: str
    identity: Mapping[str, str]
    rows: tuple[Mapping[str, str], ...]

    @property
    def work_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row["work_id"] for row in self.rows}))

    @property
    def protocol_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row["protocol_id"] for row in self.rows}))


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_atlas_csv(atlas_repo: Path, revision: str, source_path: str) -> tuple[list[dict[str, str]], str]:
    """Read a CSV exactly as committed at ``revision`` and return its SHA-256."""

    completed = subprocess.run(
        ["git", "show", f"{revision}:{source_path}"],
        cwd=atlas_repo,
        check=True,
        capture_output=True,
    )
    payload = completed.stdout
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    return rows, hashlib.sha256(payload).hexdigest()


def validate_input(rows: Sequence[Mapping[str, str]], config: Mapping[str, Any]) -> None:
    if not rows:
        raise ValueError("Atlas protocol export is empty")
    missing = sorted(set(config["atlas"]["required_columns"]) - set(rows[0]))
    if missing:
        raise ValueError(f"Atlas protocol export lacks required columns: {missing}")


def canonical_metric(row: Mapping[str, str], config: Mapping[str, Any]) -> str:
    family = row["wide_display_family"] or row["ranking_equivalence_family"]
    return config["metric_equivalence_overrides"].get(family, family)


def target_operation(row: Mapping[str, str], config: Mapping[str, Any]) -> str:
    policy = row["target_gene_policy"]
    try:
        return config["target_policy"][policy]
    except KeyError as error:
        raise ValueError(f"Unmapped target_gene_policy: {policy!r}") from error


def operational_identity(row: Mapping[str, str], config: Mapping[str, Any]) -> dict[str, str]:
    derived = dict(row)
    derived["metric_family"] = canonical_metric(row, config)
    derived["target_operation"] = target_operation(row, config)
    return {field: derived.get(field, "") for field in config["identity"]["fields"]}


def group_families(rows: Iterable[Mapping[str, str]], config: Mapping[str, Any]) -> list[Family]:
    grouped: defaultdict[tuple[tuple[str, str], ...], list[Mapping[str, str]]] = defaultdict(list)
    identities: dict[tuple[tuple[str, str], ...], Mapping[str, str]] = {}
    for row in rows:
        identity = operational_identity(row, config)
        key = tuple(identity.items())
        grouped[key].append(row)
        identities[key] = identity

    families = []
    for key, family_rows in grouped.items():
        identity = identities[key]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        families.append(Family(family_id=f"atlas-{digest}", identity=identity, rows=tuple(family_rows)))
    return sorted(families, key=lambda family: family.family_id)


def scope_decision(family: Family, config: Mapping[str, Any]) -> tuple[str, str]:
    identity = family.identity
    scope = config["scope"]

    if scope["require_empty_cross_perturbation_group"] and identity["cross_perturbation_group"]:
        return "excluded", "pooled_or_cross_perturbation"
    if identity["feature_set"] not in scope["allowed_feature_sets"]:
        return "excluded", "feature_set_out_of_scope"
    if identity["centering"] not in scope["allowed_centering"]:
        return "excluded", "centering_out_of_scope"
    if identity["reference_population"] not in scope["allowed_reference_populations"]:
        return "excluded", "reference_population_out_of_scope"
    if identity["target_operation"] in scope["excluded_target_operations"]:
        return "excluded", "target_operation_out_of_scope"

    reported_de_methods = set(de_methods(family.rows, config))
    actionable_de_methods = reported_de_methods - {"other", "unspecified"}
    postponed_de_methods = set(config["de_method_policy"]["postponed_methods"])
    if actionable_de_methods and actionable_de_methods <= postponed_de_methods:
        return "excluded", "de_method_postponed"

    population_scope = scope["allowed_by_population"].get(identity["population"])
    if population_scope is None:
        return "excluded", "population_out_of_scope"
    if identity["space"] not in population_scope["spaces"]:
        return "excluded", "space_out_of_scope"
    if identity["metric_family"] not in population_scope["metrics"]:
        return "excluded", "metric_out_of_scope"
    return "included", ""


def de_methods(rows: Iterable[Mapping[str, str]], config: Mapping[str, Any]) -> tuple[str, ...]:
    methods: set[str] = set()
    aliases = config["de_method_aliases"]
    for row in rows:
        for field in ("de_definition", "feature_de_definition"):
            value = row.get(field, "")
            if not value:
                continue
            method = json.loads(value).get("method", "unknown")
            methods.add(aliases.get(method, method))
    return tuple(sorted(methods or {"unspecified"}))


def required_extensions(family: Family) -> tuple[str, ...]:
    identity = family.identity
    extensions = {"public_concrete_protocol_api"}

    if identity["feature_set"] == "DEG" and identity["feature_ranking_basis"] == "significance":
        extensions.add("configurable_deg_selector")
    if identity["target_operation"] in {"force_include_target", "exclude_own_target"}:
        extensions.add("target_gene_policy")
    if identity["space"] == "PCA" and identity["feature_set"] != "All":
        extensions.add("subset_then_pca")

    metric = identity["metric_family"]
    if identity["population"] == "PBULK" and metric in {
        "MAE",
        "clipped-r2",
        "cosine",
        "explained-variance",
        "spearman",
        "squared-pearson",
    }:
        extensions.add(f"scalar_metric:{metric}")
    if identity["population"] == "DE":
        extensions.add("symmetric_signed_de")
        extensions.add(f"de_recovery_metric:{metric}")
    return tuple(sorted(extensions))


def implementation_status(extensions: Sequence[str]) -> str:
    substantive = [extension for extension in extensions if extension != "public_concrete_protocol_api"]
    if not substantive:
        return "direct_after_public_api"
    categories = set()
    for extension in substantive:
        category = extension.split(":", 1)[0]
        if category in {"symmetric_signed_de", "de_recovery_metric"}:
            category = "de_recovery_module"
        categories.add(category)
    if len(categories) > 1:
        return "multiple_extensions"
    category = next(iter(categories))
    return {
        "configurable_deg_selector": "selector_or_composition",
        "target_gene_policy": "selector_or_composition",
        "subset_then_pca": "selector_or_composition",
        "scalar_metric": "new_scalar_metric",
        "de_recovery_module": "de_recovery_module",
    }.get(category, "extension_required")


def task_applicability(family: Family, config: Mapping[str, Any]) -> tuple[str, ...]:
    tasks = list(config["task_applicability"]["tasks"])
    if family.identity["target_operation"] in {"force_include_target", "exclude_own_target"}:
        tasks = list(config["task_applicability"]["explicit_target_operations"])
    return tuple(tasks)


def family_record(family: Family, config: Mapping[str, Any]) -> dict[str, str | int]:
    decision, reason = scope_decision(family, config)
    extensions = required_extensions(family) if decision == "included" else ()
    raw_target_policies = sorted({row["target_gene_policy"] for row in family.rows})
    record: dict[str, str | int] = {
        "family_id": family.family_id,
        "scope_decision": decision,
        "exclusion_reason": reason,
        "independent_work_count": len(family.work_ids),
        "work_ids": ";".join(family.work_ids),
        "protocol_ids": ";".join(family.protocol_ids),
        "raw_target_policies": ";".join(raw_target_policies),
        "de_methods_reported": ";".join(de_methods(family.rows, config)),
        "required_extensions": ";".join(extensions),
        "implementation_status": implementation_status(extensions) if extensions else "not_in_scope",
        "applicable_tasks": ";".join(task_applicability(family, config)) if decision == "included" else "",
    }
    record.update(family.identity)
    canonical_k = config["canonical_parameters"]["feature_k_by_bin"].get(family.identity["feature_k_bin"], "review")
    record["canonical_feature_k"] = str(canonical_k)
    return record


def de_method_audit(
    rows: Iterable[Mapping[str, str]],
    config: Mapping[str, Any],
    *,
    in_scope_protocol_ids: set[str] | None = None,
) -> list[dict[str, str | int]]:
    evidence: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    in_scope_evidence: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    protocol_counts: Counter[tuple[str, str]] = Counter()
    raw_methods: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    papers: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    aliases = config["de_method_aliases"]

    for row in rows:
        for field in ("de_definition", "feature_de_definition"):
            value = row.get(field, "")
            if not value:
                continue
            raw_method = json.loads(value).get("method", "unknown")
            method = aliases.get(raw_method, raw_method)
            key = (field, method)
            evidence[key].add(row["work_id"])
            if in_scope_protocol_ids is not None and row["protocol_id"] in in_scope_protocol_ids:
                in_scope_evidence[key].add(row["work_id"])
            protocol_counts[key] += 1
            raw_methods[key].add(raw_method or "blank")
            papers[key].add(row["paper"])

            combined_key = ("combined", method)
            evidence[combined_key].add(row["work_id"])
            if in_scope_protocol_ids is not None and row["protocol_id"] in in_scope_protocol_ids:
                in_scope_evidence[combined_key].add(row["work_id"])
            protocol_counts[combined_key] += 1
            raw_methods[combined_key].add(raw_method or "blank")
            papers[combined_key].add(row["paper"])

    records = []
    threshold = config["atlas"]["minimum_independent_works"]
    for (field, method), works in sorted(evidence.items()):
        records.append(
            {
                "definition_field": field,
                "canonical_method": method,
                "raw_methods": ";".join(sorted(raw_methods[(field, method)])),
                "protocol_count": protocol_counts[(field, method)],
                "independent_work_count": len(works),
                "in_scope_independent_work_count": len(in_scope_evidence[(field, method)]),
                "work_ids": ";".join(sorted(works)),
                "paper_keys": ";".join(sorted(papers[(field, method)])),
                "meets_two_work_threshold": str(len(works) >= threshold).lower(),
                "scperteval_support": config["scperteval_de_support"].get(method, "unknown"),
            }
        )
    return records


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError(f"Refusing to write empty audit: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_summary(
    path: Path,
    *,
    config: Mapping[str, Any],
    source_sha256: str,
    eligible_records: Sequence[Mapping[str, Any]],
    all_records: Sequence[Mapping[str, Any]],
    de_records: Sequence[Mapping[str, Any]],
) -> None:
    statuses = Counter(str(record["implementation_status"]) for record in eligible_records)
    populations = Counter(str(record["population"]) for record in eligible_records)
    target_operations = Counter(str(record["target_operation"]) for record in eligible_records)
    excluded = Counter(
        str(record["exclusion_reason"])
        for record in all_records
        if record["scope_decision"] == "excluded"
    )

    def bullets(counter: Counter[str]) -> str:
        return "\n".join(f"- `{key}`: {value}" for key, value in sorted(counter.items()))

    method_lines = []
    for record in de_records:
        if record["definition_field"] == "combined" and record["meets_two_work_threshold"] == "true":
            method_lines.append(
                f"- `{record['canonical_method']}`: {record['independent_work_count']} works globally, "
                f"{record['in_scope_independent_work_count']} supporting included families; "
                f"scPertEval `{record['scperteval_support']}`"
            )

    text = f"""# Atlas protocol eligibility audit v1

This audit freezes the ranking-sensitivity protocol universe against PerturbationEvaluationAtlas
commit `{config['atlas']['revision']}` and `data/protocols.csv` SHA-256 `{source_sha256}`.

Eligibility means that an operationally distinct family occurs in at least
{config['atlas']['minimum_independent_works']} independent works. Reporting aggregation and DE caller are
audited separately from atomic protocol identity. `ordinary`, `unspecified`, and `not_applicable`
target policies all execute as `no_target_operation`; their raw Atlas values remain in the CSV.

## Result

- Literature-eligible operational families before scope exclusions: {len(all_records)}
- Families in the approved ranking-sensitivity scope: {len(eligible_records)}
- Families excluded by the approved scope: {len(all_records) - len(eligible_records)}

### Included by representation

{bullets(populations)}

### Included by target operation

{bullets(target_operations)}

### Implementation status

{bullets(statuses)}

### Scope exclusions

{bullets(excluded)}

## DE caller audit

The DE caller does not define atomic-family eligibility in this version. Existing scPertEval callers remain the
canonical implementations unless the literature audit below reveals a missing, sufficiently common
pipeline worth discussing before implementation.

Limma-dependent families remain visible in the scope audit but are postponed from the executable primary
universe under the frozen DE-method policy.

{chr(10).join(method_lines)}

DE-based metrics are applicable to all three benchmark tasks because each has observed and predicted
cell populations (including the already-defined control-lifted population view for point models).
Explicit target operations apply only to genetic tasks and require an unambiguous measured target gene.

## Files

- `atlas-protocol-eligibility-v1.csv`: included families only.
- `atlas-protocol-scope-audit-v1.csv`: every literature-eligible family and its inclusion/exclusion reason.
- `atlas-de-method-audit-v1.csv`: DE caller usage and current scPertEval support.
"""
    path.write_text(text)


def build_audit(atlas_repo: Path, config_path: Path, output_dir: Path, summary_path: Path) -> dict[str, int]:
    config = load_config(config_path)
    rows, source_sha256 = read_atlas_csv(
        atlas_repo,
        config["atlas"]["revision"],
        config["atlas"]["source_path"],
    )
    validate_input(rows, config)

    threshold = config["atlas"]["minimum_independent_works"]
    families = [family for family in group_families(rows, config) if len(family.work_ids) >= threshold]
    all_records = [family_record(family, config) for family in families]
    eligible_records = [record for record in all_records if record["scope_decision"] == "included"]
    included_family_ids = {
        str(record["family_id"])
        for record in eligible_records
    }
    in_scope_protocol_ids = {
        protocol_id
        for family in families
        if family.family_id in included_family_ids
        for protocol_id in family.protocol_ids
    }
    de_records = de_method_audit(rows, config, in_scope_protocol_ids=in_scope_protocol_ids)

    write_csv(output_dir / "atlas-protocol-eligibility-v1.csv", eligible_records)
    write_csv(output_dir / "atlas-protocol-scope-audit-v1.csv", all_records)
    write_csv(output_dir / "atlas-de-method-audit-v1.csv", de_records)
    write_summary(
        summary_path,
        config=config,
        source_sha256=source_sha256,
        eligible_records=eligible_records,
        all_records=all_records,
        de_records=de_records,
    )
    return {"eligible": len(eligible_records), "all": len(all_records), "de_methods": len(de_records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/evaluation"))
    parser.add_argument("--summary", type=Path, default=Path("docs/atlas-protocol-eligibility-v1.md"))
    args = parser.parse_args()
    counts = build_audit(args.atlas_repo, args.config, args.output_dir, args.summary)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
