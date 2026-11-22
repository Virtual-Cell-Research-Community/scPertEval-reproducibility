"""Protocol diversity across the Perturbation Evaluation Atlas.

Pure data functions: read the Atlas exports, assign each protocol-context pair to
a prediction task, and summarize how often papers evaluating the same task share
an evaluation protocol.

A *protocol* is one Atlas matrix column. Its identity is the ``__``-separated
tuple of evaluation choices listed in :data:`analyses.vocabulary.ATLAS_AXES`
plus an optional cross-perturbation suffix, which is treated as part of the
metric.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analyses.vocabulary import ATLAS_AXES as AXES
from analyses.vocabulary import TARGET_CONTEXT_LABELS

ATLAS_FILES = (
    "data/papers.csv",
    "data/protocols.csv",
    "data/protocol_contexts.csv",
    "data/manifest.json",
    "public/data/datasets.json",
)
ELIGIBILITY_CONFIG = Path(__file__).resolve().parents[2] / "configs/evaluation/atlas-protocol-eligibility-v1.json"


def pinned_atlas_revision(config: Path = ELIGIBILITY_CONFIG) -> str:
    """Atlas revision pinned for the ranking-sensitivity analysis, so both analyses read the same Atlas."""

    return json.loads(config.read_text())["atlas"]["revision"]


@dataclass(frozen=True)
class AtlasInputs:
    papers: list[dict[str, str]]
    protocols: list[dict[str, str]]
    contexts: list[dict[str, str]]
    manifest: dict
    dataset_names: dict[str, str]
    sha256: dict[str, str]
    revision: str


def read_atlas(atlas_repo: Path, revision: str | None) -> AtlasInputs:
    """Read Atlas exports at a git ``revision``, or from the working tree when ``None``."""

    payloads: dict[str, bytes] = {}
    for path in ATLAS_FILES:
        if revision is None:
            payloads[path] = (atlas_repo / path).read_bytes()
        else:
            payloads[path] = subprocess.run(
                ["git", "show", f"{revision}:{path}"], cwd=atlas_repo, check=True, capture_output=True
            ).stdout
    rows = {p: list(csv.DictReader(io.StringIO(b.decode("utf-8")))) for p, b in payloads.items() if p.endswith(".csv")}
    return AtlasInputs(
        papers=rows["data/papers.csv"],
        protocols=rows["data/protocols.csv"],
        contexts=rows["data/protocol_contexts.csv"],
        manifest=json.loads(payloads["data/manifest.json"]),
        dataset_names={d["id"]: d["short_name"] for d in json.loads(payloads["public/data/datasets.json"])},
        sha256={p: hashlib.sha256(b).hexdigest() for p, b in payloads.items()},
        revision=revision or "working-tree",
    )


def intervention_label(compositions_json: str) -> str | None:
    """Collapse a context's Intervention Compositions to one modality and order.

    Contexts pooling singletons and combinations of one kind are assigned to the
    combination task. Contexts mixing intervention kinds return ``None``.
    """

    compositions = json.loads(compositions_json)
    kinds = {kind for composition in compositions for kind in composition}
    if len(kinds) != 1:
        return None
    (kind,) = kinds
    order = "combination" if max(sum(c.values()) for c in compositions) > 1 else "single"
    return f"{kind.capitalize()} {order}"


def task_label(context: Mapping[str, str]) -> str | None:
    """Prediction task of one protocol-context row, or ``None`` when excluded.

    Excluded: no target-context data, dose or time generalization, and contexts
    mixing intervention kinds. These strata are too sparse to summarize.
    """

    tcd = TARGET_CONTEXT_LABELS.get(context["target_context_data"])
    if tcd is None or context["dose_generalization"] or context["time_generalization"]:
        return None
    intervention = intervention_label(context["intervention_compositions"])
    if intervention is None:
        return None
    return f"{intervention}, {tcd}"


def protocol_prefix(column: str, level: int) -> str:
    """Protocol identity using the first ``level`` axes; the last level is the full column."""

    parts = column.split("__")
    return column if level >= len(AXES) else "__".join(parts[:level])


def papers_by_task(atlas: AtlasInputs, by_dataset: bool = False) -> dict[str, dict[str, set[str]]]:
    """Map task (optionally ``task | dataset``) to work_id -> set of protocol columns."""

    column = {row["protocol_id"]: row["matrix_column"] for row in atlas.protocols}
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for context in atlas.contexts:
        task = task_label(context)
        if task is None:
            continue
        key = f"{task} | {context['dataset']}" if by_dataset else task
        groups[key][context["work_id"]].add(column[context["protocol_id"]])
    return {k: dict(v) for k, v in groups.items()}


def all_papers(atlas: AtlasInputs) -> dict[str, set[str]]:
    papers: dict[str, set[str]] = defaultdict(set)
    for row in atlas.protocols:
        papers[row["work_id"]].add(row["matrix_column"])
    return dict(papers)


def corpus_summary(atlas: AtlasInputs) -> pd.DataFrame:
    """Size of the annotated corpus: works with protocol records by publication role, records and protocols."""

    works = {row["work_id"] for row in atlas.protocols}
    roles = defaultdict(int)
    for paper in atlas.papers:
        if paper["work_id"] in works:
            roles[paper["publication_role"]] += 1
    records = [
        ("works", len(works)),
        *((f"works_{role}", count) for role, count in sorted(roles.items())),
        ("protocol_records", len(atlas.protocols)),
        ("protocol_context_records", len(atlas.contexts)),
        ("distinct_protocols", len({row["matrix_column"] for row in atlas.protocols})),
    ]
    return pd.DataFrame(records, columns=["quantity", "value"])


def pair_sharing(papers: Mapping[str, Iterable[str]]) -> float:
    """Fraction of unordered paper pairs sharing at least one protocol (NaN for fewer than two papers)."""

    sets = [set(s) for s in papers.values() if s]
    pairs = list(itertools.combinations(sets, 2))
    return sum(bool(a & b) for a, b in pairs) / len(pairs) if pairs else float("nan")


def pair_sharing_within(
    groups: Mapping[str, Mapping[str, set[str]]], attribute: Mapping[str, str], name: str
) -> pd.DataFrame:
    """Pair sharing among papers of a task that share a work attribute (e.g. publication year or role)."""

    records = []
    for group, papers in groups.items():
        values: dict[str, dict[str, set[str]]] = defaultdict(dict)
        for work, cols in papers.items():
            values[attribute.get(work, "")][work] = cols
        for value, subset in sorted(values.items()):
            records.append({"group": group, name: value, "papers": len(subset), "pair_sharing": pair_sharing(subset)})
    return pd.DataFrame.from_records(records)


def consensus_decay(groups: Mapping[str, Mapping[str, set[str]]]) -> pd.DataFrame:
    """Pair sharing and distinct-protocol counts as taxonomy axes are added."""

    records = []
    for group, papers in groups.items():
        for level in range(1, len(AXES) + 1):
            coarse = {p: {protocol_prefix(c, level) for c in cols} for p, cols in papers.items()}
            records.append(
                {
                    "group": group,
                    "level": level,
                    "axis": AXES[level - 1],
                    "papers": len(coarse),
                    "distinct_protocols": len(set().union(*coarse.values())),
                    "pair_sharing": pair_sharing(coarse),
                }
            )
    return pd.DataFrame.from_records(records)


def axis_value(column: str, axis: int) -> str:
    """Value of one taxonomy axis; the metric axis keeps the cross-perturbation suffix."""

    parts = column.split("__")
    return "__".join(parts[axis:]) if axis == len(AXES) - 1 else parts[axis]


def single_axis_agreement(groups: Mapping[str, Mapping[str, set[str]]]) -> pd.DataFrame:
    """Pair sharing when protocols are compared on one taxonomy axis at a time.

    Unlike :func:`consensus_decay`, this does not depend on the order in which axes are added.
    """

    records = []
    for group, papers in groups.items():
        for i, axis in enumerate(AXES):
            values = {p: {axis_value(c, i) for c in cols} for p, cols in papers.items()}
            records.append(
                {
                    "group": group,
                    "axis": axis,
                    "distinct_values": len(set().union(*values.values())),
                    "pair_sharing": pair_sharing(values),
                }
            )
    return pd.DataFrame.from_records(records)


def protocol_usage(papers: Mapping[str, set[str]]) -> pd.DataFrame:
    """Number of papers using each protocol, most used first."""

    counts: dict[str, int] = defaultdict(int)
    for cols in papers.values():
        for c in cols:
            counts[c] += 1
    frame = pd.DataFrame(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])), columns=["protocol", "papers"])
    frame["rank"] = range(1, len(frame) + 1)
    return frame


def rank_frequency(groups: Mapping[str, Mapping[str, set[str]]]) -> pd.DataFrame:
    """Each protocol's rank within its group and the fraction of the group's papers using it."""

    frames = []
    for group, papers in groups.items():
        usage = protocol_usage(papers)
        frames.append(usage.assign(group=group, paper_fraction=usage["papers"] / len(papers)))
    return pd.concat(frames, ignore_index=True)[["group", "rank", "protocol", "papers", "paper_fraction"]]


def task_summary(groups: Mapping[str, Mapping[str, set[str]]]) -> pd.DataFrame:
    records = []
    for group, papers in groups.items():
        usage = protocol_usage(papers)
        records.append(
            {
                "group": group,
                "papers": len(papers),
                "protocols": len(usage),
                "single_paper_protocol_fraction": (usage["papers"] == 1).mean(),
                "most_common_protocol_paper_fraction": usage["papers"].iloc[0] / len(papers),
                "pair_sharing": pair_sharing(papers),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("papers", ascending=False, ignore_index=True)
