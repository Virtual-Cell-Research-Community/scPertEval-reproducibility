"""Export the census and ranking-sensitivity results as compact JSON for the Atlas website.

The website (PerturbationEvaluationAtlas, route ``#/analysis/evaluation-diversity``) draws its own
charts from these files. Nothing is recomputed here: every number comes from the exported source
data of the census (``analyses.protocol_diversity``) and of the ranking-sensitivity analysis
(``analyses.ranking_sensitivity.atlas_analysis``), so the page shows a frozen snapshot that matches
the manuscript.

    evaluation-diversity.json   census and ranking summaries (Fig. 2 B–E and the ranking supplement)
    protocol-explorer.json      per-protocol model scores, ranks and winners for the pair explorer
    manifest.json               upstream pins and source-data checksums

From the repository root:

    uv run python -m analyses.web_export \\
      --census /path/to/figures/protocol-diversity \\
      --ranking /path/to/figures/ranking-sensitivity-atlas \\
      --out ../PerturbationEvaluationAtlas/public/data/analyses/evaluation-diversity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyses.manuscript_figures.evaluation_practice import MIN_FACTOR_CONTRASTS
from analyses.manuscript_figures.ranking_supplement import (
    CENTERING_LABELS,
    METRIC_LABELS,
    MIN_POOLED_CONTRASTS,
    MIN_TASK_CONTRASTS,
    POPULATION_LABELS,
    TARGET_LABELS,
    _counts,
    contrast_data,
    feature_label,
)
from analyses.protocol_diversity.figure import MIN_TASK_PAPERS, task_order
from analyses.ranking_sensitivity.atlas_analysis import _factor_level
from analyses.ranking_sensitivity.atlas_figures import BASELINES, MODEL_COLORS, MODEL_LABELS
from analyses.ranking_sensitivity.cross_perturbation import RETRIEVAL_PREDICTED_QUERY, decompose
from analyses.vocabulary import ATLAS_AXES, AXIS_LABELS, BENCHMARK_TASKS, CENSUS_TASKS, RANKING_FACTORS

REPO_ROOT = Path(__file__).resolve().parents[2]
ELIGIBILITY = REPO_ROOT / "configs/evaluation/atlas-protocol-eligibility.csv"
SCHEMA_VERSION = 1
SIGNIFICANT_DIGITS = 5
"""Model scores are rounded for display; ranks and winners are exported exactly."""

EXPLORER_CHOICES = {
    "representation": "evaluation_family",
    "embedding": "expression_space",
    "effect_isolation": "centering",
    "feature_selection": "feature_panel",
    "target_gene_policy": "target_policy",
    "feature_weighting": "weighting",
    "metric": "score_function",
}
"""Explorer choice (taxonomy order) → ranking-sensitivity factor that describes a protocol on it."""

EMBEDDING_LABELS = {"GEX": "Genes", "PCA": "PCA"}
EXTRA_METRIC_LABELS = {
    "topk-overlap": "Top-k overlap",
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "deg-set-precision": "DEG precision",
    "macro-f1": "Macro F1",
    "f1": "F1",
    "jaccard": "Jaccard index",
    "set-overlap-at-truth-size": "Overlap at true-set size",
    "direction-concordance": "Direction concordance",
}
"""Metrics the manuscript figures never label; names follow the Atlas framework pages."""
EXTRA_FEATURE_LABELS = {"Other · k=hvg4000_deg25_union": "4,000 HVGs ∪ top-25 DEGs"}
DISTANCE_LABELS = {"squared_euclidean": "Euclidean", "cityblock": "Manhattan", "cosine": "cosine"}


def _round(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(directory: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(directory / "source-data" / name, keep_default_na=False)


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def _count(rate: float, n: int) -> int:
    """Integer numerator of a rate over ``n`` pairs; fails if the rate is not a fraction of ``n``."""

    k = round(rate * n)
    if not np.isclose(k / n if n else 0.0, rate, atol=1e-9):
        raise ValueError(f"rate {rate} is not a fraction of {n} pairs")
    return int(k)


# Census -------------------------------------------------------------------------------------------


def census_payload(census: Path) -> dict[str, Any]:
    """Disagreement between works per evaluation choice, alone and cumulatively, for tasks with ≥ 20 works."""

    corpus = _read(census, "corpus-summary.csv").set_index("quantity")["value"].astype(int)
    summary = _read(census, "task-summary.csv")
    summary["papers"] = summary["papers"].astype(int)
    shown = summary[summary["papers"] >= MIN_TASK_PAPERS]
    tasks = task_order(shown["group"])
    summary = shown.set_index("group").loc[tasks]
    agreement = _read(census, "single-choice-agreement.csv")
    decay = _read(census, "consensus-decay.csv")
    for frame in (agreement, decay):
        frame["pair_sharing"] = frame["pair_sharing"].astype(float)
    task_rows = []
    for task, row in summary.iterrows():
        n = _pairs(int(row["papers"]))
        task_rows.append(
            {
                "name": task,
                "target_context": task.split(", ")[1].lower(),
                "works": int(row["papers"]),
                "pairs": n,
                "protocols": int(row["protocols"]),
                "single_work_protocol_fraction": float(row["single_paper_protocol_fraction"]),
                "most_common_protocol_work_fraction": float(row["most_common_protocol_paper_fraction"]),
                "disagreeing_pairs": _count(1 - float(row["pair_sharing"]), n),
                "disagreement": 1 - float(row["pair_sharing"]),
            }
        )
    pairs = {row["name"]: row["pairs"] for row in task_rows}

    def rows(frame: pd.DataFrame, extra: list[str]) -> list[dict[str, Any]]:
        out = []
        for row in frame[frame["group"].isin(tasks)].itertuples(index=False):
            n = pairs[row.group]
            out.append(
                {
                    "task": row.group,
                    "choice": row.axis,
                    **{column: int(getattr(row, column)) for column in extra},
                    "disagreeing_pairs": _count(1 - row.pair_sharing, n),
                    "pairs": n,
                    "disagreement": 1 - row.pair_sharing,
                }
            )
        return out

    return {
        "corpus": {
            "works": int(corpus["works"]),
            "model_papers": int(corpus["works_model_paper"]),
            "benchmark_papers": int(corpus["works_benchmark_paper"]),
            "competitions": int(corpus["works_competition"]),
            "distinct_protocols": int(corpus["distinct_protocols"]),
        },
        "min_task_works": MIN_TASK_PAPERS,
        "choices": [{"key": axis, "label": AXIS_LABELS[axis]} for axis in ATLAS_AXES],
        "tasks": task_rows,
        "single_choice": rows(agreement, ["distinct_values"]),
        "cumulative": rows(decay, ["level", "distinct_protocols"]),
    }


# Ranking summaries ----------------------------------------------------------------------------------


def _model(model: str) -> dict[str, Any]:
    return {
        "id": model,
        "label": MODEL_LABELS.get(model, model),
        "baseline": model in BASELINES,
        "color": MODEL_COLORS[model],
    }


def ranking_payload(ranking: Path) -> dict[str, Any]:
    disagreement = _read(ranking, "task-protocol-disagreement.csv").set_index("task")
    factors = _read(ranking, "factor-sensitivity.csv")
    shares = _read(ranking, "winner-shares.csv")
    contrasts = _read(ranking, "concrete-factor-contrasts.csv")
    rankings = _read(ranking, "protocol-rankings.csv")
    headline = rankings[rankings["headline_eligible"].astype(str) == "True"]

    tasks = []
    for task in BENCHMARK_TASKS:
        row = disagreement.loc[task]
        tasks.append(
            {
                "id": task,
                "name": CENSUS_TASKS[task],
                "benchmark_name": BENCHMARK_TASKS[task],
                "models": int(rankings.loc[rankings["task"] == task, "model"].nunique()),
                "protocols": int(headline.loc[headline["task"] == task, "protocol_id"].nunique()),
                "protocol_pairs": int(row["n_protocol_pairs"]),
                "disagreeing_pairs": int(_counts(pd.Series([row["n_winner_disagreements"]])).iloc[0]),
                "disagreement": float(row["winner_disagreement_rate"]),
                "mean_reversed_model_pairs": float(row["mean_pairwise_reversal_fraction"]),
            }
        )

    factors["n_winner_changes"] = _counts(factors["n_winner_changes"])
    factor_rows = []
    choices = list(dict.fromkeys(RANKING_FACTORS.values()))
    lookup = factors.assign(choice=factors["factor"].map(RANKING_FACTORS)).set_index(["choice", "task"])
    for choice in choices:
        for task in BENCHMARK_TASKS:
            if (choice, task) not in lookup.index:
                continue
            row = lookup.loc[(choice, task)]
            n = int(row["n_matched_contrasts"])
            factor_rows.append(
                {
                    "choice": choice,
                    "task": task,
                    "matched_pairs": n,
                    "winner_changes": int(row["n_winner_changes"]),
                    "rate": float(row["winner_change_rate"]),
                    "shown": n >= MIN_FACTOR_CONTRASTS,
                }
            )

    share_rows = [
        {
            "task": row.task,
            "model": row.model,
            "winner_credit": float(row.winner_credit),
            "protocols": int(row.n_protocols),
            "share": float(row.winner_share),
        }
        for row in shares.itertuples(index=False)
    ]

    pooled, points = contrast_data(contrasts)
    switches = []
    for row in pooled.itertuples(index=False):
        task_points = points[
            (points["factor"] == row.factor) & (points["level_a"] == row.level_a) & (points["level_b"] == row.level_b)
        ]
        switches.append(
            {
                "choice": RANKING_FACTORS[row.factor],
                "label": str(row.contrast),
                "matched_pairs": int(row.n_backgrounds),
                "winner_changes": int(row.n_winner_changes),
                "rate": float(row.rate),
                "tasks": [
                    {
                        "task": point.task,
                        "matched_pairs": int(point.n_backgrounds),
                        "winner_changes": int(point.n_winner_changes),
                        "rate": float(point.winner_change_rate),
                    }
                    for point in task_points.itertuples(index=False)
                ],
            }
        )

    models = sorted(rankings["model"].unique(), key=lambda m: (m in BASELINES, MODEL_LABELS.get(m, m).casefold()))
    return {
        "tasks": tasks,
        "models": [_model(model) for model in models],
        "choices": [{"key": choice, "label": AXIS_LABELS[choice]} for choice in choices],
        "min_factor_pairs": MIN_FACTOR_CONTRASTS,
        "factors": factor_rows,
        "winner_shares": share_rows,
        "min_switch_pairs": MIN_POOLED_CONTRASTS,
        "min_switch_task_pairs": MIN_TASK_CONTRASTS,
        "switches": switches,
    }


# Explorer --------------------------------------------------------------------------------------------


def metric_label(metric_family: str) -> str:
    spec = decompose(metric_family)
    if not spec.is_retrieval:
        labels = {**METRIC_LABELS, **EXTRA_METRIC_LABELS}
        return labels.get(metric_family, metric_family.replace("-", " ").capitalize())
    query = "prediction" if spec.strategy == RETRIEVAL_PREDICTED_QUERY else "observation"
    return f"Retrieval rank, {DISTANCE_LABELS[spec.distance]} distance, {query} as query"


def choice_label(choice: str, family: pd.Series) -> str:
    """Reader-facing value of one protocol on one evaluation choice."""

    factor = EXPLORER_CHOICES[choice]
    level = _factor_level(family, factor)
    if choice == "representation":
        return POPULATION_LABELS.get(level, level)
    if choice == "embedding":
        return EMBEDDING_LABELS.get(level, level)
    if choice == "effect_isolation":
        return CENTERING_LABELS.get(level, level)
    if choice == "feature_selection":
        if level in EXTRA_FEATURE_LABELS:
            return EXTRA_FEATURE_LABELS[level]
        if "/other · " in level:  # DEGs ranked by a basis the source does not report
            return f"{feature_label(level.replace('/other', ''))}, ranking not reported"
        return feature_label(level)
    if choice == "target_gene_policy":
        return TARGET_LABELS.get(level, level)
    if choice == "feature_weighting":
        return "Uniform" if level == "uniform" else "DE-statistic weights"
    return metric_label(level)


def _ranking(rows: pd.DataFrame, models: list[str]) -> dict[str, Any]:
    rows = rows.set_index("model").loc[models]
    winners = [i for i, model in enumerate(models) if str(rows.loc[model, "winner"]) == "True"]
    return {
        "scores": [_round(float(value)) for value in rows["score"]],
        "ranks": [float(value) for value in rows["rank"]],
        "winners": winners,
    }


def _direction(ranking: dict[str, Any]) -> str:
    scores = np.array(ranking["scores"])
    best = scores[ranking["winners"][0]]
    if np.isclose(scores.max(), scores.min()):
        return "higher"
    return "higher" if np.isclose(best, scores.max()) else "lower"


def _pair_ranking(group: pd.DataFrame, side: str, models: list[str]) -> dict[str, Any]:
    return {
        "scores": [_round(float(value)) for value in group[f"score_{side}"]],
        "ranks": [float(value) for value in group[f"rank_{side}"]],
        "winners": [i for i, model in enumerate(models) if str(group.loc[model, f"winner_{side}"]) == "True"],
    }


def explorer_payload(ranking: Path, eligibility: Path = ELIGIBILITY) -> dict[str, Any]:
    """Scores, ranks and winners of every headline protocol, plus the pairs that need their own ranks.

    A protocol pair is compared on the evaluation units both protocols define. Usually that leaves
    each protocol's ranks and winners unchanged, so the explorer can pair any two protocols from their
    own rankings. The pairs whose ranks or winners do change on the shared units are exported with
    their pair-level ranking. Only headline protocols are exported; every pair below the coverage rule
    involves a protocol that is not one, so all exported pairs are analysis-eligible.
    """

    families = pd.read_csv(eligibility, keep_default_na=False).set_index("family_id")
    rankings = _read(ranking, "protocol-rankings.csv")
    comparisons = _read(ranking, "protocol-pair-comparisons.csv")
    pair_ranks = _read(ranking, "protocol-pair-model-ranks.csv")
    tasks = []
    for task in BENCHMARK_TASKS:
        task_rows = rankings[rankings["task"] == task]
        mean = task_rows[task_rows["aggregation"] == "mean"]
        protocol_ids = sorted(mean.loc[mean["headline_eligible"].astype(str) == "True", "protocol_id"].unique())
        index = {protocol: i for i, protocol in enumerate(protocol_ids)}
        models = sorted(task_rows["model"].unique(), key=lambda m: (m in BASELINES, MODEL_LABELS.get(m, m).casefold()))
        protocols = []
        for protocol in protocol_ids:
            family = families.loc[protocol]
            by_aggregation = {
                str(aggregation): _ranking(group, models)
                for aggregation, group in task_rows[task_rows["protocol_id"] == protocol].groupby("aggregation")
            }
            protocols.append(
                {
                    "id": protocol,
                    "choices": {choice: choice_label(choice, family) for choice in EXPLORER_CHOICES},
                    "better": _direction(by_aggregation["mean"]),
                    "works": int(family["independent_work_count"]),
                    **by_aggregation,
                }
            )

        described = [tuple(protocol["choices"].values()) for protocol in protocols]
        if len(set(described)) != len(described):
            raise ValueError(f"{task}: two protocols have the same value on every evaluation choice")

        task_pairs = comparisons[comparisons["task"] == task]
        exported = task_pairs["protocol_a"].isin(index) & task_pairs["protocol_b"].isin(index)
        if (task_pairs.loc[exported, "analysis_eligible"].astype(str) != "True").any():
            raise ValueError(f"{task}: a pair of headline protocols is below the coverage rule")
        task_ranks = pair_ranks[pair_ranks["comparison_id"].isin(task_pairs.loc[exported, "comparison_id"])]
        pair_rankings = []
        for _, group in task_ranks.groupby("comparison_id", sort=True):
            group = group.set_index("model").loc[models]
            first, second = index[group["protocol_a"].iloc[0]], index[group["protocol_b"].iloc[0]]
            sides = {"a": _pair_ranking(group, "a", models), "b": _pair_ranking(group, "b", models)}
            own = {"a": protocols[first]["mean"], "b": protocols[second]["mean"]}
            if any(sides[s]["ranks"] != own[s]["ranks"] or sides[s]["winners"] != own[s]["winners"] for s in "ab"):
                pair_rankings.append({"a": first, "b": second, "ranking_a": sides["a"], "ranking_b": sides["b"]})
        matched = task_pairs[exported & (task_pairs["factor"] != "")]
        tasks.append(
            {
                "id": task,
                "name": CENSUS_TASKS[task],
                "models": models,
                "protocols": protocols,
                "matched_pairs": [
                    [index[row.protocol_a], index[row.protocol_b], RANKING_FACTORS[row.factor]]
                    for row in matched.itertuples(index=False)
                ],
                "pair_rankings": pair_rankings,
            }
        )
    all_models = sorted({m for task in tasks for m in task["models"]})
    return {
        "choices": [{"key": choice, "label": AXIS_LABELS[choice]} for choice in EXPLORER_CHOICES],
        "matched_choice_labels": {choice: AXIS_LABELS[choice] for choice in dict.fromkeys(RANKING_FACTORS.values())},
        "models": {model: _model(model) for model in all_models},
        "tasks": tasks,
    }


def pair_ranking(task: dict[str, Any], first: int, second: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """The rankings the analysis compared for protocols ``first`` and ``second`` (the website does the same)."""

    for pair in task["pair_rankings"]:
        if (pair["a"], pair["b"]) == (first, second):
            return pair["ranking_a"], pair["ranking_b"]
        if (pair["a"], pair["b"]) == (second, first):
            return pair["ranking_b"], pair["ranking_a"]
    return task["protocols"][first]["mean"], task["protocols"][second]["mean"]


def reversed_model_pairs(first: dict[str, Any], second: dict[str, Any]) -> int:
    ranks_a, ranks_b = first["ranks"], second["ranks"]
    n = len(ranks_a)
    return sum((ranks_a[i] - ranks_a[j]) * (ranks_b[i] - ranks_b[j]) < 0 for i in range(n) for j in range(i + 1, n))


def verify_explorer(explorer: dict[str, Any], comparisons: pd.DataFrame, aggregation: pd.DataFrame) -> int:
    """Check that the explorer reproduces every analysed comparison; returns the number checked."""

    checked = 0
    for task in explorer["tasks"]:
        index = {protocol["id"]: i for i, protocol in enumerate(task["protocols"])}
        rows = comparisons[
            (comparisons["task"] == task["id"]) & (comparisons["analysis_eligible"].astype(str) == "True")
        ]
        rows = rows[rows["protocol_a"].isin(index) & rows["protocol_b"].isin(index)]
        for row in rows.itertuples(index=False):
            first, second = pair_ranking(task, index[row.protocol_a], index[row.protocol_b])
            expected = (str(row.winner_changed) == "True", int(float(row.n_reversed_model_pairs)))
            found = (first["winners"] != second["winners"], reversed_model_pairs(first, second))
            if found != expected:
                raise ValueError(f"explorer disagrees with the analysis for {row.comparison_id}: {found} != {expected}")
            checked += 1
        for row in aggregation[aggregation["task"] == task["id"]].itertuples(index=False):
            protocol = task["protocols"][index[row.protocol_a]]
            changed = protocol["mean"]["winners"] != protocol["median"]["winners"]
            if changed != (str(row.winner_changed) == "True"):
                raise ValueError(f"explorer disagrees with the analysis for {row.comparison_id}")
            checked += 1
    expected_rows = (comparisons["analysis_eligible"].astype(str) == "True").sum() + len(aggregation)
    if checked != expected_rows:
        raise ValueError(f"explorer covers {checked} of {expected_rows} analysed comparisons")
    return checked


# Output ----------------------------------------------------------------------------------------------


def _git_revision() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "analyses", "configs")),
    }


def manifest(census: Path, ranking: Path, files: dict[str, Path]) -> dict[str, Any]:
    census_manifest = json.loads((census / "analysis-manifest.json").read_text())
    ranking_manifest = json.loads((ranking / "analysis-manifest.json").read_text())
    eligibility = json.loads(ELIGIBILITY.with_suffix(".json").read_text())
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "generator": {"repository": "scPertEval-reproducibility", "module": "analyses.web_export", **_git_revision()},
        "atlas_revision": census_manifest["atlas_revision"],
        "protocol_universe_atlas_revision": eligibility["atlas"]["revision"],
        "scperteval_revision": _scperteval_revision(),
        "census_inputs_sha256": census_manifest["atlas_input_sha256"],
        "ranking_source_data_sha256": ranking_manifest["source_data_sha256"],
        "ranking_rules": {
            "coverage_threshold": ranking_manifest["coverage_threshold"],
            "primary_aggregation": ranking_manifest["primary_aggregation"],
            "pair_matching": ranking_manifest["pair_matching"],
            "winner_tie_credit": ranking_manifest["winner_tie_credit"],
        },
        "files_sha256": {name: _sha256(path) for name, path in files.items()},
    }


def _scperteval_revision() -> str:
    lock = (REPO_ROOT / "uv.lock").read_text()
    marker = 'name = "scperteval"'
    block = lock[lock.index(marker) :].split("\n\n", 1)[0]
    source = next(line for line in block.splitlines() if line.startswith("source = "))
    return source.rsplit("#", 1)[1].rstrip('" }')


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def export(census: Path, ranking: Path, output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "evaluation-diversity.json": output / "evaluation-diversity.json",
        "protocol-explorer.json": output / "protocol-explorer.json",
    }
    _write_json(
        {"census": census_payload(census), "ranking": ranking_payload(ranking)}, files["evaluation-diversity.json"]
    )
    explorer = explorer_payload(ranking)
    verify_explorer(
        explorer,
        _read(ranking, "protocol-pair-comparisons.csv"),
        _read(ranking, "aggregation-comparisons.csv"),
    )
    _write_json(explorer, files["protocol-explorer.json"])
    (output / "manifest.json").write_text(json.dumps(manifest(census, ranking, files), indent=2) + "\n")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--census", type=Path, required=True, help="figures/protocol-diversity (with source-data/)")
    parser.add_argument(
        "--ranking", type=Path, required=True, help="figures/ranking-sensitivity-atlas (with source-data/)"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name, path in export(args.census, args.ranking, args.out).items():
        print(f"{name}: {path.stat().st_size / 1e3:,.0f} kB")
    return 0
