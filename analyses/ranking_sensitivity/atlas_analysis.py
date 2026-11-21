"""Assemble and analyse the completed Atlas ranking-sensitivity score bundles.

The primary analysis is deliberately descriptive and reference-free. Each literature-eligible
protocol receives one vote, protocol pairs are compared on their matched ground-truth-defined
evaluation units, and the three benchmark tasks are never pooled.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DETAIL_COLUMNS = {
    "task",
    "fold",
    "model",
    "training_seed",
    "protocol_id",
    "evaluation_unit",
    "score",
    "undefined_reason",
    "representation",
    "better",
    "score_defined",
}
AVAILABILITY_COLUMNS = {
    "task",
    "fold",
    "protocol_id",
    "evaluation_unit",
    "ground_truth_defined",
    "undefined_reason",
}
OPERATIONAL_COLUMNS = [
    "independent_work_count",
    "population",
    "space",
    "centering",
    "reference_population",
    "feature_set",
    "feature_ranking_basis",
    "feature_k_bin",
    "target_operation",
    "weighting",
    "weight_source",
    "weight_reference",
    "weight_transform",
    "metric_family",
    "canonical_feature_k",
]
FACTOR_COLUMNS = {
    "metric": ("population", "space", "metric_family"),
    "feature_panel": ("feature_set", "feature_ranking_basis", "canonical_feature_k"),
    "centering": ("centering", "reference_population"),
    "target_policy": ("target_operation",),
    "weighting": ("weighting", "weight_source", "weight_reference", "weight_transform"),
}
TASK_LABELS = {
    "essential_perturbation_generalization": "Perturbation generalization",
    "genetic_context_transfer": "Genetic context transfer",
    "chemical_context_transfer": "Chemical context transfer",
}


@dataclass(frozen=True)
class ProtocolRanking:
    scores: dict[str, float]
    ranks: dict[str, float]
    winners: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        raise ValueError(f"{column} contains non-boolean values: {sorted(normalized[invalid].unique())}")
    return normalized.eq("true")


def _read_completion(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_details(details_root: Path, families: pd.DataFrame, task_manifest: pd.DataFrame) -> pd.DataFrame:
    """Read checksum-valid detail bundles and return one validated canonical score table."""

    paths = sorted(details_root.glob("*/*/*/seed-*.csv"))
    if not paths:
        raise FileNotFoundError(f"no detail score bundles below {details_root}")
    expected_protocols = {
        task: set(group["protocol_id"].astype(str)) for task, group in task_manifest.groupby("task", sort=False)
    }
    frames: list[pd.DataFrame] = []
    bundle_keys: set[tuple[str, str, str, int]] = set()
    for path in paths:
        completion_path = path.with_suffix(".complete.json")
        if not completion_path.is_file():
            raise FileNotFoundError(f"missing completion record for {path}")
        completion = _read_completion(completion_path)
        if completion.get("status") != "complete" or completion.get("output_sha256") != _sha256(path):
            raise ValueError(f"stale or invalid completion record for {path}")
        frame = pd.read_csv(path, keep_default_na=False)
        missing = sorted(DETAIL_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing detail columns: {missing}")
        frame["score_defined"] = _as_bool(frame["score_defined"], "score_defined")
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        key_values = frame[["task", "fold", "model", "training_seed"]].drop_duplicates()
        if len(key_values) != 1:
            raise ValueError(f"{path} contains multiple bundle identities")
        key = tuple(key_values.iloc[0].tolist())
        typed_key = (str(key[0]), str(key[1]), str(key[2]), int(key[3]))
        if typed_key in bundle_keys:
            raise ValueError(f"duplicate detail bundle identity: {typed_key}")
        bundle_keys.add(typed_key)
        task = typed_key[0]
        observed_protocols = set(frame["protocol_id"].astype(str))
        if observed_protocols != expected_protocols.get(task, set()):
            raise ValueError(f"{path} protocol set does not match the frozen task manifest")
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    keys = ["task", "fold", "model", "training_seed", "protocol_id", "evaluation_unit"]
    if result.duplicated(keys).any():
        raise ValueError("detail bundles contain duplicate score rows")
    better_counts = result.groupby("protocol_id")["better"].nunique()
    if (better_counts != 1).any():
        raise ValueError("a protocol has inconsistent optimization directions")
    representation_counts = result.groupby("protocol_id")["representation"].nunique()
    if (representation_counts != 1).any():
        raise ValueError("a protocol has inconsistent representations")

    metadata = families.set_index("family_id")[OPERATIONAL_COLUMNS]
    result = result.merge(metadata, left_on="protocol_id", right_index=True, how="left", validate="many_to_one")
    if result["metric_family"].isna().any():
        raise ValueError("detail table contains a protocol absent from the frozen family table")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def validate_availability(availability: pd.DataFrame, task_manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate the independently generated ground-truth availability table."""

    missing = sorted(AVAILABILITY_COLUMNS - set(availability.columns))
    if missing:
        raise ValueError(f"availability table is missing columns: {missing}")
    result = availability[list(AVAILABILITY_COLUMNS)].copy()
    result["ground_truth_defined"] = _as_bool(result["ground_truth_defined"], "ground_truth_defined")
    keys = ["task", "fold", "protocol_id", "evaluation_unit"]
    if result.duplicated(keys).any():
        raise ValueError("availability table contains duplicate rows")
    expected = {
        task: set(group["protocol_id"].astype(str)) for task, group in task_manifest.groupby("task", sort=False)
    }
    for task, group in result.groupby("task", sort=False):
        if set(group["protocol_id"].astype(str)) != expected.get(str(task), set()):
            raise ValueError(f"availability protocols do not match the frozen manifest for {task}")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def rank_models(scores: dict[str, float], better: str) -> ProtocolRanking:
    """Rank models with average ranks and fractional winner ties."""

    if better not in {"higher", "lower"}:
        raise ValueError(f"unknown optimization direction: {better}")
    series = pd.Series(scores, dtype=float)
    if series.empty or not np.isfinite(series).all():
        raise ValueError("model scores must be non-empty and finite")
    ranks = series.rank(method="average", ascending=better == "lower")
    best = float(series.max() if better == "higher" else series.min())
    winner_mask = np.isclose(series.to_numpy(float), best, rtol=1e-9, atol=1e-12)
    winners = tuple(sorted(series.index[winner_mask].astype(str)))
    return ProtocolRanking(
        scores={str(key): float(value) for key, value in series.items()},
        ranks={str(key): float(value) for key, value in ranks.items()},
        winners=winners,
    )


def compare_rankings(first: ProtocolRanking, second: ProtocolRanking) -> dict[str, Any]:
    """Compare two rankings over an identical model panel."""

    models = sorted(first.ranks)
    if models != sorted(second.ranks):
        raise ValueError("ranking comparison requires the same model panel")
    reversed_pairs = 0
    tie_changed_pairs = 0
    for model_a, model_b in combinations(models, 2):
        delta_a = first.ranks[model_a] - first.ranks[model_b]
        delta_b = second.ranks[model_a] - second.ranks[model_b]
        if delta_a * delta_b < 0:
            reversed_pairs += 1
        elif (delta_a == 0) != (delta_b == 0):
            tie_changed_pairs += 1
    n_pairs = len(models) * (len(models) - 1) // 2
    return {
        "winner_changed": first.winners != second.winners,
        "winners_a": ";".join(first.winners),
        "winners_b": ";".join(second.winners),
        "n_model_pairs": n_pairs,
        "n_reversed_model_pairs": reversed_pairs,
        "pairwise_reversal_fraction": reversed_pairs / n_pairs if n_pairs else np.nan,
        "n_tie_changed_model_pairs": tie_changed_pairs,
        "tie_changed_fraction": tie_changed_pairs / n_pairs if n_pairs else np.nan,
    }


def _factor_level(row: pd.Series, factor: str) -> str:
    values = [str(row[column]) for column in FACTOR_COLUMNS[factor]]
    if factor == "metric":
        return f"{values[0]} · {values[1]} · {values[2]}"
    if factor == "feature_panel":
        ranking = f"/{values[1]}" if values[1] else ""
        return f"{values[0]}{ranking} · k={values[2]}"
    if factor == "centering":
        return f"{values[0]} · reference={values[1]}"
    if factor == "target_policy":
        return values[0]
    details = [value for value in values[1:] if value]
    return values[0] if not details else f"{values[0]} · {'/'.join(details)}"


def factor_for_pair(first: pd.Series, second: pd.Series) -> tuple[str, str, str] | None:
    """Return the sole differing evaluation factor, or ``None`` for multi-factor pairs."""

    changed = []
    for factor, columns in FACTOR_COLUMNS.items():
        if tuple(str(first[column]) for column in columns) != tuple(str(second[column]) for column in columns):
            changed.append(factor)
    if len(changed) != 1:
        return None
    factor = changed[0]
    level_a = _factor_level(first, factor)
    level_b = _factor_level(second, factor)
    return factor, level_a, level_b


def _availability_coverage(availability: pd.DataFrame) -> pd.DataFrame:
    by_fold = (
        availability.groupby(["task", "fold", "protocol_id"], sort=True)["ground_truth_defined"]
        .agg(n_total="size", n_defined="sum", defined_fraction="mean")
        .reset_index()
    )
    return (
        by_fold.groupby(["task", "protocol_id"], sort=True)
        .agg(
            n_folds=("fold", "nunique"),
            n_total=("n_total", "sum"),
            n_defined=("n_defined", "sum"),
            overall_defined_fraction=("defined_fraction", "mean"),
            minimum_fold_defined_fraction=("defined_fraction", "min"),
        )
        .reset_index()
    )


def _protocol_aggregate(
    details: pd.DataFrame,
    availability: pd.DataFrame,
    *,
    reduction: str,
) -> pd.DataFrame:
    if reduction not in {"mean", "median"}:
        raise ValueError("reduction must be mean or median")
    truth = availability.loc[availability["ground_truth_defined"], ["task", "fold", "protocol_id", "evaluation_unit"]]
    selected = details.merge(
        truth,
        on=["task", "fold", "protocol_id", "evaluation_unit"],
        how="inner",
        validate="many_to_one",
    )
    finite = selected["score_defined"] & np.isfinite(selected["score"])
    selected = selected.assign(finite_score=finite)
    fold_scores = (
        selected.groupby(["task", "fold", "protocol_id", "model"], sort=True)
        .agg(
            score=("score", reduction),
            n_expected=("evaluation_unit", "size"),
            n_finite=("finite_score", "sum"),
        )
        .reset_index()
    )
    fold_scores["complete"] = fold_scores["n_expected"] == fold_scores["n_finite"]
    result = (
        fold_scores.groupby(["task", "protocol_id", "model"], sort=True)
        .agg(
            score=("score", "mean"),
            n_folds=("fold", "nunique"),
            complete=("complete", "all"),
            n_evaluation_units=("n_expected", "sum"),
        )
        .reset_index()
    )
    result["aggregation"] = reduction
    return result


def protocol_rankings(
    details: pd.DataFrame,
    availability: pd.DataFrame,
    *,
    coverage_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate standard-mean and median rankings plus fractional winner shares."""

    coverage = _availability_coverage(availability)
    mean_scores = _protocol_aggregate(details, availability, reduction="mean")
    median_scores = _protocol_aggregate(details, availability, reduction="median")
    all_scores = pd.concat([mean_scores, median_scores], ignore_index=True)
    directions = details[["protocol_id", "better"]].drop_duplicates().set_index("protocol_id")["better"]
    ranking_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    coverage_lookup = coverage.set_index(["task", "protocol_id"])
    for (task, protocol, aggregation), group in all_scores.groupby(["task", "protocol_id", "aggregation"], sort=True):
        complete = bool(group["complete"].all())
        ranking = rank_models(dict(zip(group["model"], group["score"], strict=True)), directions.loc[protocol])
        protocol_coverage = coverage_lookup.loc[(task, protocol)]
        headline = bool(
            aggregation == "mean"
            and complete
            and protocol_coverage["minimum_fold_defined_fraction"] >= coverage_threshold
        )
        for model in sorted(ranking.ranks):
            ranking_rows.append(
                {
                    "task": task,
                    "protocol_id": protocol,
                    "aggregation": aggregation,
                    "model": model,
                    "score": ranking.scores[model],
                    "rank": ranking.ranks[model],
                    "winner": model in ranking.winners,
                    "winner_credit": 1 / len(ranking.winners) if model in ranking.winners else 0.0,
                    "complete_model_panel": complete,
                    "headline_eligible": headline,
                    "n_evaluation_units": int(group.loc[group["model"] == model, "n_evaluation_units"].iloc[0]),
                    "minimum_fold_defined_fraction": protocol_coverage["minimum_fold_defined_fraction"],
                }
            )
        winner_rows.append(
            {
                "task": task,
                "protocol_id": protocol,
                "aggregation": aggregation,
                "winners": ";".join(ranking.winners),
                "n_winners": len(ranking.winners),
                "complete_model_panel": complete,
                "headline_eligible": headline,
                "minimum_fold_defined_fraction": protocol_coverage["minimum_fold_defined_fraction"],
            }
        )
    rankings = pd.DataFrame(ranking_rows)
    winners = pd.DataFrame(winner_rows)
    shares = (
        rankings.loc[rankings["headline_eligible"]]
        .groupby(["task", "model"], sort=True)
        .agg(winner_credit=("winner_credit", "sum"), n_protocols=("protocol_id", "nunique"))
        .reset_index()
    )
    totals = shares.groupby("task")["winner_credit"].transform("sum")
    shares["winner_share"] = shares["winner_credit"] / totals
    shares["task_label"] = shares["task"].map(TASK_LABELS)
    return rankings, winners, shares


def _score_matrices(
    details: pd.DataFrame,
    task: str,
    models: list[str],
) -> dict[tuple[str, str], pd.DataFrame]:
    task_rows = details[details["task"] == task]
    matrices = {}
    for (fold, protocol), group in task_rows.groupby(["fold", "protocol_id"], sort=True):
        matrix = group.pivot(index="evaluation_unit", columns="model", values="score").reindex(columns=models)
        matrices[(str(fold), str(protocol))] = matrix
    return matrices


def protocol_pair_comparisons(
    details: pd.DataFrame,
    availability: pd.DataFrame,
    pair_coverage: pd.DataFrame,
    families: pd.DataFrame,
    *,
    coverage_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every protocol pair on its fold-matched ground-truth intersection."""

    directions = details[["protocol_id", "better"]].drop_duplicates().set_index("protocol_id")["better"]
    metadata = families.set_index("family_id")
    comparison_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for task, task_pairs in pair_coverage.groupby("task", sort=True):
        task_details = details[details["task"] == task]
        models = sorted(task_details["model"].unique())
        folds = sorted(task_details["fold"].unique())
        matrices = _score_matrices(details, str(task), models)
        task_availability = availability[availability["task"] == task]
        masks = {
            (str(fold), str(protocol)): group.set_index("evaluation_unit")["ground_truth_defined"].astype(bool)
            for (fold, protocol), group in task_availability.groupby(["fold", "protocol_id"], sort=True)
        }
        for pair in task_pairs.itertuples(index=False):
            protocol_a = str(pair.protocol_a)
            protocol_b = str(pair.protocol_b)
            eligible = bool(pair.minimum_fold_common_fraction >= coverage_threshold)
            fold_scores: dict[str, list[pd.Series]] = {protocol_a: [], protocol_b: []}
            common_counts: list[int] = []
            failure = ""
            if eligible:
                for fold in folds:
                    mask_a = masks[(fold, protocol_a)]
                    mask_b = masks[(fold, protocol_b)]
                    common_units = mask_a.index[mask_a & mask_b]
                    common_counts.append(len(common_units))
                    for protocol in [protocol_a, protocol_b]:
                        matrix = matrices[(fold, protocol)].reindex(index=common_units, columns=models)
                        if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy(float)).all():
                            failure = f"nonfinite_model_score:{fold}:{protocol}"
                            break
                        fold_scores[protocol].append(matrix.mean(axis=0))
                    if failure:
                        break
            if not eligible or failure:
                comparison_rows.append(
                    {
                        "comparison_id": f"{task}|{protocol_a}|{protocol_b}",
                        "task": task,
                        "protocol_a": protocol_a,
                        "protocol_b": protocol_b,
                        "analysis_eligible": False,
                        "exclusion_reason": failure or "below_coverage_threshold",
                        "minimum_fold_common_fraction": float(pair.minimum_fold_common_fraction),
                        "n_common": int(pair.n_common),
                    }
                )
                continue
            scores_a = pd.DataFrame(fold_scores[protocol_a]).mean(axis=0).to_dict()
            scores_b = pd.DataFrame(fold_scores[protocol_b]).mean(axis=0).to_dict()
            ranking_a = rank_models(scores_a, directions.loc[protocol_a])
            ranking_b = rank_models(scores_b, directions.loc[protocol_b])
            compared = compare_rankings(ranking_a, ranking_b)
            factor = factor_for_pair(metadata.loc[protocol_a], metadata.loc[protocol_b])
            factor_name, level_a, level_b = factor if factor else ("", "", "")
            comparison_id = f"{task}|{protocol_a}|{protocol_b}"
            comparison_rows.append(
                {
                    "comparison_id": comparison_id,
                    "task": task,
                    "protocol_a": protocol_a,
                    "protocol_b": protocol_b,
                    "analysis_eligible": True,
                    "exclusion_reason": "",
                    "minimum_fold_common_fraction": float(pair.minimum_fold_common_fraction),
                    "n_common": int(pair.n_common),
                    "minimum_common_units_per_fold": min(common_counts),
                    "factor": factor_name,
                    "level_a": level_a,
                    "level_b": level_b,
                    **compared,
                }
            )
            for model in models:
                rank_rows.append(
                    {
                        "comparison_id": comparison_id,
                        "task": task,
                        "protocol_a": protocol_a,
                        "protocol_b": protocol_b,
                        "factor": factor_name,
                        "level_a": level_a,
                        "level_b": level_b,
                        "model": model,
                        "score_a": ranking_a.scores[model],
                        "score_b": ranking_b.scores[model],
                        "rank_a": ranking_a.ranks[model],
                        "rank_b": ranking_b.ranks[model],
                        "winner_a": model in ranking_a.winners,
                        "winner_b": model in ranking_b.winners,
                    }
                )
    return pd.DataFrame(comparison_rows), pd.DataFrame(rank_rows)


def aggregation_comparisons(rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare mean and median reductions for each high-coverage atomic protocol."""

    rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    eligible_protocols = rankings.loc[
        (rankings["aggregation"] == "mean") & rankings["headline_eligible"], ["task", "protocol_id"]
    ].drop_duplicates()
    for item in eligible_protocols.itertuples(index=False):
        group = rankings[(rankings["task"] == item.task) & (rankings["protocol_id"] == item.protocol_id)]
        mean_rows = group[group["aggregation"] == "mean"].set_index("model")
        median_rows = group[group["aggregation"] == "median"].set_index("model")
        models = sorted(mean_rows.index)
        first = ProtocolRanking(
            scores=mean_rows["score"].to_dict(),
            ranks=mean_rows["rank"].to_dict(),
            winners=tuple(sorted(mean_rows.index[mean_rows["winner"]])),
        )
        second = ProtocolRanking(
            scores=median_rows["score"].to_dict(),
            ranks=median_rows["rank"].to_dict(),
            winners=tuple(sorted(median_rows.index[median_rows["winner"]])),
        )
        comparison_id = f"{item.task}|aggregation|{item.protocol_id}"
        rows.append(
            {
                "comparison_id": comparison_id,
                "task": item.task,
                "protocol_a": item.protocol_id,
                "protocol_b": item.protocol_id,
                "analysis_eligible": True,
                "exclusion_reason": "",
                "minimum_fold_common_fraction": float(mean_rows["minimum_fold_defined_fraction"].iloc[0]),
                "n_common": int(mean_rows["n_evaluation_units"].min()),
                "factor": "aggregation",
                "level_a": "mean",
                "level_b": "median",
                **compare_rankings(first, second),
            }
        )
        for model in models:
            rank_rows.append(
                {
                    "comparison_id": comparison_id,
                    "task": item.task,
                    "protocol_a": item.protocol_id,
                    "protocol_b": item.protocol_id,
                    "factor": "aggregation",
                    "level_a": "mean",
                    "level_b": "median",
                    "model": model,
                    "score_a": first.scores[model],
                    "score_b": second.scores[model],
                    "rank_a": first.ranks[model],
                    "rank_b": second.ranks[model],
                    "winner_a": model in first.winners,
                    "winner_b": model in second.winners,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


def summarize_comparisons(
    pairs: pd.DataFrame,
    aggregation_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = pairs[pairs["analysis_eligible"]].copy()
    task_summary = (
        eligible.groupby("task", sort=True)
        .agg(
            n_protocol_pairs=("comparison_id", "size"),
            n_winner_disagreements=("winner_changed", "sum"),
            winner_disagreement_rate=("winner_changed", "mean"),
            mean_pairwise_reversal_fraction=("pairwise_reversal_fraction", "mean"),
            median_pairwise_reversal_fraction=("pairwise_reversal_fraction", "median"),
        )
        .reset_index()
    )
    task_summary["task_label"] = task_summary["task"].map(TASK_LABELS)

    factor_rows = pd.concat([eligible[eligible["factor"].ne("")], aggregation_pairs], ignore_index=True, sort=False)
    factor_summary = (
        factor_rows.groupby(["task", "factor"], sort=True)
        .agg(
            n_matched_contrasts=("comparison_id", "size"),
            n_winner_changes=("winner_changed", "sum"),
            winner_change_rate=("winner_changed", "mean"),
            mean_pairwise_reversal_fraction=("pairwise_reversal_fraction", "mean"),
            median_pairwise_reversal_fraction=("pairwise_reversal_fraction", "median"),
        )
        .reset_index()
    )
    factor_summary["task_label"] = factor_summary["task"].map(TASK_LABELS)

    concrete = factor_rows.copy()
    swap = concrete["level_a"] > concrete["level_b"]
    concrete.loc[swap, ["level_a", "level_b"]] = concrete.loc[swap, ["level_b", "level_a"]].to_numpy()
    concrete_summary = (
        concrete.groupby(["task", "factor", "level_a", "level_b"], sort=True)
        .agg(
            n_backgrounds=("comparison_id", "size"),
            n_winner_changes=("winner_changed", "sum"),
            winner_change_rate=("winner_changed", "mean"),
            mean_pairwise_reversal_fraction=("pairwise_reversal_fraction", "mean"),
            maximum_pairwise_reversal_fraction=("pairwise_reversal_fraction", "max"),
        )
        .reset_index()
    )
    concrete_summary["task_label"] = concrete_summary["task"].map(TASK_LABELS)
    return task_summary, factor_summary, concrete_summary


def select_bump_examples(
    factor_pairs: pd.DataFrame,
    selected_examples: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Rank every candidate and mark either frozen or deterministic fallback examples."""

    candidates = factor_pairs[
        factor_pairs["analysis_eligible"] & factor_pairs["factor"].ne("") & factor_pairs["winner_changed"]
    ].copy()
    candidates = candidates.sort_values(
        ["pairwise_reversal_fraction", "minimum_fold_common_fraction", "comparison_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    rationale: dict[str, str] = {}
    if selected_examples is not None:
        selected = [str(item["comparison_id"]) for item in selected_examples]
        rationale = {str(item["comparison_id"]): str(item["rationale"]) for item in selected_examples}
        missing = sorted(set(selected) - set(candidates["comparison_id"]))
        if missing:
            raise ValueError(f"frozen bump examples are not eligible winner-changing contrasts: {missing}")
    else:
        selected = []
        for task in TASK_LABELS:
            task_candidates = candidates[candidates["task"] == task]
            if not task_candidates.empty:
                selected.append(str(task_candidates.iloc[0]["comparison_id"]))
        used_factors = set(candidates.loc[candidates["comparison_id"].isin(selected), "factor"])
        remaining = candidates[~candidates["comparison_id"].isin(selected)]
        diverse = remaining[~remaining["factor"].isin(used_factors)]
        fourth_pool = diverse if not diverse.empty else remaining
        if not fourth_pool.empty:
            selected.append(str(fourth_pool.iloc[0]["comparison_id"]))
    result = candidates.copy()
    result.insert(0, "candidate_rank", np.arange(1, len(result) + 1))
    result["selected"] = result["comparison_id"].isin(selected)
    selection_order = {comparison_id: index + 1 for index, comparison_id in enumerate(selected)}
    result["selection_order"] = result["comparison_id"].map(selection_order).astype("Int64")
    fallback = "task-covering maximum reversal, plus one factor-diverse maximum reversal"
    result["selection_rule"] = result["comparison_id"].map(rationale).fillna("")
    if selected_examples is None:
        result.loc[result["selected"], "selection_rule"] = fallback
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_canonical(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _checksums(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): _sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def analyse(
    *,
    details_root: Path,
    availability_path: Path,
    pair_coverage_path: Path,
    families_path: Path,
    task_manifest_path: Path,
    examples_path: Path | None,
    output: Path,
    coverage_threshold: float,
) -> dict[str, Any]:
    families = pd.read_csv(families_path, keep_default_na=False)
    task_manifest = pd.read_csv(task_manifest_path, keep_default_na=False)
    availability = validate_availability(pd.read_csv(availability_path, keep_default_na=False), task_manifest)
    details = collect_details(details_root, families, task_manifest)
    pair_coverage = pd.read_csv(pair_coverage_path)
    source = output / "source-data"
    _write_canonical(details, source / "canonical-scores.csv.gz")

    rankings, protocol_winners, winner_shares = protocol_rankings(
        details, availability, coverage_threshold=coverage_threshold
    )
    pairs, pair_ranks = protocol_pair_comparisons(
        details,
        availability,
        pair_coverage,
        families,
        coverage_threshold=coverage_threshold,
    )
    aggregation_pairs, aggregation_ranks = aggregation_comparisons(rankings)
    task_summary, factor_summary, concrete_summary = summarize_comparisons(pairs, aggregation_pairs)
    factor_pairs = pd.concat(
        [pairs[pairs["factor"].fillna("").ne("")], aggregation_pairs], ignore_index=True, sort=False
    )
    all_pair_ranks = pd.concat([pair_ranks, aggregation_ranks], ignore_index=True, sort=False)
    selected_examples = None
    if examples_path is not None:
        selected_examples = json.loads(examples_path.read_text())["examples"]
    candidates = select_bump_examples(factor_pairs, selected_examples)
    selected_ids = set(candidates.loc[candidates["selected"], "comparison_id"])
    bump_source = all_pair_ranks[all_pair_ranks["comparison_id"].isin(selected_ids)].merge(
        candidates[["comparison_id", "selection_order", "pairwise_reversal_fraction", "winners_a", "winners_b"]],
        on="comparison_id",
        how="left",
        validate="many_to_one",
    )

    outputs = {
        "protocol-rankings.csv": rankings,
        "protocol-winners.csv": protocol_winners,
        "winner-shares.csv": winner_shares,
        "protocol-pair-comparisons.csv": pairs,
        "protocol-pair-model-ranks.csv": pair_ranks,
        "aggregation-comparisons.csv": aggregation_pairs,
        "task-protocol-disagreement.csv": task_summary,
        "factor-sensitivity.csv": factor_summary,
        "concrete-factor-contrasts.csv": concrete_summary,
        "bump-plot-candidates.csv": candidates,
        "bump-plot-source.csv": bump_source,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, source / filename)

    manifest = {
        "schema_version": 1,
        "coverage_threshold": coverage_threshold,
        "vote_weighting": "equal",
        "winner_tie_credit": "fractional",
        "primary_aggregation": "mean within fold, then equal mean across folds",
        "pair_matching": "ground-truth-defined intersection within each task and fold",
        "tasks_pooled": False,
        "n_detail_rows": len(details),
        "n_detail_bundles": len(list(details_root.glob("*/*/*/seed-*.complete.json"))),
        "n_protocol_pairs": len(pairs),
        "n_eligible_protocol_pairs": int(pairs["analysis_eligible"].sum()),
        "inputs": {
            "availability_sha256": _sha256(availability_path),
            "pair_coverage_sha256": _sha256(pair_coverage_path),
            "families_sha256": _sha256(families_path),
            "task_manifest_sha256": _sha256(task_manifest_path),
            "examples_sha256": _sha256(examples_path) if examples_path is not None else None,
        },
    }
    (output / "analysis-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["source_data_sha256"] = _checksums(source)
    (output / "analysis-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details-root", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--pair-coverage", type=Path, required=True)
    parser.add_argument(
        "--families",
        type=Path,
        default=Path("configs/evaluation/atlas-protocol-eligibility-v1.csv"),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("configs/evaluation/atlas-task-protocol-manifest-v1.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.75)
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("configs/evaluation/ranking-sensitivity-atlas-examples-v1.json"),
    )
    args = parser.parse_args()
    analyse(
        details_root=args.details_root.resolve(),
        availability_path=args.availability.resolve(),
        pair_coverage_path=args.pair_coverage.resolve(),
        families_path=args.families.resolve(),
        task_manifest_path=args.task_manifest.resolve(),
        examples_path=args.examples.resolve() if args.examples else None,
        output=args.output.resolve(),
        coverage_threshold=args.coverage_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
