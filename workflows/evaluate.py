"""Evaluate fixed prediction bundles over the frozen post-training sensitivity grid."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kendalltau, spearmanr

from workflows.predictions import expected_test_conditions


def _dense_mean(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel()


def apply_prediction_evaluation_transform(prediction: ad.AnnData, transform: str) -> ad.AnnData:
    """Map a declared native prediction space into the shared evaluation view."""
    if transform == "identity":
        return prediction
    if transform != "normalize-total-10000-log1p":
        raise ValueError(f"evaluation transform {transform!r} is not implemented")

    result = prediction.copy()
    if sp.issparse(result.X):
        totals = np.asarray(result.X.sum(axis=1)).ravel()
        if np.any(totals <= 0):
            raise ValueError("prediction rows must have positive totals before normalization")
        result.X = sp.diags(10_000.0 / totals) @ result.X
        result.X.data = np.log1p(result.X.data).astype(np.float32)
    else:
        values = np.asarray(result.X, dtype=np.float32)
        totals = values.sum(axis=1, keepdims=True, dtype=np.float64)
        if np.any(totals <= 0):
            raise ValueError("prediction rows must have positive totals before normalization")
        result.X = np.log1p(values * (10_000.0 / totals)).astype(np.float32)
    return result


def _load_prediction_for_evaluation(prediction_path: Path) -> ad.AnnData:
    prediction = ad.read_h5ad(prediction_path)
    manifest_path = prediction_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return prediction
    manifest = json.loads(manifest_path.read_text())
    transform = str(manifest.get("expression", {}).get("evaluation_transform", "identity"))
    return apply_prediction_evaluation_transform(prediction, transform)


def observed_effect_strengths(truth: ad.AnnData, control_label: str = "control") -> dict[str, float]:
    """Full-gene L2 distance from the observed control centroid, fixed across variants."""
    labels = truth.obs["perturbation"].astype(str).to_numpy()
    control = _dense_mean(truth.X[labels == control_label])
    strengths = {}
    for perturbation in sorted(set(labels) - {control_label}):
        delta = _dense_mean(truth.X[labels == perturbation]) - control
        strengths[perturbation] = float(np.linalg.norm(delta))
    return strengths


def _seed(seed: int, label: str) -> np.random.Generator:
    return np.random.default_rng(np.array([seed, zlib.crc32(label.encode())], dtype=np.uint32))


def matched_draw(
    data: ad.AnnData,
    conditions: list[str],
    draw_size: int,
    seed: int,
    *,
    retain_controls: bool,
    control_label: str = "control",
) -> ad.AnnData:
    """Draw exactly ``draw_size`` rows per condition with condition-keyed frozen RNGs."""
    labels = data.obs["perturbation"].astype(str).to_numpy()
    keep = []
    if retain_controls:
        keep.extend(np.where(labels == control_label)[0].tolist())
    for condition in sorted(conditions):
        indices = np.where(labels == condition)[0]
        if len(indices) < draw_size:
            raise ValueError(f"{condition!r} has {len(indices)} rows, fewer than required draw size {draw_size}")
        chosen = _seed(seed, condition).choice(indices, size=draw_size, replace=False)
        keep.extend(sorted(chosen.tolist()))
    return data[np.asarray(keep)].copy()


def _load_fold(
    truth_path: Path,
    prediction_path: Path,
    split_manifest: dict,
    fold: int,
    capability: str,
) -> tuple[ad.AnnData, ad.AnnData, list[str]]:
    expected = expected_test_conditions(split_manifest, fold, capability)
    source = ad.read_h5ad(truth_path, backed="r")
    fold_spec = split_manifest["folds"][fold]
    if "test_context" in fold_spec:
        required = {"context", "evaluation_condition"}
        missing = sorted(required - set(source.obs.columns))
        if missing:
            raise ValueError(f"context-transfer truth is missing obs fields: {', '.join(missing)}")
        context_mask = source.obs["context"].astype(str) == str(fold_spec["test_context"])
        labels = source.obs["evaluation_condition"].astype(str)
        mask = (context_mask & labels.isin(["control", *expected])).to_numpy()
    else:
        labels = source.obs["perturbation"].astype(str)
        mask = labels.isin(["control", *expected]).to_numpy()
    try:
        truth = source[mask].to_memory()
    finally:
        source.file.close()
    if "test_context" in fold_spec:
        truth.obs["perturbation"] = truth.obs["evaluation_condition"].astype(str).to_numpy()
    prediction = _load_prediction_for_evaluation(prediction_path)
    observed = sorted(prediction.obs["perturbation"].astype(str).unique().tolist())
    if observed != expected:
        raise ValueError("prediction conditions do not match the selected split fold")
    return truth, prediction, expected


def _score_variants(
    truth: ad.AnnData,
    prediction: ad.AnnData,
    variants: list[dict],
    *,
    minimum_observed_cells: int,
    seed: int,
    draw_seed: int | None,
) -> list[pd.DataFrame]:
    import scperteval as spe

    protocols = sorted({variant["protocol"] for variant in variants})
    prepared = spe.prepare(
        truth,
        protocols,
        min_cells=minimum_observed_cells,
        seed=seed,
        control_label="control",
    )
    frames = []
    for variant in variants:
        result = spe.score(
            prepared,
            variant["protocol"],
            prediction,
            de_method=variant.get("de_method", "t-test"),
            center_on=variant.get("center_on"),
        )
        frame = result.per_perturbation[["perturbation", "score"]].copy()
        frame["variant"] = variant["id"]
        frame["factor"] = variant["factor"]
        frame["level"] = variant["level"]
        frame["protocol"] = variant["protocol"]
        frame["de_method"] = variant.get("de_method", "t-test")
        frame["better"] = variant["better"]
        frame["draw_seed"] = draw_seed
        frames.append(frame)
    return frames


def evaluate_bundle(
    *,
    truth_path: Path,
    prediction_path: Path,
    split_manifest_path: Path,
    grid_path: Path,
    model_id: str,
    capability: str,
    fold: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate one fixed model/fold bundle; no training or prediction occurs here."""
    split = json.loads(split_manifest_path.read_text())
    grid = json.loads(grid_path.read_text())
    truth, prediction, conditions = _load_fold(truth_path, prediction_path, split, fold, capability)
    effects = observed_effect_strengths(truth)

    point_variants = [
        variant
        for variant in grid["variants"]
        if capability in variant["capabilities"] and not variant["id"].startswith("population-")
    ]
    frames = _score_variants(
        truth,
        prediction,
        point_variants,
        minimum_observed_cells=grid["minimum_observed_cells"],
        seed=seed,
        draw_seed=None,
    )

    if capability == "population":
        population_variants = [
            variant for variant in grid["variants"] if variant["id"].startswith("population-")
        ]
        draw_size = int(grid["population"]["draw_size"])
        for draw_seed in grid["population"]["draw_seeds"]:
            drawn_truth = matched_draw(truth, conditions, draw_size, draw_seed, retain_controls=True)
            drawn_prediction = matched_draw(
                prediction,
                conditions,
                draw_size,
                draw_seed,
                retain_controls=False,
            )
            frames.extend(
                _score_variants(
                    drawn_truth,
                    drawn_prediction,
                    population_variants,
                    minimum_observed_cells=draw_size,
                    seed=seed,
                    draw_seed=draw_seed,
                )
            )

    detail = pd.concat(frames, ignore_index=True)
    detail["effect_strength"] = detail["perturbation"].map(effects)
    detail["model"] = model_id
    detail["dataset"] = split["dataset"]
    detail["split"] = split["id"]
    detail["fold"] = fold
    detail["capability"] = capability
    return detail


def summarize_evaluations(detail: pd.DataFrame) -> pd.DataFrame:
    """Average repeated draws, then aggregate uniformly, by median, and by effect strength."""
    keys = ["model", "dataset", "split", "capability", "variant", "factor", "level", "protocol", "better"]
    per_perturbation = (
        detail.groupby([*keys, "perturbation"], as_index=False, dropna=False)
        .agg(score=("score", "mean"), effect_strength=("effect_strength", "first"))
    )
    rows = []
    for values, group in per_perturbation.groupby(keys, sort=False, dropna=False):
        base = dict(zip(keys, values, strict=True))
        scores = group["score"].to_numpy(float)
        effects = group["effect_strength"].to_numpy(float)
        finite = np.isfinite(scores)
        scores = scores[finite]
        effects = effects[finite]
        if not len(scores):
            aggregates = {"mean": np.nan, "median": np.nan, "effect-weighted-mean": np.nan}
        else:
            aggregates = {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "effect-weighted-mean": float(np.average(scores, weights=effects)) if effects.sum() > 0 else np.nan,
            }
        for aggregation, score in aggregates.items():
            rows.append({**base, "aggregation": aggregation, "score": score, "n_perturbations": len(scores)})
    return pd.DataFrame(rows)


def rank_models(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank models independently within every fixed variant and aggregation."""
    result = summary.copy()
    group_keys = ["dataset", "split", "variant", "aggregation"]
    ranks = pd.Series(index=result.index, dtype=float)
    for _, group in result.groupby(group_keys, dropna=False):
        directions = group["better"].unique()
        if len(directions) != 1:
            raise ValueError("a variant has inconsistent metric directions")
        ranks.loc[group.index] = group["score"].rank(
            method="min",
            ascending=directions[0] == "lower",
            na_option="bottom",
        )
    result["rank"] = ranks
    return result


def _one_factor_scenarios(
    summary: pd.DataFrame,
    *,
    reference_variant: str = "reference",
    reference_aggregation: str = "mean",
) -> list[dict[str, str]]:
    """Return the frozen one-factor-at-a-time comparisons represented in ``summary``.

    Metric/protocol variants are compared using the reference aggregation. Aggregation
    variants are compared using the reference protocol. This deliberately excludes the
    Cartesian combinations (for example MMD plus median aggregation), which would change two
    evaluation choices at once.
    """
    scenarios = [
        {
            "id": "reference",
            "variant": reference_variant,
            "aggregation": reference_aggregation,
            "factor": "reference",
            "level": "reference",
            "stratum": "profile",
        }
    ]
    metadata = summary[["variant", "factor", "level"]].drop_duplicates()
    for row in metadata.itertuples(index=False):
        if row.variant == reference_variant:
            continue
        scenarios.append(
            {
                "id": str(row.variant),
                "variant": str(row.variant),
                "aggregation": reference_aggregation,
                "factor": str(row.factor),
                "level": str(row.level),
                "stratum": "population" if row.factor == "population_metric" else "profile",
            }
        )
    aggregations = sorted(summary.loc[summary["variant"] == reference_variant, "aggregation"].unique())
    for aggregation in aggregations:
        if aggregation == reference_aggregation:
            continue
        scenarios.append(
            {
                "id": f"aggregation-{aggregation}",
                "variant": reference_variant,
                "aggregation": str(aggregation),
                "factor": "aggregation",
                "level": str(aggregation),
                "stratum": "profile",
            }
        )
    return scenarios


def _rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    directions = result["better"].dropna().unique()
    if len(directions) != 1:
        raise ValueError("a scenario has inconsistent metric directions")
    result["rank"] = result["score"].rank(
        method="min",
        ascending=directions[0] == "lower",
        na_option="bottom",
    )
    return result


def _rank_correlations(reference: np.ndarray, alternative: np.ndarray) -> tuple[float, float]:
    if len(reference) < 2:
        return np.nan, np.nan
    return float(spearmanr(reference, alternative).statistic), float(kendalltau(reference, alternative).statistic)


def _pairwise_reversal_fraction(reference: np.ndarray, alternative: np.ndarray) -> float:
    reversals = 0
    comparable = 0
    for left in range(len(reference)):
        for right in range(left + 1, len(reference)):
            reference_order = np.sign(reference[left] - reference[right])
            alternative_order = np.sign(alternative[left] - alternative[right])
            if reference_order == 0 or alternative_order == 0:
                continue
            comparable += 1
            reversals += int(reference_order != alternative_order)
    return float(reversals / comparable) if comparable else np.nan


def compare_to_reference(
    summary: pd.DataFrame,
    *,
    reference_variant: str = "reference",
    reference_aggregation: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every one-factor scenario with the reference using identical competitors.

    Reference ranks are recomputed on the model intersection for each alternative. This is
    essential for population-only metrics: a rank change must not be caused merely by dropping
    point-only models from the competitor set.
    """
    required = {
        "model",
        "dataset",
        "split",
        "variant",
        "factor",
        "level",
        "aggregation",
        "score",
        "better",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing columns: {', '.join(missing)}")

    scenarios = _one_factor_scenarios(
        summary,
        reference_variant=reference_variant,
        reference_aggregation=reference_aggregation,
    )
    reference_scenario = scenarios[0]
    rank_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for (dataset, split), study in summary.groupby(["dataset", "split"], dropna=False):
        reference = study[
            (study["variant"] == reference_scenario["variant"])
            & (study["aggregation"] == reference_scenario["aggregation"])
        ]
        if reference.empty:
            raise ValueError(f"missing reference scenario for dataset={dataset!r}, split={split!r}")
        for scenario in scenarios[1:]:
            alternative = study[
                (study["variant"] == scenario["variant"])
                & (study["aggregation"] == scenario["aggregation"])
            ]
            common_models = sorted(set(reference["model"]) & set(alternative["model"]))
            if len(common_models) < 2:
                continue
            ref = _rank_frame(reference[reference["model"].isin(common_models)])
            alt = _rank_frame(alternative[alternative["model"].isin(common_models)])
            if ref["model"].duplicated().any() or alt["model"].duplicated().any():
                raise ValueError("each model must have exactly one aggregate score per scenario")
            paired = ref[["model", "score", "rank"]].merge(
                alt[["model", "score", "rank"]],
                on="model",
                suffixes=("_reference", "_alternative"),
                validate="one_to_one",
            ).sort_values("model")
            reference_ranks = paired["rank_reference"].to_numpy(float)
            alternative_ranks = paired["rank_alternative"].to_numpy(float)
            spearman, kendall = _rank_correlations(reference_ranks, alternative_ranks)
            reference_winners = sorted(paired.loc[paired["rank_reference"] == 1, "model"])
            alternative_winners = sorted(paired.loc[paired["rank_alternative"] == 1, "model"])
            for row in paired.itertuples(index=False):
                rank_rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "scenario": scenario["id"],
                        "factor": scenario["factor"],
                        "level": scenario["level"],
                        "stratum": scenario["stratum"],
                        "model": row.model,
                        "reference_score": row.score_reference,
                        "alternative_score": row.score_alternative,
                        "reference_rank": row.rank_reference,
                        "alternative_rank": row.rank_alternative,
                        "rank_change": row.rank_alternative - row.rank_reference,
                    }
                )
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "scenario": scenario["id"],
                    "factor": scenario["factor"],
                    "level": scenario["level"],
                    "stratum": scenario["stratum"],
                    "n_models": len(common_models),
                    "models": ";".join(common_models),
                    "spearman": spearman,
                    "kendall": kendall,
                    "pairwise_reversal_fraction": _pairwise_reversal_fraction(
                        reference_ranks, alternative_ranks
                    ),
                    "winner_changed": reference_winners != alternative_winners,
                    "reference_winners": ";".join(reference_winners),
                    "alternative_winners": ";".join(alternative_winners),
                }
            )
    return pd.DataFrame(rank_rows), pd.DataFrame(comparison_rows)


def _aggregate_scores(values: np.ndarray, effects: np.ndarray, aggregation: str) -> float:
    if aggregation == "mean":
        return float(np.mean(values))
    if aggregation == "median":
        return float(np.median(values))
    if aggregation == "effect-weighted-mean":
        return float(np.average(values, weights=effects)) if effects.sum() > 0 else np.nan
    raise ValueError(f"unsupported aggregation: {aggregation}")


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).any():
        return np.nan, np.nan, np.nan
    return tuple(float(value) for value in np.nanquantile(array, [0.025, 0.5, 0.975]))


def paired_perturbation_bootstrap(
    detail: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    reference_variant: str = "reference",
    reference_aggregation: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap perturbations jointly across models and reference/alternative scores."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    keys = ["model", "dataset", "split", "capability", "variant", "factor", "level", "protocol", "better"]
    per_perturbation = (
        detail.groupby([*keys, "perturbation"], as_index=False, dropna=False)
        .agg(score=("score", "mean"), effect_strength=("effect_strength", "first"))
    )
    scenario_summary = summarize_evaluations(detail)
    scenarios = _one_factor_scenarios(
        scenario_summary,
        reference_variant=reference_variant,
        reference_aggregation=reference_aggregation,
    )
    model_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for (dataset, split), study in per_perturbation.groupby(["dataset", "split"], dropna=False):
        reference = study[study["variant"] == reference_variant]
        for scenario in scenarios[1:]:
            alternative = study[study["variant"] == scenario["variant"]]
            common_models = sorted(set(reference["model"]) & set(alternative["model"]))
            if len(common_models) < 2:
                continue
            ref = reference[reference["model"].isin(common_models)]
            alt = alternative[alternative["model"].isin(common_models)]
            ref_direction = ref["better"].dropna().unique()
            alt_direction = alt["better"].dropna().unique()
            if len(ref_direction) != 1 or len(alt_direction) != 1:
                raise ValueError("bootstrap scenario has inconsistent metric directions")

            score_columns: dict[tuple[str, str], pd.Series] = {}
            for model in common_models:
                score_columns[("reference", model)] = ref[ref["model"] == model].set_index("perturbation")["score"]
                score_columns[("alternative", model)] = alt[alt["model"] == model].set_index("perturbation")["score"]
            matrix = pd.DataFrame(score_columns).dropna()
            effects = (
                pd.concat([ref[["perturbation", "effect_strength"]], alt[["perturbation", "effect_strength"]]])
                .drop_duplicates("perturbation")
                .set_index("perturbation")
                .reindex(matrix.index)["effect_strength"]
            )
            finite = np.isfinite(matrix.to_numpy(float)).all(axis=1) & np.isfinite(effects.to_numpy(float))
            matrix = matrix.loc[finite]
            effects = effects.loc[finite]
            if matrix.empty:
                continue

            rng = _seed(seed, f"{dataset}|{split}|{scenario['id']}")
            rank_changes = {model: [] for model in common_models}
            spearman_values: list[float] = []
            kendall_values: list[float] = []
            reversal_values: list[float] = []
            winner_changes: list[float] = []
            score_matrix = matrix.to_numpy(float)
            effect_values = effects.to_numpy(float)
            column_index = {column: index for index, column in enumerate(matrix.columns)}
            for _ in range(n_bootstrap):
                sampled = rng.integers(0, len(matrix), size=len(matrix))
                sampled_effects = effect_values[sampled]
                reference_scores = np.array(
                    [
                        _aggregate_scores(
                            score_matrix[sampled, column_index[("reference", model)]],
                            sampled_effects,
                            reference_aggregation,
                        )
                        for model in common_models
                    ]
                )
                alternative_scores = np.array(
                    [
                        _aggregate_scores(
                            score_matrix[sampled, column_index[("alternative", model)]],
                            sampled_effects,
                            scenario["aggregation"],
                        )
                        for model in common_models
                    ]
                )
                reference_ranks = pd.Series(reference_scores).rank(
                    method="min", ascending=ref_direction[0] == "lower"
                ).to_numpy(float)
                alternative_ranks = pd.Series(alternative_scores).rank(
                    method="min", ascending=alt_direction[0] == "lower"
                ).to_numpy(float)
                for index, model in enumerate(common_models):
                    rank_changes[model].append(alternative_ranks[index] - reference_ranks[index])
                spearman, kendall = _rank_correlations(reference_ranks, alternative_ranks)
                spearman_values.append(spearman)
                kendall_values.append(kendall)
                reversal_values.append(_pairwise_reversal_fraction(reference_ranks, alternative_ranks))
                winner_changes.append(float(set(np.where(reference_ranks == 1)[0]) != set(np.where(alternative_ranks == 1)[0])))

            for model, values in rank_changes.items():
                low, median, high = _quantiles(values)
                model_rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "scenario": scenario["id"],
                        "factor": scenario["factor"],
                        "level": scenario["level"],
                        "stratum": scenario["stratum"],
                        "model": model,
                        "n_perturbations": len(matrix),
                        "n_bootstrap": n_bootstrap,
                        "rank_change_median": median,
                        "rank_change_ci_low": low,
                        "rank_change_ci_high": high,
                        "probability_rank_changed": float(np.mean(np.asarray(values) != 0)),
                        "probability_rank_improved": float(np.mean(np.asarray(values) < 0)),
                        "probability_rank_worsened": float(np.mean(np.asarray(values) > 0)),
                    }
                )
            spearman_low, spearman_median, spearman_high = _quantiles(spearman_values)
            kendall_low, kendall_median, kendall_high = _quantiles(kendall_values)
            reversal_low, reversal_median, reversal_high = _quantiles(reversal_values)
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "scenario": scenario["id"],
                    "factor": scenario["factor"],
                    "level": scenario["level"],
                    "stratum": scenario["stratum"],
                    "n_models": len(common_models),
                    "n_perturbations": len(matrix),
                    "n_bootstrap": n_bootstrap,
                    "spearman_median": spearman_median,
                    "spearman_ci_low": spearman_low,
                    "spearman_ci_high": spearman_high,
                    "kendall_median": kendall_median,
                    "kendall_ci_low": kendall_low,
                    "kendall_ci_high": kendall_high,
                    "pairwise_reversal_fraction_median": reversal_median,
                    "pairwise_reversal_fraction_ci_low": reversal_low,
                    "pairwise_reversal_fraction_ci_high": reversal_high,
                    "winner_change_probability": float(np.mean(winner_changes)),
                }
            )
    return pd.DataFrame(model_rows), pd.DataFrame(comparison_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("bundle")
    evaluate.add_argument("--truth", type=Path, required=True)
    evaluate.add_argument("--prediction", type=Path, required=True)
    evaluate.add_argument("--split-manifest", type=Path, required=True)
    evaluate.add_argument("--grid", type=Path, required=True)
    evaluate.add_argument("--model-id", required=True)
    evaluate.add_argument("--capability", choices=["point", "population"], required=True)
    evaluate.add_argument("--fold", type=int, required=True)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("inputs", type=Path, nargs="+")
    aggregate.add_argument("--detail-output", type=Path, required=True)
    aggregate.add_argument("--summary-output", type=Path, required=True)
    aggregate.add_argument("--ranking-output", type=Path, required=True)
    aggregate.add_argument("--rank-change-output", type=Path)
    aggregate.add_argument("--comparison-output", type=Path)
    aggregate.add_argument("--bootstrap-rank-output", type=Path)
    aggregate.add_argument("--bootstrap-comparison-output", type=Path)
    aggregate.add_argument("--bootstrap-replicates", type=int, default=1000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    if args.command == "bundle":
        detail = evaluate_bundle(
            truth_path=args.truth,
            prediction_path=args.prediction,
            split_manifest_path=args.split_manifest,
            grid_path=args.grid,
            model_id=args.model_id,
            capability=args.capability,
            fold=args.fold,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.output, index=False)
        print(f"wrote {len(detail)} per-perturbation scores to {args.output}")
    else:
        detail = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
        summary = summarize_evaluations(detail)
        ranking = rank_models(summary)
        rank_changes, comparisons = compare_to_reference(summary)
        bootstrap_ranks, bootstrap_comparisons = paired_perturbation_bootstrap(
            detail,
            n_bootstrap=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        outputs = [args.detail_output, args.summary_output, args.ranking_output]
        optional_outputs = {
            args.rank_change_output: rank_changes,
            args.comparison_output: comparisons,
            args.bootstrap_rank_output: bootstrap_ranks,
            args.bootstrap_comparison_output: bootstrap_comparisons,
        }
        for path in [*outputs, *(path for path in optional_outputs if path is not None)]:
            path.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.detail_output, index=False)
        summary.to_csv(args.summary_output, index=False)
        ranking.to_csv(args.ranking_output, index=False)
        for path, frame in optional_outputs.items():
            if path is not None:
                frame.to_csv(path, index=False)
        print(f"wrote combined detail, summary, and rankings for {detail['model'].nunique()} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
