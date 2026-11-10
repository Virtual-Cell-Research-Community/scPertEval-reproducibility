"""Evaluate fixed prediction bundles over the frozen post-training sensitivity grid."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from workflows.predictions import expected_test_conditions


def _dense_mean(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel()


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
    prediction = ad.read_h5ad(prediction_path)
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
        for path in [args.detail_output, args.summary_output, args.ranking_output]:
            path.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.detail_output, index=False)
        summary.to_csv(args.summary_output, index=False)
        ranking.to_csv(args.ranking_output, index=False)
        print(f"wrote combined detail, summary, and rankings for {detail['model'].nunique()} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
