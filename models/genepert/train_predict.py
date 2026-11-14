"""Train the GenePert ridge model and archive one centroid per held-out perturbation.

This is a small compatibility adapter around the method in ``zou-group/GenePert``. The
upstream implementation L2-normalizes one GenePT embedding per perturbed gene and learns a
multi-output ridge regression from those embeddings to perturbation-level mean expression.
It evaluates several ridge penalties against the test data in its demonstration notebook;
that selection would leak here, so this adapter freezes ``alpha=10`` from the published K562
example before seeing any of our fold outcomes. Because no fold data are used for model
selection, the complete non-test development pool (train plus validation perturbations) is
used for fitting.

The H5AD is scanned in row blocks and only development/control expression is read. This keeps
memory bounded and, importantly, never reads held-out response expression while fitting.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

if hasattr(ad, "settings"):
    ad.settings.allow_write_nullable_strings = True

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from workflows.model_runs import add_point_run_arguments, begin_point_model_run

DATA_DIR = HERE.parent / "data"
GENEPT_PICKLE = (
    HERE.parent
    / "sclambda"
    / "cache"
    / "GenePT_emebdding_v2"
    / "GenePT_gene_protein_embedding_model_3_text.pickle."
)
IMPLEMENTATION_REVISION = "zou-group/GenePert@31c0034211010696ba76dc62cb875af51bfe6f4a"
DEFAULT_ALPHA = 10.0


def l2_normalized_embeddings(
    embeddings: dict[str, np.ndarray],
    perturbations: list[str],
) -> np.ndarray:
    """Return upstream-compatible normalized GenePT features in requested row order."""
    missing = sorted(set(perturbations) - set(embeddings))
    if missing:
        raise ValueError(f"GenePT embeddings are missing perturbations: {missing[:20]}")
    matrix = np.vstack([np.asarray(embeddings[name], dtype=np.float32) for name in perturbations])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("GenePT embeddings contain non-finite or zero-length vectors")
    return matrix / norms


def grouped_expression_means(
    adata: ad.AnnData,
    conditions: list[str],
    *,
    condition_key: str = "condition",
    block_rows: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute selected condition means with one bounded sequential scan of ``X``.

    Conditions not requested here—including the fold's test perturbations—are never read
    from the expression matrix.
    """
    if condition_key not in adata.obs:
        raise ValueError(f"model view is missing obs[{condition_key!r}]")
    if len(set(conditions)) != len(conditions):
        raise ValueError("requested conditions are not unique")
    lookup = {name: index for index, name in enumerate(conditions)}
    labels = adata.obs[condition_key].astype(str).to_numpy()
    group_indices = np.fromiter((lookup.get(label, -1) for label in labels), dtype=np.int64)
    sums = np.zeros((len(conditions), adata.n_vars), dtype=np.float64)
    counts = np.zeros(len(conditions), dtype=np.int64)

    for start in range(0, adata.n_obs, block_rows):
        stop = min(start + block_rows, adata.n_obs)
        block_groups = group_indices[start:stop]
        selected = np.flatnonzero(block_groups >= 0)
        if not len(selected):
            continue
        block = adata.X[start:stop]
        block = block[selected]
        selected_groups = block_groups[selected]
        for group in np.unique(selected_groups):
            rows = selected_groups == group
            total = block[rows].sum(axis=0)
            sums[group] += np.asarray(total).ravel()
            counts[group] += int(rows.sum())

    missing = [condition for condition, count in zip(conditions, counts, strict=True) if count == 0]
    if missing:
        raise ValueError(f"model view contains no cells for conditions: {missing[:20]}")
    means = sums / counts[:, None]
    if not np.isfinite(means).all():
        raise ValueError("computed expression means contain non-finite values")
    return means.astype(np.float32), counts


def fit_genepert(
    embeddings: dict[str, np.ndarray],
    development_perturbations: list[str],
    development_means: np.ndarray,
    test_perturbations: list[str],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[Ridge, np.ndarray]:
    """Fit the frozen GenePert ridge and predict held-out mean log1p expression."""
    if development_means.shape[0] != len(development_perturbations):
        raise ValueError("development means do not align with development perturbations")
    train_x = l2_normalized_embeddings(embeddings, development_perturbations)
    test_x = l2_normalized_embeddings(embeddings, test_perturbations)
    model = Ridge(alpha=alpha)
    model.fit(train_x, development_means)
    predictions = np.asarray(model.predict(test_x), dtype=np.float32)
    if not np.isfinite(predictions).all():
        raise ValueError("GenePert produced non-finite predictions")
    return model, predictions


def _bare_perturbations(condition_strings: list[str]) -> list[str]:
    return [condition.removesuffix("+ctrl") for condition in condition_strings if condition != "ctrl"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="replogle-weissman-2022-k562-essential",
        help="models/data/<dataset> prepared model view",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--block-rows", type=int, default=4096)
    parser.add_argument("--embeddings", type=Path, default=GENEPT_PICKLE)
    add_point_run_arguments(parser, deterministic=True)
    args = parser.parse_args()
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive")

    run = begin_point_model_run(
        args,
        repository_root=REPOSITORY_ROOT,
        model_id="genepert",
        environment="models/pixi.lock#genepert",
        configuration={
            "deterministic": True,
            "method": "multi-output-ridge",
            "alpha": args.alpha,
            "embedding_normalization": "row-l2",
            "development_pool": "train-plus-validation",
            "target": "mean-normalized-log1p-expression",
            "block_rows": args.block_rows,
        },
        deterministic=True,
    )
    if run.skipped:
        print(f"skipping verified completed run: {run.handle.directory}")
        return 0
    if not args.embeddings.is_file():
        raise FileNotFoundError(
            f"{args.embeddings} missing; run `pixi run -e genepert fetch-embeddings` first"
        )

    fold_path = DATA_DIR / args.dataset / "folds" / f"fold_{args.fold}.pkl"
    with fold_path.open("rb") as stream:
        fold = pickle.load(stream)
    development = sorted(_bare_perturbations([*fold["train"], *fold["val"]]))
    test = sorted(_bare_perturbations(fold["test"]))

    adata = ad.read_h5ad(run.raw_path, backed="r")
    try:
        means, counts = grouped_expression_means(
            adata,
            [f"{gene}+ctrl" for gene in development],
            block_rows=args.block_rows,
        )
        genes = pd.Index(np.asarray(adata.var_names).astype(str))
    finally:
        adata.file.close()

    with args.embeddings.open("rb") as stream:
        embeddings = pickle.load(stream)
    model, predictions = fit_genepert(
        embeddings,
        development,
        means,
        test,
        alpha=args.alpha,
    )

    prediction = ad.AnnData(
        X=predictions,
        obs=pd.DataFrame({"perturbation": test}, index=[f"genepert::{gene}" for gene in test]),
    )
    prediction.var_names = genes
    run.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(run.prediction_path, compression="gzip")

    checkpoint = run.checkpoint_dir / "ridge.npz"
    np.savez_compressed(
        checkpoint,
        coef=np.asarray(model.coef_, dtype=np.float32),
        intercept=np.asarray(model.intercept_, dtype=np.float32),
        development_perturbations=np.asarray(development),
        test_perturbations=np.asarray(test),
        alpha=np.asarray(args.alpha),
    )
    run.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "alpha": args.alpha,
                "n_development_perturbations": len(development),
                "n_test_perturbations": len(test),
                "n_genes": prediction.n_vars,
                "minimum_development_cells": int(counts.min()),
                "median_development_cells": float(np.median(counts)),
                "maximum_development_cells": int(counts.max()),
            }
        ]
    ).to_csv(run.metrics_path, index=False)
    run.complete(
        implementation_revision=IMPLEMENTATION_REVISION,
        checkpoints=[checkpoint],
    )
    print(f"wrote {prediction.shape} GenePert predictions to {run.prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
