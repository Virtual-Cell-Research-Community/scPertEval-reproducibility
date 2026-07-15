"""Baselines that need no training: each predicts a perturbation's profile as a fixed function
of the dataset's own control/perturbed cells.

Reuses scPertEval's own ``Dataset`` internals rather than reimplementing them, so ``no_change``
matches the framework's own ``control`` calibration source (src/scperteval/sources.py). There is
no plain (non-fold-scoped) ``mean`` baseline here on purpose: a leave-one-out mean over *every*
perturbation the dataset has — including ones in the current test fold — leaks test-fold
information into the baseline no real train/test protocol would have access to (a trained model
only ever sees its own fold's training perturbations). The fold-scoped mean that respects that
split is :func:`fold_mean_baseline` below, wired up in ``models/compare.py`` via
``FOLD_SCOPED_BASELINES`` — not something this module's own CLI can drive directly, since it
needs a fold's train/test split, not just a dataset path.

Add a new baseline by writing a function ``(ds: Dataset) -> np.ndarray`` and registering it in
``BASELINES`` below; give it its own ``train_predict.py`` (alongside ``models/gears``,
``models/scgpt``) instead if it actually needs a fitting step.

Usage::

    python models/baselines/baselines.py no_change data/replogle22k562_processed_complete.h5ad predictions.h5ad

Point ``dataset`` at the full processed dataset, not a held-out-only ground truth: scPertEval's
``score`` only looks up the perturbations it needs, so a baseline file covering every
perturbation still scores correctly against a smaller, held-out-only ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scperteval.dataset import Dataset
from scperteval.types import RunConfig
from sklearn.decomposition import PCA


def no_change_baseline(ds: Dataset) -> np.ndarray:
    """One row per perturbation: the control-cell mean (predicts no change at all)."""
    return np.vstack([ds.control_mean() for _ in ds.perturbations])


BASELINES = {"no_change": no_change_baseline}


def fold_mean_baseline(raw_path: str, fold: dict, min_cells: int = 30, seed: int = 42) -> ad.AnnData:
    r"""Miller et al. 2025's mean baseline (mu_all).

    The mean of per-perturbation means over *only this fold's training perturbations*,
    predicted identically for every held-out test perturbation in the fold — this never
    averages in cells from a perturbation the fold's own model never saw, matching the paper's
    "average of averages of all *train* perturbations" definition exactly, and matching what a
    real train/test protocol actually has access to (unlike a leave-one-out mean over every
    perturbation the dataset has, which would leak test-fold information — see the module
    docstring). Needs the fold's split (``models/data/prepare_split.py``'s ``fold_<i>.pkl``
    format: condition strings, ``ctrl`` folded into ``"train"``).

    Parameters
    ----------
    raw_path : str
        Path to the raw (unsplit) ``.h5ad`` all folds are drawn from.
    fold : dict
        One fold's ``{"train": [...], "val": [...], "test": [...]}`` split (condition strings).

    Returns
    -------
    anndata.AnnData
        One row per fold-test perturbation, every row identical (mu_all).
    """
    adata = ad.read_h5ad(raw_path)
    train_genes = {c.split("+")[0] for c in fold["train"] if c != "ctrl"}
    test_genes = sorted({c.split("+")[0] for c in fold["test"] if c != "ctrl"})
    pert = adata.obs["perturbation"].to_numpy()
    train_mask = np.isin(pert, list(train_genes)) | (pert == "control")
    cfg = RunConfig(dataset=raw_path, protocols=[], min_cells=min_cells, seed=seed)
    train_ds = Dataset(adata[train_mask].copy(), cfg)
    mu_all = train_ds.allpert_mean()
    pred = ad.AnnData(X=np.tile(mu_all, (len(test_genes), 1)).astype(np.float32), obs={"perturbation": test_genes})
    pred.var_names = train_ds.var_names
    return pred


def _pseudobulk(adata: ad.AnnData, groups: list[str]) -> np.ndarray:
    """``(genes, len(groups))`` mean-expression matrix, one column per perturbation label."""
    pert = adata.obs["perturbation"].to_numpy()
    cols = [np.asarray(adata.X[np.where(pert == g)[0]].mean(axis=0)).ravel() for g in groups]
    return np.vstack(cols).T


def linear_baseline(
    raw_path: str, fold: dict, pca_dim: int = 10, ridge: float = 0.1, min_cells: int = 30, seed: int = 42
) -> ad.AnnData:
    r"""Ahlmann-Eltze et al.'s linear baseline: PCA embeddings + ridge bilinear regression.

    Ported from const-ae/linear_perturbation_prediction-Paper's
    ``benchmark/src/run_linear_pretrained_model.R`` (``solve_y_axb``), with its script
    defaults (``pca_dim=10``, one ``ridge`` penalty shared by both embeddings).

    For single-gene perturbations, a perturbation's embedding is *that gene's own* PCA
    embedding (perturbation names equal gene names) — so a held-out gene's embedding needs no
    separate fitting step, only the shared low-rank map ``K`` between the two embedding
    spaces, fit once on the fold's training perturbations:

    1. ``gene_emb = PCA(raw train pseudobulk, k)`` -> ``(genes, k)``; doubles as the
       perturbation embedding, transposed (``control``'s embedding is the zero vector by
       convention, since it is not itself a measured gene).
    2. ``K`` minimizes ``||Y_train - (gene_emb @ K @ pert_emb_train + center)||^2`` via double
       ridge regression, where ``Y_train`` is each train perturbation's pseudobulk minus the
       control mean, and ``center`` is ``Y_train``'s row (per-gene) mean.
    3. Predict a held-out gene ``g`` via ``gene_emb @ K @ pert_emb[:, g] + center + control_mean``.

    Parameters
    ----------
    raw_path : str
        Path to the raw (unsplit) ``.h5ad`` all folds are drawn from.
    fold : dict
        One fold's ``{"train": [...], "val": [...], "test": [...]}`` split (condition strings).
    pca_dim : int
        Embedding dimension for both the gene and perturbation PCA embeddings (script default
        10; capped automatically to the number of training conditions available — relevant
        only at smoke scale, where a fold's training set is far smaller than a real dataset's).
    ridge : float
        Ridge penalty shared by both embeddings' regularization terms (script default 0.1).

    Returns
    -------
    anndata.AnnData
        One row per fold-test perturbation.
    """
    adata = ad.read_h5ad(raw_path)
    train_genes = sorted({c.split("+")[0] for c in fold["train"] if c != "ctrl"})
    test_genes = sorted({c.split("+")[0] for c in fold["test"] if c != "ctrl"})

    train_conditions = ["control", *train_genes]
    pseudobulk_train = _pseudobulk(adata, train_conditions)  # (genes, 1 + n_train)
    control_mean = pseudobulk_train[:, 0]

    k = min(pca_dim, pseudobulk_train.shape[1] - 1)
    gene_emb = PCA(n_components=k, random_state=seed).fit_transform(pseudobulk_train)  # (genes, k)
    gene_row = {g: i for i, g in enumerate(adata.var_names)}

    def pert_embedding(gene: str) -> np.ndarray:
        return np.zeros(k) if gene == "control" else gene_emb[gene_row[gene]]

    b_train = np.vstack([pert_embedding(g) for g in train_conditions]).T  # (k, 1 + n_train)
    y_train = pseudobulk_train - control_mean[:, None]  # delta from control, same columns as b_train

    center = y_train.mean(axis=1)
    y_centered = y_train - center[:, None]
    reg_a = np.linalg.inv(gene_emb.T @ gene_emb + ridge * np.eye(k))
    reg_b = np.linalg.inv(b_train @ b_train.T + ridge * np.eye(k))
    coef = reg_a @ gene_emb.T @ y_centered @ b_train.T @ reg_b  # (k, k)

    b_test = np.vstack([pert_embedding(g) for g in test_genes]).T  # (k, n_test)
    pred = gene_emb @ coef @ b_test + center[:, None] + control_mean[:, None]  # (genes, n_test)

    out = ad.AnnData(X=pred.T.astype(np.float32), obs={"perturbation": test_genes})
    out.var_names = adata.var_names
    return out


FOLD_SCOPED_BASELINES = {"mean": fold_mean_baseline, "linear": linear_baseline}


def build_predictions(baseline: str, dataset_path: str, min_cells: int = 30, seed: int = 42) -> ad.AnnData:
    """One predicted row per perturbation, from the named baseline function."""
    cfg = RunConfig(dataset=dataset_path, protocols=[], min_cells=min_cells, seed=seed)
    ds = Dataset.load(dataset_path, cfg)
    rows = BASELINES[baseline](ds)
    pred = ad.AnnData(X=rows.astype(np.float32), obs={"perturbation": ds.perturbations})
    pred.var_names = ds.var_names
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", choices=sorted(BASELINES), help="which baseline to compute")
    parser.add_argument(
        "dataset", help="path to a preprocessed .h5ad (the full dataset, not a scoped ground truth — see above)"
    )
    parser.add_argument("out", help="path to write the predictions .h5ad")
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pred = build_predictions(args.baseline, args.dataset, min_cells=args.min_cells, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(args.out)
    print(f"wrote {pred.shape} {args.baseline!r} predictions to {args.out}")


if __name__ == "__main__":
    main()
