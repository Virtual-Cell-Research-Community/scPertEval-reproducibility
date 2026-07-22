"""Baselines that need no training: each predicts a perturbation's profile as a fixed function
of the dataset's own control/perturbed cells. Pure AnnData/numpy/sklearn -- no scPertEval import,
so this drops straight into a notebook cell.

There is no plain (non-fold-scoped) ``mean`` baseline here on purpose: a leave-one-out mean over
*every* perturbation the dataset has -- including ones in the current test fold -- leaks
test-fold information a real train/test protocol wouldn't have. :func:`fold_mean_baseline` is the
fold-scoped version that respects that split. :func:`no_change_baseline` is fold-scoped for the
same reason (consistency, not leakage -- its control-mean prediction doesn't depend on the
fold): it only predicts the fold's held-out test perturbations, not every perturbation.

Load the raw ``.h5ad`` once and pass the same ``AnnData`` to every baseline -- it can be several
GB, so re-reading it per baseline/per fold would multiply the I/O for no reason::

    adata = ad.read_h5ad("data/replogle22k562/raw.h5ad")
    no_change = no_change_baseline(adata, fold)
    mu_all = fold_mean_baseline(adata, fold)
    linear = linear_baseline(adata, fold)

Also runnable from the CLI, computing every baseline against every fold in one process (one
``raw.h5ad`` read, not one per baseline/per fold)::

    python baselines.py --dataset replogle22k562

Same ``--dataset <name>`` convention and per-fold output shape as every ``models/*/train_predict.py``:
reads ``data/<dataset>/{raw.h5ad,folds/}``, writes ``baselines/<name>/predictions/<dataset>/fold_<i>.h5ad``
for each baseline (``no_change``, ``mean``, ``linear``) -- one owner directory each, nested under
``baselines/`` rather than sibling to ``gears/``, ``scgpt/``, ... since these aren't trained models.
Downstream tooling that already knows how to concatenate a model's ``fold_<i>.h5ad`` files needs
no special case for baselines -- every fold-scoped prediction set is laid out identically.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent  # baselines/ -- owner dirs (no_change/, mean/, linear/) nest here
DATA_DIR = HERE.parent / "data"


def _pseudobulk(adata: ad.AnnData, pert: np.ndarray, groups: list[str]) -> np.ndarray:
    """``(genes, len(groups))`` mean-expression matrix, one column per perturbation label."""
    return np.vstack([np.asarray(adata.X[pert == g].mean(axis=0)).ravel() for g in groups]).T


def _fold_genes(adata: ad.AnnData, fold: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Perturbation labels, plus this fold's train/test gene names (the literal ``"ctrl"`` entry dropped)."""
    pert = adata.obs["perturbation"].to_numpy()
    train_genes = sorted({c.split("+")[0] for c in fold["train"] if c != "ctrl"})
    test_genes = sorted({c.split("+")[0] for c in fold["test"] if c != "ctrl"})
    return pert, train_genes, test_genes


def no_change_baseline(adata: ad.AnnData, fold: dict, control_label: str = "control") -> ad.AnnData:
    """One row per fold-test perturbation: the control-cell mean, i.e. no change.

    The prediction itself doesn't depend on the fold (it's always the dataset's control mean) --
    only *which* perturbations get a row does, so this lines up with :func:`fold_mean_baseline`
    and :func:`linear_baseline`'s output shape instead of predicting every perturbation at once.

    No cell-count floor: ``data/prepare_data.py`` already prepares every perturbation to have at
    least 30 cells, so re-filtering here would be a no-op.
    """
    pert, _, test_genes = _fold_genes(adata, fold)
    control_mean = np.asarray(adata.X[pert == control_label].mean(axis=0)).ravel()
    pred = ad.AnnData(X=np.tile(control_mean, (len(test_genes), 1)).astype(np.float32), obs={"perturbation": test_genes})
    pred.var_names = adata.var_names
    return pred


def fold_mean_baseline(adata: ad.AnnData, fold: dict) -> ad.AnnData:
    r"""Miller et al. 2025's mean baseline (mu_all): mean of this fold's per-*train*-perturbation means.

    Predicted identically for every held-out test perturbation in the fold -- never averages in
    cells from a perturbation the fold's own model never saw, matching a real train/test
    protocol's own information access. Needs the fold's split (``data/prepare_split.py``'s
    ``fold_<i>.pkl`` format: condition strings, ``ctrl`` folded into ``"train"``).
    """
    pert, train_genes, test_genes = _fold_genes(adata, fold)
    mu_all = _pseudobulk(adata, pert, train_genes).mean(axis=1)
    pred = ad.AnnData(X=np.tile(mu_all, (len(test_genes), 1)).astype(np.float32), obs={"perturbation": test_genes})
    pred.var_names = adata.var_names
    return pred


def linear_baseline(adata: ad.AnnData, fold: dict, pca_dim: int = 10, ridge: float = 0.1, seed: int = 42) -> ad.AnnData:
    """Ahlmann-Eltze et al.'s linear baseline: PCA embeddings + ridge bilinear regression.

    Ported from const-ae/linear_perturbation_prediction-Paper's
    ``benchmark/src/run_linear_pretrained_model.R`` (``solve_y_axb``). A perturbation's embedding
    is *that gene's own* PCA embedding (perturbation names equal gene names), so a held-out gene
    needs no separate fitting step -- only the shared low-rank map ``K`` between the gene and
    perturbation embedding spaces, fit once on the fold's training perturbations.
    """
    pert, train_genes, test_genes = _fold_genes(adata, fold)
    train_conditions = ["control", *train_genes]
    pseudobulk_train = _pseudobulk(adata, pert, train_conditions)  # (genes, 1 + n_train)
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


def _load_folds(folds_dir: Path) -> list[dict]:
    """Every ``fold_<i>.pkl`` split in ``folds_dir`` (data/prepare_split.py's output), fold-ordered."""
    paths = sorted(folds_dir.glob("fold_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise SystemExit(f"no fold_<i>.pkl files found in {folds_dir}")
    folds = []
    for p in paths:
        with p.open("rb") as f:
            folds.append(pickle.load(f))
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", default="smoke_k562", help="data/<dataset>/{raw.h5ad,folds/} to predict on (default: smoke_k562)"
    )
    parser.add_argument("--pca-dim", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    folds = _load_folds(DATA_DIR / args.dataset / "folds")
    adata = ad.read_h5ad(DATA_DIR / args.dataset / "raw.h5ad")  # loaded once, reused for every baseline/fold below

    baselines = {
        "no_change": lambda fold: no_change_baseline(adata, fold),
        "mean": lambda fold: fold_mean_baseline(adata, fold),
        "linear": lambda fold: linear_baseline(adata, fold, pca_dim=args.pca_dim, ridge=args.ridge, seed=args.seed),
    }
    for name, predict in baselines.items():
        out_dir = HERE / name / "predictions" / args.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fold in enumerate(folds):
            predict(fold).write_h5ad(out_dir / f"fold_{i}.h5ad")
        print(f"wrote {len(folds)} fold predictions to {out_dir}")


if __name__ == "__main__":
    main()
