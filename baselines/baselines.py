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
for each baseline (``no_change``, ``mean``, ``linear``, ``interpolated``) -- one owner directory each,
nested under ``baselines/`` rather than sibling to ``gears/``, ``scgpt/``, ... since these aren't
trained models. Downstream tooling that already knows how to concatenate a model's
``fold_<i>.h5ad`` files needs no special case for baselines -- every fold-scoped prediction set is
laid out identically.

:func:`interpolated_baseline` is laid out the same way and is a genuine prediction like
``no_change``/``mean``/``linear`` -- scoreable against ground truth via ``sp.score()`` the same
way. But since it has privileged access to a real technical duplicate of the very perturbation
it predicts, scoring it reads more like an oracle upper bound (Miller et al. 2025's *positive
control*) than a competing model on equal footing with the other three. It doubles as a
fold-restricted ``positive=`` calibration source too: ``models/compare.py`` can register it (via
``SOURCES.register()``) in place of scPertEval's own dataset-wide ``interpolated``, which pools
its DE-weighting background from every perturbation regardless of fold. Its technical-duplicate
half must be computed by the *exact* same seeded split scPertEval's own
``scperteval.dataset.Dataset._index()`` uses (see :func:`_tech_dup_halves`) -- an independently-drawn
half could overlap scPertEval's own ``gt_half``, leaking ground-truth cells into whichever of
these two roles it's playing.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy import stats
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


def _tech_dup_halves(
    adata: ad.AnnData, control_label: str = "control", min_cells: int = 30, seed: int = 42
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every perturbation's (first-half, second-half) cell indices.

    Deliberately duplicates ``scperteval.dataset.Dataset._index()``'s split -- same seed, same
    ``min_cells`` floor, same one-shared-RNG-consumed-sequentially-over-sorted-unique-labels
    order -- rather than drawing an independent random half here. ``raw.h5ad`` is the exact
    same file (verified by hash) scPertEval's own ``Dataset`` indexes when
    ``models/compare.py`` calls ``sp.prepare()`` with its unmodified ``seed=42``/``min_cells=30``
    defaults, so this reproduces its ``halves`` dict bit-for-bit. That match matters:
    :func:`interpolated_baseline` uses the *second* half as its technical duplicate, and
    scPertEval's own ``gt_half`` control (the ground truth this whole benchmark scores against)
    is always the *first* half of the same split. Get this split wrong -- a different seed, a
    different perturbation-iteration order, a different ``min_cells`` -- and the two halves are
    no longer guaranteed disjoint, silently leaking ground-truth cells into a positive control.

    This file otherwise imports nothing from scperteval on purpose (see module docstring), so
    nothing enforces this staying in sync automatically -- if
    ``scperteval.dataset.Dataset._index()``'s split algorithm ever changes, this must change
    with it.
    """
    pert = adata.obs["perturbation"].to_numpy()
    rng = np.random.default_rng(seed)
    halves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in np.unique(pert[pert != control_label]):
        idx = np.where(pert == p)[0]
        if len(idx) < min_cells:
            continue
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        h = len(shuffled) // 2
        halves[p] = (np.sort(shuffled[:h]), np.sort(shuffled[h:]))
    return halves


def _moments(X) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-gene mean, sample variance (Bessel-corrected), and cell count.

    Duplicates ``scperteval.blocks.de._moments`` exactly (sparse- and dense-aware, same
    formula) -- keep in sync if that changes.
    """
    n = X.shape[0]
    if sp.issparse(X):
        m = np.asarray(X.mean(0)).ravel()
        msq = np.asarray(X.multiply(X).mean(0)).ravel()
    else:
        X = np.asarray(X)
        m = X.mean(0)
        msq = (X * X).mean(0)
    v = (msq - m * m) * (n / max(n - 1, 1))
    return m, np.maximum(v, 0.0), n


def _bh(pvalue: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values.

    Duplicates ``scperteval.blocks.de.bh`` exactly -- keep in sync if that changes.
    """
    p = np.asarray(pvalue, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    adj = p[order] * idx.size / np.arange(1, idx.size + 1)
    out[order] = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0.0, 1.0)
    return out


def _welch_pvalue_adj(mt, vt, nt, mr, vr, nr) -> np.ndarray:
    """BH-adjusted Welch's t-test p-values from per-gene moments.

    Duplicates ``scperteval.blocks.de.ttest_from_moments``'s formula exactly (only
    ``pvalue_adj`` is needed here, not the statistic itself) -- keep in sync if that changes.
    """
    se2 = vt / nt + vr / nr
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mt - mr) / np.sqrt(se2)
        df = se2**2 / ((vt / nt) ** 2 / max(nt - 1, 1) + (vr / nr) ** 2 / max(nr - 1, 1))
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    df = np.where(np.isfinite(df) & (df > 0), df, 1.0)
    pval = np.nan_to_num(2.0 * stats.t.sf(np.abs(t), df), nan=1.0)
    return _bh(pval)


def interpolated_baseline(
    adata: ad.AnnData, fold: dict, halves: dict[str, tuple[np.ndarray, np.ndarray]], control_label: str = "control"
) -> ad.AnnData:
    r"""Miller et al. 2025's interpolated-duplicate positive control, fold-restricted.

    Per test gene, a per-gene blend of that gene's own held-out half (the technical duplicate)
    and this fold's train-mean baseline (:func:`fold_mean_baseline`'s ``mu_all``), weighted by
    how significantly the duplicate differs from this fold's train-only background: genes where
    it doesn't differ (not perturbation-responsive) fall back to the mean baseline; genes where
    it does are trusted as real signal. Like ``no_change``/``mean``/``linear`` it's a genuine
    prediction, scoreable via ``sp.score()`` the same way -- but its privileged access to a real
    technical duplicate of the very perturbation it predicts makes it read more like an oracle
    upper bound than a baseline on equal footing with the other three. ``models/compare.py`` can
    additionally register it (via ``SOURCES.register()``) as a fold-restricted ``positive=``
    calibration source, in place of scPertEval's own dataset-wide ``interpolated`` (see the module
    docstring's note on this function).

    The background for the significance test is *every* cell from this fold's train
    perturbations, pooled -- not a subsample. scPertEval's own (dataset-wide) equivalent caps
    its DE background at ``subsample=8192`` cells, but that cap exists for the O(n^2)
    population-representation protocols' tractability, not because a t-test's moments need it
    (they're an O(n) reduction regardless of n). This is a deliberate deviation from the
    default, not an oversight, and it introduces no leakage risk the way an incorrect
    :func:`_tech_dup_halves` split would -- this background is never itself compared against
    ground truth, only used to weight the blend. Nor does it need explicit leave-one-out
    exclusion: the perturbation being scored is always in this fold's *test* set, never its
    train set, so it can never appear in its own background.

    ``halves`` must come from :func:`_tech_dup_halves` called on the same ``adata`` -- see that
    function's docstring for why an independently-drawn half would be incorrect here.
    """
    pert, train_genes, test_genes = _fold_genes(adata, fold)
    mu_all = _pseudobulk(adata, pert, train_genes).mean(axis=1)  # fold-restricted blend target

    mr, vr, nr = _moments(adata.X[np.isin(pert, train_genes)])

    rows = []
    for g in test_genes:
        if g not in halves:
            raise ValueError(
                f"{g!r} has no technical-duplicate split (fewer than min_cells cells?) -- "
                f"_tech_dup_halves and scperteval's own Dataset._index() must agree on every "
                f"perturbation this fold expects a row for"
            )
        second_half = halves[g][1]
        tech = np.asarray(adata.X[second_half].mean(axis=0)).ravel()
        mt, vt, nt = _moments(adata.X[second_half])
        alpha = np.nan_to_num(1.0 - _welch_pvalue_adj(mt, vt, nt, mr, vr, nr), nan=0.0)
        rows.append(alpha * tech + (1.0 - alpha) * mu_all)

    pred = ad.AnnData(X=np.vstack(rows).astype(np.float32), obs={"perturbation": test_genes})
    pred.var_names = adata.var_names
    return pred


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
    halves = _tech_dup_halves(adata)  # ditto -- one global split, reused per fold (see its docstring)

    baselines = {
        "no_change": lambda fold: no_change_baseline(adata, fold),
        "mean": lambda fold: fold_mean_baseline(adata, fold),
        "linear": lambda fold: linear_baseline(adata, fold, pca_dim=args.pca_dim, ridge=args.ridge, seed=args.seed),
        "interpolated": lambda fold: interpolated_baseline(adata, fold, halves),
    }
    for name, predict in baselines.items():
        out_dir = HERE / name / "predictions" / args.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fold in enumerate(folds):
            predict(fold).write_h5ad(out_dir / f"fold_{i}.h5ad")
        print(f"wrote {len(folds)} fold predictions to {out_dir}")


if __name__ == "__main__":
    main()
