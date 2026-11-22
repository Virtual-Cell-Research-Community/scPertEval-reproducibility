"""Dataset-scope retrieval-rank metric and the dataset-wide retrieval feature panel.

scPertEval's :func:`~scperteval.protocols.metrics.rank_retrieval` orders candidates by squared
Euclidean distance only. The Atlas retrieval families additionally use L1 (MAE) and cosine
distances, and some families remove the query perturbation's own target gene before computing
distances. :func:`retrieval_rank` reproduces ``rank_retrieval`` exactly — distance-matrix
orientation, tie-breaking noise, normalization — and only generalizes the distance and the
query-specific gene mask, so the scPertEval pin does not change.

Retrieval is dataset-scope: every perturbation of the fold is stacked into one candidate matrix, so
all views must share one gene axis. Per-perturbation spaces such as ``spaces.own_target()`` would
silently misalign rows and are therefore never used here. Target inclusion uses the dataset-wide
``spaces.perturbed_genes()``; target exclusion is applied per query inside the metric.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scperteval.blocks import spaces
from scperteval.blocks.spaces import SubsetSpace
from scperteval.caching import _once
from scperteval.protocols.metrics import _sq_dists
from scperteval.types import MetricOutcome

DISTANCES = ("squared_euclidean", "cityblock", "cosine")


def _cosine_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """``1 - cos`` with zero-norm vectors treated as carrying no direction (similarity 0)."""
    x_norm = np.linalg.norm(X, axis=1)
    y_norm = np.linalg.norm(Y, axis=1)
    denominator = np.outer(x_norm, y_norm)
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.where(denominator > 0, (X @ Y.T) / denominator, 0.0)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def pairwise_distances(X: np.ndarray, Y: np.ndarray, distance: str) -> np.ndarray:
    """``(len(X), len(Y))`` distances; squared Euclidean is bit-identical to ``rank_retrieval``."""
    if distance == "squared_euclidean":
        return _sq_dists(X, Y)
    if distance == "cityblock":
        return cdist(X, Y, metric="cityblock")
    if distance == "cosine":
        return _cosine_distances(X, Y)
    raise ValueError(f"unknown retrieval distance {distance!r}; expected one of {DISTANCES}")


def _query_masked_distances(
    P: np.ndarray,
    G: np.ndarray,
    excluded: Sequence[np.ndarray],
    distance: str,
    *,
    transpose: bool,
) -> np.ndarray:
    """Distance matrix ``D[i, j] = d(P_i, G_j)`` where each query drops its own excluded genes.

    With ``transpose`` each prediction ``P_i`` is the query (row ``i`` uses query ``i``'s mask);
    otherwise each observed profile ``G_j`` is the query (column ``j`` uses query ``j``'s mask).
    """
    n, n_genes = G.shape
    D = np.empty((n, n), dtype=np.float64)
    for query, genes in enumerate(excluded):
        keep = np.ones(n_genes, dtype=bool)
        keep[genes] = False
        if transpose:
            D[query, :] = pairwise_distances(P[query : query + 1, keep], G[:, keep], distance)[0]
        else:
            D[:, query] = pairwise_distances(P[:, keep], G[query : query + 1, keep], distance)[:, 0]
    return D


def retrieval_rank(
    gt,
    prediction,
    ctx,
    *,
    distance: str = "squared_euclidean",
    transpose: bool = False,
    exclude_query_target: bool = False,
    gene_space: SubsetSpace | None = None,
) -> list[MetricOutcome]:
    """Cross-perturbation retrieval rank with a configurable distance; lower is better.

    ``D[i, j] = d(P_i, G_j)`` between predicted and observed centroids. By default each observed
    perturbation is the query and its own prediction is ranked among all predictions (column-wise);
    ``transpose=True`` makes each prediction the query and ranks its matching observed profile among
    all observed profiles. Exact ties are broken by the same seeded ``[0, 1e-12)`` noise as
    ``rank_retrieval`` and ranks are normalized by ``n - 1``, so 0 is a perfect top-1 retrieval.

    ``exclude_query_target`` removes the query perturbation's own target gene(s) from every distance
    that query computes (the Virtual Cell Challenge PDS convention). A query whose target is not
    measured is undefined (``target_gene_not_measured``) but remains a candidate for other queries.
    Centroids lie on the full gene axis unless ``gene_space`` names the dataset-wide panel they were
    sliced to; a measured target outside that panel has nothing to remove, so its query stays defined.
    With cosine distance, an observed profile with zero norm is undefined (``constant_truth``).
    """
    G = np.vstack(gt).astype(np.float64, copy=False)
    P = np.vstack(prediction).astype(np.float64, copy=False)
    if G.shape != P.shape:
        raise ValueError(f"ground truth {G.shape} and prediction {P.shape} centroids differ in shape")
    if not (np.isfinite(G).all() and np.isfinite(P).all()):
        raise ValueError("retrieval rank received non-finite centroids")
    n = G.shape[0]
    undefined: list[str | None] = [None] * n

    if exclude_query_target:
        perturbations = list(ctx.perturbations)
        if len(perturbations) != n:
            raise ValueError("query-target exclusion needs one centroid per prepared perturbation")
        n_full = len(ctx.ds.var_names)
        if gene_space is None:
            panel = np.arange(n_full)
        else:
            if gene_space.per_pert:
                raise ValueError("query-target exclusion needs a dataset-wide gene panel")
            panel = np.arange(n_full)[gene_space.select(ctx, None)]
        if G.shape[1] != panel.size:
            raise ValueError(f"centroids have {G.shape[1]} genes but the declared gene panel has {panel.size}")
        position = np.full(n_full, -1, dtype=int)
        position[panel] = np.arange(panel.size)
        excluded = []
        for index, perturbation in enumerate(perturbations):
            genes = np.asarray(ctx.ds.target_gene_indices(perturbation), dtype=int)
            if genes.size == 0:
                undefined[index] = "target_gene_not_measured"
            on_panel = position[genes]
            excluded.append(on_panel[on_panel >= 0])
        D = _query_masked_distances(P, G, excluded, distance, transpose=transpose)
    else:
        D = pairwise_distances(P, G, distance)

    if distance == "cosine":
        for index in np.flatnonzero(np.linalg.norm(G, axis=1) == 0):
            undefined[index] = undefined[index] or "constant_truth"

    # Identical to scperteval.protocols.metrics.rank_retrieval from here on.
    if transpose:
        D = D.T
    noise = np.random.default_rng(42).uniform(0, 1e-12, size=D.shape)
    ranks = np.argsort(np.argsort(D + noise, axis=0), axis=0)
    values = np.diag(ranks).astype(np.float64) / max(n - 1, 1)
    return [
        MetricOutcome(float(value)) if reason is None else MetricOutcome(float("nan"), reason)
        for value, reason in zip(values, undefined, strict=True)
    ]


@dataclass(frozen=True)
class PerUnitDEGUnion(SubsetSpace):
    """Dataset-wide union of every evaluation unit's top truth-derived DEGs.

    PerturBench-style retrieval panels add each condition's top DE genes to a shared HVG panel.
    The union is computed once per prepared fold from ground-truth DE against controls, so every
    perturbation and every model is scored on the same gene axis.
    """

    padj: float = 0.05
    top_k: int = 25
    rank_by: str = "absolute_effect"
    reference: str = "control"

    @property
    def per_pert(self) -> bool:
        return False

    @property
    def cache_key(self) -> Hashable:
        return ("atlas-per-unit-deg-union", self.padj, self.top_k, self.rank_by, self.reference)

    def _select_raw(self, ctx, pert: str | None) -> np.ndarray:
        selector = spaces.degs(self.padj, top_k=self.top_k, rank_by=self.rank_by, reference=self.reference)

        def call() -> np.ndarray:
            genes = np.arange(len(ctx.ds.var_names))
            selected = [genes[selector._select_raw(ctx, unit)] for unit in ctx.perturbations]
            return np.unique(np.concatenate([np.asarray(item, dtype=int) for item in selected]))

        key = ("select", (self.cache_key, ctx.cfg.truth, ctx.cfg.de_method))
        return _once(ctx._store, key, call)

    def precompute(self, ctx) -> None:
        self._select_raw(ctx, None)
