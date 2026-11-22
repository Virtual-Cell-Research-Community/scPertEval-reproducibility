from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scperteval as spe
from scperteval.blocks import spaces
from scperteval.protocols import metrics
from scperteval.types import Protocol

from analyses.ranking_sensitivity.atlas_runtime import (
    RETRIEVAL_HVG_K,
    RETRIEVAL_UNION_DEGS,
    build_protocols,
    load_eligible_families,
)
from analyses.ranking_sensitivity.cross_perturbation import (
    DIRECT,
    RETRIEVAL_OBSERVED_QUERY,
    RETRIEVAL_PREDICTED_QUERY,
    decompose,
)
from analyses.ranking_sensitivity.retrieval import retrieval_rank


def _centroids(seed: int = 0, n: int = 9, genes: int = 7) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, genes))
    P = G + rng.normal(scale=0.8, size=(n, genes))
    P[3] = P[4]  # exact ties among candidates exercise the shared tie-breaking noise
    G[6] = G[7]
    return list(G), list(P)


def _values(outcomes) -> np.ndarray:
    return np.array([outcome.value for outcome in outcomes])


def _brute_force(G, P, distance, *, transpose, excluded=None):
    """Literal definition: count candidates closer than the match (no tie noise needed here)."""
    G, P = np.vstack(G), np.vstack(P)
    n = len(G)

    def dist(x, y, keep):
        x, y = x[keep], y[keep]
        if distance == "squared_euclidean":
            return float(((x - y) ** 2).sum())
        if distance == "cityblock":
            return float(np.abs(x - y).sum())
        return 1 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    out = []
    for query in range(n):
        keep = np.ones(G.shape[1], dtype=bool)
        if excluded is not None:
            keep[excluded[query]] = False
        if transpose:
            d = [dist(P[query], G[j], keep) for j in range(n)]
        else:
            d = [dist(P[j], G[query], keep) for j in range(n)]
        out.append(sum(value < d[query] for value in d) / (n - 1))
    return np.array(out)


@pytest.mark.parametrize("transpose", [False, True])
def test_squared_euclidean_variant_matches_scperteval_rank_retrieval_exactly(transpose):
    for seed in range(5):
        G, P = _centroids(seed)
        expected = metrics.rank_retrieval(G, P, None, transpose=transpose)
        observed = _values(retrieval_rank(G, P, None, distance="squared_euclidean", transpose=transpose))
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("distance", ["squared_euclidean", "cityblock", "cosine"])
@pytest.mark.parametrize("transpose", [False, True])
def test_every_distance_matches_the_literal_rank_definition_without_ties(distance, transpose):
    rng = np.random.default_rng(11)
    G = list(rng.normal(size=(8, 5)))
    P = list(np.vstack(G) + rng.normal(scale=1.0, size=(8, 5)))
    observed = _values(retrieval_rank(G, P, None, distance=distance, transpose=transpose))
    np.testing.assert_allclose(observed, _brute_force(G, P, distance, transpose=transpose), atol=0)


def test_transpose_follows_rank_retrieval_orientation():
    # Prediction 0 collapses onto observed 1: as a query it retrieves the wrong observation
    # (transposed), but observation 0 still finds prediction 0 closest among all predictions.
    G = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([5.0, 5.0])]
    P = [np.array([0.9, 0.0]), np.array([3.0, 0.0]), np.array([5.0, 5.0])]
    assert _values(retrieval_rank(G, P, None, transpose=True))[0] == 0.5
    assert _values(retrieval_rank(G, P, None, transpose=False))[0] == 0.0


def test_zero_prediction_has_no_direction_and_ties_with_every_candidate():
    G = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    P = [np.zeros(2), np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    observed = retrieval_rank(G, P, None, distance="cosine", transpose=True)
    assert np.isfinite(_values(observed)).all()
    assert all(outcome.undefined_reason is None for outcome in observed)


@pytest.mark.parametrize("transpose", [False, True])
def test_query_target_exclusion_drops_only_the_query_target_and_keeps_unmeasured_candidates(transpose):
    rng = np.random.default_rng(3)
    names = ["g0", "g1", "g2", "g3", "g4", "g5"]
    perturbations = ["g0", "g2", "missing", "g5", "g1"]
    G = list(rng.normal(size=(5, 6)))
    P = list(np.vstack(G) + rng.normal(scale=0.7, size=(5, 6)))
    positions = {name: index for index, name in enumerate(names)}
    ctx = SimpleNamespace(
        perturbations=perturbations,
        ds=SimpleNamespace(
            var_names=names,
            target_gene_indices=lambda pert: np.array([positions[pert]] if pert in positions else [], dtype=int),
        ),
    )
    excluded = [np.array([positions[p]] if p in positions else [], dtype=int) for p in perturbations]
    outcomes = retrieval_rank(G, P, ctx, distance="cityblock", transpose=transpose, exclude_query_target=True)
    expected = _brute_force(G, P, "cityblock", transpose=transpose, excluded=excluded)
    assert outcomes[2].undefined_reason == "target_gene_not_measured"
    assert np.isnan(outcomes[2].value)
    defined = [index for index in range(5) if index != 2]
    np.testing.assert_allclose(_values(outcomes)[defined], expected[defined], atol=1e-12)


@pytest.mark.parametrize("transpose", [False, True])
def test_query_target_exclusion_on_a_panel_maps_targets_and_ignores_targets_off_the_panel(transpose):
    rng = np.random.default_rng(5)
    names = ["g0", "g1", "g2", "g3", "g4", "g5"]
    panel = np.array([1, 2, 4, 5])  # g0 and g3 are measured but outside the panel
    perturbations = ["g0", "g2", "missing", "g5", "g1"]
    G = list(rng.normal(size=(5, panel.size)))
    P = list(np.vstack(G) + rng.normal(scale=0.7, size=(5, panel.size)))
    positions = {name: index for index, name in enumerate(names)}
    ctx = SimpleNamespace(
        perturbations=perturbations,
        ds=SimpleNamespace(
            var_names=names,
            target_gene_indices=lambda pert: np.array([positions[pert]] if pert in positions else [], dtype=int),
        ),
    )
    gene_space = SimpleNamespace(per_pert=False, select=lambda ctx, pert: panel)
    on_panel = {int(gene): index for index, gene in enumerate(panel)}
    excluded = [
        np.array([on_panel[positions[p]]] if p in positions and positions[p] in on_panel else [], dtype=int)
        for p in perturbations
    ]
    outcomes = retrieval_rank(
        G, P, ctx, distance="cosine", transpose=transpose, exclude_query_target=True, gene_space=gene_space
    )
    expected = _brute_force(G, P, "cosine", transpose=transpose, excluded=excluded)
    assert outcomes[0].undefined_reason is None  # measured target off the panel: nothing to remove
    assert outcomes[2].undefined_reason == "target_gene_not_measured"
    defined = [index for index in range(5) if index != 2]
    np.testing.assert_allclose(_values(outcomes)[defined], expected[defined], atol=1e-12)


def test_cross_perturbation_decomposition_pairs_retrieval_with_the_same_direct_distance():
    assert decompose("pearson").strategy == DIRECT
    mse = decompose("transposed-retrieval-rank-MSE")
    assert (mse.strategy, mse.score_function, mse.distance, mse.transpose) == (
        RETRIEVAL_PREDICTED_QUERY,
        "squared-error",
        "squared_euclidean",
        True,
    )
    rmse = decompose("retrieval-rank-rmse")
    assert (rmse.strategy, rmse.score_function, rmse.transpose) == (RETRIEVAL_OBSERVED_QUERY, "squared-error", False)
    assert decompose("transposed-retrieval-rank-MAE").score_function == "MAE"
    assert decompose("retrieval-rank-cosine").distance == "cosine"


def test_universe_contains_exactly_thirteen_dataset_wide_retrieval_protocols():
    protocols = {protocol.name: protocol for protocol in build_protocols()}
    assert len(protocols) == 115
    retrieval = {name: protocol for name, protocol in protocols.items() if protocol.group == "atlas-retrieval"}
    assert len(retrieval) == 13
    assert all(protocol.scope == "dataset" and protocol.better == "lower" for protocol in retrieval.values())
    # Stacking every perturbation into one matrix requires one shared gene axis.
    assert all(not protocol.space.per_pert for protocol in retrieval.values())
    families = load_eligible_families().set_index("family_id")
    assert set(families.loc[list(retrieval), "cross_perturbation_group"]) == {"identity_matching"}
    assert set(families.drop(index=list(retrieval))["cross_perturbation_group"]) == {""}
    exclusion = sorted(name for name, p in retrieval.items() if p.metric.keywords["exclude_query_target"])
    assert exclusion == ["atlas-2715e9a4439c", "atlas-b94b1c70ad98", "atlas-d81fdc541750", "atlas-e1e926ba033b"]
    # Panel exclusion maps the query target onto the panel; the full-axis family keeps the original path.
    for name in exclusion:
        keywords = retrieval[name].metric.keywords
        if families.loc[name, "feature_set"] == "All":
            assert "gene_space" not in keywords
        else:
            assert keywords["gene_space"].cache_key == retrieval[name].space.cache_key


def test_union_panel_parameters_match_the_frozen_config():
    config = json.loads(Path("configs/evaluation/atlas-protocol-eligibility.json").read_text())
    panel = config["canonical_parameters"]["retrieval"]["union_panel"]
    assert RETRIEVAL_HVG_K == panel["hvg_k"]
    assert RETRIEVAL_UNION_DEGS.top_k == panel["per_unit_deg_top_k"]
    assert RETRIEVAL_UNION_DEGS.padj == panel["deg_adjusted_p_value_max"]
    assert RETRIEVAL_UNION_DEGS.rank_by == panel["deg_rank_by"]
    assert RETRIEVAL_UNION_DEGS.reference == panel["deg_reference"]


def _synthetic_fold(seed: int = 5) -> tuple[ad.AnnData, ad.AnnData]:
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(12)]
    units = ["G0", "G3", "G5", "G7", "G9"]
    parts, labels, pred_parts, pred_labels = (
        [rng.poisson(2, size=(40, 12)).astype(np.float32)],
        ["control"] * 40,
        [],
        [],
    )
    for index, unit in enumerate(units):
        values = rng.poisson(2, size=(30, 12)).astype(np.float32)
        values[:, (2 * index) % 12] += 6
        parts.append(values)
        labels += [unit] * 30
        noisy = values + rng.poisson(1, size=values.shape).astype(np.float32)
        pred_parts.append(noisy)
        pred_labels += [unit] * 30
    truth = ad.AnnData(np.vstack(parts))
    truth.var_names = genes
    truth.obs["perturbation"] = pd.Categorical(labels)
    prediction = ad.AnnData(np.vstack(pred_parts))
    prediction.var_names = genes
    prediction.obs["perturbation"] = pd.Categorical(pred_labels)
    return truth, prediction


def test_end_to_end_mse_protocol_equals_scperteval_transpose_rank():
    truth, prediction = _synthetic_fold()
    protocols = {protocol.name: protocol for protocol in build_protocols()}
    ours = protocols["atlas-fbb25436e8b0"]  # transposed MSE, control-centred, all genes
    reference = Protocol(
        "reference-transpose-rank",
        partial(metrics.rank_retrieval, transpose=True),
        representation="centroid",
        scope="dataset",
        space=spaces.full(),
        centering="control_mean",
        better="lower",
        perfect=0.0,
    )
    prepared = spe.prepare(truth, [ours, reference], min_cells=10, seed=42, workers=1, name="synthetic")
    a = spe.score(prepared, ours, prediction).per_perturbation.set_index("perturbation")["score"]
    b = spe.score(prepared, reference, prediction).per_perturbation.set_index("perturbation")["score"]
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_end_to_end_truth_self_retrieval_is_perfect_and_target_policies_resolve():
    truth, _ = _synthetic_fold()
    added = [p for p in build_protocols() if p.group == "atlas-retrieval"]
    prepared = spe.prepare(truth, added, min_cells=10, seed=42, workers=1, name="synthetic")
    availability = spe.truth_availability(prepared, added)
    assert len(availability) == len(added) * 5
    assert availability["ground_truth_defined"].all()
    assert (availability["self_score"] == 0).all()
