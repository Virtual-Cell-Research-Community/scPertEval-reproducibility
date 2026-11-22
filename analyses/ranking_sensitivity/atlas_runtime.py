"""Resolve the frozen Atlas semantic families to concrete scPertEval protocols."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
from scperteval.blocks import spaces
from scperteval.blocks.spaces import FeatureSpace, SubsetSpace
from scperteval.protocols import metrics
from scperteval.types import Protocol

DEFAULT_ELIGIBILITY = Path("configs/evaluation/atlas-protocol-eligibility-v1.csv")

PROFILE_METRICS = {
    "MAE": metrics.mae,
    "clipped-r2": metrics.clipped_r2,
    "cosine": metrics.cosine,
    "explained-variance": metrics.explained_variance,
    "pearson": metrics.pearson,
    "r2": metrics.r2,
    "spearman": metrics.spearman,
    "squared-error": metrics.mse,
    "squared-pearson": metrics.squared_pearson,
}
POPULATION_METRICS = {
    "energy-distance": metrics.energy_distance,
    "mmd-rbf": metrics.unbiased_mmd_median,
    "sinkhorn-divergence": metrics.sinkhorn_w2,
}
TARGET_POLICIES = {
    "no_target_operation",
    "force_include_target",
    "exclude_own_target",
}

DE_METRICS = {
    "average-precision": metrics.de_auprc,
    "auroc": metrics.de_auroc,
    "class-precision": metrics.de_precision,
    "deg-identification-auprc": metrics.de_auprc,
    "deg-set-intersection": metrics.de_recall,
    "direction-concordance": metrics.de_direction_concordance,
    "f1": metrics.de_f1,
    "jaccard": metrics.de_jaccard,
    "macro-f1": metrics.de_macro_f1,
    "set-overlap-at-truth-size": metrics.de_overlap_at_truth_size,
    "spearman": metrics.de_spearman,
}


def load_eligible_families(path: Path = DEFAULT_ELIGIBILITY) -> pd.DataFrame:
    """Load and minimally validate the frozen executable family table."""
    frame = pd.read_csv(path, keep_default_na=False)
    if len(frame) != 101 or frame["family_id"].nunique() != 101:
        raise ValueError(f"expected 101 unique eligible Atlas families, found {len(frame)} rows")
    if set(frame["scope_decision"]) != {"included"}:
        raise ValueError("eligibility table contains non-included families")
    return frame


def _canonical_k(row: pd.Series) -> int | None:
    value = str(row["canonical_feature_k"])
    return int(value) if value.isdigit() else None


def _target_operation(space: SubsetSpace, operation: str) -> SubsetSpace:
    """Apply one frozen target-gene operation to a subset space."""
    if operation not in TARGET_POLICIES:
        raise ValueError(f"unsupported target operation {operation!r}")
    if operation == "force_include_target":
        return space | spaces.own_target()
    if operation == "exclude_own_target":
        return space - spaces.own_target()
    return space


def gene_space(row: pd.Series) -> SubsetSpace:
    """Resolve one row's truth-derived gene universe and target policy."""
    feature_set = str(row["feature_set"]).lower()
    if feature_set not in {"all", "deg", "hvg", "heg"}:
        raise ValueError(f"{row['family_id']}: unsupported feature set {row['feature_set']!r}")
    rank = str(row["feature_ranking_basis"])
    rank_by = {"significance": "significance", "effect": "absolute_effect", "": "absolute_effect"}[rank]
    top_k = _canonical_k(row)
    if feature_set == "all":
        base = spaces.full()
    elif feature_set == "deg":
        base = spaces.degs(padj=0.05, top_k=top_k, rank_by=rank_by, reference="control")
    elif feature_set == "hvg":
        base = spaces.hvg(top_k or 2000)
    else:
        base = spaces.heg(top_k or 1000)
    return _target_operation(base, str(row["target_operation"]))


def _space(row: pd.Series, subset: SubsetSpace) -> FeatureSpace:
    if str(row["space"]) == "GEX":
        return subset
    if str(row["space"]) != "PCA":
        raise ValueError(f"{row['family_id']}: unsupported space {row['space']!r}")
    if subset.cache_key == spaces.full().cache_key:
        return spaces.pca(50)
    return spaces.pca(50).on(subset)


def _de_metric(row: pd.Series):
    """Translate the former DEComparison semantics to one pure DE metric.

    A DEG feature panel defined the positive/relevant genes inside the complete classification
    universe in the old API.  Keep that universe complete here and bind the selector knobs to the
    metric; applying ``spaces.degs`` first would remove all negatives and make AUROC undefined.
    """
    family_id = str(row["family_id"])
    family = str(row["metric_family"])
    top_k = _canonical_k(row)
    rank = str(row["feature_ranking_basis"])
    rank_by = {"significance": "significance", "effect": "absolute_effect", "": "absolute_effect"}[rank]
    if family in {"set-overlap-at-n", "topk-overlap"}:
        return partial(metrics.de_overlap, k=top_k or 50, rank_by=rank_by)
    metric = DE_METRICS.get(family)
    if metric is None:
        raise ValueError(f"{family_id}: unsupported DE metric {family!r}")
    kwargs: dict[str, object] = {"padj": 0.05, "top_k": top_k, "rank_by": rank_by}
    if family == "spearman":
        kwargs["degs_only"] = str(row["feature_set"]).lower() == "deg"
    return partial(metric, **kwargs)


def _de_space(row: pd.Series) -> SubsetSpace:
    """Return the DE classification universe while preserving the old panel semantics."""
    feature_set = str(row["feature_set"]).lower()
    if feature_set == "deg":
        if str(row["target_operation"]) != "no_target_operation":
            raise ValueError("target operations on DEG-positive sets need an explicit label policy")
        return spaces.full()
    return gene_space(row)


def build_protocol(row: pd.Series) -> Protocol:
    """Build one fully concrete protocol from its frozen semantic row."""
    family_id = str(row["family_id"])
    subset = gene_space(row)
    population = str(row["population"])
    metric_family = str(row["metric_family"])
    centering = "control_mean" if str(row["centering"]) == "control-centered" else None

    if population == "PBULK":
        metric = PROFILE_METRICS.get(metric_family)
        if metric is None:
            raise ValueError(f"{family_id}: unsupported profile metric {metric_family!r}")
        lower = metric_family in {"MAE", "squared-error"}
        return Protocol(
            family_id,
            metric,
            representation="centroid",
            space=_space(row, subset),
            centering=centering,
            reference="control",
            better="lower" if lower else "higher",
            perfect=0.0 if lower else 1.0,
            group="atlas-profile",
        )
    if population == "SC":
        metric = POPULATION_METRICS.get(metric_family)
        if metric is None:
            raise ValueError(f"{family_id}: unsupported population metric {metric_family!r}")
        return Protocol(
            family_id,
            metric,
            representation="population",
            space=_space(row, subset),
            reference="control",
            better="lower",
            perfect=0.0,
            group="atlas-population",
            requires_extra="sinkhorn" if metric_family == "sinkhorn-divergence" else None,
        )
    if population == "DE":
        return Protocol(
            family_id,
            _de_metric(row),
            representation="de",
            space=_de_space(row),
            reference="control",
            neg_reference="control",
            better="higher",
            perfect=1.0,
            group="atlas-de",
        )
    raise ValueError(f"{family_id}: unsupported population representation {population!r}")


def build_protocols(path: Path = DEFAULT_ELIGIBILITY) -> list[Protocol]:
    """Build all 101 concrete protocols in stable manifest order."""
    return [build_protocol(row) for _, row in load_eligible_families(path).iterrows()]


def target_key_for_task(task: str, obs_columns) -> str | None:
    """Resolve the gene-target column needed by ``spaces.own_target``.

    Essential-screen labels are bare gene symbols and need no separate key.  Genetic-context
    labels are namespaced evaluation IDs, while ``model_perturbation`` retains the bare target.
    Prefer an explicit ``target_gene`` column whenever a prepared view provides one.
    """
    columns = set(obs_columns)
    if "target_gene" in columns:
        return "target_gene"
    if task == "genetic_context_transfer" and "model_perturbation" in columns:
        return "model_perturbation"
    return None
