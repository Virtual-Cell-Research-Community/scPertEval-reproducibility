"""Resolve the frozen Atlas semantic families to concrete scPertEval protocols."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scperteval.blocks.panels import GenePanel, panel_pca_space, panel_space
from scperteval.blocks.spaces import SPACES
from scperteval.protocols import metrics
from scperteval.protocols.de_metrics import DEComparison
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
    "no_target_operation": "no_special",
    "force_include_target": "force_include_target",
    "exclude_own_target": "exclude_own_target",
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


def gene_panel(row: pd.Series) -> GenePanel:
    """Resolve one row's truth-derived feature panel and target policy."""
    feature_set = str(row["feature_set"]).lower()
    if feature_set not in {"all", "deg", "hvg", "heg"}:
        raise ValueError(f"{row['family_id']}: unsupported feature set {row['feature_set']!r}")
    rank = str(row["feature_ranking_basis"])
    rank_by = {"significance": "significance", "effect": "absolute_effect", "": None}[rank]
    top_k = _canonical_k(row)
    return GenePanel(
        feature_set=feature_set,
        top_k=top_k,
        padj_max=0.05 if feature_set == "deg" else None,
        rank_by=rank_by,
        target_policy=TARGET_POLICIES[str(row["target_operation"])],
        reference="control",
    )


def _space(row: pd.Series, panel: GenePanel) -> str:
    if str(row["space"]) == "GEX":
        return panel_space(panel)
    if str(row["space"]) != "PCA":
        raise ValueError(f"{row['family_id']}: unsupported space {row['space']!r}")
    if panel.feature_set == "all" and panel.target_policy == "no_special":
        return SPACES.instance("pca", 50)
    return panel_pca_space(panel, 50)


def build_protocol(row: pd.Series) -> Protocol:
    """Build one fully concrete protocol from its frozen semantic row."""
    family_id = str(row["family_id"])
    panel = gene_panel(row)
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
            space=_space(row, panel),
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
            space=_space(row, panel),
            reference="control",
            better="lower",
            perfect=0.0,
            group="atlas-population",
            requires_extra="sinkhorn" if metric_family == "sinkhorn-divergence" else None,
        )
    if population == "DE":
        top_k = _canonical_k(row)
        metric = DEComparison(metric_family, panel=panel, padj_max=0.05, top_k=top_k)
        return Protocol(
            family_id,
            metric,
            representation="de",
            space="full",
            reference="control",
            neg_reference="control",
            better="higher",
            perfect=1.0,
            group="atlas-de",
        )
    raise ValueError(f"{family_id}: unsupported population representation {population!r}")


def build_protocols(path: Path = DEFAULT_ELIGIBILITY) -> list[Protocol]:
    """Build all 99 concrete protocols in stable manifest order."""
    return [build_protocol(row) for _, row in load_eligible_families(path).iterrows()]
