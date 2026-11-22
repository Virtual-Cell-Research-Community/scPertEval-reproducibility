"""Resolve the frozen protocol-behavior manifest into concrete scPertEval protocols."""

from __future__ import annotations

from scperteval.blocks import spaces
from scperteval.protocols import PROTOCOLS, metrics
from scperteval.types import Protocol

_SPACE_EXPECTATIONS = {
    "pearson_pert_degs_padj=0.05": ("degs_0.05", spaces.degs(0.05)),
    "pearson_pert_top_k=50": ("top_50", spaces.top(50)),
    "mse_degs_padj=0.05": ("degs_0.05", spaces.degs(0.05)),
    "mse_top_k=50": ("top_50", spaces.top(50)),
    "unbiased_mmd_median_top_k=50": ("top_50", spaces.top(50)),
    "unbiased_mmd_median_pca_k=50": ("pca_50", spaces.pca(50)),
    "energy_distance_top_k=50": ("top_50", spaces.top(50)),
    "energy_distance_pca_k=50": ("pca_50", spaces.pca(50)),
    "sinkhorn_w2_top_k=50": ("top_50", spaces.top(50)),
    "sinkhorn_w2_pca_k=50": ("pca_50", spaces.pca(50)),
}


def _resolve_one(spec: str) -> Protocol:
    """Resolve one frozen catalog specification into a concrete protocol object."""
    if "=" not in spec:
        protocol = PROTOCOLS.get(spec)
        if protocol is None:
            raise ValueError(f"unknown frozen protocol {spec!r}")
        if protocol.parameterised:
            protocol = protocol.resolve(protocol.param.default)
    else:
        family, raw_value = spec.split("=", 1)
        protocol = PROTOCOLS.get(family)
        if protocol is None or protocol.param is None:
            raise ValueError(f"unknown parameterised frozen protocol {spec!r}")
        protocol = protocol.resolve(protocol.param.cast(raw_value))
    if protocol.parameterised:
        raise AssertionError(f"protocol did not resolve concretely: {spec}")
    if protocol.name != spec:
        raise ValueError(f"protocol {spec!r} resolved under unexpected name {protocol.name!r}")
    expected_space = _SPACE_EXPECTATIONS.get(spec)
    if expected_space is not None and protocol.space != expected_space[0]:
        raise ValueError(
            f"protocol {spec!r} resolved to space {protocol.space!r}, expected {expected_space[0]!r}"
        )
    return protocol


def resolve_frozen_protocols(specs: list[str]) -> list[Protocol]:
    """Resolve protocol names in manifest order and validate the DE metric primitives."""
    protocols = [_resolve_one(spec) for spec in specs]
    by_name = {protocol.name: protocol for protocol in protocols}
    expected_de_metrics = {
        "de_auprc": metrics.de_auprc,
        "de_auroc": metrics.de_auroc,
    }
    for name, metric in expected_de_metrics.items():
        if name in by_name and by_name[name].metric is not metric:
            raise ValueError(f"protocol {name!r} no longer resolves to the expected pure DE metric")
    overlap = by_name.get("de_overlap_k=50")
    if overlap is not None and getattr(overlap.metric, "func", None) is not metrics.de_overlap:
        raise ValueError("de_overlap_k=50 no longer resolves to the expected pure DE metric")
    return protocols
