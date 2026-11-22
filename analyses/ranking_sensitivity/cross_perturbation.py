"""Decompose Atlas metric families into cross-perturbation strategy and score function.

Retrieval families encode two evaluation choices in one Atlas metric label: whether the matching
perturbation is compared directly or retrieved among all perturbations of the fold, and which
distance orders the candidates. Keeping both explicit lets the runtime build the metric and lets the
analysis classify "direct versus retrieval with the same distance" as its own factor.

This module is pure Python so the analysis layer can use it without importing scPertEval.
"""

from __future__ import annotations

from dataclasses import dataclass

DIRECT = "direct"
RETRIEVAL_PREDICTED_QUERY = "retrieval_predicted_query"
RETRIEVAL_OBSERVED_QUERY = "retrieval_observed_query"

_PREFIXES = (
    ("transposed-retrieval-rank-", RETRIEVAL_PREDICTED_QUERY),
    ("retrieval-rank-", RETRIEVAL_OBSERVED_QUERY),
)

RETRIEVAL_DISTANCES = {
    "MSE": "squared_euclidean",
    "rmse": "squared_euclidean",
    "MAE": "cityblock",
    "cosine": "cosine",
}
"""Atlas distance suffix → candidate-ordering distance.

MSE, RMSE and Euclidean distance order a fixed-dimensional candidate set identically, so all three
use the squared Euclidean distance of ``scperteval.protocols.metrics.rank_retrieval``.
"""

DIRECT_SCORE_FUNCTION = {
    "squared_euclidean": "squared-error",
    "cityblock": "MAE",
    "cosine": "cosine",
}
"""Retrieval distance → the direct profile score function with the same distance."""


@dataclass(frozen=True)
class CrossPerturbation:
    """The cross-perturbation strategy and the score function shared with direct comparison."""

    strategy: str
    score_function: str
    distance: str | None = None

    @property
    def is_retrieval(self) -> bool:
        return self.strategy != DIRECT

    @property
    def transpose(self) -> bool:
        """``True`` when each prediction queries all observed perturbations (``rank_retrieval`` naming)."""
        return self.strategy == RETRIEVAL_PREDICTED_QUERY


def decompose(metric_family: str) -> CrossPerturbation:
    """Split one Atlas ``metric_family`` into strategy and comparable score function."""

    for prefix, strategy in _PREFIXES:
        if metric_family.startswith(prefix):
            suffix = metric_family.removeprefix(prefix)
            try:
                distance = RETRIEVAL_DISTANCES[suffix]
            except KeyError as error:
                raise ValueError(f"unsupported retrieval distance in {metric_family!r}") from error
            return CrossPerturbation(strategy, DIRECT_SCORE_FUNCTION[distance], distance)
    return CrossPerturbation(DIRECT, metric_family)
