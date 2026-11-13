"""Run TabICLv2 without its bundled PRESAGE evaluation.

The upstream multiview model computes fused predictions before invoking a
single-cell Wilcoxon DEG workflow and PRESAGE-specific metrics.  scPertEval
evaluates every model centrally from the saved predictions, so repeating that
model-specific evaluation is both redundant and prohibitively expensive for
the genome-wide screen.

The pinned upstream implementation does not expose an evaluation switch.  This
module therefore replaces only its two evaluation hooks for the duration of a
single ``fit`` call.  Prediction, modality fusion, validation, and artifact
serialization remain upstream code, and the original hooks are restored even
when fitting raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Protocol


class _FittableModel(Protocol):
    def fit(self, **kwargs: Any) -> Any: ...


class _PredictionOnlyEvaluator:
    """No-op evaluator satisfying the upstream serialization contract."""

    def evaluate_split(self, *_args: Any, **_kwargs: Any) -> tuple[dict, dict]:
        return {}, {}


def _skip_degs(**_kwargs: Any) -> dict:
    return {}


def _build_noop_evaluator(**_kwargs: Any) -> tuple[_PredictionOnlyEvaluator, None, None]:
    return _PredictionOnlyEvaluator(), None, None


@contextmanager
def prediction_only_evaluation(
    upstream_module: ModuleType | Any | None = None,
) -> Iterator[None]:
    """Temporarily bypass upstream DEG computation and evaluation."""
    if upstream_module is None:
        from tfm_perturbation.models.transcriptomics import tabicl_multiview as upstream_module

    required = ("compute_or_load_presage_degs", "_build_presage_evaluator")
    missing = [name for name in required if not hasattr(upstream_module, name)]
    if missing:
        raise RuntimeError(
            "Pinned TabICLv2 implementation no longer exposes the expected "
            f"evaluation hooks: {missing}"
        )

    original_degs = upstream_module.compute_or_load_presage_degs
    original_evaluator = upstream_module._build_presage_evaluator
    upstream_module.compute_or_load_presage_degs = _skip_degs
    upstream_module._build_presage_evaluator = _build_noop_evaluator
    try:
        yield
    finally:
        upstream_module.compute_or_load_presage_degs = original_degs
        upstream_module._build_presage_evaluator = original_evaluator


def fit_prediction_only(
    model: _FittableModel,
    *,
    upstream_module: ModuleType | Any | None = None,
    **fit_kwargs: Any,
) -> Any:
    """Fit TabICLv2 while retaining predictions but skipping upstream metrics."""
    with prediction_only_evaluation(upstream_module):
        return model.fit(**fit_kwargs)
