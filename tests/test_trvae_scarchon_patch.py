"""Focused regression tests for the trvaep compatibility layer."""

from __future__ import annotations

import pytest

from models.trvae.scarchon_patch import scalarize_validation_metrics


class TensorMetric:
    """Small tensor-shaped test double that keeps this unit test Torch-free."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def numel(self) -> int:
        return len(self.values)

    def item(self) -> float:
        return self.values[0]


def test_scalarizes_one_element_tensors() -> None:
    metrics = scalarize_validation_metrics((1.0, TensorMetric([2.5])))
    assert metrics == (1.0, 2.5)


def test_rejects_vector_valued_metrics() -> None:
    with pytest.raises(ValueError, match="validation metrics must be scalar"):
        scalarize_validation_metrics((TensorMetric([1.0, 2.0]),))
