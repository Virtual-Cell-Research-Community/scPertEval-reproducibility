"""Compatibility helpers for TabICLv2's PRESAGE datamodule."""

from pathlib import Path
from typing import Protocol


class _SplitDataModule(Protocol):
    split_json_path: str

    def setup(self) -> None: ...


def initialize_with_nonempty_validation(
    datamodule: _SplitDataModule,
    *,
    validation_split_path: Path,
    model_split_path: Path,
) -> None:
    """Load upstream data, then expose TabICL's merged development pool.

    The pinned upstream PRESAGE datamodule rejects an empty validation split
    during setup. TabICLv2 performs no validation-based selection in our
    configuration, so its actual model split deliberately merges the frozen
    validation perturbations into its in-context training pool. Load against
    the ordinary non-empty split first, then switch the path consumed by
    ``TabICLMultiviewModel.fit`` to the merged development split.
    """
    datamodule.split_json_path = str(validation_split_path)
    datamodule.setup()
    datamodule.split_json_path = str(model_split_path)
