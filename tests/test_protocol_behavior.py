from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from analyses.protocol_behavior.cli import (
    DEFAULT_GWPS_DATASET,
    DEFAULT_OUTPUT_DIR,
    build_parser,
    run_gwps_de_pilot,
    select_quartile_targets,
)


def test_protocol_command_defaults_to_the_resumable_gwps_pilot():
    args = build_parser().parse_args([])

    assert args.pilot == "gwps-de"
    assert args.dataset == DEFAULT_GWPS_DATASET
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.force is False


def test_select_quartile_targets_is_balanced_and_deterministic():
    rows = []
    for index in range(80):
        rows.extend({"perturbation": f"g{index:02d}", "is_control": False} for _ in range(index + 10))
    rows.extend({"perturbation": "control", "is_control": True} for _ in range(200))
    obs = pd.DataFrame(rows)

    first = select_quartile_targets(obs, n_per_quartile=3, seed=42)
    second = select_quartile_targets(obs, n_per_quartile=3, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("quartile").size().to_dict() == {0: 3, 1: 3, 2: 3, 3: 3}
    assert first["perturbation"].is_unique
    assert "control" not in set(first["perturbation"])


def test_select_quartile_targets_checks_metadata_and_size():
    with pytest.raises(ValueError, match="metadata"):
        select_quartile_targets(pd.DataFrame({"perturbation": ["a"]}), n_per_quartile=1)
    tiny = pd.DataFrame(
        {
            "perturbation": ["a", "b", "control"],
            "is_control": [False, False, True],
        }
    )
    with pytest.raises(ValueError, match="at least 4"):
        select_quartile_targets(tiny, n_per_quartile=1)


def test_gwps_de_pilot_writes_and_reuses_complete_grid(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    matrices = [rng.normal(size=(100, 8)).astype(np.float32)]
    labels = ["control"] * 100
    controls = [True] * 100
    for index, n_cells in enumerate((50, 60, 70, 80)):
        matrices.append(rng.normal(loc=index + 1, size=(n_cells, 8)).astype(np.float32))
        labels.extend([f"g{index}"] * n_cells)
        controls.extend([False] * n_cells)
    dataset = ad.AnnData(np.vstack(matrices))
    dataset.obs["perturbation"] = labels
    dataset.obs["is_control"] = controls
    dataset.var_names = [f"gene{index}" for index in range(dataset.n_vars)]
    dataset_path = tmp_path / "dataset.h5ad"
    dataset.write_h5ad(dataset_path)

    output = tmp_path / "pilot"
    summary = run_gwps_de_pilot(
        dataset_path,
        output,
        n_per_quartile=1,
        subsample=40,
        workers=1,
    )
    assert summary.is_file()
    artifacts = list(output.glob("*__vs__*/*/de.h5"))
    assert len(artifacts) == 8
    measured_summary = summary.read_text()

    def unexpected_prepare(*_args, **_kwargs):
        raise AssertionError("a complete grid must be reused without loading expression data")

    monkeypatch.setattr("analyses.protocol_behavior.cli.spe.prepare", unexpected_prepare)
    run_gwps_de_pilot(
        dataset_path,
        output,
        n_per_quartile=1,
        subsample=40,
        workers=1,
    )
    import json

    assert summary.read_text() == measured_summary
    statuses = {stage["status"] for stage in json.loads(summary.read_text())["stages"]}
    assert statuses == {"computed"}
