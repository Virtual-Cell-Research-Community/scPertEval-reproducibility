from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from analyses.protocol_behavior.cli import (
    DEFAULT_GWPS_DATASET,
    build_parser,
    main,
    run_gwps_de_pilot,
    select_quartile_targets,
)
from analyses.protocol_behavior.production import (
    _declared_perturbations,
    _reusable_record,
    file_sha256,
    load_manifest,
    production_stages,
    run_stage,
)
from analyses.protocol_behavior.render import (
    FIGURE_STEMS,
    _table_text_color,
    _wrap_table_label,
    render_and_validate,
)


def test_protocol_command_defaults_to_the_production_workflow():
    args = build_parser().parse_args([])

    assert args.pilot is None
    assert args.dataset == DEFAULT_GWPS_DATASET
    assert args.output_dir is None
    assert args.force is False


def test_table_renderer_wraps_headers_and_contrasts_dark_cells():
    assert _wrap_table_label("Genetic K562 essential") == "Genetic K562\nessential"
    assert _table_text_color(np.array([0.0, 0.0, 0.0, 1.0])) == "white"
    assert _table_text_color(np.array([1.0, 1.0, 1.0, 1.0])) == "black"


def test_production_manifest_declares_all_analysis_units_and_chunks():
    manifest = load_manifest(Path("configs/protocol_behavior/production-v1.json"))
    stages = production_stages(manifest)

    assert len(manifest["datasets"]) == 8
    assert len(manifest["protocols"]) + len(manifest["gpu_protocols"]) == 22
    assert len(stages) == 64
    assert len({stage.id for stage in stages}) == len(stages)
    assert sum(stage.kind == "de" for stage in stages) == 48
    assert sum(stage.kind == "drf-cpu" for stage in stages) == 8
    assert sum(stage.kind == "drf-gpu" for stage in stages) == 8
    chemical = next(item for item in manifest["datasets"] if item["id"] == "chemical-a549")
    assert len(_declared_perturbations(chemical)) == 33


@pytest.mark.parametrize(
    ("stage_class", "expected_count", "expected_prefix"),
    [
        ("de", 48, "de."),
        ("drf-cpu", 8, "drf.cpu."),
        ("drf-gpu", 8, "drf.gpu."),
    ],
)
def test_production_stage_classes_are_independently_listable(stage_class, expected_count, expected_prefix, capsys):
    assert main(["--list-stages", "--stage-class", stage_class]) == 0

    stages = capsys.readouterr().out.splitlines()
    assert len(stages) == expected_count
    assert all(stage.startswith(expected_prefix) for stage in stages)


def test_analysis_unit_can_resolve_from_an_external_data_root(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    dataset = root / "models/data/example/raw.h5ad"
    dataset.parent.mkdir(parents=True)
    data = ad.AnnData(np.ones((2, 2), dtype=np.float32))
    data.obs["perturbation"] = ["control", "g1"]
    data.obs["is_control"] = [True, False]
    data.write_h5ad(dataset)
    monkeypatch.setenv("SCPERTEVAL_DATA_ROOT", str(root))

    from analyses.protocol_behavior.production import _load_analysis_unit

    observed = _load_analysis_unit({"path": "models/data/example/raw.h5ad"})

    assert observed.shape == (2, 2)


def test_stage_record_reuse_normalizes_absolute_and_relative_artifact_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = Path("artifact.csv")
    artifact.write_text("complete\n")
    record = Path("stage.json")
    record.write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": "manifest",
                "artifact_sha256": {str(artifact.resolve()): file_sha256(artifact)},
            }
        )
    )

    assert _reusable_record(record, "manifest", [artifact])


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


def test_tiny_production_workflow_renders_and_validates(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    matrices = [rng.normal(size=(30, 12)).astype(np.float32)]
    labels = ["control"] * 30
    controls = [True] * 30
    for index in range(4):
        matrices.append(rng.normal(loc=index / 2, size=(10, 12)).astype(np.float32))
        labels.extend([f"g{index}"] * 10)
        controls.extend([False] * 10)
    data = ad.AnnData(np.vstack(matrices))
    data.obs["perturbation"] = labels
    data.obs["is_control"] = controls
    data.var_names = [f"gene{index}" for index in range(data.n_vars)]
    dataset = tmp_path / "tiny.h5ad"
    data.write_h5ad(dataset)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "tiny",
                "seed": 42,
                "minimum_cells": 5,
                "reference_subsample": 20,
                "workers": 1,
                "protocols": ["pearson", "mse"],
                "gpu_protocols": ["sinkhorn_w2_pca_k=2"],
                "de": {
                    "target": "gt_all_cells",
                    "references": ["all_perturbed", "control"],
                    "methods": ["t-test", "t-test_overestim_var", "MWU"],
                },
                "datasets": [{"id": "tiny", "label": "Tiny", "path": str(dataset)}],
            }
        )
    )
    output = tmp_path / "results"
    manifest = load_manifest(manifest_path)
    for stage in production_stages(manifest):
        assert run_stage(manifest_path, stage.id, output, workers=1).is_file()

    first_stage = production_stages(manifest)[0]

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("a checksum-valid production stage must be reused before loading its H5AD")

    monkeypatch.setattr("analyses.protocol_behavior.production.ad.read_h5ad", unexpected_read)
    assert run_stage(manifest_path, first_stage.id, output, workers=1).is_file()

    completion = render_and_validate(manifest_path, output)

    assert json.loads(completion.read_text())["status"] == "complete"
    assert all((output / "figures" / f"{stem}.png").stat().st_size > 0 for stem in FIGURE_STEMS)
