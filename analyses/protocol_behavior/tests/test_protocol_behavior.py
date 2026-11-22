from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from analyses.protocol_behavior.cli import build_parser, main
from analyses.protocol_behavior.production import (
    _declared_perturbations,
    _reusable_record,
    _valid_drf,
    create_input_record,
    default_results,
    expected_scperteval_revision,
    file_sha256,
    load_manifest,
    production_stages,
    run_stage,
    stage_status,
)
from analyses.protocol_behavior.protocols import resolve_frozen_protocols
from analyses.protocol_behavior.render import (
    FIGURE_STEMS,
    _concordance_rows,
    _k_grid,
    _plot_jaccard,
    _table_text_color,
    _wrap_table_label,
    render_and_validate,
)
from scperteval.blocks import spaces as current_spaces


def test_protocol_command_defaults_to_the_production_workflow():
    args = build_parser().parse_args([])

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


def test_current_manifest_replaces_gwps_with_k562_essential():
    manifest = load_manifest(Path("configs/protocol_behavior/production-v2.json"))

    assert manifest["status"] == "ready-scperteval-main-staging"
    assert manifest["scperteval_revision"] == "58ccc91095ccb30bbe8f4470dff8b398d2eb4525"
    assert default_results(manifest) == Path("results/protocol_behavior/production-v2-scperteval-main-staging")
    assert manifest["datasets"][0]["id"] == "essential-k562"
    assert manifest["high_memory_cpu_drf_stage"] == "drf.cpu.essential-k562"
    assert len(production_stages(manifest)) == 64


def test_frozen_protocol_lists_resolve_to_concrete_current_api_objects():
    manifest = load_manifest(Path("configs/protocol_behavior/production-v2.json"))
    names = manifest["protocols"] + manifest["gpu_protocols"]

    resolved = resolve_frozen_protocols(names)

    assert [protocol.name for protocol in resolved] == names
    assert all(not protocol.parameterised for protocol in resolved)
    protocol_spaces = {protocol.name: protocol.space for protocol in resolved}
    assert protocol_spaces["pearson_pert_degs_padj=0.05"] == "degs_0.05"
    assert protocol_spaces["mse_top_k=50"] == "top_50"
    assert protocol_spaces["sinkhorn_w2_pca_k=50"] == "pca_50"
    assert current_spaces.degs(0.05).cache_key[1] == "degs"
    assert current_spaces.top(50).cache_key[1] == "top"
    assert current_spaces.pca(50).cache_key[1] == "pca"


def test_immutable_v1_manifest_resolves_its_historical_scperteval_revision():
    manifest = load_manifest(Path("configs/protocol_behavior/production-v1.json"))

    assert expected_scperteval_revision(manifest) == "89fc21f3e59673651b3162355d9a32495e274253"


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


def test_stage_listing_can_exclude_high_memory_dataset_by_id(capsys):
    assert (
        main(
            [
                "--list-stages",
                "--stage-class",
                "drf-cpu",
                "--exclude-stage",
                "drf.cpu.essential-k562",
            ]
        )
        == 0
    )

    stages = capsys.readouterr().out.splitlines()
    assert len(stages) == 7
    assert "drf.cpu.essential-k562" not in stages


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


def test_stage_record_reuse_resolves_relative_artifact_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = Path("artifact.csv")
    artifact.write_text("complete\n")
    record = Path("stage.json")
    record.write_text(
        json.dumps(
            {
                "status": "complete",
                "inputs": {"manifest": "example"},
                "artifact_sha256": {artifact.name: file_sha256(artifact)},
            }
        )
    )

    assert _reusable_record(record, {"manifest": "example"}, [artifact])


def test_stage_record_reuse_rejects_absolute_artifact_paths(tmp_path):
    stage = tmp_path / "results" / "stages" / "example"
    stage.mkdir(parents=True)
    artifact = stage / "artifact.csv"
    artifact.write_text("complete\n")
    record = stage / "stage.json"
    record.write_text(
        json.dumps(
            {
                "status": "complete",
                "inputs": {"manifest": "example"},
                "artifact_sha256": {
                    "/home/another-user/mount/results/stages/example/artifact.csv": file_sha256(artifact)
                },
            }
        )
    )

    assert not _reusable_record(record, {"manifest": "example"}, [artifact])


def test_drf_validation_requires_every_protocol_perturbation_pair_once(tmp_path):
    drf_path = tmp_path / "drf.csv"
    timing_path = tmp_path / "timing.csv"
    protocols = ["pearson", "mse"]
    perturbations = ["g1", "g2"]
    pd.DataFrame(
        [
            {
                "protocol": protocol,
                "perturbation": perturbation,
                "positive": "interpolated",
                "negative": "all_perturbed_mean",
                "raw_positive": 1.0,
                "raw_negative": 0.0,
                "undefined_reason": np.nan,
                "drf": 1.0,
                "dataset": "tiny",
                "de_method": "t-test",
                "subsample": 20,
                "seed": 42,
            }
            for protocol in protocols
            for perturbation in perturbations
        ]
    ).to_csv(drf_path, index=False)
    pd.DataFrame([{"dataset": "tiny", "protocol": protocol, "seconds": 1.0} for protocol in protocols]).to_csv(
        timing_path, index=False
    )

    assert _valid_drf(drf_path, timing_path, protocols, perturbations, seed=42, subsample=20)
    truncated = pd.read_csv(drf_path).iloc[:-1]
    truncated.to_csv(drf_path, index=False)
    assert not _valid_drf(drf_path, timing_path, protocols, perturbations, seed=42, subsample=20)


def test_concordance_uses_legacy_grid_and_reports_aucc():
    assert _k_grid(1000).tolist() == [1, 2, 4, 8, 16, 32, 50, 64, 125, 128, 250, 256, 375, 500]
    perturbations = ["g1", "g2"]
    statistic = np.arange(1000, dtype=float)[None, :].repeat(2, axis=0)
    padj = np.linspace(1e-8, 1.0, 1000)[None, :].repeat(2, axis=0)

    rows = _concordance_rows("example", perturbations, statistic, padj, statistic, padj)

    assert [row["k"] for row in rows] == _k_grid(1000).tolist()
    assert all(row["median"] == pytest.approx(1.0) for row in rows)
    assert all(row["median_aucc"] == pytest.approx(1.0) for row in rows)


def test_jaccard_renderer_supports_multiple_dataset_rows(tmp_path):
    labels = ["criterion-a", "criterion-b"]
    matrices = [(f"dataset-{index}", np.eye(2)) for index in range(8)]
    median_sizes = [np.array([10.0, 20.0]) for _dataset in matrices]
    spearman_values = [np.linspace(0, 1, 10) for _dataset in matrices]
    output = tmp_path / "jaccard.png"

    _plot_jaccard(matrices, labels, median_sizes, spearman_values, output, "Example")

    assert output.stat().st_size > 0


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
    data_manifest = tmp_path / "raw.manifest.json"
    data_manifest.write_text(json.dumps({"output": {"path": dataset.name, "sha256": file_sha256(dataset)}}))
    cohort_manifest = tmp_path / "cohort.json"
    cohort_manifest.write_text(json.dumps({"perturbations": [f"g{index}" for index in range(4)]}))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "tiny",
                "scperteval_revision": "58ccc91095ccb30bbe8f4470dff8b398d2eb4525",
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
                "datasets": [
                    {
                        "id": "tiny",
                        "label": "Tiny",
                        "path": str(dataset),
                        "data_manifest": str(data_manifest),
                        "data_manifest_output_key": "output",
                        "cohort_manifest": str(cohort_manifest),
                    }
                ],
            }
        )
    )
    output = tmp_path / "results"
    assert create_input_record(manifest_path, output).is_file()
    manifest = load_manifest(manifest_path)
    for stage in production_stages(manifest):
        assert run_stage(manifest_path, stage.id, output, workers=1).is_file()
    for reference in ("all_perturbed", "control"):
        for method in ("t-test", "t-test_overestim_var", "MWU"):
            with h5py.File(output / "stages" / f"de.tiny.{reference}.{method}" / "de.h5") as handle:
                assert method in handle
                assert [value.decode() for value in handle["perturbations"][:]] == ["g0", "g1", "g2", "g3"]

    first_stage = production_stages(manifest)[0]

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("a checksum-valid production stage must be reused before loading its H5AD")

    monkeypatch.setattr("analyses.protocol_behavior.production.ad.read_h5ad", unexpected_read)
    assert run_stage(manifest_path, first_stage.id, output, workers=1).is_file()

    completion = render_and_validate(manifest_path, output, baseline_results=output)

    completion_payload = json.loads(completion.read_text())
    assert completion_payload["status"] == "complete"
    assert completion_payload["n_compute_stages"] == 8
    assert stage_status(manifest_path, output)["counts"] == {"complete": 8, "failed": 0, "pending": 0}
    assert all(not Path(artifact["path"]).is_absolute() for artifact in completion_payload["artifacts"])
    first_record = json.loads((output / "stages" / first_stage.id / "stage.json").read_text())
    assert first_record["inputs"]["scperteval_revision"] == "58ccc91095ccb30bbe8f4470dff8b398d2eb4525"
    assert all(not Path(name).is_absolute() for name in first_record["artifact_sha256"])
    source = pd.read_csv(output / "figures" / "deg_jaccard_with_counts.csv")
    assert set(source["record_type"]) == {"jaccard", "set_size", "spearman"}
    assert all((output / "figures" / f"{stem}.png").stat().st_size > 0 for stem in FIGURE_STEMS)
    assert (output / "figures" / "v1_v2_equivalence.csv").is_file()
    with h5py.File(output / "stages" / "de.tiny.control.MWU" / "de.h5") as handle:
        assert handle.attrs["target"] == "gt_all_cells"
        assert handle.attrs["reference"] == "control"
        assert int(handle.attrs["subsample"]) == 20
        assert int(handle.attrs["seed"]) == 42
        assert list(handle) == ["MWU", "genes", "perturbations"]


def test_input_record_rejects_a_changed_environment_fingerprint(tmp_path, monkeypatch):
    manifest = load_manifest(Path("configs/protocol_behavior/production-v2.json"))
    output = tmp_path / "results"
    output.mkdir()
    inputs = {
        "status": "complete",
        "environment": {"manifest_sha256": "old"},
        "datasets": {},
    }
    from analyses.protocol_behavior.production import _canonical_sha256, validate_input_record

    inputs["input_fingerprint_sha256"] = _canonical_sha256(inputs)
    (output / "inputs.json").write_text(json.dumps(inputs))
    monkeypatch.setattr("analyses.protocol_behavior.production.load_manifest", lambda _path: manifest)

    with pytest.raises(RuntimeError, match="different code or evaluator environment"):
        validate_input_record(Path("configs/protocol_behavior/production-v2.json"), output)
