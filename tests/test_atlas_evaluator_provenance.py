from __future__ import annotations

import json
from pathlib import Path

from analyses.ranking_sensitivity.engine import (
    assert_scperteval_revision,
    expected_scperteval_revision,
)
from scripts.evaluate_atlas_production_benchmark import (
    Selection,
    Track,
    _completion_record,
    _output_is_current,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_frozen_engine_revision_matches_the_installed_direct_dependency():
    root = Path(__file__).parents[1]
    expected = "58ccc91095ccb30bbe8f4470dff8b398d2eb4525"
    assert expected_scperteval_revision(root) == expected
    assert assert_scperteval_revision(root) == expected


def test_old_completion_records_are_stale_after_evaluator_migration(tmp_path: Path):
    revision = "58ccc91095ccb30bbe8f4470dff8b398d2eb4525"
    _write(
        tmp_path / "configs/evaluation/ranking-sensitivity-atlas-v1.json",
        json.dumps({"evaluation_engine": {"revision": revision}}),
    )
    _write(tmp_path / "configs/evaluation/atlas-protocol-eligibility-v1.csv", "family_id\nexample\n")
    _write(tmp_path / "configs/evaluation/atlas-task-protocol-manifest-v1.csv", "task,protocol_id\ntask,example\n")
    _write(tmp_path / "analyses/ranking_sensitivity/atlas_runtime.py", "# atlas runtime\n")
    _write(tmp_path / "analyses/ranking_sensitivity/prediction_runtime.py", "# prediction runtime\n")
    _write(tmp_path / "analyses/ranking_sensitivity/retrieval.py", "# retrieval\n")
    _write(tmp_path / "analyses/ranking_sensitivity/cross_perturbation.py", "# cross perturbation\n")
    split = _write(tmp_path / "split.json", "{}\n")
    experiment = _write(tmp_path / "experiment.json", "{}\n")
    prediction_manifest = _write(
        tmp_path / "predictions.manifest.json",
        json.dumps({"artifact": {"sha256": "a" * 64}}),
    )
    output = _write(tmp_path / "seed-42.csv", "score\n1\n")
    track = Track("track", "task", "run", tmp_path / "truth.h5ad", split, experiment, ("fold",), ("model",), (42,))

    current = _completion_record(
        track=track,
        model="model",
        fold_id="fold",
        seed=42,
        prediction_manifest=prediction_manifest,
        output=output,
        project_root=tmp_path,
    )
    assert current["schema_version"] == 2
    assert current["inputs"]["scperteval_revision"] == revision
    assert len(current["inputs"]["evaluator_sha256"]) == 64

    legacy = dict(current)
    legacy["schema_version"] = 1
    legacy["inputs"] = {
        key: value for key, value in current["inputs"].items() if key not in {"scperteval_revision", "evaluator_sha256"}
    }
    output.with_suffix(".complete.json").write_text(json.dumps(legacy))
    assert not _output_is_current(
        track=track,
        model="model",
        fold_id="fold",
        seed=42,
        prediction_manifest=prediction_manifest,
        output=output,
        project_root=tmp_path,
    )


def test_incremental_v2_records_bind_the_added_protocol_selection(tmp_path: Path):
    revision = "58ccc91095ccb30bbe8f4470dff8b398d2eb4525"
    _write(
        tmp_path / "configs/evaluation/ranking-sensitivity-atlas-v1.json",
        json.dumps({"evaluation_engine": {"revision": revision}}),
    )
    for version in ("v1", "v2"):
        _write(tmp_path / f"configs/evaluation/atlas-protocol-eligibility-{version}.csv", f"family_id\n{version}\n")
        _write(
            tmp_path / f"configs/evaluation/atlas-task-protocol-manifest-{version}.csv",
            f"task,protocol_id\ntask,{version}\n",
        )
    for name in ("atlas_runtime", "prediction_runtime", "retrieval", "cross_perturbation"):
        _write(tmp_path / f"analyses/ranking_sensitivity/{name}.py", f"# {name}\n")
    split = _write(tmp_path / "split.json", "{}\n")
    prediction_manifest = _write(
        tmp_path / "predictions.manifest.json",
        json.dumps({"artifact": {"sha256": "b" * 64}}),
    )
    output = _write(tmp_path / "seed-42.csv", "score\n1\n")
    track = Track("track", "task", "run", tmp_path / "truth.h5ad", split, split, ("fold",), ("model",), (42,))
    common = {
        "track": track,
        "model": "model",
        "fold_id": "fold",
        "seed": 42,
        "prediction_manifest": prediction_manifest,
        "output": output,
        "project_root": tmp_path,
    }
    record = _completion_record(**common, selection=Selection("v2", "v1"), protocol_ids=["b", "a"])
    assert record["schema_version"] == 3
    assert record["protocol_selection"]["protocol_ids"] == ["a", "b"]
    assert record["protocol_selection"]["added_since"] == "v1"
    assert record["inputs"]["prediction_sha256"] == "b" * 64
    output.with_suffix(".complete.json").write_text(json.dumps(record))
    assert _output_is_current(**common, selection=Selection("v2", "v1"), protocol_ids=["a", "b"])
    assert not _output_is_current(**common, selection=Selection("v2", "v1"), protocol_ids=["a"])
    assert not _output_is_current(**common)
