from __future__ import annotations

import json
from pathlib import Path

from analyses.ranking_sensitivity.engine import (
    assert_scperteval_revision,
    expected_scperteval_revision,
)
from scripts.evaluate_atlas_production_benchmark import (
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
        tmp_path / "configs/evaluation/ranking-sensitivity-atlas.json",
        json.dumps({"evaluation_engine": {"revision": revision}}),
    )
    _write(tmp_path / "configs/evaluation/atlas-protocol-eligibility.csv", "family_id\nexample\n")
    _write(tmp_path / "configs/evaluation/atlas-task-protocol-manifest.csv", "task,protocol_id\ntask,example\n")
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
