from __future__ import annotations

import json
from pathlib import Path

from workflows.prepare_protocol_data import build_steps, main


def test_protocol_data_plan_covers_six_sources_and_three_views(tmp_path):
    config = {
        "schema_version": 1,
        "id": "test",
        "canonical_datasets": ["a", "b", "c", "d", "e", "f"],
        "views": [
            {
                "id": "gwps",
                "builder": "gwps",
                "inputs": ["a"],
                "config": "configs/gwps.json",
                "output": "models/data/gwps/raw.h5ad",
                "perturbation_key": "target_gene",
                "control_key": "control_class",
                "control_value": "non-targeting",
                "split_id": "gwps-five-fold",
            },
            {
                "id": "genetic",
                "builder": "context",
                "inputs": ["b", "c", "d", "e"],
                "config": "configs/genetic.json",
                "output": "models/data/genetic/raw.h5ad",
                "counts_output": "models/data/genetic/counts.h5ad",
            },
            {
                "id": "chemical",
                "builder": "context",
                "inputs": ["f"],
                "config": "configs/chemical.json",
                "output": "models/data/chemical/raw.h5ad",
                "counts_output": "models/data/chemical/counts.h5ad",
            },
        ],
    }

    steps = build_steps(config, repository_root=tmp_path, data_root=tmp_path / "data")

    assert sum(step.kind == "canonical" for step in steps) == 6
    assert sum(step.kind == "view" for step in steps) == 3
    assert steps[-1].id == "chemical"


def test_prepare_protocol_data_plan_mode_does_not_execute(tmp_path, capsys):
    config = tmp_path / "data-plan.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "empty",
                "canonical_datasets": [],
                "views": [],
            }
        )
    )

    assert main(["--config", str(config), "--plan"]) == 0
    assert "empty" in capsys.readouterr().out


def test_current_protocol_data_config_uses_the_essential_screen():
    config = json.loads(Path("configs/protocol_behavior/data-v2.json").read_text())

    steps = build_steps(config)
    essential = next(
        step for step in steps if step.id == "replogle-weissman-2022-k562-essential" and step.kind == "view"
    )

    assert len(config["canonical_datasets"]) == 5
    assert len(config["views"]) == 3
    assert essential.kind == "view"
    assert "models.data.prepare_data" in essential.commands[0]
    assert "replogle-weissman-2022-k562-essential" in essential.commands[0]
