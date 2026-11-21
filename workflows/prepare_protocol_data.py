"""Prepare every public dataset and frozen view used by protocol-behavior analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflows.preparation_utils import file_checksum

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/protocol_behavior/data-v2.json")
DEFAULT_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class PreparationStep:
    """One canonical acquisition or benchmark-view preparation unit."""

    id: str
    kind: str
    commands: tuple[tuple[str, ...], ...]
    outputs: tuple[Path, ...]


def _canonical_path(dataset_id: str, data_root: Path) -> Path:
    return data_root / "processed" / dataset_id / "adata.h5ad"


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the protocol-data declaration."""
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"unsupported protocol-data schema in {path}")
    required = {"id", "canonical_datasets", "views"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"protocol-data config is missing: {', '.join(missing)}")
    canonical = list(map(str, config["canonical_datasets"]))
    if len(canonical) != len(set(canonical)):
        raise ValueError("protocol-data config contains duplicate canonical datasets")
    known = set(canonical)
    for view in config["views"]:
        unknown = sorted(set(view["inputs"]) - known)
        if unknown:
            raise ValueError(f"view {view['id']!r} references unknown canonical inputs: {', '.join(unknown)}")
    return config


def _context_command(view: dict[str, Any], repository_root: Path, data_root: Path) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "models.data.prepare_context",
        "--config",
        str(repository_root / view["config"]),
    ]
    for dataset_id in view["inputs"]:
        command.extend(["--input", f"{dataset_id}={_canonical_path(dataset_id, data_root)}"])
    command.extend(["--output", str(repository_root / view["output"])])
    if view.get("counts_output"):
        command.extend(["--counts-output", str(repository_root / view["counts_output"])])
    return tuple(command)


def _gwps_commands(view: dict[str, Any], repository_root: Path, data_root: Path) -> tuple[tuple[str, ...], ...]:
    dataset_id = str(view["inputs"][0])
    prepare = [
        sys.executable,
        "-m",
        "models.data.prepare_data",
        "--input",
        str(_canonical_path(dataset_id, data_root)),
        "--perturbation-key",
        str(view["perturbation_key"]),
        "--control-key",
        str(view["control_key"]),
        "--control-value",
        str(view["control_value"]),
        "--experiment-manifest",
        str(repository_root / view["config"]),
        "--dataset-name",
        str(view["id"]),
        "--cell-type",
        str(view.get("cell_type", "K562")),
    ]
    split = [
        sys.executable,
        "-m",
        "models.data.prepare_split",
        str(repository_root / view["output"]),
        "--dataset-id",
        str(view["id"]),
        "--split-id",
        str(view["split_id"]),
    ]
    return tuple(map(tuple, (prepare, split)))


def _perturbation_command(
    view: dict[str, Any],
    repository_root: Path,
    data_root: Path,
) -> tuple[str, ...]:
    """Prepare one frozen single-context perturbation view without model-specific splitting."""
    dataset_id = str(view["inputs"][0])
    expected_output = repository_root / "models" / "data" / str(view["id"]) / "raw.h5ad"
    declared_output = (repository_root / view["output"]).resolve()
    if expected_output.resolve() != declared_output:
        raise ValueError(
            "perturbation builder output must be models/data/<view id>/raw.h5ad: "
            f"{view['output']}"
        )
    command = [
        sys.executable,
        "-m",
        "models.data.prepare_data",
        "--input",
        str(_canonical_path(dataset_id, data_root)),
        "--perturbation-key",
        str(view.get("perturbation_key", "perturbation")),
        "--control-key",
        str(view.get("control_key", "is_control")),
        "--experiment-manifest",
        str(repository_root / view["config"]),
        "--dataset-name",
        str(view["id"]),
        "--cell-type",
        str(view.get("cell_type", "K562")),
    ]
    if view.get("control_value") is not None:
        command.extend(["--control-value", str(view["control_value"])])
    return tuple(command)


def build_steps(
    config: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[PreparationStep]:
    """Resolve the declared public preparation into deterministic executable steps."""
    repository_root = repository_root.resolve()
    data_root = data_root.resolve()
    steps = []
    for dataset_id in config["canonical_datasets"]:
        output = _canonical_path(dataset_id, data_root)
        command_name = "validate" if output.is_file() else "all"
        command = (
            sys.executable,
            "-m",
            "workflows.data",
            "--root",
            str(data_root),
            command_name,
            str(dataset_id),
        )
        steps.append(PreparationStep(str(dataset_id), "canonical", (command,), (output,)))

    for view in config["views"]:
        outputs = [repository_root / view["output"]]
        if view.get("counts_output"):
            outputs.append(repository_root / view["counts_output"])
        if view["builder"] == "gwps":
            commands = _gwps_commands(view, repository_root, data_root)
        elif view["builder"] == "perturbation":
            commands = (_perturbation_command(view, repository_root, data_root),)
        elif view["builder"] == "context":
            commands = (_context_command(view, repository_root, data_root),)
        else:
            raise ValueError(f"unknown protocol-data view builder: {view['builder']}")
        steps.append(PreparationStep(str(view["id"]), "view", commands, tuple(outputs)))
    return steps


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_record(path: Path, config_path: Path, config: dict[str, Any], steps: list[PreparationStep]) -> Path:
    missing = [output for step in steps for output in step.outputs if not output.is_file()]
    if missing:
        raise FileNotFoundError(f"protocol-data preparation did not produce: {missing[0]}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": config["id"],
                "config": str(config_path),
                "config_sha256": _config_sha256(config_path),
                "repository_commit": commit,
                "steps": [
                    {
                        "id": step.id,
                        "kind": step.kind,
                        "commands": [list(command) for command in step.commands],
                        "outputs": [
                            {
                                "path": str(output.relative_to(REPOSITORY_ROOT))
                                if output.is_relative_to(REPOSITORY_ROOT)
                                else str(output),
                                "sha256": file_checksum(output),
                                "size_bytes": output.stat().st_size,
                            }
                            for output in step.outputs
                        ],
                    }
                    for step in steps
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _record_is_valid(path: Path, config_path: Path, steps: list[PreparationStep]) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("config_sha256") != _config_sha256(config_path):
        return False
    recorded = {
        (step["id"], item["path"]): item["sha256"]
        for step in record.get("steps", [])
        for item in step.get("outputs", [])
    }
    for step in steps:
        for output in step.outputs:
            key = (
                step.id,
                str(output.relative_to(REPOSITORY_ROOT))
                if output.is_relative_to(REPOSITORY_ROOT)
                else str(output),
            )
            if not output.is_file() or recorded.get(key) != file_checksum(output):
                return False
    return True


def build_parser() -> argparse.ArgumentParser:
    """Build the public preparation command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--plan", action="store_true", help="print the resolved steps without executing them")
    parser.add_argument("--validate-only", action="store_true", help="validate the existing preparation record")
    parser.add_argument("--force", action="store_true", help="rerun preparation even when the record is valid")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run or display the complete public protocol-data preparation."""
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    steps = build_steps(config, repository_root=REPOSITORY_ROOT, data_root=args.data_root)
    if args.plan:
        print(json.dumps({"id": config["id"], "steps": [step.__dict__ for step in steps]}, indent=2, default=str))
        return 0
    record_path = (
        args.record
        or REPOSITORY_ROOT / "models" / "data" / f"{config['id']}.manifest.json"
    ).resolve()
    valid = _record_is_valid(record_path, config_path, steps)
    if args.validate_only:
        if not valid:
            raise RuntimeError(f"protocol-data preparation record is missing or invalid: {record_path}")
        print(f"validated protocol data: {record_path}")
        return 0
    if valid and not args.force:
        print(f"reused validated protocol data: {record_path}")
        return 0
    for step in steps:
        for command in step.commands:
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    record = _write_record(record_path, config_path, config, steps)
    print(f"prepared and recorded protocol data: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
