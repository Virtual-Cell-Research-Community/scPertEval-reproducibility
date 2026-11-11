"""Shared, resumable run-directory contract for model adapters and baselines."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from workflows.preparation_utils import file_checksum


@dataclass(frozen=True)
class RunHandle:
    directory: Path
    resolved_sha256: str
    skipped: bool


def _safe_component(value: str | int) -> str:
    component = str(value).strip().replace("/", "-").replace(" ", "-")
    if not component or component in {".", ".."}:
        raise ValueError(f"invalid run identifier component: {value!r}")
    return component


def run_directory(
    root: Path,
    *,
    experiment: str,
    model: str,
    fold: str | int,
    seed: int,
) -> Path:
    """Return the one canonical directory for a model × fold × seed run."""
    return (
        root
        / _safe_component(experiment)
        / _safe_component(model)
        / _safe_component(fold)
        / f"seed-{int(seed)}"
    )


def repository_revision(cwd: Path | None = None) -> str:
    safe_directory = str((cwd or Path.cwd()).resolve())
    result = subprocess.run(
        ["git", "-c", f"safe.directory={safe_directory}", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _artifact_path(run_dir: Path, recorded: str) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else run_dir / path


def completed_run_is_valid(run_dir: Path, *, resolved_sha256: str | None = None) -> bool:
    """Return true only for a complete record whose declared artifacts still match."""
    path = run_dir / "completion.json"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if record.get("status") != "complete":
        return False
    if resolved_sha256 is not None and record.get("resolved_run_sha256") != resolved_sha256:
        return False
    for artifact in record.get("artifacts", []):
        artifact_path = _artifact_path(run_dir, artifact["path"])
        if not artifact_path.is_file() or file_checksum(artifact_path) != artifact["sha256"]:
            return False
    return bool(record.get("artifacts"))


def begin_run(
    root: Path,
    *,
    experiment: str,
    model: str,
    fold: str | int,
    seed: int,
    configuration: dict,
    inputs: dict[str, Path],
    command: list[str],
    repo_root: Path,
    force: bool = False,
) -> RunHandle:
    """Resolve a run and decide whether its verified completed artifacts can be reused."""
    directory = run_directory(
        root,
        experiment=experiment,
        model=model,
        fold=fold,
        seed=seed,
    )
    resolved = {
        "schema_version": 1,
        "identity": {
            "experiment": experiment,
            "model": model,
            "fold": fold,
            "training_seed": int(seed),
        },
        "repository_revision": repository_revision(repo_root),
        "command": command,
        "configuration": configuration,
        "inputs": {
            name: {"path": str(path), "sha256": file_checksum(path)}
            for name, path in sorted(inputs.items())
        },
    }
    content = _json_bytes(resolved)
    resolved_sha256 = _sha256_bytes(content)
    if not force and completed_run_is_valid(directory, resolved_sha256=resolved_sha256):
        return RunHandle(directory=directory, resolved_sha256=resolved_sha256, skipped=True)
    previous = directory / "resolved-run.json"
    if previous.is_file() and file_checksum(previous) != resolved_sha256 and not force:
        raise ValueError(
            f"run directory contains a different resolved configuration: {directory}; "
            "use a new seed/configuration or pass force explicitly"
        )

    for child in ["checkpoints", "logs", "predictions"]:
        (directory / child).mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / "resolved-run.json", content)
    return RunHandle(directory=directory, resolved_sha256=resolved_sha256, skipped=False)


def complete_run(handle: RunHandle, artifacts: dict[str, Path]) -> Path:
    """Write the completion marker last, after checksumming every required artifact."""
    if handle.skipped:
        raise ValueError("cannot complete a run that was skipped")
    if not artifacts:
        raise ValueError("a completed run must declare at least one artifact")
    records = []
    run_root = handle.directory.resolve()
    for role, path in sorted(artifacts.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        try:
            recorded_path = str(resolved.relative_to(run_root))
        except ValueError:
            recorded_path = str(resolved)
        records.append(
            {
                "role": role,
                "path": recorded_path,
                "sha256": file_checksum(path),
                "size_bytes": path.stat().st_size,
            }
        )
    completion = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
        "resolved_run_sha256": handle.resolved_sha256,
        "artifacts": records,
    }
    path = handle.directory / "completion.json"
    _atomic_write(path, _json_bytes(completion))
    return path
