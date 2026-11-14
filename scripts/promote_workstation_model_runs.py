#!/usr/bin/env python3
"""Atomically promote validated workstation run bundles to shared storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_bundle(run_dir: Path) -> bool:
    completion_path = run_dir / "completion.json"
    resolved_path = run_dir / "resolved-run.json"
    if not completion_path.is_file() or not resolved_path.is_file():
        return False
    try:
        completion = json.loads(completion_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if completion.get("status") != "complete":
        return False
    if completion.get("resolved_run_sha256") != checksum(resolved_path):
        return False
    artifacts = completion.get("artifacts", [])
    if not artifacts:
        return False
    for artifact in artifacts:
        path = Path(artifact["path"])
        artifact_path = path if path.is_absolute() else run_dir / path
        if not artifact_path.is_file() or checksum(artifact_path) != artifact["sha256"]:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be positive")

    for fold in args.folds:
        relative = Path(args.experiment) / args.model / f"fold-{fold}" / f"seed-{args.seed}"
        source = args.source_root / relative
        destination = args.destination_root / relative

        while not (source / "completion.json").is_file():
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] waiting for {relative}", flush=True)
            time.sleep(args.poll_seconds)
        if not valid_bundle(source):
            raise RuntimeError(f"source completion bundle failed validation: {source}")

        if destination.exists():
            if valid_bundle(destination):
                print(f"already promoted and valid: {destination}", flush=True)
                continue
            raise RuntimeError(f"destination exists but is not valid: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.promoting-{os.getpid()}")
        if temporary.exists():
            raise RuntimeError(f"temporary promotion path already exists: {temporary}")
        print(f"promoting {source} -> {destination}", flush=True)
        shutil.copytree(source, temporary, copy_function=shutil.copy2)
        if not valid_bundle(temporary):
            raise RuntimeError(f"copied bundle failed validation: {temporary}")
        os.replace(temporary, destination)
        print(f"promoted and validated: {destination}", flush=True)


if __name__ == "__main__":
    main()
