"""Shared archival contract for unseen-perturbation model adapters."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from workflows.predictions import build_prediction_manifest
from workflows.run_contract import RunHandle, begin_run, complete_run, repository_revision


@dataclass(frozen=True)
class PointModelRun:
    """Resolved paths and provenance for one point-prediction model run."""

    handle: RunHandle
    repository_root: Path
    raw_path: Path
    split_manifest_path: Path
    data_manifest_path: Path
    prediction_path: Path
    metrics_path: Path
    checkpoint_dir: Path
    model_id: str
    dataset: str
    fold: int
    seed: int
    environment: str
    command: str

    @property
    def skipped(self) -> bool:
        return self.handle.skipped

    def complete(
        self,
        *,
        implementation_revision: str,
        checkpoints: list[Path] | None = None,
        extra_artifacts: dict[str, Path] | None = None,
        evaluation_transform: str = "identity",
    ) -> Path:
        """Validate predictions, write their manifest, then mark the run complete."""
        prediction_manifest = build_prediction_manifest(
            prediction_path=self.prediction_path,
            truth_path=self.raw_path,
            dataset_manifest_path=self.data_manifest_path,
            split_manifest_path=self.split_manifest_path,
            capability="point",
            model_id=self.model_id,
            implementation_revision=implementation_revision,
            adapter_revision=repository_revision(self.repository_root),
            fold=self.fold,
            seed=self.seed,
            environment=self.environment,
            command=self.command,
            evaluation_transform=evaluation_transform,
        )
        manifest_path = self.prediction_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(prediction_manifest, indent=2) + "\n")

        artifacts: dict[str, Path] = {
            "metrics": self.metrics_path,
            "prediction": self.prediction_path,
            "prediction_manifest": manifest_path,
        }
        for index, checkpoint in enumerate(checkpoints or []):
            artifacts[f"checkpoint_{index}"] = checkpoint
        artifacts.update(extra_artifacts or {})
        return complete_run(
            self.handle,
            artifacts,
            precomputed_checksums={"prediction": prediction_manifest["artifact"]["sha256"]},
        )


def add_point_run_arguments(parser: argparse.ArgumentParser, *, deterministic: bool = False) -> None:
    """Add the common fold/seed/archive options to a model adapter parser."""
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "archival identity for this deterministic run (default: 42)"
            if deterministic
            else "training and archival seed (default: 42)"
        ),
    )
    parser.add_argument("--runs-root", type=Path, help="archive root (default: repository runs/)")
    parser.add_argument(
        "--data-manifest",
        type=Path,
        help="prepared model-view manifest (default: raw.manifest.json when present)",
    )
    parser.add_argument("--force", action="store_true", help="rerun a verified completed run")


def begin_point_model_run(
    args: argparse.Namespace,
    *,
    repository_root: Path,
    model_id: str,
    environment: str,
    configuration: dict,
    deterministic: bool = False,
) -> PointModelRun:
    """Resolve and begin one unseen-perturbation point-model run."""
    if deterministic and args.seed != 42:
        raise ValueError(f"{model_id} is deterministic; use its canonical archival seed 42")

    data_dir = repository_root / "models" / "data" / args.dataset
    raw_path = data_dir / "raw.h5ad"
    split_manifest_path = data_dir / "split_manifest.json"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not split_manifest_path.is_file():
        raise FileNotFoundError(split_manifest_path)
    data_manifest_path = args.data_manifest or raw_path.with_suffix(".manifest.json")
    if not data_manifest_path.is_file():
        # Smoke fixtures predate standalone model-view manifests. Their split manifest
        # still pins the raw H5AD checksum and therefore provides a complete input identity.
        data_manifest_path = split_manifest_path

    runs_root = args.runs_root or repository_root / "runs"
    command = " ".join([sys.executable, *sys.argv])
    handle = begin_run(
        runs_root,
        experiment=args.dataset,
        model=model_id,
        fold=f"fold-{args.fold}",
        seed=args.seed,
        configuration=configuration,
        inputs={
            "model_view_manifest": data_manifest_path,
            "split_manifest": split_manifest_path,
        },
        command=[sys.executable, *sys.argv],
        repo_root=repository_root,
        force=args.force,
    )
    return PointModelRun(
        handle=handle,
        repository_root=repository_root,
        raw_path=raw_path,
        split_manifest_path=split_manifest_path,
        data_manifest_path=data_manifest_path,
        prediction_path=handle.directory / "predictions" / "predictions.h5ad",
        metrics_path=handle.directory / "logs" / "metrics.csv",
        checkpoint_dir=handle.directory / "checkpoints",
        model_id=model_id,
        dataset=args.dataset,
        fold=args.fold,
        seed=args.seed,
        environment=environment,
        command=command,
    )
