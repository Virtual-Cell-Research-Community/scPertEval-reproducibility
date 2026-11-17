"""Materialize population-valued predictions from completed centroid runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import anndata as ad

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_production_benchmark import _tracks
from workflows.evaluate import apply_prediction_evaluation_transform
from workflows.population_lift import (
    lift_prediction_population,
    lifted_manifest,
    manifest_is_current,
)
from workflows.preparation_utils import file_checksum

# The released TxPert workflow uses batch-average controls and therefore yields batch centroids,
# not within-batch cell populations. Lift its mean predictions in the same way as point models.
FORCE_LIFT = {"txpert-public"}


def _atomic_write_h5ad(data: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".h5ad")
    os.close(descriptor)
    try:
        data.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize_one(
    *,
    project_root: Path,
    run_id: str,
    model: str,
    fold_index: int,
    fold_id: str,
    seed: int,
    truth_path: Path,
    split_path: Path,
    truth_identity: str,
    split_sha256: str,
) -> tuple[str, int, bool]:
    run = project_root / "runs" / run_id / model / fold_id / f"seed-{seed}"
    source_path = run / "predictions" / "predictions.h5ad"
    source_manifest_path = source_path.with_suffix(".manifest.json")
    output_path = run / "predictions" / "population-lifted.h5ad"
    output_manifest_path = output_path.with_suffix(".manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_sha256 = source_manifest["artifact"]["sha256"]
    if manifest_is_current(
        output_manifest_path,
        source_prediction_sha256=source_sha256,
        truth_identity=truth_identity,
        split_manifest_sha256=split_sha256,
    ):
        manifest = json.loads(output_manifest_path.read_text())
        return str(output_path), int(manifest["artifact"]["n_obs"]), True

    source = ad.read_h5ad(source_path)
    source = apply_prediction_evaluation_transform(
        source,
        str(source_manifest.get("expression", {}).get("evaluation_transform", "identity")),
    )
    truth = ad.read_h5ad(truth_path, backed="r")
    split = json.loads(split_path.read_text())
    try:
        lifted = lift_prediction_population(truth, source, split, fold_index, seed=seed)
    finally:
        truth.file.close()
    _atomic_write_h5ad(lifted, output_path)
    manifest = lifted_manifest(
        output_path=output_path,
        source_prediction_path=source_path,
        truth_path=truth_path,
        split_manifest_path=split_path,
        model_id=model,
        fold=fold_index,
        seed=seed,
        source_prediction_sha256=source_sha256,
        truth_identity=truth_identity,
        split_manifest_sha256=split_sha256,
    )
    temporary_manifest = output_manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, output_manifest_path)
    return str(output_path), lifted.n_obs, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--track",
        action="append",
        choices=["essential", "genetic-context", "chemical-context"],
    )
    parser.add_argument("--model", action="append", help="restrict to one or more model IDs")
    parser.add_argument("--fold", action="append", help="restrict to one or more fold IDs")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    selected = set(args.track or ["essential", "genetic-context", "chemical-context"])
    tasks = []
    for track_id, track in _tracks(project_root).items():
        if track_id not in selected:
            continue
        truth_manifest = track.truth.with_suffix(".manifest.json")
        if truth_manifest.is_file():
            truth_identity = f"sha256:{json.loads(truth_manifest.read_text())['artifact']['sha256']}"
        else:
            status = track.truth.stat()
            truth_identity = f"stat-v1:{status.st_size}:{status.st_mtime_ns}"
        split_sha256 = file_checksum(track.split_manifest)
        for model in track.models:
            if args.model and model not in set(args.model):
                continue
            for fold_index, fold_id in enumerate(track.folds):
                if args.fold and fold_id not in set(args.fold):
                    continue
                for seed in track.seeds:
                    source = (
                        project_root
                        / "runs"
                        / track.run_id
                        / model
                        / fold_id
                        / f"seed-{seed}"
                        / "predictions"
                        / "predictions.h5ad"
                    )
                    manifest = json.loads(source.with_suffix(".manifest.json").read_text())
                    if manifest["capability"] != "point" and model not in FORCE_LIFT:
                        continue
                    tasks.append(
                        {
                            "project_root": project_root,
                            "run_id": track.run_id,
                            "model": model,
                            "fold_index": fold_index,
                            "fold_id": fold_id,
                            "seed": seed,
                            "truth_path": track.truth,
                            "split_path": track.split_manifest,
                            "truth_identity": truth_identity,
                            "split_sha256": split_sha256,
                        }
                    )

    print(f"discovered {len(tasks)} point or batch-centroid bundles to lift", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(materialize_one, **task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            path, rows, skipped = future.result()
            state = "current" if skipped else "completed"
            print(f"{state} {task['model']}/{task['fold_id']}: {rows} cells -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
