"""Evaluate every completed production prediction bundle and build rankings.

The script is deliberately restart-safe: each bundle writes a CSV plus a small
completion record atomically, and subsequent invocations skip outputs whose
inputs and output checksum still match. Tracks can be evaluated independently
on different machines and aggregated once all detail files are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from workflows.evaluate import (
    compare_to_reference,
    evaluate_bundle,
    paired_perturbation_bootstrap,
    rank_models,
    summarize_evaluations,
)
from workflows.preparation_utils import file_checksum


@dataclass(frozen=True)
class Track:
    id: str
    run_id: str
    truth: Path
    split_manifest: Path
    experiment_manifest: Path
    folds: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationJob:
    track: str
    model: str
    fold_id: str
    fold_index: int
    seed: int
    capability: str
    truth: Path
    split_manifest: Path
    grid: Path
    prediction: Path
    prediction_manifest: Path
    output: Path


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode())


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracks(project_root: Path) -> dict[str, Track]:
    experiments = project_root / "configs" / "experiments"
    model_data = project_root / "models" / "data"

    essential_experiment = experiments / "replogle-weissman-2022-k562-essential-perturbation-generalization.json"
    essential = _json(essential_experiment)
    essential_split = model_data / "replogle-weissman-2022-k562-essential" / "split_manifest.json"

    genetic_experiment = experiments / "replogle-nadig-four-context-50-cell.json"
    genetic = _json(genetic_experiment)
    genetic_folds = tuple(fold["id"] for fold in genetic["folds"])

    chemical_experiment = experiments / "sciplex3-nine-drug-context-transfer.json"
    chemical = _json(chemical_experiment)
    chemical_folds = tuple(fold["id"] for fold in chemical["folds"])

    return {
        "essential": Track(
            id="essential",
            run_id="replogle-weissman-2022-k562-essential",
            truth=model_data / "replogle-weissman-2022-k562-essential" / "raw.h5ad",
            split_manifest=essential_split,
            experiment_manifest=essential_experiment,
            folds=tuple(f"fold-{index}" for index in range(5)),
            models=(*essential["models"], *essential["baselines"]),
            seeds=tuple(essential["training_seeds"]),
        ),
        "genetic-context": Track(
            id="genetic-context",
            run_id=genetic["id"],
            truth=model_data / genetic["id"] / "raw.h5ad",
            split_manifest=genetic_experiment,
            experiment_manifest=genetic_experiment,
            folds=genetic_folds,
            models=tuple(genetic["models"]),
            seeds=tuple(genetic["training_seeds"]),
        ),
        "chemical-context": Track(
            id="chemical-context",
            run_id=chemical["id"],
            truth=model_data / chemical["id"] / "raw.h5ad",
            split_manifest=chemical_experiment,
            experiment_manifest=chemical_experiment,
            folds=chemical_folds,
            models=tuple(chemical["models"]),
            seeds=tuple(chemical["training_seeds"]),
        ),
    }


def _completion_record(job: EvaluationJob) -> dict:
    return {
        "schema_version": 1,
        "status": "complete",
        "track": job.track,
        "model": job.model,
        "fold": job.fold_id,
        "fold_index": job.fold_index,
        "training_seed": job.seed,
        "capability": job.capability,
        "inputs": {
            "prediction_sha256": _json(job.prediction_manifest)["artifact"]["sha256"],
            "prediction_manifest_sha256": _manifest_sha256(job.prediction_manifest),
            "split_manifest_sha256": _manifest_sha256(job.split_manifest),
            "evaluation_grid_sha256": _manifest_sha256(job.grid),
        },
    }


def _completion_path(output: Path) -> Path:
    return output.with_suffix(".complete.json")


def _output_is_current(job: EvaluationJob) -> bool:
    completion = _completion_path(job.output)
    if not job.output.is_file() or not completion.is_file():
        return False
    try:
        record = _json(completion)
        expected = _completion_record(job)
    except (KeyError, OSError, json.JSONDecodeError):
        return False
    output_sha256 = record.pop("output_sha256", None)
    return record == expected and output_sha256 == file_checksum(job.output)


def discover_jobs(
    project_root: Path,
    results_root: Path,
    grid: Path,
    selected_tracks: set[str],
) -> list[EvaluationJob]:
    jobs: list[EvaluationJob] = []
    for track_id, track in _tracks(project_root).items():
        if track_id not in selected_tracks:
            continue
        for model in track.models:
            for fold_index, fold_id in enumerate(track.folds):
                for seed in track.seeds:
                    run = project_root / "runs" / track.run_id / model / fold_id / f"seed-{seed}"
                    completion = run / "completion.json"
                    prediction = run / "predictions" / "predictions.h5ad"
                    prediction_manifest = prediction.with_suffix(".manifest.json")
                    if not completion.is_file() or _json(completion).get("status") != "complete":
                        raise FileNotFoundError(f"run is not complete: {run}")
                    if not prediction.is_file() or not prediction_manifest.is_file():
                        raise FileNotFoundError(f"run has no prediction bundle: {run}")
                    manifest = _json(prediction_manifest)
                    jobs.append(
                        EvaluationJob(
                            track=track_id,
                            model=model,
                            fold_id=fold_id,
                            fold_index=fold_index,
                            seed=seed,
                            capability=manifest["capability"],
                            truth=track.truth,
                            split_manifest=track.split_manifest,
                            grid=grid,
                            prediction=prediction,
                            prediction_manifest=prediction_manifest,
                            output=(
                                results_root
                                / "details"
                                / track_id
                                / model
                                / fold_id
                                / f"seed-{seed}.csv"
                            ),
                        )
                    )
    return jobs


def evaluate_job(job: EvaluationJob) -> tuple[str, int, bool]:
    if _output_is_current(job):
        return str(job.output), len(pd.read_csv(job.output)), True

    detail = evaluate_bundle(
        truth_path=job.truth,
        prediction_path=job.prediction,
        split_manifest_path=job.split_manifest,
        grid_path=job.grid,
        model_id=job.model,
        capability=job.capability,
        fold=job.fold_index,
        seed=job.seed,
    )
    detail["track"] = job.track
    detail["fold_id"] = job.fold_id
    detail["training_seed"] = job.seed
    split = _json(job.split_manifest)
    fold = split["folds"][job.fold_index]
    detail["held_out_context"] = fold.get("test_context")

    _atomic_csv(detail, job.output)
    completion = _completion_record(job)
    completion["output_sha256"] = file_checksum(job.output)
    _atomic_bytes(
        _completion_path(job.output),
        (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode(),
    )
    return str(job.output), len(detail), False


def aggregate(results_root: Path, jobs: list[EvaluationJob], *, bootstrap_replicates: int) -> None:
    missing = [job.output for job in jobs if not _output_is_current(job)]
    if missing:
        raise RuntimeError(f"cannot aggregate with {len(missing)} missing or stale detail outputs")
    detail = pd.concat([pd.read_csv(job.output) for job in jobs], ignore_index=True)
    summary = summarize_evaluations(detail)
    ranking = rank_models(summary)
    rank_changes, comparisons = compare_to_reference(summary)
    bootstrap_ranks, bootstrap_comparisons = paired_perturbation_bootstrap(
        detail,
        n_bootstrap=bootstrap_replicates,
        seed=42,
    )
    frames = {
        "detail.csv": detail,
        "summary.csv": summary,
        "rankings.csv": ranking,
        "rank-changes.csv": rank_changes,
        "comparisons.csv": comparisons,
        "bootstrap-rank-changes.csv": bootstrap_ranks,
        "bootstrap-comparisons.csv": bootstrap_comparisons,
    }
    for name, frame in frames.items():
        _atomic_csv(frame, results_root / name)
    record = {
        "schema_version": 1,
        "status": "complete",
        "n_bundles": len(jobs),
        "n_detail_rows": len(detail),
        "bootstrap_replicates": bootstrap_replicates,
        "outputs": {name: file_checksum(results_root / name) for name in frames},
    }
    _atomic_bytes(
        results_root / "completion.json",
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--grid", type=Path)
    parser.add_argument(
        "--track",
        action="append",
        choices=["essential", "genetic-context", "chemical-context"],
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results_root = args.results_root.resolve()
    grid = (args.grid or project_root / "configs" / "evaluation" / "ranking-sensitivity-v1.json").resolve()
    selected_tracks = set(args.track or ["essential", "genetic-context", "chemical-context"])
    jobs = discover_jobs(project_root, results_root, grid, selected_tracks)
    print(f"discovered {len(jobs)} evaluation bundles across {', '.join(sorted(selected_tracks))}", flush=True)

    if not args.aggregate_only:
        pending = [job for job in jobs if not _output_is_current(job)]
        print(f"{len(jobs) - len(pending)} current; {len(pending)} pending", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(evaluate_job, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                path, rows, skipped = future.result()
                state = "skipped" if skipped else "completed"
                print(f"{state} {job.track}/{job.model}/{job.fold_id}/seed-{job.seed}: {rows} rows -> {path}", flush=True)

    if args.aggregate or args.aggregate_only:
        aggregate(results_root, jobs, bootstrap_replicates=args.bootstrap_replicates)
        print(f"wrote aggregate ranking-sensitivity results to {results_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
