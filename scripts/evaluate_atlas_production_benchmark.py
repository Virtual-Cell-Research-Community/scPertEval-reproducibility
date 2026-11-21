"""Evaluate retained production predictions with frozen Atlas-derived protocols.

The unit of execution is one benchmark fold. Truth-derived DE, panel, and PCA state is prepared
once and reused across every selected model. Each model writes an atomic detail CSV and completion
record, so rerunning the command skips checksum-valid outputs without retraining or regenerating
predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analyses.ranking_sensitivity.prediction_runtime import (
    evaluate_prediction,
    prepare_prediction_fold,
    task_protocols,
)
from analyses.ranking_sensitivity.truth_coverage_runtime import load_truth_fold
from workflows.evaluate import _load_prediction_for_evaluation
from workflows.preparation_utils import file_checksum


@dataclass(frozen=True)
class Track:
    key: str
    task: str
    run_id: str
    truth: Path
    split_manifest: Path
    experiment_manifest: Path
    fold_ids: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def tracks(project_root: Path, model_data_root: Path | None = None) -> dict[str, Track]:
    experiments = project_root / "configs" / "experiments"
    model_data = model_data_root or project_root / "models" / "data"
    specs = {
        "essential": (
            "essential_perturbation_generalization",
            "replogle-weissman-2022-k562-essential",
            "replogle-weissman-2022-k562-essential-perturbation-generalization.json",
        ),
        "genetic-context": (
            "genetic_context_transfer",
            "replogle-nadig-four-context-50-cell",
            "replogle-nadig-four-context-50-cell.json",
        ),
        "chemical-context": (
            "chemical_context_transfer",
            "sciplex3-nine-drug-context-transfer",
            "sciplex3-nine-drug-context-transfer.json",
        ),
    }
    resolved: dict[str, Track] = {}
    for key, (task, run_id, filename) in specs.items():
        experiment = experiments / filename
        config = _json(experiment)
        fold_ids = tuple(fold["id"] for fold in config.get("folds", []))
        if not fold_ids:
            fold_ids = tuple(f"fold-{index}" for index in range(5))
        split_manifest = experiment
        if key == "essential":
            split_manifest = model_data / run_id / "split_manifest.json"
        resolved[key] = Track(
            key=key,
            task=task,
            run_id=run_id,
            truth=model_data / run_id / "raw.h5ad",
            split_manifest=split_manifest,
            experiment_manifest=experiment,
            fold_ids=fold_ids,
            models=(*config["models"], *config.get("baselines", [])),
            seeds=tuple(config["training_seeds"]),
        )
    return resolved


def _run_directory(runs_root: Path, track: Track, model: str, fold_id: str, seed: int) -> Path:
    return runs_root / track.run_id / model / fold_id / f"seed-{seed}"


def _prediction_bundle(run: Path) -> tuple[Path, Path]:
    native = run / "predictions" / "predictions.h5ad"
    lifted = run / "predictions" / "population-lifted.h5ad"
    prediction = lifted if lifted.is_file() else native
    manifest = prediction.with_suffix(".manifest.json")
    if not prediction.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"run has no complete evaluation prediction bundle: {run}")
    return prediction, manifest


def _completion_record(
    *,
    track: Track,
    model: str,
    fold_id: str,
    seed: int,
    prediction_manifest: Path,
    output: Path,
    project_root: Path,
) -> dict:
    prediction = _json(prediction_manifest)
    return {
        "schema_version": 1,
        "status": "complete",
        "task": track.task,
        "model": model,
        "fold": fold_id,
        "training_seed": seed,
        "inputs": {
            "prediction_sha256": prediction["artifact"]["sha256"],
            "prediction_manifest_sha256": _sha256(prediction_manifest),
            "split_manifest_sha256": _sha256(track.split_manifest),
            "experiment_manifest_sha256": _sha256(track.experiment_manifest),
            "eligible_families_sha256": _sha256(
                project_root / "configs/evaluation/atlas-protocol-eligibility-v1.csv"
            ),
            "task_protocol_manifest_sha256": _sha256(
                project_root / "configs/evaluation/atlas-task-protocol-manifest-v1.csv"
            ),
        },
        "output_sha256": file_checksum(output),
    }


def _output_is_current(
    *,
    track: Track,
    model: str,
    fold_id: str,
    seed: int,
    prediction_manifest: Path,
    output: Path,
    project_root: Path,
) -> bool:
    completion = output.with_suffix(".complete.json")
    if not output.is_file() or not completion.is_file():
        return False
    try:
        observed = _json(completion)
        expected = _completion_record(
            track=track,
            model=model,
            fold_id=fold_id,
            seed=seed,
            prediction_manifest=prediction_manifest,
            output=output,
            project_root=project_root,
        )
    except (KeyError, OSError, json.JSONDecodeError):
        return False
    return observed == expected


def evaluate_fold(
    *,
    project_root: Path,
    runs_root: Path,
    results_root: Path,
    track: Track,
    fold_id: str,
    selected_models: set[str] | None,
    workers: int,
) -> None:
    if fold_id not in track.fold_ids:
        raise ValueError(f"unknown fold {fold_id!r} for {track.key}; expected {track.fold_ids}")
    models = [model for model in track.models if selected_models is None or model in selected_models]
    unknown = sorted((selected_models or set()) - set(track.models))
    if unknown:
        raise ValueError(f"models are not part of {track.key}: {unknown}")
    pending: list[tuple[str, int, Path, Path, Path]] = []
    for model in models:
        for seed in track.seeds:
            run = _run_directory(runs_root, track, model, fold_id, seed)
            completion = run / "completion.json"
            if not completion.is_file() or _json(completion).get("status") != "complete":
                raise FileNotFoundError(f"production run is not complete: {run}")
            prediction_path, prediction_manifest = _prediction_bundle(run)
            output = results_root / "details" / track.task / model / fold_id / f"seed-{seed}.csv"
            if _output_is_current(
                track=track,
                model=model,
                fold_id=fold_id,
                seed=seed,
                prediction_manifest=prediction_manifest,
                output=output,
                project_root=project_root,
            ):
                print(f"current {track.task}/{model}/{fold_id}/seed-{seed}", flush=True)
                continue
            pending.append((model, seed, prediction_path, prediction_manifest, output))
    if not pending:
        return

    fold_index = track.fold_ids.index(fold_id)
    split = _json(track.split_manifest)
    truth, expected_units = load_truth_fold(track.truth, split, fold_index if track.key == "essential" else fold_id)
    protocols = task_protocols(track.task)
    prepared = prepare_prediction_fold(
        truth,
        task=track.task,
        fold_label=fold_id,
        protocols=protocols,
        workers=workers,
    )
    fold_spec = split["folds"][fold_index]
    held_out_context = fold_spec.get("test_context")

    for model, seed, prediction_path, prediction_manifest, output in pending:
        prediction = _load_prediction_for_evaluation(prediction_path)
        detail = evaluate_prediction(
            prepared,
            prediction,
            task=track.task,
            fold_label=fold_id,
            model=model,
            training_seed=seed,
            expected_units=expected_units,
            protocols=protocols,
            held_out_context=held_out_context,
        )
        _atomic_csv(detail, output)
        record = _completion_record(
            track=track,
            model=model,
            fold_id=fold_id,
            seed=seed,
            prediction_manifest=prediction_manifest,
            output=output,
            project_root=project_root,
        )
        _atomic_bytes(
            output.with_suffix(".complete.json"),
            (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
        )
        print(
            f"completed {track.task}/{model}/{fold_id}/seed-{seed}: {len(detail):,} rows",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--model-data-root",
        type=Path,
        help="directory containing prepared dataset subdirectories (default: PROJECT_ROOT/models/data)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="authoritative production runs directory (default: PROJECT_ROOT/runs)",
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--track", choices=["essential", "genetic-context", "chemical-context"], required=True)
    parser.add_argument("--fold", action="append", help="fold id; repeat to select multiple (default: every fold)")
    parser.add_argument("--model", action="append", help="model id; repeat to select multiple (default: every model)")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    model_data_root = args.model_data_root.resolve() if args.model_data_root else None
    runs_root = args.runs_root.resolve() if args.runs_root else project_root / "runs"
    track = tracks(project_root, model_data_root)[args.track]
    folds = args.fold or list(track.fold_ids)
    for fold_id in folds:
        evaluate_fold(
            project_root=project_root,
            runs_root=runs_root,
            results_root=args.results_root.resolve(),
            track=track,
            fold_id=fold_id,
            selected_models=set(args.model) if args.model else None,
            workers=args.workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
