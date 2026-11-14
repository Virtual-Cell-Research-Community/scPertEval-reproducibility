"""Archive leakage-free point baselines for perturbation generalization."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import anndata as ad

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.baselines.baselines import (
    _tech_dup_halves,
    fold_mean_baseline,
    interpolated_baseline,
    linear_baseline,
    no_change_baseline,
)
from workflows.predictions import build_prediction_manifest
from workflows.run_contract import begin_run, complete_run, repository_revision

MODEL_FUNCTIONS = {
    "no-change": no_change_baseline,
    "fold-mean": fold_mean_baseline,
    "linear": linear_baseline,
    "interpolated-technical-duplicate": interpolated_baseline,
}
DEFAULT_MODELS = ("no-change", "fold-mean", "linear")


def _load_fold(path: Path) -> dict[str, list[str]]:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _fold_genes(values: list[str]) -> set[str]:
    return {value.split("+")[0] for value in values if value != "ctrl"}


def validate_fold_conversion(fold: dict[str, list[str]], manifest_fold: dict) -> None:
    """Prove the model-native pickle represents the frozen manifest fold."""
    for split_name in ("train", "val", "test"):
        if _fold_genes(fold[split_name]) != set(manifest_fold[split_name]):
            raise ValueError(f"model-native {split_name} split differs from frozen manifest")


def _development_fold(fold: dict[str, list[str]]) -> dict[str, list[str]]:
    """Expose train plus validation to methods that perform no model selection."""
    return {**fold, "train": [*fold["train"], *fold["val"]]}


def predict_perturbation_baseline(
    model: str,
    data: ad.AnnData,
    fold: dict[str, list[str]],
    *,
    seed: int = 42,
    pca_dim: int = 10,
    ridge: float = 0.1,
    technical_duplicate_halves: dict | None = None,
) -> ad.AnnData:
    """Compute one baseline using the full non-test development pool."""
    development_fold = _development_fold(fold)
    if model == "no-change":
        return no_change_baseline(data, development_fold)
    if model == "fold-mean":
        return fold_mean_baseline(data, development_fold)
    if model == "linear":
        return linear_baseline(
            data,
            development_fold,
            pca_dim=pca_dim,
            ridge=ridge,
            seed=seed,
        )
    if model == "interpolated-technical-duplicate":
        if technical_duplicate_halves is None:
            raise ValueError("technical-duplicate halves are required for the oracle control")
        return interpolated_baseline(data, development_fold, technical_duplicate_halves)
    raise ValueError(f"unknown perturbation baseline: {model}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="replogle-weissman-2022-k562-essential",
        help="models/data/<dataset> prepared model view",
    )
    parser.add_argument("--runs-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--fold", type=int, action="append", help="fold to run; repeatable (default: all)")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_FUNCTIONS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--seed", type=int, default=42, help="deterministic archival seed")
    parser.add_argument("--pca-dim", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("perturbation baselines are deterministic; use archival seed 42")

    data_dir = REPOSITORY_ROOT / "models" / "data" / args.dataset
    raw_path = data_dir / "raw.h5ad"
    split_manifest_path = data_dir / "split_manifest.json"
    data_manifest_path = raw_path.with_suffix(".manifest.json")
    if not data_manifest_path.is_file():
        data_manifest_path = split_manifest_path
    split = json.loads(split_manifest_path.read_text())
    fold_indices = args.fold or list(range(len(split["folds"])))
    if any(index < 0 or index >= len(split["folds"]) for index in fold_indices):
        raise ValueError(f"folds must be in 0..{len(split['folds']) - 1}")

    revision = repository_revision(REPOSITORY_ROOT)
    command = [sys.executable, *sys.argv]
    handles = {}
    native_folds = {}
    for fold_index in fold_indices:
        native_fold = _load_fold(data_dir / "folds" / f"fold_{fold_index}.pkl")
        validate_fold_conversion(native_fold, split["folds"][fold_index])
        native_folds[fold_index] = native_fold
        for model in args.models:
            handles[(fold_index, model)] = begin_run(
                args.runs_root,
                experiment=args.dataset,
                model=model,
                fold=f"fold-{fold_index}",
                seed=args.seed,
                configuration={
                    "deterministic": True,
                    "development_pool": "train-plus-validation",
                    "pca_dim": args.pca_dim if model == "linear" else None,
                    "ridge": args.ridge if model == "linear" else None,
                    "oracle_control": model == "interpolated-technical-duplicate",
                },
                inputs={
                    "model_view_manifest": data_manifest_path,
                    "split_manifest": split_manifest_path,
                },
                command=command,
                repo_root=REPOSITORY_ROOT,
                force=args.force,
            )

    pending = {key for key, handle in handles.items() if not handle.skipped}
    if not pending:
        print("all requested perturbation baseline runs are already checksum-valid")
        return 0

    data = ad.read_h5ad(raw_path)
    halves = (
        _tech_dup_halves(data) if any(model == "interpolated-technical-duplicate" for _, model in pending) else None
    )
    for fold_index in fold_indices:
        fold = native_folds[fold_index]
        for model in args.models:
            handle = handles[(fold_index, model)]
            if handle.skipped:
                print(f"skipping verified completed run: {handle.directory}")
                continue
            prediction = predict_perturbation_baseline(
                model,
                data,
                fold,
                seed=args.seed,
                pca_dim=args.pca_dim,
                ridge=args.ridge,
                technical_duplicate_halves=halves,
            )

            output = handle.directory / "predictions" / "predictions.h5ad"
            output.parent.mkdir(parents=True, exist_ok=True)
            prediction.write_h5ad(output, compression="gzip")
            manifest = build_prediction_manifest(
                prediction_path=output,
                truth_path=raw_path,
                dataset_manifest_path=data_manifest_path,
                split_manifest_path=split_manifest_path,
                capability="point",
                model_id=model,
                implementation_revision=revision,
                adapter_revision=revision,
                fold=fold_index,
                seed=args.seed,
                environment="uv.lock",
                command=" ".join(command),
            )
            manifest_path = output.with_suffix(".manifest.json")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            complete_run(
                handle,
                {"prediction": output, "prediction_manifest": manifest_path},
                precomputed_checksums={"prediction": manifest["artifact"]["sha256"]},
            )
            print(f"wrote {prediction.shape} {model} predictions for fold {fold_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
