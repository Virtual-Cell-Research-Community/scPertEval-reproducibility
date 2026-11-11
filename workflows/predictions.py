"""Validate and package one fold-scoped prediction H5AD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from workflows.preparation_utils import file_checksum


def ordered_strings_sha256(values: list[str]) -> str:
    """Hash an ordered sequence of identifiers."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _finite_matrix(matrix, block_rows: int = 4096) -> bool:
    for start in range(0, matrix.shape[0], block_rows):
        block = matrix[start : start + block_rows]
        values = block.data if sp.issparse(block) else np.asarray(block)
        if not np.isfinite(values).all():
            return False
    return True


def _fold(split: dict, fold: int | str) -> dict:
    folds = split["folds"]
    if isinstance(fold, int):
        if fold < 0 or fold >= len(folds):
            raise ValueError(f"fold {fold} is outside 0..{len(folds) - 1}")
        return folds[fold]
    for candidate in folds:
        if str(candidate.get("id")) == str(fold):
            return candidate
    raise ValueError(f"split has no fold {fold!r}")


def _format_number(value: float) -> str:
    return f"{float(value):g}"


def expected_test_conditions(split: dict, fold: int | str, capability: str) -> list[str]:
    """Expand the frozen cohort declaration into canonical test-condition identifiers."""
    fold_spec = _fold(split, fold)
    if "test" in fold_spec:
        return sorted(map(str, fold_spec["test"]))
    context = fold_spec["test_context"]
    if "perturbations" in split:
        return sorted(f"{context}::{perturbation}" for perturbation in split["perturbations"])
    if "drugs" in split and "doses" in split:
        excluded = {
            (item["drug"], float(item["dose"]))
            for item in split.get("population_exclusions", [])
        }
        conditions = []
        for drug in split["drugs"]:
            for dose in split["doses"]:
                if capability == "population" and (drug["label"], float(dose)) in excluded:
                    continue
                conditions.append(f"{context}::{drug['canonical_label']}::{_format_number(dose)}nM")
        return sorted(conditions)
    raise ValueError("split fold does not declare explicit or expandable test conditions")


def validate_prediction_h5ad(
    prediction_path: Path,
    truth_path: Path,
    split: dict,
    fold: int | str,
    capability: str,
    minimum_cells_per_condition: int | None = None,
    maximum_cells_per_condition: int = 100,
) -> tuple[int, int, list[str]]:
    """Validate genes, conditions, row multiplicity, and numeric values."""
    expected = expected_test_conditions(split, fold, capability)
    prediction = ad.read_h5ad(prediction_path, backed="r")
    truth = ad.read_h5ad(truth_path, backed="r")
    try:
        if not prediction.obs_names.is_unique or not prediction.var_names.is_unique:
            raise ValueError("prediction H5AD has duplicate observation or gene identifiers")
        pred_genes = list(map(str, prediction.var_names))
        truth_genes = list(map(str, truth.var_names))
        if set(pred_genes) != set(truth_genes) or len(pred_genes) != len(truth_genes):
            raise ValueError("prediction and truth H5ADs do not contain the same unique gene set")
        if "perturbation" not in prediction.obs:
            raise ValueError("prediction H5AD is missing obs['perturbation']")
        labels = prediction.obs["perturbation"].astype(str)
        observed = sorted(labels.unique().tolist())
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise ValueError(f"prediction conditions do not match test fold; missing={missing}, extra={extra}")
        counts = labels.value_counts()
        if capability == "point":
            if not (counts == 1).all():
                raise ValueError("point prediction bundle must contain exactly one row per test condition")
        elif capability == "population":
            if minimum_cells_per_condition is None:
                raise ValueError("population prediction bundle requires minimum_cells_per_condition")
            if minimum_cells_per_condition > maximum_cells_per_condition:
                raise ValueError("population prediction minimum exceeds the archival maximum")
            below = counts[counts < minimum_cells_per_condition]
            if len(below):
                raise ValueError(f"population prediction bundle has undersized conditions: {below.to_dict()}")
            above = counts[counts > maximum_cells_per_condition]
            if len(above):
                raise ValueError(f"population prediction bundle exceeds archival cap: {above.to_dict()}")
        else:
            raise ValueError(f"unknown prediction capability: {capability}")
        if not _finite_matrix(prediction.X):
            raise ValueError("prediction H5AD contains NaN or infinite values")
        return prediction.n_obs, prediction.n_vars, pred_genes
    finally:
        prediction.file.close()
        truth.file.close()


def build_prediction_manifest(
    *,
    prediction_path: Path,
    truth_path: Path,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    capability: str,
    model_id: str,
    implementation_revision: str,
    adapter_revision: str,
    fold: int | str,
    seed: int,
    environment: str,
    command: str,
    minimum_cells_per_condition: int | None = None,
    maximum_cells_per_condition: int = 100,
    expression_space: str = "log1p_expression",
    evaluation_transform: str = "identity",
    dataset_id: str | None = None,
) -> dict:
    """Validate an artifact and return its schema-v1 archival manifest."""
    split = json.loads(split_manifest_path.read_text())
    n_obs, n_vars, genes = validate_prediction_h5ad(
        prediction_path,
        truth_path,
        split,
        fold,
        capability,
        minimum_cells_per_condition,
        maximum_cells_per_condition,
    )
    dataset_id = dataset_id or split.get("dataset") or dataset_manifest_path.stem
    prediction_id = f"{model_id}-{dataset_id}-{split['id']}-fold-{fold}-seed-{seed}"
    manifest = {
        "schema_version": 1,
        "prediction_id": prediction_id,
        "capability": capability,
        "model": {
            "id": model_id,
            "implementation_revision": implementation_revision,
            "adapter_revision": adapter_revision,
        },
        "dataset": {
            "id": dataset_id,
            "manifest_sha256": file_checksum(dataset_manifest_path),
        },
        "split": {
            "id": split["id"],
            "fold": fold,
            "manifest_sha256": file_checksum(split_manifest_path),
        },
        "expression": {
            "space": expression_space,
            "genes_sha256": ordered_strings_sha256(genes),
            "evaluation_space": "log1p_expression",
            "evaluation_transform": evaluation_transform,
        },
        "artifact": {
            "path": prediction_path.name,
            "format": "h5ad",
            "sha256": file_checksum(prediction_path),
            "n_obs": n_obs,
            "n_vars": n_vars,
            "perturbation_key": "perturbation",
        },
        "provenance": {
            "training_seed": seed,
            "environment": environment,
            "command": command,
        },
    }
    if capability == "population":
        manifest["population"] = {
            "minimum_cells_per_condition": minimum_cells_per_condition,
            "maximum_cells_per_condition": maximum_cells_per_condition,
        }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--capability", choices=["point", "population"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--minimum-cells-per-condition", type=int)
    parser.add_argument("--maximum-cells-per-condition", type=int, default=100)
    parser.add_argument(
        "--expression-space",
        choices=["raw_counts", "normalized_expression", "log1p_expression", "effect"],
        default="log1p_expression",
    )
    parser.add_argument(
        "--evaluation-transform",
        choices=[
            "identity",
            "log1p",
            "normalize-total-10000-log1p",
            "add-held-out-context-control",
            "add-held-out-context-control-then-log1p",
            "add-held-out-context-control-then-normalize-total-10000-log1p",
        ],
        default="identity",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fold: int | str = int(args.fold) if args.fold.isdigit() else args.fold
    manifest = build_prediction_manifest(
        prediction_path=args.prediction,
        truth_path=args.truth,
        dataset_manifest_path=args.dataset_manifest,
        split_manifest_path=args.split_manifest,
        capability=args.capability,
        model_id=args.model_id,
        implementation_revision=args.implementation_revision,
        adapter_revision=args.adapter_revision,
        fold=fold,
        seed=args.seed,
        environment=args.environment,
        command=args.command,
        minimum_cells_per_condition=args.minimum_cells_per_condition,
        maximum_cells_per_condition=args.maximum_cells_per_condition,
        expression_space=args.expression_space,
        evaluation_transform=args.evaluation_transform,
    )
    output = args.output or args.prediction.with_suffix(".manifest.json")
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"validated and wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
