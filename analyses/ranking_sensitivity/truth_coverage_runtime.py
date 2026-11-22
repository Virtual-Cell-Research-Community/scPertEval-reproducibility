"""Truth-self-evaluate frozen Atlas protocols for one benchmark fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd
import scperteval as spe
from scperteval.types import Protocol

from workflows.predictions import expected_test_conditions

from .atlas_runtime import build_protocols, target_key_for_task
from .engine import assert_scperteval_revision

DEFAULT_TASK_MANIFEST = Path("configs/evaluation/atlas-task-protocol-manifest.csv")

REASON_MAP = {
    "metric_undefined": "constant_truth",
    "constant_ground_truth_de_statistic": "constant_truth",
    "no_ground_truth_positive_degs": "no_ground_truth_positives",
    "single_ground_truth_class": "single_ground_truth_class",
    "empty_gene_space": "empty_feature_set",
    "empty_truth_derived_gene_panel": "empty_feature_set",
    "target_gene_not_measured": "target_gene_not_measured",
    "non_finite_de_statistic": "non_finite_ground_truth_de_statistic",
}


def load_truth_fold(
    truth_path: Path,
    split: dict,
    fold: int | str,
    capability: str = "population",
) -> tuple[ad.AnnData, list[str]]:
    """Load exactly one held-out fold plus its matching controls."""
    expected = expected_test_conditions(split, fold, capability)
    source = ad.read_h5ad(truth_path, backed="r")
    try:
        fold_spec = (
            split["folds"][fold]
            if isinstance(fold, int)
            else next(item for item in split["folds"] if str(item.get("id")) == str(fold))
        )
        if "test_context" in fold_spec:
            required = {"context", "evaluation_condition"}
            missing = sorted(required - set(source.obs.columns))
            if missing:
                raise ValueError(f"context-transfer truth is missing obs fields: {', '.join(missing)}")
            context_mask = source.obs["context"].astype(str) == str(fold_spec["test_context"])
            labels = source.obs["evaluation_condition"].astype(str)
            mask = (context_mask & labels.isin(["control", *expected])).to_numpy()
        else:
            labels = source.obs["perturbation"].astype(str)
            mask = labels.isin(["control", *expected]).to_numpy()
        truth = source[mask].to_memory()
    finally:
        source.file.close()
    if "test_context" in fold_spec:
        truth.obs["perturbation"] = truth.obs["evaluation_condition"].astype(str).to_numpy()
    observed = sorted(set(truth.obs["perturbation"].astype(str)) - {"control"})
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(f"truth fold does not match frozen evaluation units: missing={missing}, extra={extra}")
    return truth, expected


def applicable_protocol_ids(task: str, path: Path = DEFAULT_TASK_MANIFEST) -> set[str]:
    manifest = pd.read_csv(path)
    selected = set(manifest.loc[manifest["task"] == task, "protocol_id"].astype(str))
    if not selected:
        raise ValueError(f"task manifest contains no protocols for {task!r}")
    return selected


def task_protocols(task: str) -> list[Protocol]:
    """Resolve the frozen protocols applicable to one task."""
    applicable = applicable_protocol_ids(task)
    protocols = [protocol for protocol in build_protocols() if protocol.name in applicable]
    if {protocol.name for protocol in protocols} != applicable:
        raise ValueError("not every applicable protocol has a concrete runtime implementation")
    return protocols


def evaluate_fold(
    truth: ad.AnnData,
    *,
    task: str,
    fold_label: str,
    expected_units: list[str],
    minimum_cells: int = 50,
    workers: int = 0,
) -> pd.DataFrame:
    """Return the coverage-audit availability schema for one fold."""
    protocols = task_protocols(task)
    prepared = spe.prepare(
        truth,
        protocols,
        min_cells=minimum_cells,
        seed=42,
        control_label="control",
        perturbation_key="perturbation",
        target_key=target_key_for_task(task, truth.obs.columns),
        workers=workers,
        name=f"{task}__{fold_label}",
    )
    result = spe.truth_availability(prepared, protocols, de_method="t-test")
    observed = sorted(result["perturbation"].unique().tolist())
    if observed != sorted(expected_units):
        raise ValueError("scPertEval filtered or added evaluation units relative to the frozen fold")
    result["undefined_reason"] = (
        result["undefined_reason"].fillna("").map(lambda reason: REASON_MAP.get(str(reason), str(reason)))
    )
    return pd.DataFrame(
        {
            "task": task,
            "fold": fold_label,
            "protocol_id": result["protocol"],
            "evaluation_unit": result["perturbation"],
            "ground_truth_defined": result["ground_truth_defined"],
            "undefined_reason": result["undefined_reason"],
            "self_score": result["self_score"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", required=True, help="integer index for k-fold, otherwise the context-fold id")
    parser.add_argument("--fold-label", help="coverage-config label (default: fold argument)")
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_scperteval_revision(Path.cwd())
    split = json.loads(args.split_manifest.read_text())
    fold: int | str = int(args.fold) if args.fold.isdigit() else args.fold
    truth, expected = load_truth_fold(args.truth, split, fold)
    result = evaluate_fold(
        truth,
        task=args.task,
        fold_label=args.fold_label or str(args.fold),
        expected_units=expected,
        minimum_cells=args.minimum_cells,
        workers=args.workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"wrote {len(result):,} protocol/evaluation-unit rows to {args.out}")


if __name__ == "__main__":
    main()
