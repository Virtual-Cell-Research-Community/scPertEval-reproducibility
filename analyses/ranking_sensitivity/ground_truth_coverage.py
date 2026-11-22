"""Audit protocol coverage using ground-truth-defined availability only.

The input contains every frozen evaluation unit for every applicable protocol.
It must not contain model identifiers, predictions, or model scores. The audit
reports individual protocol coverage and the matched intersection for every
unordered protocol pair within each task and fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CONFIG = Path("configs/evaluation/ranking-sensitivity-atlas.json")
REQUIRED_COLUMNS = {
    "task",
    "fold",
    "protocol_id",
    "evaluation_unit",
    "ground_truth_defined",
    "undefined_reason",
}
FORBIDDEN_COLUMNS = {"model", "prediction", "predicted", "score"}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        values = sorted(normalized[invalid].unique())
        raise ValueError(f"ground_truth_defined contains non-boolean values: {values}")
    return normalized.eq("true")


def expected_fold_sizes(config: dict[str, Any]) -> dict[tuple[str, str], int]:
    return {(str(task["id"]), str(fold)): int(size) for task in config["tasks"] for fold, size in task["folds"].items()}


def validate_availability(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    expected_protocols: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Validate and normalize a complete protocol-by-evaluation-unit table."""

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"availability table is missing columns: {', '.join(missing)}")
    forbidden = sorted(FORBIDDEN_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError("ground-truth coverage input must not contain model outputs: " + ", ".join(forbidden))

    result = frame.copy()
    for column in ["task", "fold", "protocol_id", "evaluation_unit"]:
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"availability table contains an empty {column}")
    result["ground_truth_defined"] = _as_bool(result["ground_truth_defined"])
    result["undefined_reason"] = result["undefined_reason"].fillna("").astype(str).str.strip()

    keys = ["task", "fold", "protocol_id", "evaluation_unit"]
    duplicates = result.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError("availability table contains duplicate protocol/evaluation-unit rows")
    if result.loc[~result["ground_truth_defined"], "undefined_reason"].eq("").any():
        raise ValueError("undefined ground-truth rows require an undefined_reason")
    if result.loc[result["ground_truth_defined"], "undefined_reason"].ne("").any():
        raise ValueError("defined ground-truth rows must not carry an undefined_reason")
    if config is not None:
        allowed_reasons = set(config["ground_truth_coverage"].get("allowed_undefined_reasons", []))
        observed_reasons = set(result.loc[~result["ground_truth_defined"], "undefined_reason"])
        unknown_reasons = sorted(observed_reasons - allowed_reasons) if allowed_reasons else []
        if unknown_reasons:
            raise ValueError(f"availability table contains undeclared undefined reasons: {unknown_reasons}")

    for (task, fold), group in result.groupby(["task", "fold"], sort=False):
        universes = {
            protocol: frozenset(rows["evaluation_unit"]) for protocol, rows in group.groupby("protocol_id", sort=False)
        }
        if len(set(universes.values())) != 1:
            sizes = ", ".join(f"{protocol}={len(units)}" for protocol, units in sorted(universes.items()))
            raise ValueError(f"protocols do not share a complete unit universe for {task}/{fold}: {sizes}")

    if config is not None:
        expected = expected_fold_sizes(config)
        observed_folds = set(zip(result["task"], result["fold"], strict=False))
        unknown = sorted(observed_folds - set(expected))
        if unknown:
            raise ValueError(f"availability table contains undeclared task/fold values: {unknown}")
        missing_folds = sorted(set(expected) - observed_folds)
        if missing_folds:
            raise ValueError(f"availability table is missing declared task/fold values: {missing_folds}")
        for (task, fold), group in result.groupby(["task", "fold"], sort=False):
            observed = int(group["evaluation_unit"].nunique())
            if observed != expected[(task, fold)]:
                raise ValueError(f"{task}/{fold} has {observed} evaluation units, expected {expected[(task, fold)]}")
            if expected_protocols is not None:
                observed_protocols = set(group["protocol_id"])
                declared_protocols = expected_protocols.get(task, set())
                if observed_protocols != declared_protocols:
                    missing_protocols = sorted(declared_protocols - observed_protocols)
                    extra_protocols = sorted(observed_protocols - declared_protocols)
                    raise ValueError(
                        f"{task}/{fold} protocol manifest mismatch: "
                        f"missing={missing_protocols}, extra={extra_protocols}"
                    )
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _reason_counts(group: pd.DataFrame) -> str:
    counts = (
        group.loc[~group["ground_truth_defined"], "undefined_reason"].value_counts(sort=False).sort_index().to_dict()
    )
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


def protocol_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ground-truth availability for every task/fold/protocol."""

    rows = []
    for (task, fold, protocol), group in frame.groupby(["task", "fold", "protocol_id"], sort=True):
        n_total = len(group)
        n_defined = int(group["ground_truth_defined"].sum())
        rows.append(
            {
                "task": task,
                "fold": fold,
                "protocol_id": protocol,
                "n_total": n_total,
                "n_defined": n_defined,
                "defined_fraction": n_defined / n_total,
                "undefined_reason_counts": _reason_counts(group),
            }
        )
    return pd.DataFrame(rows)


def pairwise_coverage_by_fold(frame: pd.DataFrame, candidate_threshold: float) -> pd.DataFrame:
    """Return the matched common ground-truth set for every protocol pair."""

    if not 0 < candidate_threshold <= 1:
        raise ValueError("candidate_threshold must be in (0, 1]")
    rows = []
    for (task, fold), group in frame.groupby(["task", "fold"], sort=True):
        availability = group.pivot(
            index="evaluation_unit", columns="protocol_id", values="ground_truth_defined"
        ).sort_index()
        protocols = sorted(str(protocol) for protocol in availability.columns)
        n_total = len(availability)
        for protocol_a, protocol_b in combinations(protocols, 2):
            defined_a = availability[protocol_a].to_numpy(bool)
            defined_b = availability[protocol_b].to_numpy(bool)
            common = defined_a & defined_b
            fraction = float(common.mean())
            rows.append(
                {
                    "task": task,
                    "fold": fold,
                    "protocol_a": protocol_a,
                    "protocol_b": protocol_b,
                    "n_total": n_total,
                    "n_a_defined": int(defined_a.sum()),
                    "n_b_defined": int(defined_b.sum()),
                    "n_common": int(common.sum()),
                    "common_fraction": fraction,
                    "meets_candidate_threshold": fraction >= candidate_threshold,
                }
            )
    return pd.DataFrame(rows)


def summarize_protocol_pairs(fold_coverage: pd.DataFrame, candidate_threshold: float) -> pd.DataFrame:
    """Aggregate pair coverage across folds without hiding a weak held-out context."""

    rows = []
    for (task, protocol_a, protocol_b), group in fold_coverage.groupby(["task", "protocol_a", "protocol_b"], sort=True):
        n_total = int(group["n_total"].sum())
        n_common = int(group["n_common"].sum())
        minimum = float(group["common_fraction"].min())
        rows.append(
            {
                "task": task,
                "protocol_a": protocol_a,
                "protocol_b": protocol_b,
                "n_folds": len(group),
                "n_total": n_total,
                "n_common": n_common,
                "common_fraction": n_common / n_total,
                "minimum_fold_common_fraction": minimum,
                "meets_candidate_threshold_all_folds": minimum >= candidate_threshold,
            }
        )
    return pd.DataFrame(rows)


def coverage_summary(
    pair_summary: pd.DataFrame,
    candidate_threshold: float,
    *,
    candidate_is_frozen: bool = False,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task, group in pair_summary.groupby("task", sort=True):
        values = group["minimum_fold_common_fraction"].to_numpy(float)
        tasks[str(task)] = {
            "n_protocol_pairs": len(group),
            "n_meeting_candidate_threshold": int((values >= candidate_threshold).sum()),
            "fraction_meeting_candidate_threshold": float((values >= candidate_threshold).mean()),
            "minimum": float(values.min()),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "maximum": float(values.max()),
        }
    return {
        "schema_version": 1,
        "candidate_minimum_common_fraction": candidate_threshold,
        "candidate_is_frozen": candidate_is_frozen,
        "tasks": tasks,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve_repo_path(config_path: Path, declared_path: str) -> Path:
    path = Path(declared_path)
    if path.is_absolute():
        return path
    repository_root = config_path.resolve().parents[2]
    return repository_root / path


def load_task_protocol_manifest(config: dict[str, Any], config_path: Path) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Load and checksum the frozen task-specific protocol declaration."""

    declaration = config["protocol_universe"]
    path = _resolve_repo_path(config_path, declaration["task_manifest"])
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha256 != declaration["task_manifest_sha256"]:
        raise ValueError(
            "task protocol manifest checksum mismatch: "
            f"observed={observed_sha256}, expected={declaration['task_manifest_sha256']}"
        )
    manifest = pd.read_csv(path, dtype=str).fillna("")
    required = {"task", "protocol_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"task protocol manifest is missing columns: {', '.join(missing)}")
    if manifest.duplicated(["task", "protocol_id"]).any():
        raise ValueError("task protocol manifest contains duplicate task/protocol rows")
    expected = {
        str(task): set(group["protocol_id"].astype(str)) for task, group in manifest.groupby("task", sort=False)
    }
    return manifest, expected


def run_audit(availability_path: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    expected_protocols = None
    if "protocol_universe" in config:
        _, expected_protocols = load_task_protocol_manifest(config, config_path)
    frame = validate_availability(
        pd.read_csv(availability_path),
        config,
        expected_protocols=expected_protocols,
    )
    threshold = float(config["ground_truth_coverage"]["candidate_minimum_common_fraction"])
    individual = protocol_coverage(frame)
    pairs_by_fold = pairwise_coverage_by_fold(frame, threshold)
    pair_summary = summarize_protocol_pairs(pairs_by_fold, threshold)
    summary = coverage_summary(
        pair_summary,
        threshold,
        candidate_is_frozen=bool(config["ground_truth_coverage"].get("candidate_is_frozen", False)),
    )
    summary["inputs"] = {
        "availability": str(availability_path),
        "availability_sha256": hashlib.sha256(availability_path.read_bytes()).hexdigest(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }

    _atomic_write(output_dir / "protocol-coverage-by-fold.csv", individual.to_csv(index=False).encode())
    _atomic_write(output_dir / "pairwise-coverage-by-fold.csv", pairs_by_fold.to_csv(index=False).encode())
    _atomic_write(output_dir / "pairwise-coverage-summary.csv", pair_summary.to_csv(index=False).encode())
    _atomic_write(
        output_dir / "coverage-summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.availability, args.config, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
