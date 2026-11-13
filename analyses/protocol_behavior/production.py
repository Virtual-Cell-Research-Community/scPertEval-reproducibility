"""Manifest-driven production stages for the protocol-behavior study."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scperteval as spe

DEFAULT_MANIFEST = Path("configs/protocol_behavior/production-v1.json")
DEFAULT_RESULTS = Path("results/protocol_behavior/production-v1")


@dataclass(frozen=True)
class Stage:
    id: str
    kind: str
    dataset: str | None = None
    method: str | None = None
    reference: str | None = None


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    required = {
        "schema_version",
        "id",
        "seed",
        "minimum_cells",
        "reference_subsample",
        "protocols",
        "gpu_protocols",
        "de",
        "datasets",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"protocol manifest is missing: {', '.join(missing)}")
    if manifest["schema_version"] != 1:
        raise ValueError(f"unsupported protocol manifest schema: {manifest['schema_version']}")
    ids = [dataset["id"] for dataset in manifest["datasets"]]
    if len(ids) != len(set(ids)):
        raise ValueError("protocol manifest contains duplicate dataset ids")
    if not manifest["protocols"]:
        raise ValueError("protocol manifest must declare at least one protocol")
    return manifest


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_stages(manifest: dict[str, Any]) -> list[Stage]:
    stages: list[Stage] = []
    for dataset in manifest["datasets"]:
        dataset_id = dataset["id"]
        for reference in manifest["de"]["references"]:
            for method in manifest["de"]["methods"]:
                stages.append(Stage(f"de.{dataset_id}.{reference}.{method}", "de", dataset_id, method, reference))
        stages.append(Stage(f"drf.cpu.{dataset_id}", "drf-cpu", dataset_id))
        if manifest["gpu_protocols"]:
            stages.append(Stage(f"drf.gpu.{dataset_id}", "drf-gpu", dataset_id))
    return stages


def _max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024**2 if __import__("platform").system() == "Linux" else 1024**3)


def _dataset_spec(manifest: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [item for item in manifest["datasets"] if item["id"] == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"unknown production dataset {dataset_id!r}")
    return matches[0]


def _load_analysis_unit(spec: dict[str, Any]) -> ad.AnnData:
    path = Path(spec["path"])
    data_root = os.environ.get("SCPERTEVAL_DATA_ROOT")
    if data_root and not path.is_absolute():
        path = Path(data_root) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    data = ad.read_h5ad(path)
    if "context_key" in spec:
        key = spec["context_key"]
        if key not in data.obs:
            raise ValueError(f"{path} is missing context field {key!r}")
        mask = data.obs[key].astype(str) == str(spec["context_value"])
        if not mask.any():
            raise ValueError(f"{path} has no cells for {key}={spec['context_value']!r}")
        data = data[mask].copy()
    required = {"perturbation", "is_control"}
    missing = sorted(required - set(data.obs))
    if missing:
        raise ValueError(f"{path} is missing analysis fields: {', '.join(missing)}")
    data.obs["perturbation"] = np.where(
        data.obs["is_control"].astype(bool),
        "control",
        data.obs["perturbation"].astype(str),
    )
    return data


def _eligible_perturbations(data: ad.AnnData, minimum_cells: int) -> list[str]:
    labels = data.obs["perturbation"].astype(str)
    controls = data.obs["is_control"].astype(bool)
    counts = labels[~controls].value_counts()
    return sorted(counts[counts >= minimum_cells].index.astype(str))


def _record_path(output: Path, stage: Stage) -> Path:
    return output / "stages" / stage.id / "stage.json"


def _write_record(path: Path, record: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def _valid_de(path: Path, stage: Stage, perturbations: list[str]) -> bool:
    if not path.is_file() or stage.method is None or stage.reference is None:
        return False
    try:
        with h5py.File(path) as handle:
            observed = [value.decode() for value in handle["perturbations"][:]]
            return (
                handle.attrs["target"] == "gt_all_cells"
                and handle.attrs["reference"] == stage.reference
                and observed == perturbations
                and handle[stage.method]["statistic"].shape == (len(perturbations), len(handle["genes"]))
                and handle[stage.method]["pvalue_adj"].shape == (len(perturbations), len(handle["genes"]))
            )
    except (KeyError, OSError, ValueError):
        return False


def _reusable_record(record_path: Path, manifest_digest: str, artifacts: list[Path]) -> bool:
    if not record_path.is_file() or any(not artifact.is_file() for artifact in artifacts):
        return False
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = record.get("artifact_sha256", {})
    return (
        record.get("status") == "complete"
        and record.get("manifest_sha256") == manifest_digest
        and set(expected) == {str(artifact) for artifact in artifacts}
        and all(expected[str(artifact)] == file_sha256(artifact) for artifact in artifacts)
    )


def run_de_stage(
    manifest: dict[str, Any],
    stage: Stage,
    output: Path,
    *,
    workers: int,
    manifest_digest: str,
    force: bool = False,
) -> Path:
    assert stage.dataset and stage.method and stage.reference
    spec = _dataset_spec(manifest, stage.dataset)
    stage_dir = output / "stages" / stage.id
    artifact = stage_dir / "de.h5"
    record_path = stage_dir / "stage.json"

    if not force and _reusable_record(record_path, manifest_digest, [artifact]):
        return record_path

    data = _load_analysis_unit(spec)
    perturbations = _eligible_perturbations(data, int(manifest["minimum_cells"]))

    stage_dir.mkdir(parents=True, exist_ok=True)
    temporary = stage_dir / ".scperteval-output"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    started = time.perf_counter()
    prepared = spe.prepare(
        data,
        [],
        subsample=int(manifest["reference_subsample"]),
        seed=int(manifest["seed"]),
        min_cells=int(manifest["minimum_cells"]),
        workers=workers,
        name=stage.dataset,
    )
    spe.de(
        prepared,
        stage.method,
        target=manifest["de"]["target"],
        reference=stage.reference,
        perturbations=perturbations,
        out_dir=temporary,
    )
    candidates = list(temporary.glob("*__de.h5"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one DE artifact from scPertEval, found {len(candidates)}")
    candidates[0].replace(artifact)
    temporary.rmdir()
    if not _valid_de(artifact, stage, perturbations):
        raise RuntimeError(f"invalid completed DE artifact: {artifact}")
    return _write_record(
        record_path,
        {
            "stage": stage.id,
            "status": "complete",
            "manifest_sha256": manifest_digest,
            "artifact": str(artifact),
            "artifact_sha256": {str(artifact): file_sha256(artifact)},
            "n_cells": data.n_obs,
            "n_genes": data.n_vars,
            "n_perturbations": len(perturbations),
            "seconds": time.perf_counter() - started,
            "max_rss_gib": _max_rss_gib(),
            "size_bytes": artifact.stat().st_size,
        },
    )


def _valid_drf(drf_path: Path, timing_path: Path, protocols: list[str]) -> bool:
    if not drf_path.is_file() or not timing_path.is_file():
        return False
    try:
        drf = pd.read_csv(drf_path)
        timing = pd.read_csv(timing_path)
    except (OSError, pd.errors.ParserError):
        return False
    return (
        {"protocol", "perturbation", "drf"} <= set(drf)
        and set(protocols) == set(drf["protocol"].astype(str))
        and set(protocols) == set(timing["protocol"].astype(str))
        and np.isfinite(timing["seconds"].to_numpy(dtype=float)).all()
    )


def run_drf_stage(
    manifest: dict[str, Any],
    stage: Stage,
    output: Path,
    *,
    workers: int,
    manifest_digest: str,
    gpu: bool,
    force: bool = False,
) -> Path:
    assert stage.dataset
    spec = _dataset_spec(manifest, stage.dataset)
    stage_dir = output / "stages" / stage.id
    drf_path = stage_dir / "drf.csv"
    timing_path = stage_dir / "timing.csv"
    record_path = stage_dir / "stage.json"
    protocols = list(manifest["gpu_protocols"] if gpu else manifest["protocols"])
    if not force and _reusable_record(record_path, manifest_digest, [drf_path, timing_path]) and _valid_drf(
        drf_path, timing_path, protocols
    ):
        return record_path

    stage_dir.mkdir(parents=True, exist_ok=True)
    data = _load_analysis_unit(spec)
    started = time.perf_counter()
    prepared = spe.prepare(
        data,
        protocols,
        subsample=int(manifest["reference_subsample"]),
        seed=int(manifest["seed"]),
        min_cells=int(manifest["minimum_cells"]),
        workers=workers,
        name=stage.dataset,
    )
    frames: list[pd.DataFrame] = []
    timings: list[dict[str, Any]] = []
    for protocol in protocols:
        protocol_started = time.perf_counter()
        result = spe.calibrate(prepared, protocol, de_method="t-test", calibrator="drf")
        elapsed = time.perf_counter() - protocol_started
        frames.append(result.per_perturbation)
        timings.append({"dataset": stage.dataset, "protocol": protocol, "seconds": elapsed})
    pd.concat(frames, ignore_index=True).to_csv(drf_path, index=False)
    pd.DataFrame(timings).to_csv(timing_path, index=False)
    if not _valid_drf(drf_path, timing_path, protocols):
        raise RuntimeError(f"invalid completed DRF artifacts for {stage.dataset}")
    return _write_record(
        record_path,
        {
            "stage": stage.id,
            "status": "complete",
            "manifest_sha256": manifest_digest,
            "artifacts": [str(drf_path), str(timing_path)],
            "artifact_sha256": {
                str(drf_path): file_sha256(drf_path),
                str(timing_path): file_sha256(timing_path),
            },
            "n_cells": data.n_obs,
            "n_genes": data.n_vars,
            "n_perturbations": int(pd.concat(frames)["perturbation"].nunique()),
            "seconds": time.perf_counter() - started,
            "max_rss_gib": _max_rss_gib(),
        },
    )


def run_stage(
    manifest_path: Path,
    stage_id: str,
    output: Path,
    *,
    workers: int | None = None,
    force: bool = False,
) -> Path:
    manifest = load_manifest(manifest_path)
    matches = [stage for stage in production_stages(manifest) if stage.id == stage_id]
    if len(matches) != 1:
        raise ValueError(f"unknown production stage {stage_id!r}")
    stage = matches[0]
    actual_workers = int(workers or manifest.get("workers", 16))
    digest = manifest_sha256(manifest_path)
    if stage.kind == "de":
        return run_de_stage(
            manifest,
            stage,
            output,
            workers=actual_workers,
            manifest_digest=digest,
            force=force,
        )
    if stage.kind in {"drf-cpu", "drf-gpu"}:
        return run_drf_stage(
            manifest,
            stage,
            output,
            workers=actual_workers,
            manifest_digest=digest,
            gpu=stage.kind == "drf-gpu",
            force=force,
        )
    raise AssertionError(stage.kind)
