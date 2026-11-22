"""Manifest-driven production stages for the protocol-behavior study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import resource
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scperteval as spe

from .protocols import resolve_frozen_protocols

DEFAULT_MANIFEST = Path("configs/protocol_behavior/production-v2.json")
DEFAULT_RESULTS = Path("results/protocol_behavior/production-v2-scperteval-main-staging")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PROJECT = Path(__file__).resolve().parent
PROTOCOL_PYPROJECT = PROTOCOL_PROJECT / "pyproject.toml"
PROTOCOL_LOCK = PROTOCOL_PROJECT / "uv.lock"
WORKFLOW_FILES = (
    Path(__file__).resolve(),
    PROTOCOL_PROJECT / "cli.py",
    PROTOCOL_PROJECT / "protocols.py",
    PROTOCOL_PROJECT / "render.py",
)
LEGACY_SCPERTEVAL_REVISIONS = {
    "protocol-behavior-production-v1": "89fc21f3e59673651b3162355d9a32495e274253",
}


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


def default_results(manifest: dict[str, Any]) -> Path:
    """Return the manifest-declared safe output tree, with a legacy fallback."""
    return Path(manifest.get("output_dir", DEFAULT_RESULTS))


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_scperteval_revision() -> str | None:
    """Return the resolved Git commit recorded by the installed direct dependency."""
    distribution = importlib.metadata.distribution("scperteval")
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return None
    metadata = json.loads(direct_url)
    return metadata.get("vcs_info", {}).get("commit_id")


def expected_scperteval_revision(manifest: dict[str, Any]) -> str:
    """Resolve an explicit revision, with one checksum-preserving legacy declaration."""
    expected = manifest.get("scperteval_revision") or LEGACY_SCPERTEVAL_REVISIONS.get(manifest.get("id"))
    if not expected:
        raise ValueError("protocol manifest must pin scperteval_revision")
    return str(expected)


def assert_scperteval_revision(manifest: dict[str, Any]) -> str:
    """Refuse to run scientific stages with a different metric implementation."""
    expected = expected_scperteval_revision(manifest)
    observed = installed_scperteval_revision()
    if observed != expected:
        raise RuntimeError(
            "installed scPertEval revision does not match the protocol manifest: "
            f"observed={observed!r}, expected={expected!r}"
        )
    return observed


def assert_manifest_runnable(manifest: dict[str, Any]) -> None:
    """Keep review-gated production manifests inspectable but non-executable."""
    status = str(manifest.get("status", "ready"))
    if status.startswith("planned-pending-"):
        raise RuntimeError(f"protocol manifest is not approved for execution: status={status}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def repository_revision() -> str:
    """Return the exact reproducibility-repository commit executing the workflow."""
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def environment_fingerprint(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint the evaluator environment and workflow sources used by every stage."""
    sources = {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in WORKFLOW_FILES}
    return {
        "manifest_sha256": manifest_sha256(manifest_path),
        "protocol_pyproject_sha256": file_sha256(PROTOCOL_PYPROJECT),
        "protocol_uv_lock_sha256": file_sha256(PROTOCOL_LOCK),
        "workflow_sources_sha256": _canonical_sha256(sources),
        "workflow_sources": sources,
        "repository_revision": repository_revision(),
        "scperteval_revision": expected_scperteval_revision(manifest),
    }


def _resolve_repository_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _resolve_data_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    data_root = Path(os.environ.get("SCPERTEVAL_DATA_ROOT", REPOSITORY_ROOT))
    return data_root / value


def _declared_data_output(spec: dict[str, Any]) -> tuple[Path, dict[str, Any], Path]:
    manifest_name = spec.get("data_manifest")
    output_key = spec.get("data_manifest_output_key")
    if not manifest_name or not output_key:
        raise ValueError(f"dataset {spec['id']!r} must declare data_manifest and data_manifest_output_key")
    provenance_path = _resolve_data_path(manifest_name)
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text())
    declared = provenance.get(output_key)
    if not isinstance(declared, dict) or "sha256" not in declared:
        raise ValueError(f"{provenance_path} has no checksummed {output_key!r} output")
    dataset_path = _resolve_data_path(spec["path"])
    declared_name = declared.get("filename") or declared.get("path")
    if declared_name and Path(str(declared_name)).name != dataset_path.name:
        raise ValueError(f"{provenance_path} describes {declared_name!r}, not expected dataset {dataset_path.name!r}")
    return dataset_path, declared, provenance_path


def create_input_record(manifest_path: Path, output: Path) -> Path:
    """Validate prepared data and write the deterministic production input record."""
    manifest = load_manifest(manifest_path)
    assert_manifest_runnable(manifest)
    assert_scperteval_revision(manifest)
    hashed_data: dict[Path, str] = {}
    datasets: dict[str, Any] = {}
    for spec in manifest["datasets"]:
        dataset_path, declared, provenance_path = _declared_data_output(spec)
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        resolved_dataset = dataset_path.resolve()
        if resolved_dataset not in hashed_data:
            hashed_data[resolved_dataset] = file_sha256(dataset_path)
        observed = hashed_data[resolved_dataset]
        if observed != declared["sha256"]:
            raise RuntimeError(
                f"prepared dataset checksum mismatch for {spec['id']}: "
                f"observed={observed}, declared={declared['sha256']}"
            )
        cohort_path = _resolve_repository_path(spec["cohort_manifest"])
        if not cohort_path.is_file():
            raise FileNotFoundError(cohort_path)
        stat = dataset_path.stat()
        datasets[spec["id"]] = {
            "path": str(Path(spec["path"])),
            "sha256": observed,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "data_manifest": str(Path(spec["data_manifest"])),
            "data_manifest_sha256": file_sha256(provenance_path),
            "cohort_manifest": str(Path(spec["cohort_manifest"])),
            "cohort_manifest_sha256": file_sha256(cohort_path),
        }
    payload = {
        "status": "complete",
        "environment": environment_fingerprint(manifest_path, manifest),
        "datasets": datasets,
    }
    payload["input_fingerprint_sha256"] = _canonical_sha256(payload)
    return _atomic_json(output / "inputs.json", payload)


def validate_input_record(manifest_path: Path, output: Path, *, rehash_data: bool = False) -> dict[str, Any]:
    """Validate the production input record without rehashing large H5ADs per stage."""
    manifest = load_manifest(manifest_path)
    path = output / "inputs.json"
    if not path.is_file():
        raise RuntimeError(f"missing production input record {path}; run --preflight first")
    payload = json.loads(path.read_text())
    fingerprint = payload.pop("input_fingerprint_sha256", None)
    if fingerprint != _canonical_sha256(payload):
        raise RuntimeError("production input record fingerprint is invalid")
    payload["input_fingerprint_sha256"] = fingerprint
    if payload.get("status") != "complete":
        raise RuntimeError("production input validation is not complete")
    if payload.get("environment") != environment_fingerprint(manifest_path, manifest):
        raise RuntimeError("production input record belongs to a different code or evaluator environment")
    for spec in manifest["datasets"]:
        recorded = payload.get("datasets", {}).get(spec["id"])
        if not recorded:
            raise RuntimeError(f"production input record is missing dataset {spec['id']}")
        dataset_path, declared, provenance_path = _declared_data_output(spec)
        stat = dataset_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (recorded["size_bytes"], recorded["mtime_ns"]):
            raise RuntimeError(f"prepared dataset changed after preflight: {dataset_path}")
        if file_sha256(provenance_path) != recorded["data_manifest_sha256"]:
            raise RuntimeError(f"prepared-data manifest changed after preflight: {provenance_path}")
        cohort_path = _resolve_repository_path(spec["cohort_manifest"])
        if file_sha256(cohort_path) != recorded["cohort_manifest_sha256"]:
            raise RuntimeError(f"cohort manifest changed after preflight: {cohort_path}")
        if declared["sha256"] != recorded["sha256"]:
            raise RuntimeError(f"prepared-data manifest now declares a different checksum: {provenance_path}")
        if rehash_data and file_sha256(dataset_path) != recorded["sha256"]:
            raise RuntimeError(f"prepared dataset checksum changed after preflight: {dataset_path}")
    return payload


def stage_inputs(manifest_path: Path, output: Path) -> dict[str, Any]:
    """Return the exact provenance block every reusable stage must match."""
    record = validate_input_record(manifest_path, output)
    return {
        **record["environment"],
        "input_record_sha256": file_sha256(output / "inputs.json"),
    }


def _artifact_key(path: Path, record_path: Path) -> str:
    """Store an artifact relative to its stage record so mount aliases remain irrelevant."""
    return str(path.resolve().relative_to(record_path.parent.resolve()))


def resolve_record_artifact(record_path: Path, name: str) -> Path:
    """Resolve one relative artifact path against its stage record."""
    stored = Path(name)
    if stored.is_absolute() or ".." in stored.parts:
        raise ValueError(f"stage artifact path must be relative: {name}")
    return (record_path.parent / stored).resolve()


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


def _declared_perturbations(spec: dict[str, Any]) -> list[str] | None:
    cohort_path = spec.get("cohort_manifest")
    if cohort_path is None:
        return None
    cohort = json.loads(_resolve_repository_path(cohort_path).read_text())
    if "perturbations" in cohort:
        return list(map(str, cohort["perturbations"]))
    if {"drugs", "doses", "population_exclusions"} <= set(cohort):
        canonical = {item["label"]: item["canonical_label"] for item in cohort["drugs"]}
        excluded = {(canonical[item["drug"]], float(item["dose"])) for item in cohort["population_exclusions"]}
        return sorted(
            f"{item['canonical_label']}::{float(dose):g}nM"
            for item in cohort["drugs"]
            for dose in cohort["doses"]
            if (item["canonical_label"], float(dose)) not in excluded
        )
    raise ValueError(f"cannot resolve a perturbation cohort from {cohort_path}")


def _load_analysis_unit(spec: dict[str, Any], *, minimum_cells: int | None = None) -> ad.AnnData:
    path = _resolve_data_path(spec["path"])
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
    declared = _declared_perturbations(spec)
    if declared is not None:
        labels = data.obs["perturbation"].astype(str)
        controls = data.obs["is_control"].astype(bool)
        observed = set(labels[~controls])
        missing = sorted(set(declared) - observed)
        if missing:
            raise ValueError(f"{spec['id']} is missing {len(missing)} declared perturbations: {missing[:5]}")
        data = data[(controls | labels.isin(declared)).to_numpy()].copy()
        if minimum_cells is not None:
            counts = data.obs.loc[~data.obs["is_control"].astype(bool), "perturbation"].astype(str).value_counts()
            below = counts[counts < minimum_cells]
            if len(below):
                raise ValueError(
                    f"{spec['id']} has {len(below)} declared perturbations below {minimum_cells} cells: "
                    f"{below.index[:5].tolist()}"
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
    return _atomic_json(path, record)


def _valid_de(
    path: Path,
    stage: Stage,
    perturbations: list[str],
    *,
    target: str = "gt_all_cells",
    subsample: int | None = None,
    seed: int | None = None,
) -> bool:
    if not path.is_file() or stage.method is None or stage.reference is None:
        return False
    try:
        with h5py.File(path) as handle:
            observed = [value.decode() for value in handle["perturbations"][:]]
            return (
                handle.attrs["target"] == target
                and handle.attrs["reference"] == stage.reference
                and (subsample is None or int(handle.attrs["subsample"]) == subsample)
                and (seed is None or int(handle.attrs["seed"]) == seed)
                and observed == perturbations
                and handle[stage.method]["statistic"].shape == (len(perturbations), len(handle["genes"]))
                and handle[stage.method]["pvalue_adj"].shape == (len(perturbations), len(handle["genes"]))
            )
    except (KeyError, OSError, ValueError):
        return False


def _reusable_record(record_path: Path, inputs: dict[str, Any], artifacts: list[Path]) -> bool:
    if not record_path.is_file() or any(not artifact.is_file() for artifact in artifacts):
        return False
    try:
        record = json.loads(record_path.read_text())
        expected = {
            resolve_record_artifact(record_path, name): digest
            for name, digest in record.get("artifact_sha256", {}).items()
        }
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    resolved = {artifact.resolve(): artifact for artifact in artifacts}
    return (
        record.get("status") == "complete"
        and record.get("inputs") == inputs
        and set(expected) == set(resolved)
        and all(expected[path] == file_sha256(artifact) for path, artifact in resolved.items())
    )


def run_de_stage(
    manifest: dict[str, Any],
    stage: Stage,
    output: Path,
    *,
    workers: int,
    inputs: dict[str, Any],
    force: bool = False,
) -> Path:
    assert stage.dataset and stage.method and stage.reference
    spec = _dataset_spec(manifest, stage.dataset)
    stage_dir = output / "stages" / stage.id
    artifact = stage_dir / "de.h5"
    record_path = stage_dir / "stage.json"

    if not force and _reusable_record(record_path, inputs, [artifact]):
        return record_path

    data = _load_analysis_unit(spec, minimum_cells=int(manifest["minimum_cells"]))
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
    if not _valid_de(
        artifact,
        stage,
        perturbations,
        target=str(manifest["de"]["target"]),
        subsample=int(manifest["reference_subsample"]),
        seed=int(manifest["seed"]),
    ):
        raise RuntimeError(f"invalid completed DE artifact: {artifact}")
    artifact_key = _artifact_key(artifact, record_path)
    return _write_record(
        record_path,
        {
            "stage": stage.id,
            "status": "complete",
            "inputs": inputs,
            "artifact": artifact_key,
            "artifact_sha256": {artifact_key: file_sha256(artifact)},
            "n_cells": data.n_obs,
            "n_genes": data.n_vars,
            "n_perturbations": len(perturbations),
            "seconds": time.perf_counter() - started,
            "max_rss_gib": _max_rss_gib(),
            "size_bytes": artifact.stat().st_size,
        },
    )


def _valid_drf(
    drf_path: Path,
    timing_path: Path,
    protocols: list[str],
    perturbations: list[str] | None = None,
    *,
    seed: int | None = None,
    subsample: int | None = None,
) -> bool:
    if not drf_path.is_file() or not timing_path.is_file():
        return False
    try:
        drf = pd.read_csv(drf_path)
        timing = pd.read_csv(timing_path)
    except (OSError, pd.errors.ParserError):
        return False
    required = {
        "protocol",
        "perturbation",
        "positive",
        "negative",
        "raw_positive",
        "raw_negative",
        "undefined_reason",
        "drf",
        "dataset",
        "de_method",
        "subsample",
        "seed",
    }
    if not required <= set(drf) or not {"dataset", "protocol", "seconds"} <= set(timing):
        return False
    expected_protocols = set(protocols)
    if set(drf["protocol"].astype(str)) != expected_protocols:
        return False
    if set(timing["protocol"].astype(str)) != expected_protocols or timing["protocol"].duplicated().any():
        return False
    if not np.isfinite(timing["seconds"].to_numpy(dtype=float)).all():
        return False
    if seed is not None and set(drf["seed"].astype(int)) != {seed}:
        return False
    if subsample is not None and set(drf["subsample"].astype(int)) != {subsample}:
        return False
    if set(drf["de_method"].astype(str)) != {"t-test"}:
        return False
    pairs = drf[["protocol", "perturbation"]].astype(str)
    if pairs.duplicated().any():
        return False
    if perturbations is None:
        return True
    expected_perturbations = set(perturbations)
    if set(pairs["perturbation"]) != expected_perturbations:
        return False
    expected_pairs = {(protocol, perturbation) for protocol in protocols for perturbation in perturbations}
    return set(pairs.itertuples(index=False, name=None)) == expected_pairs


def run_drf_stage(
    manifest: dict[str, Any],
    stage: Stage,
    output: Path,
    *,
    workers: int,
    inputs: dict[str, Any],
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
    protocol_objects = resolve_frozen_protocols(protocols)
    declared_perturbations = _declared_perturbations(spec)
    if (
        not force
        and _reusable_record(record_path, inputs, [drf_path, timing_path])
        and _valid_drf(
            drf_path,
            timing_path,
            protocols,
            declared_perturbations,
            seed=int(manifest["seed"]),
            subsample=int(manifest["reference_subsample"]),
        )
    ):
        return record_path

    stage_dir.mkdir(parents=True, exist_ok=True)
    data = _load_analysis_unit(spec, minimum_cells=int(manifest["minimum_cells"]))
    perturbations = _eligible_perturbations(data, int(manifest["minimum_cells"]))
    started = time.perf_counter()
    prepared = spe.prepare(
        data,
        protocol_objects,
        subsample=int(manifest["reference_subsample"]),
        seed=int(manifest["seed"]),
        min_cells=int(manifest["minimum_cells"]),
        workers=workers,
        name=stage.dataset,
    )
    frames: list[pd.DataFrame] = []
    timings: list[dict[str, Any]] = []
    for protocol in protocol_objects:
        protocol_started = time.perf_counter()
        result = spe.calibrate(prepared, protocol, de_method="t-test", calibrator="drf")
        elapsed = time.perf_counter() - protocol_started
        frames.append(result.per_perturbation)
        timings.append({"dataset": stage.dataset, "protocol": protocol.name, "seconds": elapsed})
    drf_temporary = drf_path.with_name(f".{drf_path.name}.tmp")
    timing_temporary = timing_path.with_name(f".{timing_path.name}.tmp")
    pd.concat(frames, ignore_index=True).to_csv(drf_temporary, index=False)
    pd.DataFrame(timings).to_csv(timing_temporary, index=False)
    drf_temporary.replace(drf_path)
    timing_temporary.replace(timing_path)
    if not _valid_drf(
        drf_path,
        timing_path,
        protocols,
        perturbations,
        seed=int(manifest["seed"]),
        subsample=int(manifest["reference_subsample"]),
    ):
        raise RuntimeError(f"invalid completed DRF artifacts for {stage.dataset}")
    artifact_keys = {_artifact_key(path, record_path): path for path in (drf_path, timing_path)}
    return _write_record(
        record_path,
        {
            "stage": stage.id,
            "status": "complete",
            "inputs": inputs,
            "artifacts": list(artifact_keys),
            "artifact_sha256": {key: file_sha256(path) for key, path in artifact_keys.items()},
            "n_cells": data.n_obs,
            "n_genes": data.n_vars,
            "n_perturbations": len(perturbations),
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
    assert_manifest_runnable(manifest)
    assert_scperteval_revision(manifest)
    matches = [stage for stage in production_stages(manifest) if stage.id == stage_id]
    if len(matches) != 1:
        raise ValueError(f"unknown production stage {stage_id!r}")
    stage = matches[0]
    actual_workers = int(workers or manifest.get("workers", 16))
    inputs = stage_inputs(manifest_path, output)
    failure_path = output / "stages" / stage.id / "failure.json"
    try:
        if stage.kind == "de":
            result = run_de_stage(
                manifest,
                stage,
                output,
                workers=actual_workers,
                inputs=inputs,
                force=force,
            )
        elif stage.kind in {"drf-cpu", "drf-gpu"}:
            result = run_drf_stage(
                manifest,
                stage,
                output,
                workers=actual_workers,
                inputs=inputs,
                gpu=stage.kind == "drf-gpu",
                force=force,
            )
        else:
            raise AssertionError(stage.kind)
    except Exception as error:
        _write_record(
            failure_path,
            {
                "stage": stage.id,
                "status": "failed",
                "inputs": inputs,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    failure_path.unlink(missing_ok=True)
    return result


def stage_status(manifest_path: Path, output: Path) -> dict[str, Any]:
    """Summarize checksum-valid complete, failed, and pending production stages."""
    manifest = load_manifest(manifest_path)
    stages = production_stages(manifest)
    try:
        inputs = stage_inputs(manifest_path, output)
        input_error = None
    except (OSError, RuntimeError, ValueError) as error:
        inputs = None
        input_error = str(error)
    groups: dict[str, list[str]] = {"complete": [], "failed": [], "pending": []}
    for stage in stages:
        stage_dir = output / "stages" / stage.id
        artifacts = [stage_dir / "de.h5"] if stage.kind == "de" else [stage_dir / "drf.csv", stage_dir / "timing.csv"]
        record = stage_dir / "stage.json"
        if inputs is not None and _reusable_record(record, inputs, artifacts):
            groups["complete"].append(stage.id)
        elif (stage_dir / "failure.json").is_file():
            groups["failed"].append(stage.id)
        else:
            groups["pending"].append(stage.id)
    return {
        "expected": len(stages),
        "counts": {name: len(values) for name, values in groups.items()},
        "stages": groups,
        "input_error": input_error,
    }
