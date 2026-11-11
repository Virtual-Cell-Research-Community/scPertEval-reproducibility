"""Acquire, prepare, and validate the public benchmark datasets."""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path
from typing import Any

import anndata as ad
import requests

from workflows.preparation import prepare_sciplex, prepare_scperturb
from workflows.preparation_utils import assert_raw_counts, file_checksum

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = REPOSITORY_ROOT / "configs" / "datasets"
CELL_CYCLE_PATH = REPOSITORY_ROOT / "data" / "references" / "regev_lab_cell_cycle_genes.csv"


def load_manifest(dataset_id: str) -> dict[str, Any]:
    path = MANIFEST_ROOT / f"{dataset_id}.toml"
    if not path.is_file():
        known = ", ".join(sorted(item.stem for item in MANIFEST_ROOT.glob("*.toml")))
        raise FileNotFoundError(f"unknown dataset {dataset_id!r}; available datasets: {known}")
    manifest = tomllib.loads(path.read_text())
    if manifest.get("id") != dataset_id:
        raise ValueError(f"manifest ID mismatch in {path}")
    return manifest


def data_root(explicit: Path | None = None) -> Path:
    return explicit or Path(os.environ.get("SCPERTEVAL_DATA_ROOT", REPOSITORY_ROOT / "data"))


def source_artifacts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    source = manifest["source"]
    if "artifacts" in source:
        return source["artifacts"]
    return [
        {
            "name": source["filename"],
            "url": source["url"],
            "size_bytes": source["size_bytes"],
            "checksum_type": source["checksum_type"],
            "checksum": source["checksum"],
        }
    ]


def raw_directory(dataset_id: str, root: Path) -> Path:
    return root / "raw" / dataset_id


def processed_path(dataset_id: str, root: Path) -> Path:
    return root / "processed" / dataset_id / "adata.h5ad"


def verify_file(path: Path, artifact: dict[str, Any]) -> None:
    expected_size = int(artifact["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(f"unexpected size for {path}: {path.stat().st_size} != {expected_size}")
    actual = file_checksum(path, artifact["checksum_type"])
    if actual != artifact["checksum"]:
        raise ValueError(f"unexpected {artifact['checksum_type']} for {path}: {actual}")


def download_dataset(dataset_id: str, root: Path, *, force: bool = False) -> list[Path]:
    """Download every registered source artifact and verify it before use."""
    manifest = load_manifest(dataset_id)
    destination = raw_directory(dataset_id, root)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for artifact in source_artifacts(manifest):
        output = destination / artifact["name"]
        if output.exists() and not force:
            verify_file(output, artifact)
            paths.append(output)
            continue
        partial = output.with_suffix(output.suffix + ".partial")
        with requests.get(artifact["url"], stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if block:
                        stream.write(block)
        verify_file(partial, artifact)
        partial.replace(output)
        paths.append(output)
    return paths


def stage_dataset_sources(
    dataset_id: str,
    source_directory: Path,
    root: Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Stage already-downloaded, checksum-verified sources as symlinks.

    This is intended for maintainers validating public regeneration against an internal
    canonical object without storing a second copy of the immutable upstream downloads.
    Public users normally use ``download`` instead.
    """
    manifest = load_manifest(dataset_id)
    destination = raw_directory(dataset_id, root)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for artifact in source_artifacts(manifest):
        source = source_directory / artifact["name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        verify_file(source, artifact)
        output = destination / artifact["name"]
        if output.exists() or output.is_symlink():
            if not force:
                verify_file(output, artifact)
                paths.append(output)
                continue
            output.unlink()
        output.symlink_to(source.resolve())
        paths.append(output)
    return paths


def prepare_dataset(dataset_id: str, root: Path, *, force: bool = False) -> Path:
    """Prepare one canonical raw-count object from its verified sources."""
    manifest = load_manifest(dataset_id)
    for path, artifact in zip(
        [raw_directory(dataset_id, root) / item["name"] for item in source_artifacts(manifest)],
        source_artifacts(manifest),
        strict=True,
    ):
        verify_file(path, artifact)

    output = processed_path(dataset_id, root)
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force explicitly")
    adapter = manifest["preparation"]["adapter"]
    if adapter == "scperturb-crispri":
        source = raw_directory(dataset_id, root) / manifest["source"]["filename"]
        prepared = prepare_scperturb(
            source,
            cell_cycle_path=CELL_CYCLE_PATH,
            merge_disambiguated=manifest["preparation"].get("merge_disambiguated", False),
            filter_residual_expression=manifest["preparation"].get("filter_residual_expression", True),
        )
    elif adapter == "sciplex3":
        prepared = prepare_sciplex(raw_directory(dataset_id, root))
    else:
        raise ValueError(f"unknown preparation adapter: {adapter}")

    output.parent.mkdir(parents=True, exist_ok=True)
    compression = manifest["preparation"].get("compression")
    prepared.write_h5ad(output, compression=compression)
    return output


def validate_dataset(dataset_id: str, root: Path, *, exact_checksum: bool = True) -> Path:
    """Validate structure and, by default, the exact reference artifact checksum."""
    manifest = load_manifest(dataset_id)
    expected = manifest["reference_output"]
    path = processed_path(dataset_id, root)
    if not path.is_file():
        raise FileNotFoundError(path)
    if exact_checksum and "semantic_sha256" not in expected and path.stat().st_size != int(expected["size_bytes"]):
        raise ValueError(
            f"unexpected output size for {path}: got {path.stat().st_size}, expected {expected['size_bytes']}"
        )
    prepared = ad.read_h5ad(path, backed="r")
    try:
        if prepared.shape != (int(expected["n_obs"]), int(expected["n_vars"])):
            raise ValueError(f"unexpected output shape for {path}: {prepared.shape}")
        missing_obs = sorted(set(manifest["contract"]["required_obs"]) - set(prepared.obs.columns))
        if missing_obs:
            raise ValueError(f"prepared dataset is missing obs fields: {', '.join(missing_obs)}")
        if not prepared.obs_names.is_unique or not prepared.var_names.is_unique:
            raise ValueError("prepared dataset has duplicate cell or gene identifiers")
        sample = prepared.X[: min(128, prepared.n_obs), : min(128, prepared.n_vars)]
        assert_raw_counts(sample)
    finally:
        prepared.file.close()
    if exact_checksum:
        if "semantic_sha256" in expected:
            from workflows.semantic_h5ad import semantic_h5ad_sha256

            actual = semantic_h5ad_sha256(path)
            if actual != expected["semantic_sha256"]:
                raise ValueError(f"unexpected semantic output checksum for {path}: {actual}")
        else:
            actual = file_checksum(path)
            if actual != expected["sha256"]:
                raise ValueError(f"unexpected reference-output SHA-256 for {path}: {actual}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="data root (default: data/ or SCPERTEVAL_DATA_ROOT)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["download", "stage", "prepare", "validate", "all"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("dataset", help="dataset manifest ID")
        if command in {"download", "stage", "prepare", "all"}:
            sub.add_argument("--force", action="store_true")
        if command == "stage":
            sub.add_argument("--from-directory", type=Path, required=True)
        if command == "validate":
            sub.add_argument("--structural-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = data_root(args.root)
    if args.command in {"download", "all"}:
        for path in download_dataset(args.dataset, root, force=args.force):
            print(f"verified source: {path}")
    if args.command == "stage":
        for path in stage_dataset_sources(
            args.dataset,
            args.from_directory,
            root,
            force=args.force,
        ):
            print(f"staged verified source: {path}")
    if args.command in {"prepare", "all"}:
        print(f"prepared: {prepare_dataset(args.dataset, root, force=args.force)}")
    if args.command in {"validate", "all"}:
        structural_only = getattr(args, "structural_only", False)
        print(f"validated: {validate_dataset(args.dataset, root, exact_checksum=not structural_only)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
