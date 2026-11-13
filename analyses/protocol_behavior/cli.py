"""Run protocol-behavior reproductions and their cost pilots."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scperteval as spe

from .production import DEFAULT_MANIFEST, DEFAULT_RESULTS, load_manifest, production_stages, run_stage

METHODS = ("t-test", "t-test_overestim_var", "MWU")
FULL_TARGET = "gt_all_cells"
SPLIT_TARGET = "gt_half"
REFERENCES = ("all_perturbed", "control")
DEFAULT_GWPS_DATASET = Path("models/data/replogle-weissman-2022-k562-gwps/raw.h5ad")
DEFAULT_OUTPUT_DIR = Path("results/protocol_behavior/pilots/gwps-de")


def _comparison_grid() -> list[tuple[str, str, str]]:
    comparisons = [
        (target, reference, method) for target in (FULL_TARGET,) for reference in REFERENCES for method in METHODS
    ]
    comparisons.extend((SPLIT_TARGET, reference, "MWU") for reference in REFERENCES)
    return comparisons


def select_quartile_targets(
    obs: pd.DataFrame,
    *,
    perturbation_key: str = "perturbation",
    control_key: str = "is_control",
    n_per_quartile: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Select equal deterministic samples from four treated-cell-count quartiles.

    Labels break cell-count ties before quartile assignment. The returned order is stable and is
    reused by every DE comparison.
    """
    if n_per_quartile < 1:
        raise ValueError("n_per_quartile must be positive")
    missing = sorted({perturbation_key, control_key} - set(obs.columns))
    if missing:
        raise ValueError(f"dataset is missing pilot metadata: {', '.join(missing)}")
    labels = obs[perturbation_key].astype(str)
    controls = obs[control_key].astype(bool).to_numpy()
    counts = labels[~controls].value_counts().sort_index()
    if len(counts) < 4 * n_per_quartile:
        raise ValueError(f"pilot needs at least {4 * n_per_quartile} treated perturbations; found {len(counts)}")

    table = counts.rename("n_cells").to_frame()
    table.index.name = "perturbation"
    # Rank after sorting labels so ties are deterministic while each quartile remains equally sized.
    table["quartile"] = pd.qcut(table["n_cells"].rank(method="first"), 4, labels=False).astype(int)
    rng = np.random.default_rng(seed)
    selected = []
    for quartile in range(4):
        candidates = table.index[table["quartile"] == quartile].to_numpy(dtype=str)
        chosen = rng.choice(candidates, size=n_per_quartile, replace=False)
        selected.append(table.loc[sorted(chosen)].copy())
    result = pd.concat(selected).reset_index()
    result["selection_seed"] = seed
    return result


def _max_rss_gib() -> float:
    # Linux reports KiB and macOS bytes. Production pilots run on Linux.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024**2 if os.uname().sysname == "Linux" else 1024**3
    return value / divisor


def _valid_de_artifact(
    path: Path,
    *,
    method: str,
    target: str,
    reference: str,
    perturbations: list[str],
) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path) as handle:
            observed = [value.decode() for value in handle["perturbations"][:]]
            return (
                handle.attrs["target"] == target
                and handle.attrs["reference"] == reference
                and observed == perturbations
                and handle[method]["statistic"].shape[0] == len(perturbations)
                and handle[method]["pvalue_adj"].shape == handle[method]["statistic"].shape
            )
    except (KeyError, OSError, ValueError):
        return False


def _run_de_stage(
    prepared: spe.Prepared,
    *,
    method: str,
    target: str,
    reference: str,
    perturbations: list[str],
    output_dir: Path,
    force: bool,
) -> dict:
    stage_dir = output_dir / f"{target}__vs__{reference}" / method
    artifact = stage_dir / "de.h5"
    stage_record = stage_dir / "stage.json"
    if not force and _valid_de_artifact(
        artifact,
        method=method,
        target=target,
        reference=reference,
        perturbations=perturbations,
    ):
        record = json.loads(stage_record.read_text()) if stage_record.is_file() else {}
        record.update(
            {
                "method": method,
                "target": target,
                "reference": reference,
                "artifact": str(artifact),
                "status": "reused",
            }
        )
        return record

    stage_dir.mkdir(parents=True, exist_ok=True)
    temporary = stage_dir / ".scperteval-output"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    started = time.perf_counter()
    spe.de(
        prepared,
        method,
        target=target,
        reference=reference,
        perturbations=perturbations,
        out_dir=temporary,
    )
    candidates = list(temporary.glob("*__de.h5"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one DE artifact from scPertEval, found {len(candidates)}")
    candidates[0].replace(artifact)
    temporary.rmdir()
    elapsed = time.perf_counter() - started
    if not _valid_de_artifact(
        artifact,
        method=method,
        target=target,
        reference=reference,
        perturbations=perturbations,
    ):
        raise RuntimeError(f"invalid completed DE artifact: {artifact}")
    record = {
        "method": method,
        "target": target,
        "reference": reference,
        "artifact": str(artifact),
        "status": "computed",
        "seconds": elapsed,
        "max_rss_gib": _max_rss_gib(),
        "size_bytes": artifact.stat().st_size,
    }
    stage_record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _reusable_stages(output_dir: Path, perturbations: list[str]) -> list[dict] | None:
    """Return validated existing stage records, or ``None`` if any stage needs computation."""
    stages = []
    for target, reference, method in _comparison_grid():
        stage_dir = output_dir / f"{target}__vs__{reference}" / method
        artifact = stage_dir / "de.h5"
        stage_record = stage_dir / "stage.json"
        if not stage_record.is_file() or not _valid_de_artifact(
            artifact,
            method=method,
            target=target,
            reference=reference,
            perturbations=perturbations,
        ):
            return None
        record = json.loads(stage_record.read_text())
        record.update(
            {
                "method": method,
                "target": target,
                "reference": reference,
                "artifact": str(artifact),
                "status": "reused",
            }
        )
        stages.append(record)
    return stages


def _write_summary(
    path: Path,
    *,
    dataset: Path,
    perturbations: list[str],
    n_per_quartile: int,
    seed: int,
    subsample: int,
    workers: int,
    load_seconds: float,
    stages: list[dict],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "dataset": str(dataset.resolve()),
                "selection": {
                    "strategy": "16-per-treated-cell-count-quartile" if n_per_quartile == 16 else "per-quartile",
                    "n_per_quartile": n_per_quartile,
                    "seed": seed,
                    "n_perturbations": len(perturbations),
                },
                "reference_subsample": subsample,
                "scperteval_version": spe.__version__,
                "workers": workers,
                "load_seconds": load_seconds,
                "max_rss_gib_after_load": _max_rss_gib(),
                "stages": stages,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def run_gwps_de_pilot(
    dataset: Path,
    output_dir: Path,
    *,
    n_per_quartile: int = 16,
    seed: int = 42,
    subsample: int = 8192,
    workers: int = 16,
    force: bool = False,
) -> Path:
    """Run the agreed GWPS DE cost grid and return its summary path."""
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    backed = ad.read_h5ad(dataset, backed="r")
    try:
        selected = select_quartile_targets(
            backed.obs,
            n_per_quartile=n_per_quartile,
            seed=seed,
        )
    finally:
        backed.file.close()
    selected.to_csv(output_dir / "selected-perturbations.csv", index=False)
    perturbations = selected["perturbation"].tolist()

    if not force:
        reusable = _reusable_stages(output_dir, perturbations)
        if reusable is not None:
            summary = output_dir / "summary.json"
            if summary.is_file():
                return summary
            return _write_summary(
                summary,
                dataset=dataset,
                perturbations=perturbations,
                n_per_quartile=n_per_quartile,
                seed=seed,
                subsample=subsample,
                workers=workers,
                load_seconds=0.0,
                stages=reusable,
            )

    load_started = time.perf_counter()
    prepared = spe.prepare(
        dataset,
        [],
        subsample=subsample,
        seed=seed,
        min_cells=50,
        workers=workers,
        name="replogle-weissman-2022-k562-gwps-de-pilot",
    )
    load_seconds = time.perf_counter() - load_started

    stages = [
        _run_de_stage(
            prepared,
            method=method,
            target=target,
            reference=reference,
            perturbations=perturbations,
            output_dir=output_dir,
            force=force,
        )
        for target, reference, method in _comparison_grid()
    ]
    return _write_summary(
        output_dir / "summary.json",
        dataset=dataset,
        perturbations=perturbations,
        n_per_quartile=n_per_quartile,
        seed=seed,
        subsample=subsample,
        workers=workers,
        load_seconds=load_seconds,
        stages=stages,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        choices=["gwps-de"],
        help="run the bounded GWPS cost pilot instead of the production workflow",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_GWPS_DATASET,
        help="prepared GWPS model view (default: %(default)s)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", help=argparse.SUPPRESS)
    parser.add_argument("--list-stages", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage-class", choices=["all", "cpu", "gpu"], default="all", help=argparse.SUPPRESS)
    parser.add_argument("--finalize", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--targets-per-quartile", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pilot:
        output = args.output_dir or DEFAULT_OUTPUT_DIR
        summary = run_gwps_de_pilot(
            args.dataset,
            output,
            n_per_quartile=args.targets_per_quartile,
            seed=args.seed,
            subsample=args.subsample,
            workers=args.workers,
            force=args.force,
        )
        print(f"completed GWPS DE pilot: {summary}")
        return 0

    output = args.output_dir or DEFAULT_RESULTS
    manifest = load_manifest(args.manifest)
    stages = production_stages(manifest)
    if args.list_stages:
        if args.stage_class == "cpu":
            stages = [stage for stage in stages if stage.kind != "drf-gpu"]
        elif args.stage_class == "gpu":
            stages = [stage for stage in stages if stage.kind == "drf-gpu"]
        print("\n".join(stage.id for stage in stages))
        return 0
    if args.stage:
        record = run_stage(
            args.manifest,
            args.stage,
            output,
            workers=args.workers,
            force=args.force,
        )
        print(f"completed production stage {args.stage}: {record}")
        return 0
    if args.finalize:
        from .render import render_and_validate

        completion = render_and_validate(args.manifest, output)
        print(f"rendered and validated protocol-behavior production results: {completion}")
        return 0

    for stage in stages:
        run_stage(args.manifest, stage.id, output, workers=args.workers, force=args.force)
    from .render import render_and_validate

    completion = render_and_validate(args.manifest, output)
    print(f"completed protocol-behavior production workflow: {completion}")
    return 0
