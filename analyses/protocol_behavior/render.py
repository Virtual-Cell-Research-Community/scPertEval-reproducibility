"""Render and validate the production protocol-behavior result set."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .production import (
    _declared_perturbations,
    _reusable_record,
    _valid_de,
    _valid_drf,
    file_sha256,
    load_manifest,
    production_stages,
    resolve_record_artifact,
    stage_inputs,
    validate_input_record,
)

FIGURE_STEMS = (
    "drf_table_mean",
    "drf_table_median",
    "metric_timing_table",
    "deg_jaccard_with_counts",
    "deg_concordance",
    "deg_jaccard_overestim_with_counts",
    "deg_concordance_overestim",
)
TTEST_STAT_THRESHOLDS = (2.0, 4.0, 8.0)
MWU_STAT_THRESHOLDS = (0.15, 0.33, 0.47)
PADJ_THRESHOLDS = (0.1, 0.05, 0.01)


def _wrap_table_label(label: object, *, width: int = 15) -> str:
    return "\n".join(
        textwrap.wrap(
            str(label),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _table_text_color(background: np.ndarray) -> str:
    red, green, blue = background[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.45 else "black"


def _save_table(frame: pd.DataFrame, path: Path, *, title: str, cmap: str = "RdBu_r") -> None:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    width = max(8.0, 3.0 + 1.35 * len(frame.columns))
    height = max(3.0, 1.5 + 0.34 * len(frame))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            hi = lo + 1.0
        colors = plt.get_cmap(cmap)((np.nan_to_num(values, nan=lo) - lo) / (hi - lo))
        colors[~np.isfinite(values)] = (0.95, 0.95, 0.95, 1.0)
    else:
        colors = np.full((*values.shape, 4), 0.95)
        colors[..., 3] = 1.0
    text = [["-" if not np.isfinite(v) else f"{v:.3g}" for v in row] for row in values]
    table = ax.table(
        cellText=text,
        rowLabels=frame.index.astype(str),
        colLabels=[_wrap_table_label(label) for label in frame.columns],
        cellColours=colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for column in range(len(frame.columns)):
        header = table[(0, column)]
        header.set_height(header.get_height() * 1.55)
    for row in range(len(frame)):
        for column in range(len(frame.columns)):
            table[(row + 1, column)].get_text().set_color(_table_text_color(colors[row, column]))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_drf(manifest: dict[str, Any], output: Path, figures: Path) -> list[Path]:
    labels = {item["id"]: item["label"] for item in manifest["datasets"]}
    drf_frames = []
    timing_frames = []
    for dataset_id, label in labels.items():
        for resource in ("cpu", "gpu"):
            stage_dir = output / "stages" / f"drf.{resource}.{dataset_id}"
            if not stage_dir.is_dir():
                continue
            drf = pd.read_csv(stage_dir / "drf.csv")
            timing = pd.read_csv(stage_dir / "timing.csv")
            drf["analysis_unit"] = label
            timing["analysis_unit"] = label
            drf_frames.append(drf)
            timing_frames.append(timing)
    drf = pd.concat(drf_frames, ignore_index=True)
    timing = pd.concat(timing_frames, ignore_index=True)
    written = []
    for statistic in ("mean", "median"):
        grouped = drf.groupby(["protocol", "analysis_unit"], observed=True)["drf"]
        table = (grouped.mean() if statistic == "mean" else grouped.median()).unstack("analysis_unit")
        csv_path = figures / f"drf_table_{statistic}.csv"
        png_path = figures / f"drf_table_{statistic}.png"
        table.to_csv(csv_path)
        _save_table(table, png_path, title=f"Dynamic range fraction ({statistic})")
        written.extend((csv_path, png_path))
    timing_table = timing.pivot(index="protocol", columns="analysis_unit", values="seconds")
    timing_table["TOTAL"] = timing_table.sum(axis=1)
    csv_path = figures / "metric_timing_table.csv"
    png_path = figures / "metric_timing_table.png"
    timing_table.to_csv(csv_path)
    _save_table(timing_table, png_path, title="Computation time (seconds)", cmap="Greys")
    written.extend((csv_path, png_path))
    return written


def _read_de(path: Path, method: str) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    with h5py.File(path) as handle:
        perturbations = [value.decode() for value in handle["perturbations"][:]]
        genes = [value.decode() for value in handle["genes"][:]]
        statistic = handle[method]["statistic"][:]
        padj = handle[method]["pvalue_adj"][:]
    return perturbations, genes, statistic, padj


def _de_pair(
    output: Path, dataset_id: str, ttest_method: str
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = output / "stages"
    ttest_path = base / f"de.{dataset_id}.all_perturbed.{ttest_method}" / "de.h5"
    mwu_path = base / f"de.{dataset_id}.all_perturbed.MWU" / "de.h5"
    perturbations, genes, tstat, tpadj = _read_de(ttest_path, ttest_method)
    mwu_perturbations, mwu_genes, mstat, mpadj = _read_de(mwu_path, "MWU")
    if perturbations != mwu_perturbations or genes != mwu_genes:
        raise RuntimeError(f"unaligned DE artifacts for {dataset_id}")
    return perturbations, tstat, tpadj, mstat, mpadj


def _criteria(ttest_label: str) -> list[tuple[str, str, str, float]]:
    result = [(f"{ttest_label} |stat|>{value:g}", "t", "stat", value) for value in TTEST_STAT_THRESHOLDS]
    result += [(f"{ttest_label} padj<{value:g}", "t", "padj", value) for value in PADJ_THRESHOLDS]
    result += [(f"MWU |delta|>{value:g}", "m", "stat", value) for value in MWU_STAT_THRESHOLDS]
    result += [(f"MWU padj<{value:g}", "m", "padj", value) for value in PADJ_THRESHOLDS]
    return result


def _mask(
    criterion: tuple[str, str, str, float], tstat: np.ndarray, tpadj: np.ndarray, mstat: np.ndarray, mpadj: np.ndarray
) -> np.ndarray:
    _, method, kind, threshold = criterion
    statistic, padj = (tstat, tpadj) if method == "t" else (mstat, mpadj)
    values = statistic if kind == "stat" else padj
    return np.isfinite(values) & ((np.abs(values) > threshold) if kind == "stat" else (values < threshold))


def _mean_jaccard_and_sizes(
    criteria: list[tuple[str, str, str, float]],
    tstat: np.ndarray,
    tpadj: np.ndarray,
    mstat: np.ndarray,
    mpadj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros((len(criteria), len(criteria)), dtype=float)
    sizes = np.zeros((tstat.shape[0], len(criteria)), dtype=float)
    for row in range(tstat.shape[0]):
        masks = [_mask(item, tstat[row], tpadj[row], mstat[row], mpadj[row]) for item in criteria]
        sizes[row] = [mask.sum() for mask in masks]
        for left, left_mask in enumerate(masks):
            for right, right_mask in enumerate(masks):
                union = np.count_nonzero(left_mask | right_mask)
                total[left, right] += np.count_nonzero(left_mask & right_mask) / union if union else 0.0
    return total / max(tstat.shape[0], 1), np.median(sizes, axis=0)


def _spearman_values(tpadj: np.ndarray, mpadj: np.ndarray) -> np.ndarray:
    values = np.full(tpadj.shape[0], np.nan, dtype=float)
    for row in range(tpadj.shape[0]):
        valid = np.isfinite(tpadj[row]) & np.isfinite(mpadj[row])
        if valid.sum() < 2:
            continue
        left = -np.log10(np.clip(tpadj[row, valid], 1e-300, None))
        right = -np.log10(np.clip(mpadj[row, valid], 1e-300, None))
        if np.ptp(left) == 0 or np.ptp(right) == 0:
            continue
        correlation = spearmanr(left, right).statistic
        values[row] = float(correlation) if np.isfinite(correlation) else np.nan
    return values


def _k_grid(n_genes: int, kmax: int = 500) -> np.ndarray:
    """Return the legacy Squair-style doubling grid plus fractional anchors."""
    limit = min(int(kmax), int(n_genes))
    if limit < 1:
        return np.array([], dtype=int)
    values = set()
    k = 1
    while k <= limit:
        values.add(k)
        k *= 2
    values.update(max(1, round(fraction * limit)) for fraction in (0.1, 0.25, 0.5, 0.75, 1.0))
    values.add(limit)
    return np.array(sorted(values), dtype=int)


def _concordance_rows(
    dataset: str,
    perturbations: list[str],
    tstat: np.ndarray,
    tpadj: np.ndarray,
    mstat: np.ndarray,
    mpadj: np.ndarray,
) -> list[dict[str, Any]]:
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    aucc_values = []
    for row, _perturbation in enumerate(perturbations):
        valid = np.isfinite(tstat[row]) & np.isfinite(tpadj[row]) & np.isfinite(mstat[row]) & np.isfinite(mpadj[row])
        if valid.sum() < 4:
            continue
        t_order = np.lexsort((-np.abs(tstat[row, valid]), np.clip(tpadj[row, valid], 1e-300, None)))
        m_order = np.lexsort((-np.abs(mstat[row, valid]), np.clip(mpadj[row, valid], 1e-300, None)))
        grid = _k_grid(len(t_order))
        values = np.array([len(set(t_order[:k]) & set(m_order[:k])) / k for k in grid], dtype=float)
        curves.append((grid, values))
        aucc_values.append(float(np.mean(values)))
    if not curves:
        return []
    shared_grid = _k_grid(min(int(grid.max()) for grid, _values in curves))
    interpolated = np.empty((len(curves), len(shared_grid)), dtype=float)
    for row, (grid, values) in enumerate(curves):
        interpolated[row] = np.interp(np.log(shared_grid), np.log(grid), values)
    median_aucc = float(np.median(aucc_values))
    return [
        {
            "dataset": dataset,
            "k": int(k),
            "median": float(np.median(interpolated[:, index])),
            "p25": float(np.percentile(interpolated[:, index], 25)),
            "p75": float(np.percentile(interpolated[:, index], 75)),
            "n_perturbations": len(curves),
            "median_aucc": median_aucc,
        }
        for index, k in enumerate(shared_grid)
    ]


def _plot_jaccard(
    matrices: list[tuple[str, np.ndarray]],
    labels: list[str],
    median_sizes: list[np.ndarray],
    spearman_values: list[np.ndarray],
    path: Path,
    title: str,
) -> None:
    columns = min(4, len(matrices))
    rows = int(np.ceil(len(matrices) / columns))
    fig = plt.figure(figsize=(4.5 * columns, 8 * rows + 4.5), layout="constrained")
    outer = fig.add_gridspec(3, 1, height_ratios=[5.5 * rows, 2.5 * rows, 4.5])
    heat_grid = outer[0].subgridspec(rows, columns)
    axes = np.asarray([[fig.add_subplot(heat_grid[row, column]) for column in range(columns)] for row in range(rows)])
    image = None
    for ax, (dataset, matrix) in zip(axes.flat, matrices):
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_title(dataset)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
    for ax in axes.flat[len(matrices) :]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.55, label="Mean Jaccard")
    distribution_grid = outer[1].subgridspec(rows, columns)
    for index, (dataset, _matrix) in enumerate(matrices):
        distribution_ax = fig.add_subplot(distribution_grid[index // columns, index % columns])
        values = spearman_values[index]
        finite = values[np.isfinite(values)]
        if finite.size:
            distribution_ax.hist(finite, bins=20, range=(0, 1), color="#4477aa", edgecolor="white")
            distribution_ax.axvline(float(np.median(finite)), color="crimson", linestyle="--", linewidth=1)
        distribution_ax.set_xlim(0, 1)
        distribution_ax.set_title(dataset, fontsize=9)
        distribution_ax.set_xlabel("Spearman correlation", fontsize=8)
        if index == 0:
            distribution_ax.set_ylabel("Perturbations", fontsize=8)
        distribution_ax.tick_params(labelsize=7)
    size_ax = fig.add_subplot(outer[2])
    size_ax.axis("off")
    size_table = size_ax.table(
        cellText=np.rint(np.column_stack(median_sizes)).astype(int),
        rowLabels=labels,
        colLabels=[dataset for dataset, _matrix in matrices],
        cellLoc="center",
        loc="center",
    )
    size_table.auto_set_font_size(False)
    size_table.set_fontsize(6)
    size_table.scale(1.0, 1.2)
    size_ax.set_title("Median number of genes selected per criterion", fontweight="bold", pad=10)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_concordance(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, group in frame.groupby("dataset", sort=False):
        aucc = float(group["median_aucc"].iloc[0])
        ax.plot(group["k"], group["median"], marker="o", markersize=2, label=f"{dataset} (AUCC={aucc:.2f})")
        ax.fill_between(group["k"], group["p25"], group["p75"], alpha=0.12)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Top-k genes")
    ax.set_ylabel("Top-k overlap (Concordance@k)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_de(manifest: dict[str, Any], output: Path, figures: Path) -> list[Path]:
    written: list[Path] = []
    for method, suffix, label in (
        ("t-test", "", "Welch t-test"),
        ("t-test_overestim_var", "_overestim", "Conservative t-test"),
    ):
        criteria = _criteria(label)
        criterion_labels = [item[0] for item in criteria]
        matrices = []
        median_sizes = []
        source_rows = []
        spearman_values = []
        concordance = []
        for spec in manifest["datasets"]:
            perturbations, tstat, tpadj, mstat, mpadj = _de_pair(output, spec["id"], method)
            matrix, dataset_sizes = _mean_jaccard_and_sizes(criteria, tstat, tpadj, mstat, mpadj)
            correlations = _spearman_values(tpadj, mpadj)
            matrices.append((spec["label"], matrix))
            median_sizes.append(dataset_sizes)
            spearman_values.append(correlations)
            source_rows.extend(
                {
                    "record_type": "jaccard",
                    "dataset": spec["label"],
                    "left_criterion": left_label,
                    "right_criterion": right_label,
                    "mean_jaccard": float(matrix[left, right]),
                }
                for left, left_label in enumerate(criterion_labels)
                for right, right_label in enumerate(criterion_labels)
            )
            source_rows.extend(
                {
                    "record_type": "set_size",
                    "dataset": spec["label"],
                    "criterion": criterion,
                    "median_set_size": float(size),
                }
                for criterion, size in zip(criterion_labels, dataset_sizes)
            )
            source_rows.extend(
                {
                    "record_type": "spearman",
                    "dataset": spec["label"],
                    "perturbation": perturbation,
                    "spearman": float(correlation),
                }
                for perturbation, correlation in zip(perturbations, correlations)
            )
            concordance.extend(_concordance_rows(spec["label"], perturbations, tstat, tpadj, mstat, mpadj))
        jaccard_stem = f"deg_jaccard{suffix}_with_counts"
        concordance_stem = f"deg_concordance{suffix}"
        jaccard_csv = figures / f"{jaccard_stem}.csv"
        jaccard_png = figures / f"{jaccard_stem}.png"
        concordance_csv = figures / f"{concordance_stem}.csv"
        concordance_png = figures / f"{concordance_stem}.png"
        pd.DataFrame(source_rows).to_csv(jaccard_csv, index=False)
        _plot_jaccard(
            matrices,
            criterion_labels,
            median_sizes,
            spearman_values,
            jaccard_png,
            f"DEG-set agreement: {label} vs MWU",
        )
        concordance_frame = pd.DataFrame(concordance)
        concordance_frame.to_csv(concordance_csv, index=False)
        _plot_concordance(concordance_frame, concordance_png, f"Top-gene concordance: {label} vs MWU")
        written.extend((jaccard_csv, jaccard_png, concordance_csv, concordance_png))
    return written


def validate_production(manifest_path: Path, output: Path) -> Path:
    manifest = load_manifest(manifest_path)
    validate_input_record(manifest_path, output, rehash_data=True)
    inputs = stage_inputs(manifest_path, output)
    errors = []
    stage_inventory = []
    for stage in production_stages(manifest):
        record = output / "stages" / stage.id / "stage.json"
        if not record.is_file():
            errors.append(f"missing stage record: {stage.id}")
            continue
        try:
            payload = json.loads(record.read_text())
        except (OSError, json.JSONDecodeError):
            errors.append(f"invalid stage record: {stage.id}")
            continue
        if payload.get("status") != "complete":
            errors.append(f"stage is not complete: {stage.id}")
        if payload.get("inputs") != inputs:
            errors.append(f"stage used different inputs or evaluator environment: {stage.id}")
        checksums = payload.get("artifact_sha256", {})
        if not checksums:
            errors.append(f"stage has no artifact checksums: {stage.id}")
        for artifact_name, expected in checksums.items():
            try:
                artifact = resolve_record_artifact(record, artifact_name)
            except ValueError as error:
                errors.append(str(error))
                continue
            if not artifact.is_file():
                errors.append(f"stage artifact is missing: {artifact}")
            elif file_sha256(artifact) != expected:
                errors.append(f"stage artifact checksum mismatch: {artifact}")
        if stage.dataset is None:
            continue
        spec = next(item for item in manifest["datasets"] if item["id"] == stage.dataset)
        perturbations = _declared_perturbations(spec)
        if perturbations is None:
            if stage.kind == "de":
                with h5py.File(record.parent / "de.h5") as handle:
                    perturbations = [value.decode() for value in handle["perturbations"][:]]
            else:
                perturbations = sorted(pd.read_csv(record.parent / "drf.csv")["perturbation"].astype(str).unique())
        stage_dir = record.parent
        stage_artifacts = [stage_dir / "de.h5"] if stage.kind == "de" else [stage_dir / "drf.csv", stage_dir / "timing.csv"]
        if not _reusable_record(record, inputs, stage_artifacts):
            errors.append(f"stage record or artifact checksums are not reusable: {stage.id}")
        if stage.kind == "de" and not _valid_de(
            stage_dir / "de.h5",
            stage,
            perturbations,
            target=str(manifest["de"]["target"]),
            subsample=int(manifest["reference_subsample"]),
            seed=int(manifest["seed"]),
        ):
            errors.append(f"DE artifact failed semantic validation: {stage.id}")
        if stage.kind in {"drf-cpu", "drf-gpu"}:
            protocols = list(manifest["gpu_protocols"] if stage.kind == "drf-gpu" else manifest["protocols"])
            if not _valid_drf(
                stage_dir / "drf.csv",
                stage_dir / "timing.csv",
                protocols,
                perturbations,
                seed=int(manifest["seed"]),
                subsample=int(manifest["reference_subsample"]),
            ):
                errors.append(f"DRF artifact failed frozen-cohort validation: {stage.id}")
        stage_inventory.append(
            {
                "stage": stage.id,
                "record_sha256": file_sha256(record),
                "artifacts": payload.get("artifact_sha256", {}),
            }
        )
    figures = output / "figures"
    for stem in FIGURE_STEMS:
        for suffix in (".csv", ".png"):
            path = figures / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty rendered artifact: {path}")
                continue
            try:
                if suffix == ".png":
                    image = plt.imread(path)
                    if image.ndim not in {2, 3} or min(image.shape[:2]) < 100:
                        errors.append(f"rendered image has invalid dimensions: {path}")
                else:
                    frame = pd.read_csv(path)
                    if frame.empty:
                        errors.append(f"rendered table has no rows: {path}")
            except (OSError, ValueError, pd.errors.ParserError) as error:
                errors.append(f"cannot read rendered artifact {path}: {error}")
    for stem in ("deg_jaccard_with_counts", "deg_jaccard_overestim_with_counts"):
        path = figures / f"{stem}.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            required = {"record_type", "dataset"}
            if not required <= set(frame) or set(frame["record_type"]) != {"jaccard", "set_size", "spearman"}:
                errors.append(f"Jaccard source table has an invalid schema: {path}")
    for stem in ("deg_concordance", "deg_concordance_overestim"):
        path = figures / f"{stem}.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            required = {"dataset", "k", "median", "p25", "p75", "n_perturbations", "median_aucc"}
            if not required <= set(frame):
                errors.append(f"concordance source table has an invalid schema: {path}")
    if errors:
        raise RuntimeError("production validation failed:\n- " + "\n- ".join(errors))
    artifacts = sorted(path for path in figures.iterdir() if path.is_file())
    completion = output / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "inputs": inputs,
                "n_compute_stages": len(production_stages(manifest)),
                "stages": stage_inventory,
                "artifacts": [
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in artifacts
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return completion


def render_and_validate(
    manifest_path: Path,
    output: Path,
    *,
    baseline_results: Path | None = None,
) -> Path:
    manifest = load_manifest(manifest_path)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    render_drf(manifest, output, figures)
    render_de(manifest, output, figures)
    if baseline_results is not None:
        from .equivalence import compare_results

        compare_results(manifest_path, output, baseline_results)
    return validate_production(manifest_path, output)
