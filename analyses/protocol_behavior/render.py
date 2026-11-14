"""Render and validate the production protocol-behavior result set."""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .production import file_sha256, load_manifest, manifest_sha256, production_stages

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        _save_table(table, png_path, title=f"Dataset-rescaling factor ({statistic})")
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


def _de_pair(output: Path, dataset_id: str, ttest_method: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _mask(criterion: tuple[str, str, str, float], tstat: np.ndarray, tpadj: np.ndarray, mstat: np.ndarray, mpadj: np.ndarray) -> np.ndarray:
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


def _concordance_rows(dataset: str, perturbations: list[str], tstat: np.ndarray, tpadj: np.ndarray, mstat: np.ndarray, mpadj: np.ndarray) -> list[dict[str, Any]]:
    grid = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 500])
    curves = []
    for row, _perturbation in enumerate(perturbations):
        valid = np.isfinite(tstat[row]) & np.isfinite(tpadj[row]) & np.isfinite(mstat[row]) & np.isfinite(mpadj[row])
        if valid.sum() < 4:
            continue
        t_order = np.lexsort((-np.abs(tstat[row, valid]), np.clip(tpadj[row, valid], 1e-300, None)))
        m_order = np.lexsort((-np.abs(mstat[row, valid]), np.clip(mpadj[row, valid], 1e-300, None)))
        limit = min(len(t_order), int(grid[-1]))
        row_values = []
        for k in grid:
            actual = min(int(k), limit)
            row_values.append(len(set(t_order[:actual]) & set(m_order[:actual])) / actual)
        curves.append(row_values)
    values = np.asarray(curves, dtype=float)
    if not len(values):
        return []
    return [
        {
            "dataset": dataset,
            "k": int(k),
            "median": float(np.median(values[:, index])),
            "p25": float(np.percentile(values[:, index], 25)),
            "p75": float(np.percentile(values[:, index], 75)),
        }
        for index, k in enumerate(grid)
    ]


def _plot_jaccard(matrices: list[tuple[str, np.ndarray]], labels: list[str], path: Path, title: str) -> None:
    columns = min(4, len(matrices))
    rows = int(np.ceil(len(matrices) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.5 * columns, 5.5 * rows),
        squeeze=False,
        layout="constrained",
    )
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
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_concordance(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, group in frame.groupby("dataset", sort=False):
        ax.plot(group["k"], group["median"], marker="o", markersize=2, label=dataset)
        ax.fill_between(group["k"], group["p25"], group["p75"], alpha=0.12)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Top-k genes")
    ax.set_ylabel("Concordance@k")
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
        sizes = []
        concordance = []
        for spec in manifest["datasets"]:
            perturbations, tstat, tpadj, mstat, mpadj = _de_pair(output, spec["id"], method)
            matrix, median_sizes = _mean_jaccard_and_sizes(criteria, tstat, tpadj, mstat, mpadj)
            matrices.append((spec["label"], matrix))
            sizes.extend(
                {"dataset": spec["label"], "criterion": criterion, "median_set_size": float(size)}
                for criterion, size in zip(criterion_labels, median_sizes)
            )
            concordance.extend(_concordance_rows(spec["label"], perturbations, tstat, tpadj, mstat, mpadj))
        jaccard_stem = f"deg_jaccard{suffix}_with_counts"
        concordance_stem = f"deg_concordance{suffix}"
        jaccard_csv = figures / f"{jaccard_stem}.csv"
        jaccard_png = figures / f"{jaccard_stem}.png"
        concordance_csv = figures / f"{concordance_stem}.csv"
        concordance_png = figures / f"{concordance_stem}.png"
        pd.DataFrame(sizes).to_csv(jaccard_csv, index=False)
        _plot_jaccard(matrices, criterion_labels, jaccard_png, f"DEG-set agreement: {label} vs MWU")
        concordance_frame = pd.DataFrame(concordance)
        concordance_frame.to_csv(concordance_csv, index=False)
        _plot_concordance(concordance_frame, concordance_png, f"Top-gene concordance: {label} vs MWU")
        written.extend((jaccard_csv, jaccard_png, concordance_csv, concordance_png))
    return written


def validate_production(manifest_path: Path, output: Path) -> Path:
    manifest = load_manifest(manifest_path)
    manifest_digest = manifest_sha256(manifest_path)
    errors = []
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
        if payload.get("manifest_sha256") != manifest_digest:
            errors.append(f"stage used a different manifest revision: {stage.id}")
        checksums = payload.get("artifact_sha256", {})
        if not checksums:
            errors.append(f"stage has no artifact checksums: {stage.id}")
        for artifact_name, expected in checksums.items():
            artifact = Path(artifact_name)
            if not artifact.is_file():
                errors.append(f"stage artifact is missing: {artifact}")
            elif file_sha256(artifact) != expected:
                errors.append(f"stage artifact checksum mismatch: {artifact}")
    figures = output / "figures"
    for stem in FIGURE_STEMS:
        for suffix in (".csv", ".png"):
            path = figures / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty rendered artifact: {path}")
    if errors:
        raise RuntimeError("production validation failed:\n- " + "\n- ".join(errors))
    artifacts = sorted(path for path in figures.iterdir() if path.is_file())
    completion = output / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256(manifest_path),
                "n_compute_stages": len(production_stages(manifest)),
                "artifacts": [
                    {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
                    for path in artifacts
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return completion


def render_and_validate(manifest_path: Path, output: Path) -> Path:
    manifest = load_manifest(manifest_path)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    render_drf(manifest, output, figures)
    render_de(manifest, output, figures)
    return validate_production(manifest_path, output)
