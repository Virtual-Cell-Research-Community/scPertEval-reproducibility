"""Compare production-v2 outputs with the seven shared frozen-v1 analysis units."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .production import load_manifest, production_stages


def _decoded(handle: h5py.File, name: str) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in handle[name][:]]


def _numeric_comparison(stage: str, column: str, old: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(old) & np.isfinite(new)
    differences = np.abs(old[finite] - new[finite])
    return {
        "stage": stage,
        "comparison": column,
        "n_values": int(old.size),
        "n_finite_shared": int(finite.sum()),
        "n_nonfinite_old": int((~np.isfinite(old)).sum()),
        "n_nonfinite_new": int((~np.isfinite(new)).sum()),
        "max_abs_difference": float(differences.max()) if differences.size else np.nan,
        "mean_abs_difference": float(differences.mean()) if differences.size else np.nan,
        "equivalent": bool(np.allclose(old, new, rtol=1e-7, atol=1e-10, equal_nan=True)),
    }


def _compare_de(stage_id: str, new_path: Path, old_path: Path, method: str) -> list[dict[str, Any]]:
    with h5py.File(old_path) as old, h5py.File(new_path) as new:
        old_perturbations, new_perturbations = _decoded(old, "perturbations"), _decoded(new, "perturbations")
        old_genes, new_genes = _decoded(old, "genes"), _decoded(new, "genes")
        rows = [
            {
                "stage": stage_id,
                "comparison": "perturbation_order",
                "n_values": len(new_perturbations),
                "equivalent": old_perturbations == new_perturbations,
            },
            {
                "stage": stage_id,
                "comparison": "gene_order",
                "n_values": len(new_genes),
                "equivalent": old_genes == new_genes,
            },
        ]
        if old_perturbations == new_perturbations and old_genes == new_genes:
            for name in ("statistic", "pvalue_adj"):
                rows.append(
                    _numeric_comparison(
                        stage_id,
                        name,
                        np.asarray(old[method][name][:], dtype=float),
                        np.asarray(new[method][name][:], dtype=float),
                    )
                )
        return rows


def _compare_drf(stage_id: str, new_path: Path, old_path: Path) -> list[dict[str, Any]]:
    old, new = pd.read_csv(old_path), pd.read_csv(new_path)
    keys = ["protocol", "perturbation"]
    merged = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True, sort=False)
    rows: list[dict[str, Any]] = [
        {
            "stage": stage_id,
            "comparison": "protocol_perturbation_pairs",
            "n_values": len(merged),
            "equivalent": bool((merged["_merge"] == "both").all()),
        }
    ]
    if (merged["_merge"] == "both").all():
        for column in ("raw_positive", "raw_negative", "drf"):
            rows.append(
                _numeric_comparison(
                    stage_id,
                    column,
                    merged[f"{column}_old"].to_numpy(dtype=float),
                    merged[f"{column}_new"].to_numpy(dtype=float),
                )
            )
    new_reasons = new.get("undefined_reason", pd.Series(index=new.index, dtype=object)).dropna()
    rows.append(
        {
            "stage": stage_id,
            "comparison": "structured_undefined_outcomes",
            "n_values": len(new),
            "n_nonfinite_new": int(new["drf"].isna().sum()),
            "undefined_reason_count": len(new_reasons),
            "undefined_reasons": ";".join(sorted(set(new_reasons.astype(str)))),
            "equivalent": True,
        }
    )
    return rows


def compare_results(manifest_path: Path, output: Path, baseline: Path) -> tuple[Path, Path]:
    """Write a machine-readable and concise comparison for every shared v1/v2 stage."""
    manifest = load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    compared_stages = 0
    compared_units: set[str] = set()
    for stage in production_stages(manifest):
        if stage.dataset == "essential-k562":
            continue
        new_dir = output / "stages" / stage.id
        old_dir = baseline / "stages" / stage.id
        if not old_dir.is_dir():
            raise FileNotFoundError(f"frozen-v1 comparison stage is missing: {old_dir}")
        compared_stages += 1
        assert stage.dataset is not None
        compared_units.add(stage.dataset)
        if stage.kind == "de":
            assert stage.method is not None
            rows.extend(_compare_de(stage.id, new_dir / "de.h5", old_dir / "de.h5", stage.method))
        else:
            rows.extend(_compare_drf(stage.id, new_dir / "drf.csv", old_dir / "drf.csv"))
    frame = pd.DataFrame(rows)
    figures = output / "figures"
    csv_path = figures / "v1_v2_equivalence.csv"
    report_path = figures / "v1_v2_equivalence.md"
    frame.to_csv(csv_path, index=False)
    comparable = frame[frame["comparison"] != "structured_undefined_outcomes"]
    failures = comparable[~comparable["equivalent"].astype(bool)]
    reasons = sorted(
        {
            reason
            for value in frame.get("undefined_reasons", pd.Series(dtype=object)).dropna().astype(str)
            for reason in value.split(";")
            if reason
        }
    )
    report_path.write_text(
        "# scPertEval evaluator equivalence report\n\n"
        f"- Shared analysis units: {len(compared_units)}\n"
        f"- Shared compute stages compared: {compared_stages}\n"
        f"- Comparison rows: {len(frame)}\n"
        f"- Non-equivalent comparison rows: {len(failures)}\n"
        f"- Structured undefined reasons in v2: {', '.join(reasons) if reasons else 'none'}\n\n"
        "Numerical comparisons use `rtol=1e-7`, `atol=1e-10`, and equal NaN positions. "
        "The new Essential K562 unit has no frozen-v1 counterpart and is intentionally excluded. "
        "Timing and aggregate figures are not equivalence targets because production-v2 replaces the GWPS unit.\n"
    )
    return csv_path, report_path
