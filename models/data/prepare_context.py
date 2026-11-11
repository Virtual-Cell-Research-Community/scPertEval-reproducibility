"""Prepare a bounded, combined model view for genetic or chemical context transfer."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from models.data.prepare_data import prepare_canonical_raw
from workflows.preparation_utils import file_checksum

SEED = 42


def _sciplex_gene_panel(data: ad.AnnData, config: dict) -> tuple[ad.AnnData, dict]:
    """Apply the frozen global HVG + per-condition DE panel to a sci-Plex view."""
    panel = config.get("gene_panel")
    if panel is None:
        return data, {}

    n_hvgs = int(panel["n_hvgs"])
    n_de_genes = int(panel["n_de_genes_per_condition"])
    hvg_flavor = str(panel["hvg_flavor"])
    de_method = str(panel["de_method"])
    conditions = sorted(set(data.obs["model_perturbation"].astype(str)) - {"control"})
    expected_conditions = int(config["n_point_conditions"])
    if len(conditions) != expected_conditions:
        raise ValueError(
            f"sci-Plex gene selection found {len(conditions)} treated conditions; "
            f"expected {expected_conditions}"
        )

    sc.pp.highly_variable_genes(
        data,
        n_top_genes=min(n_hvgs, data.n_vars),
        flavor=hvg_flavor,
        inplace=True,
    )
    hvg_genes = set(data.var_names[data.var["highly_variable"].astype(bool)])

    # Pool the three contexts for each drug-dose comparison, exactly as declared in the
    # experiment config. The data are already normalized/log1p at this point; controls from
    # every context form the common reference group.
    data.obs["_gene_panel_group"] = pd.Categorical(
        data.obs["model_perturbation"].astype(str),
        categories=["control", *conditions],
    )
    key = "_gene_panel_de"
    sc.tl.rank_genes_groups(
        data,
        groupby="_gene_panel_group",
        groups=conditions,
        reference="control",
        method=de_method,
        n_genes=min(n_de_genes, data.n_vars),
        key_added=key,
        use_raw=False,
    )
    de_by_condition = {
        condition: list(map(str, data.uns[key]["names"][condition][:n_de_genes]))
        for condition in conditions
    }
    de_genes = {gene for genes in de_by_condition.values() for gene in genes}
    selected = hvg_genes | de_genes
    if not selected:
        raise ValueError("sci-Plex gene panel selection produced an empty panel")

    data.var["gene_panel_hvg"] = data.var_names.isin(hvg_genes)
    data.var["gene_panel_de"] = data.var_names.isin(de_genes)
    data.var["gene_panel_selected"] = data.var_names.isin(selected)
    del data.obs["_gene_panel_group"]
    del data.uns[key]
    result = data[:, data.var["gene_panel_selected"].to_numpy()].copy()
    metadata = {
        "scope": "global-across-all-contexts",
        "hvg": {
            "flavor": hvg_flavor,
            "requested": n_hvgs,
            "selected": len(hvg_genes),
        },
        "differential_expression": {
            "method": de_method,
            "reference": "pooled-controls-across-all-contexts",
            "genes_per_condition": n_de_genes,
            "n_conditions": len(conditions),
            "unique_genes": len(de_genes),
        },
        "union_genes": result.n_vars,
    }
    result.uns["gene_panel"] = metadata
    result.uns["gene_panel_de_by_condition"] = de_by_condition
    return result, metadata


def _condition_rng(seed: int, context: str, condition: str) -> np.random.Generator:
    import zlib

    tags = [seed, zlib.crc32(context.encode()), zlib.crc32(condition.encode())]
    return np.random.default_rng(np.asarray(tags, dtype=np.uint32))


def _sample(
    obs: pd.DataFrame,
    *,
    context: str,
    conditions: list[str],
    minimum_cells: int,
    max_cells: int,
    max_controls: int,
    seed: int,
) -> np.ndarray:
    labels = obs["model_perturbation"].astype(str).to_numpy()
    is_control = obs["is_control"].astype(bool).to_numpy()
    keep = []
    controls = np.where(is_control)[0]
    if not len(controls):
        raise ValueError(f"{context} has no control cells")
    if len(controls) > max_controls:
        controls = _condition_rng(seed, context, "control").choice(controls, max_controls, replace=False)
    keep.extend(controls.tolist())
    for condition in sorted(conditions):
        indices = np.where((labels == condition) & ~is_control)[0]
        if len(indices) < minimum_cells:
            raise ValueError(f"{context}::{condition} has {len(indices)} cells; expected at least {minimum_cells}")
        if len(indices) > max_cells:
            indices = _condition_rng(seed, context, condition).choice(indices, max_cells, replace=False)
        keep.extend(indices.tolist())
    return np.sort(np.asarray(keep))


def _load_rows(path: Path, mask: np.ndarray) -> ad.AnnData:
    source = ad.read_h5ad(path, backed="r")
    try:
        return source[mask].to_memory()
    finally:
        source.file.close()


def _prepare_genetic_context(
    context: str,
    path: Path,
    perturbations: list[str],
    *,
    minimum_cells: int,
    max_cells: int,
    max_controls: int,
    seed: int,
) -> ad.AnnData:
    source = ad.read_h5ad(path, backed="r")
    try:
        source_labels = source.obs["perturbation"].astype(str)
        source_controls = source.obs["is_control"].astype(bool)
        mask = (source_controls | source_labels.isin(perturbations)).to_numpy()
    finally:
        source.file.close()
    data = _load_rows(path, mask)
    data.obs["model_perturbation"] = np.where(
        data.obs["is_control"].astype(bool),
        "control",
        data.obs["perturbation"].astype(str),
    )
    indices = _sample(
        data.obs,
        context=context,
        conditions=perturbations,
        minimum_cells=minimum_cells,
        max_cells=max_cells,
        max_controls=max_controls,
        seed=seed,
    )
    data = data[indices].copy()
    data = prepare_canonical_raw(data, min_cells=1)
    data.obs["model_perturbation"] = data.obs["perturbation"].astype(str)
    data.obs["context"] = context
    data.obs["cell_type"] = context
    data.obs["evaluation_condition"] = np.where(
        data.obs["perturbation"].astype(str) == "control",
        "control",
        context + "::" + data.obs["perturbation"].astype(str),
    )
    data.obs_names = pd.Index([f"{context}::{name}" for name in data.obs_names])
    return data


def _chemical_model_label(perturbation: str, dose: float) -> str:
    return f"{perturbation}::{float(dose):g}nM"


def _prepare_chemical_contexts(
    path: Path,
    config: dict,
    *,
    minimum_cells: int,
    max_cells: int,
    max_controls: int,
    seed: int,
) -> list[ad.AnnData]:
    drug_labels = {item["canonical_label"] for item in config["drugs"]}
    source = ad.read_h5ad(path, backed="r")
    try:
        source_context = source.obs["cell_line"].astype(str)
        source_perturbation = source.obs["perturbation"].astype(str)
        source_control = source.obs["is_control"].astype(bool)
        mask = (
            source_context.isin(config["contexts"])
            & (source_control | source_perturbation.isin(drug_labels))
        ).to_numpy()
    finally:
        source.file.close()
    selected = _load_rows(path, mask)
    prepared = []
    for context in config["contexts"]:
        data = selected[selected.obs["cell_line"].astype(str) == context].copy()
        is_control = data.obs["is_control"].astype(bool).to_numpy()
        data.obs["model_perturbation"] = [
            "control" if control else _chemical_model_label(perturbation, dose)
            for control, perturbation, dose in zip(
                is_control,
                data.obs["perturbation"].astype(str),
                data.obs["dose"],
                strict=True,
            )
        ]
        conditions = [
            _chemical_model_label(drug["canonical_label"], dose)
            for drug in config["drugs"]
            for dose in config["doses"]
        ]
        # The point stratum includes all 36 conditions; the three population exclusions are
        # allowed below 50 but still must be present. Enforce the configured population floor
        # only for the 33 retained conditions.
        population_excluded = {
            _chemical_model_label(
                next(item["canonical_label"] for item in config["drugs"] if item["label"] == exclusion["drug"]),
                exclusion["dose"],
            )
            for exclusion in config["population_exclusions"]
        }
        labels = data.obs["model_perturbation"].astype(str).to_numpy()
        for condition in conditions:
            count = int(np.sum(labels == condition))
            required = 1 if condition in population_excluded else minimum_cells
            if count < required:
                raise ValueError(f"{context}::{condition} has {count} cells; expected at least {required}")
        indices = _sample(
            data.obs,
            context=context,
            conditions=conditions,
            minimum_cells=1,
            max_cells=max_cells,
            max_controls=max_controls,
            seed=seed,
        )
        data = data[indices].copy()
        original_model_labels = data.obs["model_perturbation"].astype(str).to_numpy()
        data.obs["perturbation"] = original_model_labels
        data = prepare_canonical_raw(data, min_cells=1)
        data.obs["model_perturbation"] = data.obs["perturbation"].astype(str)
        data.obs["context"] = context
        data.obs["cell_type"] = context
        data.obs["evaluation_condition"] = np.where(
            data.obs["perturbation"].astype(str) == "control",
            "control",
            context + "::" + data.obs["perturbation"].astype(str),
        )
        data.obs_names = pd.Index([f"{context}::{name}" for name in data.obs_names])
        prepared.append(data)
    return prepared


def prepare_context_view(
    config: dict,
    inputs: dict[str, Path],
    *,
    minimum_cells: int,
    max_cells: int,
    max_controls: int,
    seed: int,
) -> ad.AnnData:
    """Build the combined normalized view while keeping model and evaluation IDs separate."""
    if "perturbations" in config:
        missing = sorted(set(config["contexts"]) - set(inputs))
        if missing:
            raise ValueError(f"missing context inputs: {', '.join(missing)}")
        datasets = [
            _prepare_genetic_context(
                context,
                inputs[context],
                config["perturbations"],
                minimum_cells=minimum_cells,
                max_cells=max_cells,
                max_controls=max_controls,
                seed=seed,
            )
            for context in config["contexts"]
        ]
    else:
        if set(inputs) != {config["dataset"]}:
            raise ValueError(f"chemical context view requires exactly one input named {config['dataset']!r}")
        datasets = _prepare_chemical_contexts(
            inputs[config["dataset"]],
            config,
            minimum_cells=minimum_cells,
            max_cells=max_cells,
            max_controls=max_controls,
            seed=seed,
        )
    combined = ad.concat(datasets, join="inner", merge="same", index_unique=None)
    sc.pp.filter_genes(combined, min_cells=3)
    combined.var["gene_name"] = combined.var_names
    if "drugs" in config:
        combined, _ = _sciplex_gene_panel(combined, config)
    return combined


def _write_context_metadata(output: Path, result: ad.AnnData) -> dict:
    counts = (
        result.obs.assign(_n_cells=1)
        .groupby(["context", "model_perturbation"], observed=True)["_n_cells"]
        .sum()
        .rename("n_cells")
        .reset_index()
        .sort_values(["context", "model_perturbation"])
    )
    counts_path = output.with_name(f"{output.stem}.cell-counts.tsv")
    counts.to_csv(counts_path, sep="\t", index=False)
    return {
        "path": counts_path.name,
        "sha256": file_checksum(counts_path),
        "n_rows": len(counts),
        "minimum_treated_cells": int(
            counts.loc[counts["model_perturbation"] != "control", "n_cells"].min()
        ),
        "maximum_treated_cells": int(
            counts.loc[counts["model_perturbation"] != "control", "n_cells"].max()
        ),
    }


def _write_gene_panel_metadata(output: Path, result: ad.AnnData, config: dict) -> dict | None:
    if "drugs" not in config:
        return None
    panel_path = output.with_name(f"{output.stem}.gene-panel.tsv")
    columns = ["gene_panel_hvg", "gene_panel_de", "gene_panel_selected"]
    panel = result.var[columns].copy()
    panel.index.name = "gene"
    panel.to_csv(panel_path, sep="\t")
    metadata = dict(result.uns["gene_panel"])
    assignments_path = output.with_name(f"{output.stem}.gene-panel-de.tsv")
    assignments = pd.DataFrame(
        [
            {"condition": condition, "rank": rank, "gene": gene}
            for condition, genes in sorted(result.uns["gene_panel_de_by_condition"].items())
            for rank, gene in enumerate(genes, start=1)
        ]
    )
    assignments.to_csv(assignments_path, sep="\t", index=False)
    metadata["differential_expression"]["assignments"] = {
        "path": assignments_path.name,
        "sha256": file_checksum(assignments_path),
        "n_rows": len(assignments),
    }
    return {
        **metadata,
        "path": panel_path.name,
        "sha256": file_checksum(panel_path),
    }


def _atomic_write_h5ad(data: ad.AnnData, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    os.close(fd)
    try:
        data.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("each --input must be CONTEXT=/path/to/adata.h5ad")
        context, path = value.split("=", 1)
        result[context] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True, help="CONTEXT=/path/to/canonical.h5ad")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--max-cells", type=int, default=100)
    parser.add_argument("--max-controls", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    inputs = _parse_inputs(args.input)
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    input_checksums = {name: file_checksum(path) for name, path in inputs.items()}
    if args.output.resolve() in {path.resolve() for path in inputs.values()}:
        raise ValueError("--output must not overwrite a canonical input")
    result = prepare_context_view(
        config,
        inputs,
        minimum_cells=args.minimum_cells,
        max_cells=args.max_cells,
        max_controls=args.max_controls,
        seed=args.seed,
    )
    _atomic_write_h5ad(result, args.output)
    for name, path in inputs.items():
        if file_checksum(path) != input_checksums[name]:
            raise RuntimeError(f"canonical input changed during preparation: {path}")
    counts = _write_context_metadata(args.output, result)
    gene_panel = _write_gene_panel_metadata(args.output, result, config)
    manifest = {
        "schema_version": 1,
        "experiment": config["id"],
        "config_sha256": file_checksum(args.config),
        "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": {
            name: {"path": str(path), "sha256": input_checksums[name], "verified_unchanged": True}
            for name, path in inputs.items()
        },
        "output": {
            "path": args.output.name,
            "sha256": file_checksum(args.output),
            "n_obs": result.n_obs,
            "n_vars": result.n_vars,
        },
        "sampling": {
            "minimum_cells": args.minimum_cells,
            "max_cells": args.max_cells,
            "max_controls": args.max_controls,
            "seed": args.seed,
        },
        "cell_counts": counts,
    }
    if gene_panel is not None:
        manifest["gene_panel"] = gene_panel
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {result.shape} context-transfer model view to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
