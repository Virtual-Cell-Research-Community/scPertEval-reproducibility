"""Prepare a Miller et al. 2025-consistent model view, optionally reduced to smoke-test scale.

The preferred public interface is ``--input /path/to/canonical/adata.h5ad``. Canonical
PerturbationData objects keep raw counts in ``X`` and identify controls with
``obs['is_control']``. This script creates a separate, generated model view: it applies the
historical scPertEval cell/gene filters, normalizes each cell to 10,000 counts, applies
``log1p``, and maps the canonical control cells to the model adapters' ``"control"`` label.
The canonical input is never modified.

For backwards compatibility, omitting ``--input`` loads the older pre-normalized object from
``models/data/source/<source>_processed_complete.h5ad``
— downloaded via ``docs/user-guide/datasets.md``'s ``gsutil cp gs://scperteval/processed/...``
step directly into ``models/data/source/`` rather than the project's own top-level ``data/``
dir, so this reproduction scaffold's data dependency is self-contained under ``models/``. It's
scPertEval's own generic "normalize + trim" prep (see ``docs/notebooks/02_preparing_a_dataset.ipynb``):
``filter_cells(min_genes=200)``, ``filter_genes(min_cells=3)``, ``normalize_total(1e4)`` +
``log1p``. This script then applies the parts of
Miller et al. 2025's B.2 protocol (shiftbioscience/Perturbation-Models-Outperform-Baselines's
``data/replogle22k562/get_data.py``) that scPertEval's own prep doesn't already share with it —
see ``docs/user-guide/reproducing-literature-protocols.md`` for the full comparison this closes:

- Drop perturbations whose own target gene isn't itself a measured feature. GEARS/scGPT's data
  pipeline (``gears.utils.create_cell_graph_dataset_for_prediction``) hard-crashes on these at
  predict time, and even where it doesn't crash (``gears.pertdata.PertData.get_pert_idx``'s own
  ``try/except`` at train time, which falls back to ``pert_idx=None`` — the same value used for
  real control cells) it silently trains on that cell's real post-perturbation expression while
  flagging the graph as if no gene had been targeted at all. Not something Miller's own
  ``get_data.py`` resolves either — it only prints a ``Missing genes`` diagnostic after the fact
  and leaves the cells in — but dropped here rather than left to crash or silently mistrain a model.
- Drop perturbations ``gears_uncovered_genes()`` flags: measured in our own panel, but missing
  from GEARS' own bundled GO-annotation reference (``gene2go_all.pkl``, fetched dynamically —
  see that function) regardless — same crash/silent-mislabeling failure mode as the previous
  bullet, just a different root cause (GEARS' own fixed external reference not covering the
  gene, rather than us not measuring it). Only 5 of replogle22k562's 1789 perturbed genes
  (0.28%: C19orf53, C7orf26, FAU, GNB1L, SEM1) are affected as of this reference snapshot.
- Drop perturbations ``sclambda_uncovered_genes()`` flags: same idea, different model/reference
  — scLAMBDA's own GenePT gene-embedding pickle (fetched dynamically — see that function)
  doesn't cover every gene either. Unlike the GEARS/GO-graph case, this one crashes
  unconditionally on *every* run (scLAMBDA builds its gene_embeddings dict from every
  perturbed gene in the whole dataset up front, not per-fold), so leaving it unfixed wouldn't
  just bias training on some folds — it would hard-crash every single one. 14 of
  replogle22k562's 1789 perturbed genes (0.78%: C12orf45, C1orf109, C7orf26, C9orf16,
  CCDC130, INTS2, MBTPS1, PHB, POLR2B, SPATA5, SPATA5L1, WDR61, WDR92, ZNF720) are affected
  as of this reference snapshot.
- Per-perturbation downsampling: non-control conditions capped at ``round(mean)`` cells (over
  the surviving population), control capped at ``MAX_CELLS_CONTROL`` (8192, Miller's own value).
- Gene panel restricted to the top-``N_HVGS`` (8192, Miller's value) HVGs union every remaining
  perturbed gene (guaranteed already measured, since the drop above removed anything that
  wasn't) — reduces training cost without ever losing a perturbation's own target gene.

Not applied: Miller's own >=12-cells (then >=4-cells post-downsampling) perturbation filters —
both are a no-op for replogle22k562 specifically (its minimum per-perturbation count is already
30, well above either threshold) — kept as explicit checks below rather than silently skipped,
so a future dataset where they'd actually matter doesn't fall through unnoticed.

Usage::

    python -m models.data.prepare_data \
      --input data/derived/replogle-weissman-2022-k562-essential/adata.h5ad \
      --dataset-name replogle-weissman-2022-k562-essential \
      --compatible-models gears sclambda
    python -m models.data.prepare_data \
      --input /path/to/replogle-weissman-2022-k562-gwps/adata.h5ad \
      --perturbation-key target_gene \
      --control-key control_class \
      --control-value non-targeting \
      --experiment-manifest configs/experiments/replogle-weissman-2022-k562-gwps-perturbation-generalization.json \
      --validate-metadata-only
    python -m models.data.prepare_data --smoke --dataset-name smoke_k562 --n-perts 12 --cells-per-pert 100
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from workflows.preparation_utils import assert_raw_counts

ad.settings.allow_write_nullable_strings = True

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source"

MIN_CELLS_PER_PERT = 12  # Miller's first filter, pre-downsampling
MIN_CELLS_PER_PERT_POST = 4  # Miller's second filter, post-downsampling
MAX_CELLS_CONTROL = 8192  # Miller's control cap
N_HVGS = 8192  # Miller's HVG count
SEED = 42

# GEARS' own bundled GO-annotation reference (PertData.set_pert_genes's gene2go_all.pkl,
# ~67832 genes globally) doesn't cover every gene — see gears_uncovered_genes() below for why
# perturbations targeting an uncovered gene need to be dropped for every model, not just
# gears/scgpt. Same file GEARS' own PertData.__init__ downloads unconditionally (fixed
# Dataverse URL below, taken directly from its source) into gears/pert_data/ — fetched here
# too if missing, via a plain HTTP GET (no gears/torch import needed to read a pickled dict),
# so this script doesn't depend on having run any gears/scgpt code first.
GEARS_GENE2GO_URL = "https://dataverse.harvard.edu/api/access/datafile/6153417"
GEARS_GENE2GO_PATH = HERE.parent / "gears" / "pert_data" / "gene2go_all.pkl"

# scLAMBDA's own GenePT gene-embedding reference (~134k gene symbols from NCBI, OpenAI
# text-embedding-3-large embeddings of gene summaries) doesn't cover every gene either — see
# sclambda_uncovered_genes() below. scLAMBDA's own gene_embeddings dict construction
# (models/sclambda/train_predict.py) does a bare per-gene dict lookup with no existence
# check, hard-crashing (KeyError) on any gene missing from it — and since that dict is built
# from every perturbed gene in the whole dataset (not just one fold's split), this crashes
# every single run, immediately, not just some folds. Same zip GEARS' own `pixi run -e
# sclambda fetch-embeddings` task downloads (URL/target path below, taken directly from
# models/pixi.toml) — fetched here too if missing, so this script doesn't depend on having
# run that task first.
SCLAMBDA_GENEPT_URL = "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1"
SCLAMBDA_GENEPT_PATH = (
    HERE.parent / "sclambda" / "cache" / "GenePT_emebdding_v2" / "GenePT_gene_protein_embedding_model_3_text.pickle."
)


def _target_genes(condition: str) -> list[str]:
    """A condition's own perturbed gene(s) — ``"control"`` has none, a combo splits on ``"+"``
    (scPertEval's own combination convention, docs/user-guide/datasets.md)."""
    return [] if condition == "control" else condition.split("+")


def _fetch(url: str, save_path: Path) -> None:
    """Downloads ``url`` to ``save_path`` if it isn't already there — shared by
    ``gears_uncovered_genes``/``sclambda_uncovered_genes`` below, both of which fetch a
    reference file via a plain HTTP GET rather than importing gears/scgpt/torch just to reuse
    their own model-specific download helpers (this script's whole point is running from any
    model's environment — see the module docstring).

    Two things this guards against, both hit for real running this script standalone on a
    fresh checkout: Harvard Dataverse (unlike Zenodo, empirically) returns HTTP 403 for
    urllib's own default User-Agent ("Python-urllib/x.y") — a common bot-protection heuristic,
    confirmed by testing the exact same request with/without a browser-like header — so one is
    set explicitly rather than switching HTTP clients (``requests`` isn't installed in every
    model's environment; ``urllib`` always is). And this can write to a path GEARS' own
    PertData.__init__ (via gears/patch.py's ``_atomic_download``) also populates — same
    check-then-act gap that patch fixes for GEARS' own downloads, fixed the same way here:
    write to a private temp file in the same directory first, only ``os.replace()`` into
    ``save_path`` once complete, so a concurrent reader's existence check never sees a
    partially-written file.
    """
    if save_path.exists():
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} to {save_path} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; scPertEval)"})
    fd, tmp_path = tempfile.mkstemp(dir=str(save_path.parent), prefix=".download-")
    try:
        with urllib.request.urlopen(req) as response, os.fdopen(fd, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except BaseException:
        os.remove(tmp_path)
        raise
    os.replace(tmp_path, save_path)


def gears_uncovered_genes(genes: set[str]) -> set[str]:
    """Which of ``genes`` GEARS' own bundled GO-annotation reference doesn't cover — see
    GEARS_GENE2GO_PATH above for why these need dropping. Downloads that reference itself
    (into the same path GEARS' own PertData.__init__ would use, so no duplicate download
    later) if it isn't already there."""
    _fetch(GEARS_GENE2GO_URL, GEARS_GENE2GO_PATH)
    with open(GEARS_GENE2GO_PATH, "rb") as f:
        gene2go = pickle.load(f)
    return genes - set(gene2go.keys())


def sclambda_uncovered_genes(genes: set[str]) -> set[str]:
    """Which of ``genes`` scLAMBDA's own GenePT gene-embedding reference doesn't cover — see
    SCLAMBDA_GENEPT_PATH above for why these need dropping. Downloads+extracts that reference
    itself (into the same path `pixi run -e sclambda fetch-embeddings` would use, so no
    duplicate download later) if it isn't already there."""
    if not SCLAMBDA_GENEPT_PATH.exists():
        cache_dir = SCLAMBDA_GENEPT_PATH.parent.parent
        zip_path = cache_dir / "GenePT_emebdding_v2.zip"
        _fetch(SCLAMBDA_GENEPT_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
        zip_path.unlink()
    with open(SCLAMBDA_GENEPT_PATH, "rb") as f:
        genept = pickle.load(f)
    return genes - set(genept.keys())


def canonical_annotations(
    obs: pd.DataFrame,
    *,
    perturbation_key: str = "perturbation",
    control_key: str = "is_control",
    control_value: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve source-specific annotations into canonical labels and a control mask."""
    required = {perturbation_key, control_key}
    missing = sorted(required - set(obs.columns))
    if missing:
        raise ValueError(f"canonical input is missing obs fields: {', '.join(missing)}")

    perturbations = obs[perturbation_key].astype("string")
    if perturbations.isna().any():
        raise ValueError(f"obs[{perturbation_key!r}] contains missing perturbation labels")
    perturbation_values = perturbations.astype(str).to_numpy()

    controls = obs[control_key]
    if controls.isna().any():
        raise ValueError(f"obs[{control_key!r}] contains missing control annotations")
    if control_value is None:
        unique = set(controls.unique().tolist())
        if not unique <= {True, False}:
            raise ValueError(
                f"obs[{control_key!r}] is not boolean; provide --control-value explicitly"
            )
        control_mask = controls.astype(bool).to_numpy()
    else:
        control_mask = (controls.astype(str) == str(control_value)).to_numpy()

    if not control_mask.any():
        raise ValueError("canonical input has no control cells")
    if control_mask.all():
        raise ValueError("canonical input has no perturbed cells")
    empty = np.isin(perturbation_values, ["", "nan", "<NA>"])
    if np.any(empty & ~control_mask):
        raise ValueError("canonical input has perturbed cells without a perturbation label")
    return np.where(control_mask, "control", perturbation_values), control_mask


def validate_frozen_experiment_metadata(
    adata: ad.AnnData,
    experiment: dict,
    *,
    perturbation_key: str = "perturbation",
    control_key: str = "is_control",
    control_value: str | None = None,
) -> dict:
    """Validate a large canonical object using only obs/var metadata and frozen declarations."""
    labels, control_mask = canonical_annotations(
        adata.obs,
        perturbation_key=perturbation_key,
        control_key=control_key,
        control_value=control_value,
    )
    canonical = experiment["canonical_input"]
    declared_fields = {
        "perturbation_field": perturbation_key,
        "control_field": control_key,
        "control_value": control_value,
    }
    mismatched_fields = {
        key: {"declared": canonical.get(key), "requested": value}
        for key, value in declared_fields.items()
        if key in canonical and canonical.get(key) != value
    }
    if mismatched_fields:
        raise ValueError(f"source-field mapping disagrees with the experiment: {mismatched_fields}")
    expected_shape = (int(canonical["n_obs"]), int(canonical["n_vars"]))
    if adata.shape != expected_shape:
        raise ValueError(
            "canonical dimensions do not match the experiment declaration: "
            f"observed={adata.shape}, expected={expected_shape}"
        )
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("canonical input has duplicate observation or gene identifiers")

    frozen = list(map(str, experiment["perturbations"]))
    frozen_set = set(frozen)
    if len(frozen_set) != len(frozen):
        raise ValueError("experiment declaration contains duplicate perturbations")
    counts = pd.Series(labels[~control_mask]).value_counts()
    missing = sorted(frozen_set - set(counts.index))
    minimum = int(experiment["minimum_treated_cells_per_perturbation"])
    below = {
        label: int(counts[label])
        for label in frozen
        if label in counts and counts[label] < minimum
    }
    unmeasured = sorted(frozen_set - set(map(str, adata.var_names)))
    if missing or below or unmeasured:
        raise ValueError(
            "frozen perturbation cohort does not match canonical metadata: "
            f"missing={missing[:10]}, below_minimum={dict(list(below.items())[:10])}, "
            f"unmeasured={unmeasured[:10]}"
        )

    expected_targeted = experiment.get("eligibility", {}).get("observed_perturbations")
    eligibility = experiment.get("eligibility", {})
    n_targeted = int((~control_mask).sum())
    if expected_targeted is not None and n_targeted != int(expected_targeted):
        raise ValueError(
            "targeted-cell count does not match the experiment declaration: "
            f"observed={n_targeted}, expected={expected_targeted}"
        )
    n_targets_at_minimum = int((counts >= minimum).sum())
    measured_genes = set(map(str, adata.var_names))
    n_measured_targets_at_minimum = int(
        sum(label in measured_genes for label in counts[counts >= minimum].index)
    )
    declared_counts = {
        "n_targets_total": int(counts.size),
        "n_targets_at_least_50_cells": n_targets_at_minimum,
        "n_targets_measured_at_least_50_cells": n_measured_targets_at_minimum,
        "n_targets_shared": len(frozen),
    }
    count_mismatches = {
        key: {"declared": int(eligibility[key]), "observed": observed}
        for key, observed in declared_counts.items()
        if key in eligibility and int(eligibility[key]) != observed
    }
    if count_mismatches:
        raise ValueError(f"cohort summary disagrees with the experiment: {count_mismatches}")
    frozen_counts = counts.reindex(frozen)
    return {
        "dataset": experiment["dataset"],
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "source_fields": {
            "perturbation_key": perturbation_key,
            "control_key": control_key,
            "control_value": control_value,
        },
        "n_control_cells": int(control_mask.sum()),
        "n_targeted_cells": n_targeted,
        "n_source_targets": int(counts.size),
        "n_targets_at_minimum": n_targets_at_minimum,
        "n_measured_targets_at_minimum": n_measured_targets_at_minimum,
        "n_frozen_targets": len(frozen),
        "n_frozen_targeted_cells": int(frozen_counts.sum()),
        "frozen_cells_minimum": int(frozen_counts.min()),
        "frozen_cells_median": float(frozen_counts.median()),
        "frozen_cells_maximum": int(frozen_counts.max()),
        "all_frozen_targets_measured": True,
        "minimum_cells_requirement": minimum,
        "status": "valid",
    }


def prepare_frozen_experiment_raw(
    adata: ad.AnnData,
    experiment: dict,
    *,
    perturbation_key: str = "perturbation",
    control_key: str = "is_control",
    control_value: str | None = None,
    target_sum: float = 1e4,
    smoke: tuple[int, int, int, int] | None = None,
) -> ad.AnnData:
    """Create the exact frozen-cohort model view.

    When ``smoke`` is provided as ``(n_perts, cells_per_pert, control_cells, seed)``,
    choose the bounded cell population from backed metadata before materializing expression.
    This avoids loading and normalizing the full cohort merely to discard almost all rows.
    """
    perturbations, control_mask = canonical_annotations(
        adata.obs,
        perturbation_key=perturbation_key,
        control_key=control_key,
        control_value=control_value,
    )
    frozen = set(map(str, experiment["perturbations"]))
    keep = control_mask | np.isin(perturbations, list(frozen))
    cohort_indices = np.flatnonzero(keep)
    cohort_perturbations = perturbations[keep]
    cohort_controls = control_mask[keep]

    observed = set(cohort_perturbations[~cohort_controls])
    if observed != frozen:
        raise ValueError(
            "materialized cohort does not match the frozen perturbations: "
            f"missing={sorted(frozen - observed)}, extra={sorted(observed - frozen)}"
        )
    expected_genes = int(experiment["gene_panel"]["n_genes"])
    if adata.n_vars != expected_genes:
        raise ValueError(
            f"materialized gene panel has {adata.n_vars} genes; expected {expected_genes}"
        )
    expected_cells = int(control_mask.sum()) + int(
        pd.Series(perturbations[~control_mask]).isin(frozen).sum()
    )
    if len(cohort_indices) != expected_cells:
        raise ValueError(
            f"materialized cohort has {len(cohort_indices)} cells; expected {expected_cells}"
        )

    if smoke is not None:
        n_perts, cells_per_pert, control_cells, seed = smoke
        selected = smoke_subsample_indices(
            cohort_perturbations,
            n_perts=n_perts,
            cells_per_pert=cells_per_pert,
            control_cells=control_cells,
            seed=seed,
        )
        cohort_indices = cohort_indices[selected]
        cohort_perturbations = cohort_perturbations[selected]
        cohort_controls = cohort_controls[selected]

    result = adata[cohort_indices].to_memory() if adata.isbacked else adata[cohort_indices].copy()
    result.obs["perturbation"] = cohort_perturbations
    result.obs["is_control"] = cohort_controls
    assert_raw_counts(result.X)

    # Canonical raw-count objects may retain a stale Scanpy log1p marker in ``uns`` from
    # source processing even though X has been restored to counts. assert_raw_counts above
    # is the authoritative value check; remove the stale marker before applying this view's
    # one deliberate transform so Scanpy does not emit a misleading double-log warning.
    result.uns.pop("log1p", None)
    sc.pp.normalize_total(result, target_sum=target_sum)
    sc.pp.log1p(result)
    # Older AnnData versions used by scGPT cannot read the newer H5AD ``null``
    # encoding produced for ``base=None``. An absent base has the same natural-log
    # meaning and remains readable across all pinned model environments.
    result.uns["log1p"].pop("base", None)
    return result


def prepare_canonical_raw(
    adata: ad.AnnData,
    *,
    min_genes: int = 200,
    min_cells: int = 3,
    target_sum: float = 1e4,
    retain_counts_layer: bool = False,
    perturbation_key: str = "perturbation",
    control_key: str = "is_control",
    control_value: str | None = None,
) -> ad.AnnData:
    """Create the normalized model-input view from a canonical raw-count object.

    The filtering and normalization deliberately reproduce the generic scPertEval processing
    previously applied to the hosted ``*_processed_complete.h5ad`` files. Keeping this step
    here, rather than in the dataset preparation workflow, preserves the registry contract
    that canonical ``X`` always contains raw counts.
    """
    assert_raw_counts(adata.X)

    result = adata.copy()
    perturbation, control_mask = canonical_annotations(
        result.obs,
        perturbation_key=perturbation_key,
        control_key=control_key,
        control_value=control_value,
    )
    result.obs["perturbation"] = perturbation
    result.obs["is_control"] = control_mask
    sc.pp.filter_cells(result, min_genes=min_genes)
    sc.pp.filter_genes(result, min_cells=min_cells)
    if retain_counts_layer:
        result.layers["counts"] = result.X.copy()
    sc.pp.normalize_total(result, target_sum=target_sum)
    sc.pp.log1p(result)
    result.uns["log1p"].pop("base", None)
    return result


def apply_miller_fixes(
    adata: ad.AnnData,
    seed: int = SEED,
    compatible_models: set[str] | None = None,
) -> ad.AnnData:
    """The parts of Miller et al. 2025's B.2 protocol scPertEval's own prep doesn't already
    apply — see the module docstring for what each step does and why."""
    rng = np.random.default_rng(seed)
    if compatible_models is None:
        # Preserve the historical six-model scaffold when callers do not opt into a selected
        # model set. The public ranking study passes ``{"gears", "sclambda"}`` explicitly.
        compatible_models = {"gears", "sclambda"}

    pert = adata.obs["perturbation"].to_numpy()
    var_set = set(adata.var_names)
    measured = np.array([all(g in var_set for g in _target_genes(c)) for c in pert])
    adata = adata[measured].copy()
    pert = adata.obs["perturbation"].to_numpy()

    if "gears" in compatible_models:
        # See gears_uncovered_genes() above.
        perturbed_genes = {g for c in pert for g in _target_genes(c)}
        uncovered = gears_uncovered_genes(perturbed_genes)
        if uncovered:
            print(f"dropping {len(uncovered)} perturbations GEARS' own GO reference doesn't cover: {sorted(uncovered)}")
        covered = np.array([not (set(_target_genes(c)) & uncovered) for c in pert])
        adata = adata[covered].copy()
        pert = adata.obs["perturbation"].to_numpy()

    if "sclambda" in compatible_models:
        # See sclambda_uncovered_genes() above.
        perturbed_genes = {g for c in pert for g in _target_genes(c)}
        uncovered = sclambda_uncovered_genes(perturbed_genes)
        if uncovered:
            print(
                f"dropping {len(uncovered)} perturbations scLAMBDA's own GenePT reference doesn't cover: "
                f"{sorted(uncovered)}"
            )
        covered = np.array([not (set(_target_genes(c)) & uncovered) for c in pert])
        adata = adata[covered].copy()
        pert = adata.obs["perturbation"].to_numpy()

    counts = pd.Series(pert).value_counts()
    eligible = counts[(counts >= MIN_CELLS_PER_PERT) | (counts.index == "control")].index
    adata = adata[np.isin(pert, eligible)].copy()
    pert = adata.obs["perturbation"].to_numpy()

    counts = pd.Series(pert).value_counts()
    max_cells = round(counts.drop("control", errors="ignore").mean())
    keep_idx = []
    for cond in counts.index:
        idx = np.where(pert == cond)[0]
        cap = MAX_CELLS_CONTROL if cond == "control" else max_cells
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep_idx.append(idx)
    adata = adata[np.sort(np.concatenate(keep_idx))].copy()
    pert = adata.obs["perturbation"].to_numpy()

    counts = pd.Series(pert).value_counts()
    eligible = counts[(counts >= MIN_CELLS_PER_PERT_POST) | (counts.index == "control")].index
    adata = adata[np.isin(pert, eligible)].copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVGS)
    hvg_genes = set(adata.var_names[adata.var["highly_variable"]])
    perturbed_genes = {g for c in adata.obs["perturbation"].unique() for g in _target_genes(c)}
    genes_to_keep = sorted(hvg_genes | perturbed_genes)
    adata = adata[:, genes_to_keep].copy()
    # highly_variable_genes() adds several non-string .var columns (means, dispersions, ...)
    # only needed transiently to pick genes_to_keep above — drop them rather than carry them
    # into the object-dtype normalization below, which anndata can't write for booleans/floats.
    adata.var = adata.var[[]]

    return adata


def smoke_subsample(
    adata: ad.AnnData, n_perts: int, cells_per_pert: int, control_cells: int, seed: int
) -> ad.AnnData:
    """Further reduce an already Miller-fixed population to a tiny smoke-test scale — the same
    selection logic this script used to apply directly to the raw processed file, now layered
    on top of the shared Miller-consistent population instead."""
    pert = adata.obs["perturbation"].to_numpy()
    keep_idx = smoke_subsample_indices(
        pert,
        n_perts=n_perts,
        cells_per_pert=cells_per_pert,
        control_cells=control_cells,
        seed=seed,
    )
    return adata[keep_idx].copy()


def smoke_subsample_indices(
    perturbations: np.ndarray,
    *,
    n_perts: int,
    cells_per_pert: int,
    control_cells: int,
    seed: int,
) -> np.ndarray:
    """Choose bounded smoke-test rows using annotations alone."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(perturbations).groupby(perturbations, sort=False).indices
    eligible = np.sort(
        np.asarray(
            [label for label, indices in groups.items() if label != "control" and len(indices) >= cells_per_pert]
        )
    )
    chosen_perts = rng.choice(eligible, size=min(n_perts, len(eligible)), replace=False)

    keep_idx = []
    control_idx = groups.get("control", np.asarray([], dtype=int))
    keep_idx.append(rng.choice(control_idx, size=min(control_cells, len(control_idx)), replace=False))
    for p in chosen_perts:
        idx = groups[p]
        keep_idx.append(rng.choice(idx, size=min(cells_per_pert, len(idx)), replace=False))
    return np.sort(np.concatenate(keep_idx))


def finalize_model_annotations(adata: ad.AnnData, cell_type: str) -> ad.AnnData:
    """Add model adapter fields and make string-like annotations H5AD-safe.

    Canonical metadata can contain object columns with a mixture of strings, integer batch
    identifiers, and missing values. Casting every annotation to ``object`` preserves that
    mixture and h5py cannot serialize it as a variable-length string array. Numeric and boolean
    columns remain native; only object, categorical, and pandas-string columns are normalized to
    concrete strings.
    """
    pert = adata.obs["perturbation"].astype(str).to_numpy()
    adata.obs["condition"] = np.where(pert == "control", "ctrl", [f"{gene}+ctrl" for gene in pert])
    adata.obs["cell_type"] = cell_type
    adata.var["gene_name"] = adata.var_names.astype(str)

    adata.obs.index = pd.Index(adata.obs.index.astype(str), name=adata.obs.index.name)
    adata.var.index = pd.Index(adata.var.index.astype(str), name=adata.var.index.name)
    for frame in [adata.obs, adata.var]:
        for column in frame.columns:
            dtype = frame[column].dtype
            if dtype == object or isinstance(dtype, (pd.CategoricalDtype, pd.StringDtype)):
                frame[column] = np.asarray(frame[column].astype("string").fillna(""), dtype=str)
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", default="replogle22k562", help="data/<source>_processed_complete.h5ad to prepare (default: replogle22k562)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="canonical raw-count H5AD to prepare (preferred; overrides --source)",
    )
    parser.add_argument(
        "--perturbation-key",
        default="perturbation",
        help="source obs field containing perturbation labels (default: perturbation)",
    )
    parser.add_argument(
        "--control-key",
        default="is_control",
        help="source obs field identifying controls (default: is_control)",
    )
    parser.add_argument(
        "--control-value",
        help="value in --control-key that denotes control; omit for a boolean field",
    )
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        help="frozen experiment JSON used to validate cohort metadata",
    )
    parser.add_argument(
        "--validate-metadata-only",
        action="store_true",
        help="validate backed obs/var metadata without loading or processing X",
    )
    parser.add_argument(
        "--dataset-name", default=None, help="output name under models/data/<name>/ (default: --source's value)"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="additionally subsample to a tiny smoke-test scale (see --n-perts etc.)"
    )
    parser.add_argument("--n-perts", type=int, default=12, help="[--smoke] number of perturbations to keep (default: 12)")
    parser.add_argument("--cells-per-pert", type=int, default=100, help="[--smoke] per-perturbation cell cap (default: 100)")
    parser.add_argument("--control-cells", type=int, default=500, help="[--smoke] control cell cap (default: 500)")
    parser.add_argument("--cell-type", default="K562", help="cell_type label to assign (default: K562)")
    parser.add_argument(
        "--compatible-models",
        nargs="+",
        choices=["gears", "state", "sclambda"],
        default=["gears", "sclambda"],
        help="models whose external gene references the view must satisfy (default: gears sclambda)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    args = parser.parse_args()

    source_path = args.input or SOURCE_DIR / f"{args.source}_processed_complete.h5ad"
    if not source_path.exists():
        raise FileNotFoundError(f"input dataset does not exist: {source_path}")
    dataset_name = args.dataset_name or args.source
    out_path = HERE / dataset_name / "raw.h5ad"

    experiment = None
    if args.experiment_manifest is not None:
        experiment = json.loads(args.experiment_manifest.read_text())
    if args.validate_metadata_only:
        if args.input is None:
            raise ValueError("--validate-metadata-only requires --input")
        if experiment is None:
            raise ValueError("--validate-metadata-only requires --experiment-manifest")
        adata = ad.read_h5ad(source_path, backed="r")
        try:
            report = validate_frozen_experiment_metadata(
                adata,
                experiment,
                perturbation_key=args.perturbation_key,
                control_key=args.control_key,
                control_value=args.control_value,
            )
        finally:
            adata.file.close()
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.input is not None and experiment is not None:
        source = ad.read_h5ad(source_path, backed="r")
        try:
            validate_frozen_experiment_metadata(
                source,
                experiment,
                perturbation_key=args.perturbation_key,
                control_key=args.control_key,
                control_value=args.control_value,
            )
            adata = prepare_frozen_experiment_raw(
                source,
                experiment,
                perturbation_key=args.perturbation_key,
                control_key=args.control_key,
                control_value=args.control_value,
                smoke=(args.n_perts, args.cells_per_pert, args.control_cells, args.seed)
                if args.smoke
                else None,
            )
        finally:
            source.file.close()
        print(
            "frozen canonical raw-count cohort normalized for models: "
            f"{adata.shape}, {adata.obs['perturbation'].nunique() - 1} perturbations"
        )
    else:
        adata = ad.read_h5ad(source_path)
        if args.input is not None:
            adata = prepare_canonical_raw(
                adata,
                perturbation_key=args.perturbation_key,
                control_key=args.control_key,
                control_value=args.control_value,
            )
            print(f"canonical raw-count input normalized for models: {adata.shape}")
        adata = apply_miller_fixes(
            adata,
            seed=args.seed,
            compatible_models=set(args.compatible_models),
        )
        print(
            f"after Miller fixes: {adata.shape}, "
            f"{adata.obs['perturbation'].nunique()} perturbations"
        )

    if args.smoke and not (args.input is not None and experiment is not None):
        adata = smoke_subsample(adata, args.n_perts, args.cells_per_pert, args.control_cells, args.seed)
        print(f"after smoke subsample: {adata.shape}, {adata.obs['perturbation'].nunique()} perturbations")

    adata = finalize_model_annotations(adata, args.cell_type)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_out = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        adata.write_h5ad(temporary_out)
        os.replace(temporary_out, out_path)
    finally:
        temporary_out.unlink(missing_ok=True)
    print(f"wrote {adata.shape} to {out_path}")


if __name__ == "__main__":
    main()
