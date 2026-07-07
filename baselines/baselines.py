"""Baselines that need no training: each predicts a perturbation's profile as a fixed function
of the dataset's own control/perturbed cells.

Reuses scPertEval's own ``Dataset`` internals rather than reimplementing them, so each baseline's
definition matches the framework's own calibration sources (src/scperteval/sources.py):
``mean`` mirrors ``all_perturbed_mean``/``global_mean``, ``no_change`` mirrors ``control``.
Add a new baseline by writing a function ``(ds: Dataset) -> np.ndarray`` and registering it in
``BASELINES`` below; give it its own ``train_predict.py`` (alongside ``models/gears``,
``models/scgpt``) instead if it actually needs a fitting step.

Usage::

    python models/baselines/baselines.py mean data/replogle22k562_processed_complete.h5ad predictions.h5ad
    python models/baselines/baselines.py no_change data/replogle22k562_processed_complete.h5ad predictions.h5ad

Point ``dataset`` at the full processed dataset, not a held-out-only ground truth: scPertEval's
``score`` only looks up the perturbations it needs, so a baseline file covering every
perturbation still scores correctly against a smaller, held-out-only ground truth, while the
leave-one-out mean is computed over the true full perturbation set rather than just the handful
of other held-out perturbations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scperteval.dataset import Dataset
from scperteval.types import RunConfig


def mean_baseline(ds: Dataset) -> np.ndarray:
    """One row per perturbation: the leave-one-out mean of all other perturbations."""
    return np.vstack([ds.allpert_mean_except(p) for p in ds.perturbations])


def no_change_baseline(ds: Dataset) -> np.ndarray:
    """One row per perturbation: the control-cell mean (predicts no change at all)."""
    return np.vstack([ds.control_mean() for _ in ds.perturbations])


BASELINES = {"mean": mean_baseline, "no_change": no_change_baseline}


def build_predictions(baseline: str, dataset_path: str, min_cells: int = 30, seed: int = 42) -> ad.AnnData:
    """One predicted row per perturbation, from the named baseline function."""
    cfg = RunConfig(dataset=dataset_path, protocols=[], min_cells=min_cells, seed=seed)
    ds = Dataset.load(dataset_path, cfg)
    rows = BASELINES[baseline](ds)
    pred = ad.AnnData(X=rows.astype(np.float32), obs={"perturbation": ds.perturbations})
    pred.var_names = ds.var_names
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", choices=sorted(BASELINES), help="which baseline to compute")
    parser.add_argument(
        "dataset", help="path to a preprocessed .h5ad (the full dataset, not a scoped ground truth — see above)"
    )
    parser.add_argument("out", help="path to write the predictions .h5ad")
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pred = build_predictions(args.baseline, args.dataset, min_cells=args.min_cells, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(args.out)
    print(f"wrote {pred.shape} {args.baseline!r} predictions to {args.out}")


if __name__ == "__main__":
    main()
