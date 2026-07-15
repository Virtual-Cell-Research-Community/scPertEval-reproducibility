"""Smoke test: train ArcInstitute/state (STATE Transition model) on a small subsample and
write a scPertEval-compatible predictions.h5ad.

Uses our own 5-fold cross-validation split (models/data/prepare_split.py) rather than the
official Replogle-Nadig Colab's own hepg2 few-shot holdout — see that script's docstring for
why full gene coverage across folds beats one arbitrary holdout. STATE's own `cell-load`
few-shot TOML mechanism (hold out specific genes within a cell type) maps directly onto our
existing fold format: models/data/*_kfold_splits.json already stores bare gene symbols
matching our RAW file's own ``obs['perturbation']`` convention, so no new per-fold file is
needed (unlike PRESAGE's own convention — see ``to_presage_custom_split``).

Unlike gears/presage/sclambda, `arc-state` is a properly packaged, PyPI-published project with
a real CLI (`state tx ...`, Hydra-config driven) — there's no importable ``Model``/``.train()``
class to call directly, so this script drives the official CLI via subprocess, the same way a
human would from the shell. This mirrors the official Colab
(https://colab.research.google.com/drive/1Ih-KtTEsPqDQnjTh6etVv_f-gRAA86ZN, "STATE Training on
Replogle-Nadig") with a few deliberate deviations:

- HVGs are computed manually with scanpy (as the Colab itself does — data is already
  log-transformed), skipping `state tx preprocess_train`'s own normalize_total+log1p: our
  shared RAW file is already on that scale (models/data/prepare_data.py), and re-normalizing
  would double-transform it — same reasoning as PRESAGE's ``SmokePRESAGEDataModule``.
- `data.kwargs.output_space` is left at its default ("all", the full measured gene panel)
  rather than overridden to "gene" (HVG-only) like the Colab's own run. "gene" would restrict
  predictions to the ~2000 HVG-selected genes, not directly comparable gene-for-gene with
  gears/scgpt/presage/sclambda's predictions.h5ad (all four score across the full panel);
  "all" is an equally official, first-class mode of the same CLI, just not the one the Colab
  happens to demo.
- `state tx predict` is run with `--predict-only --pseudobulk` rather than the Colab's bare
  invocation (which also runs STATE's own internal cell-eval metrics suite). scPertEval scores
  raw predictions itself downstream — same reasoning as PRESAGE's `training.eval_test=False` —
  and `--pseudobulk` aggregates to one row per (cell_type, perturbation), matching every other
  model's predictions.h5ad convention (STATE predicts per-cell by default).
- `model=pertsets` (the repo's current GPT2-backed preset) is used rather than the Colab's
  `model=state` + hidden_dim=328 (which errors under the repo's current `state.yaml` — see
  below) or the preprint's own Table 3 numbers for Replogle-Nadig (hidden_dim=128,
  cell_set_len=32, GPT2, ~10M params). Table 3's exact figures don't correspond to any config
  actually checked into the repo's git history (a full sweep of real `replogle_*.yaml` configs
  from past commits clusters at hidden_dim 328-896, never 128) — the closest thing to an
  authoritative "this is what we ran" artifact is `replogle_best.yaml` (hidden_dim=672,
  cell_set_len=512, LLaMA), which itself doesn't match Table 3's GPT2 claim either. Rather than
  chase an unreproducible historical config, this takes today's shipped `pertsets` preset as
  its architecture base — a defensible, easy-to-justify choice regardless of which exact
  configuration produced the paper's own numbers. (For reference, the Colab's `hidden_dim=328`
  doesn't evenly divide `model=state`'s current 12 attention heads — `transformers`'
  LlamaConfig validates that strictly and raises at construction — while `pertsets.yaml`
  already has GPT2 + hidden_dim=328, the Colab's exact values, suggesting `model=state` used to
  mean what `pertsets.yaml` now is before some repo-side rename.)
- `model.kwargs.cell_set_len` *is* overridden to the Colab's 64 (unlike hidden_dim, which
  already matches `pertsets.yaml`'s own default) — not `replogle_best.yaml`'s 512, and not
  `pertsets.yaml`'s own 512 default either. cell_set_len is the transformer's sequence length
  (`n_positions`/`max_position_embeddings`), i.e. how many cells get processed together as one
  set — checked directly against the raw replogle22k562 file, cells-per-perturbation is 125 at
  the median and 44 at the 5th percentile, so 512 would leave most perturbations padded/
  undersized; 64 is independently justified by that distribution, not just Colab fidelity.
- `model.kwargs.batch_encoder` stays at `pertsets.yaml`'s own default (`False`), unlike the
  Colab's `True` — our `obs['gem_group']` is a single placeholder value for every cell (no real
  batch/plate structure, see `prepare_dataset()` below), so encoding it would just be a
  constant contributing nothing. The Colab's own dataset apparently had real batch structure
  its `True` was meant to capture; that doesn't apply here.
- `training.max_steps`/`val_freq`/`batch_size`/`lr` all take the Colab's own values (80000/
  2000/64/1e-3) rather than state's bare library defaults (400000/2000/16/1e-4 — val_freq
  happens to already agree) — the Colab is the validated reference run, not just the
  framework's blanket defaults. `training.ckpt_every_n_steps` is passed through only if
  explicitly set: it's a no-op in the installed state version (`get_checkpoint_callbacks()`
  accepts it as `_ckpt_every_n_steps` and never uses it — `every_n_train_steps=val_freq` on
  its `ModelCheckpoint` is what actually controls checkpoint cadence), and the Colab never
  sets it either. `data.kwargs.num_workers` takes the Colab's 4 rather than the data config
  group's own default of 12 (`state/configs/data/perturbation.yaml`) — that same file confirms
  `cell_type_key`/`batch_col`/`output_space` already match what this script relies on
  (`cell_type`/`gem_group`/`all`) without needing an explicit override, and that `pert_col`/
  `control_pert` (default `gene`/`DMSO_TF`, assuming a different dataset's column naming and
  drug-perturbation controls) are correctly overridden below, not redundantly. `use_wandb=false`
  since no wandb login is available here
  (same spirit as PRESAGE's `training.offline=True`). All of the above are full-scale values;
  smoke-scale runs need much lower ones passed explicitly (e.g. `--max-steps 20 --val-freq 10
  --batch-size 4 --num-workers 0`).
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ad.settings.allow_write_nullable_strings = True

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
PERT_DATA_DIR = HERE / "pert_data"
RUN_DIR = HERE / "runs"
OUT_DIR = HERE / "smoke_data"


def prepare_dataset(raw: Path, dataset_dir: Path, cell_type: str, num_hvgs: int) -> Path:
    """Write our shared RAW smoke data in cell-load's expected form: a dataset directory
    containing one h5ad, with obs['gem_group'] (cell-load's default batch_col — a single
    placeholder value, we have no real batch/plate info) and obsm['X_hvg'] (HVGs computed
    directly on the already-normalized data, no redundant normalize_total/log1p — see the
    module docstring)."""
    adata = ad.read_h5ad(raw)
    adata.obs["gem_group"] = "1"

    sc.pp.highly_variable_genes(adata, n_top_genes=num_hvgs)
    hvg = adata[:, adata.var["highly_variable"]].X
    adata.obsm["X_hvg"] = hvg.toarray() if hasattr(hvg, "toarray") else np.asarray(hvg)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    out_path = dataset_dir / f"{cell_type}.h5ad"
    adata.write_h5ad(out_path)
    return out_path


def build_toml(fold: dict[str, list[str]], out_path: Path, dataset_name: str, dataset_dir: Path, cell_type: str) -> Path:
    """cell-load's few-shot TOML format (see its README) — bare gene symbols, same convention
    our RAW file's obs['perturbation'] already uses, no relabeling needed."""
    val = ", ".join(f'"{g}"' for g in fold["val"])
    test = ", ".join(f'"{g}"' for g in fold["test"])
    out_path.write_text(
        f"""[datasets]
{dataset_name} = "{dataset_dir}"

[training]
{dataset_name} = "train"

[zeroshot]

[fewshot]
[fewshot."{dataset_name}.{cell_type}"]
val = [{val}]
test = [{test}]
"""
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="smoke_k562",
        help="models/data/<dataset>/{raw.h5ad,kfold_splits.json} to train/predict on (default: smoke_k562)",
    )
    parser.add_argument("--fold", type=int, default=0, help="which fold in *_kfold_splits.json to train/predict on")
    parser.add_argument("--cell-type", default="K562", help="cell_type label for the dataset (default: K562)")
    parser.add_argument("--num-hvgs", type=int, default=2000, help="number of HVGs computed for data.kwargs.embed_key=X_hvg (default: 2000)")
    # All the following defaults come from ArcInstitute's own "STATE Training on
    # Replogle-Nadig" Colab (the same one this script's docstring already references),
    # not state's own bare library defaults (training/default.yaml: max_steps=400000,
    # val_freq=2000, batch_size=16, lr=1e-4) — the Colab is the actual validated
    # reference run, not just the framework's blanket defaults. Smoke scale needs much
    # lower values passed explicitly (e.g. --max-steps 20 --val-freq 10 --batch-size 4).
    parser.add_argument("--max-steps", type=int, default=80000, help="training.max_steps (default: 80000, full-scale)")
    # training.val_freq is the *actual* checkpoint-cadence knob (every_n_train_steps=
    # val_freq in state's own ModelCheckpoint callback — see --ckpt-every-n-steps below),
    # not just a validation-logging frequency. The Colab doesn't override this, i.e. it
    # uses the library's own default (2000) as-is.
    parser.add_argument("--val-freq", type=int, default=2000, help="training.val_freq (default: 2000)")
    parser.add_argument(
        "--ckpt-every-n-steps",
        type=int,
        default=None,
        help=(
            "training.ckpt_every_n_steps — currently a no-op in the installed state version: "
            "get_checkpoint_callbacks() accepts it as `_ckpt_every_n_steps` but never uses it; "
            "every_n_train_steps=val_freq is what actually controls checkpoint cadence. Kept "
            "here only for forward-compat / explicit intent; omitted from the command entirely "
            "unless set, matching the Colab (which never sets it either)."
        ),
    )
    # perturbation.yaml's own data-config default is 12; the Colab explicitly overrides
    # to 4. At smoke scale even 12 reliably deadlocked `state tx predict` (hung
    # indefinitely, 0% CPU, no progress) — untested whether 4 is safe at smoke scale too,
    # so smoke runs should pass --num-workers 0 to be safe.
    parser.add_argument("--num-workers", type=int, default=4, help="data.kwargs.num_workers (default: 4, per the Colab)")
    parser.add_argument("--batch-size", type=int, default=64, help="training.batch_size (default: 64, per the Colab)")
    parser.add_argument("--lr", type=float, default=1e-3, help="training.lr (default: 1e-3, per the Colab; library default is 1e-4)")
    # model.kwargs.cell_set_len: how many cells the transformer backbone processes as one
    # set (n_positions/max_position_embeddings in pertsets.yaml) — must fit within what's
    # actually available per perturbation or sets end up padded/undersized. pertsets.yaml's
    # own default (512) is tuned for state's own larger pretraining corpora: on
    # replogle22k562 specifically, cells-per-perturbation is 125 at the median and 44 at
    # the 5th percentile (checked directly against the raw file), so 512 would leave most
    # perturbations undersized. The Colab's 64 fits comfortably under that median without
    # being so small it wastes most perturbations' cells — independently justified here,
    # not just because the Colab used it.
    parser.add_argument("--cell-set-len", type=int, default=64, help="model.kwargs.cell_set_len (default: 64, see above)")
    args = parser.parse_args()

    kfold_splits = DATA_DIR / args.dataset / "kfold_splits.json"
    dataset_dir = PERT_DATA_DIR / args.dataset
    fold = json.loads(kfold_splits.read_text())[args.fold]
    run_name = f"{args.dataset}_fold_{args.fold}"

    prepare_dataset(DATA_DIR / args.dataset / "raw.h5ad", dataset_dir, args.cell_type, args.num_hvgs)
    # Written into pert_data/ (already gitignored via */pert_data/) rather than directly under
    # models/state/, alongside the dataset's own h5ad — both are generated, per-fold artifacts.
    toml_path = build_toml(
        fold, dataset_dir.parent / f"{args.dataset}_fold_{args.fold}.toml", args.dataset, dataset_dir, args.cell_type
    )

    train_overrides = [
        f"data.kwargs.toml_config_path={toml_path}",
        "data.kwargs.embed_key=X_hvg",
        "data.kwargs.pert_col=perturbation",
        "data.kwargs.control_pert=control",
        # `data_module.save_state()` persists this, so `predict`'s reloaded data
        # module inherits it without a separate override.
        f"data.kwargs.num_workers={args.num_workers}",
        f"training.max_steps={args.max_steps}",
        f"training.val_freq={args.val_freq}",
        f"training.batch_size={args.batch_size}",
        f"training.lr={args.lr}",
        f"model.kwargs.cell_set_len={args.cell_set_len}",
        # batch_encoder stays at pertsets.yaml's own default (False), unlike the Colab's
        # True — our obs['gem_group'] is a single placeholder value for every cell (no
        # real batch/plate structure, see prepare_dataset() above), so encoding it would
        # just be a constant contributing nothing; the Colab's own dataset apparently had
        # real batch structure its True was meant to capture.
        "model=pertsets",
        "use_wandb=false",
        "overwrite=true",
        f"output_dir={RUN_DIR}",
        f"name={run_name}",
    ]
    if args.ckpt_every_n_steps is not None:
        train_overrides.append(f"training.ckpt_every_n_steps={args.ckpt_every_n_steps}")

    subprocess.run(["state", "tx", "train", *train_overrides], check=True)

    run_output_dir = RUN_DIR / run_name
    subprocess.run(
        [
            "state",
            "tx",
            "predict",
            "--output-dir",
            str(run_output_dir),
            # best.ckpt (the val_loss-monitored ModelCheckpoint callback, see
            # get_checkpoint_callbacks in state's own tx/utils/__init__.py) rather than
            # final.ckpt (an unconditional snapshot taken right after training ends,
            # whatever step that happened to be) — matches gears/scLAMBDA/PRESAGE's own
            # convention of predicting with the best validation checkpoint, not the last one.
            "--checkpoint",
            "best.ckpt",
            "--predict-only",
            "--pseudobulk",
        ],
        check=True,
    )

    pred_path = run_output_dir / "eval_best.ckpt" / "adata_pred.h5ad"
    pred = ad.read_h5ad(pred_path)
    # --pseudobulk aggregates by (context, perturbation) and, with should_yield_control_cells
    # defaulting to true, includes a "control" group alongside the actual test genes — drop
    # it, matching gears/scgpt/presage/sclambda's convention of one row per test perturbation.
    pred = pred[pred.obs["perturbation"] != "control"].copy()

    with open(run_output_dir / "var_dims.pkl", "rb") as f:
        var_dims = pickle.load(f)

    out_path = OUT_DIR / f"{args.dataset}_predictions_fold{args.fold}.h5ad"
    pred_adata = ad.AnnData(
        X=np.asarray(pred.X, dtype=np.float32),
        obs={"perturbation": pred.obs["perturbation"].to_numpy()},
    )
    pred_adata.var_names = pd.Index(np.asarray(var_dims["gene_names"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_adata.write_h5ad(out_path)
    print(f"wrote {pred_adata.shape} predictions to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
