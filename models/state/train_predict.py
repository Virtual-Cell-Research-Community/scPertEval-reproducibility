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
- `model=pertsets` (the repo's current GPT2-backed preset) is used as the architecture base
  rather than the Colab's `model=state` (LLaMA-backed). The preprint's own Table 3 gives a
  "Replogle-Nadig" calibration matching `pertsets.yaml` on every field except `hidden_dim`
  (Table 3: 128, ~10M params vs. `pertsets.yaml`'s default 328) — but that Table 3 hidden_dim
  value does NOT hold up empirically and should not be used. Two independent, real-data checks
  both contradict it: (1) a full-scale diagnostic at `hidden_dim=128` plateaued at val_loss
  ~1.4-1.5, substantially worse than `hidden_dim=328`'s ~0.86-0.96 (both cell_set_len=32,
  otherwise identical); (2) Arc's own actually-released checkpoints for this exact cell line
  (huggingface.co/arcinstitute/ST-HVG-Replogle, `fewshot/k562/config.yaml` and
  `zeroshot/k562/config.yaml` — both trained from scratch, `checkpoint: null`) use
  `hidden_dim: 328`, not 128. `hidden_dim` is kept at `pertsets.yaml`'s own default (328,
  37.7M params measured at cell_set_len=32/n_encoder=n_decoder=4) rather than overridden.
  Separately, note the paper's own ~10M-param figure doesn't reproduce with either backbone we
  actually tested on this data: 37.7M for this GPT2 config, 53.0M for a from-scratch `model=state`
  (LLaMA, hidden_dim=328, n_encoder=n_decoder=1, matching Arc's real k562 config) — so Table 3's
  params column looks stale/inapplicable here, not just its hidden_dim value.
  `model=state` itself was also re-tested directly and, contrary to an earlier note in this
  docstring, does NOT error on hidden_dim=328: the LLaMA backbone's `head_dim` is passed
  explicitly in `state.yaml` (64) rather than derived from hidden_dim/num_heads, so the
  divisibility concern doesn't apply. It was still not adopted here: Arc's HF checkpoints are
  for their own "Replogle-Nadig" task/splits, not scPertEval's own gene-holdout replogle22k562
  — a different setting close enough to be directionally useful (as the hidden_dim cross-check
  above), not close enough to justify porting the whole architecture family over on that basis
  alone. `pertsets.yaml`'s GPT2 backbone is kept as the one already validated end-to-end on our
  own data and task.
- `model.kwargs.cell_set_len` is kept at the Colab's 64 (down from `pertsets.yaml`'s own 512
  default, but NOT overridden further to the preprint's Table 3 value of 32, despite Table 3
  otherwise matching `pertsets.yaml` closely — see the hidden_dim discussion above on why Table
  3 shouldn't be trusted blindly here). cell_set_len is the transformer's sequence length
  (`n_positions`/`max_position_embeddings`), i.e. how many cells get processed together as one
  set — `cell_load`'s own sampler (`data_modules/samplers.py`) chunks each perturbation's cells
  into consecutive, non-overlapping blocks of this size, padding any leftover partial block up
  to full size by resampling *from that same leftover pool* with replacement. Measured directly
  against replogle22k562: at 64, 22.5% of all cell-draws end up being duplicates (61,399 of
  273,472); at 32, that drops to 9.8% (22,935 of 235,008) — a real difference, and initially
  used to justify overriding to 32. That was tested directly, though: two full-scale diagnostics
  (hidden_dim=328 fixed, otherwise identical, same ~11-epoch budget) show cell_set_len=32
  plateaus at val_loss ~1.19-1.27, while cell_set_len=64 plateaus at ~0.87-0.89, matching the
  hidden_dim=328/cell_set_len=64 baseline that originally motivated this whole investigation.
  So despite the real duplicate-cell-draw cost, 64 measurably trains better on this data — kept
  at 64, matching both the Colab and Arc's own real k562 checkpoints
  (huggingface.co/arcinstitute/ST-HVG-Replogle), not Table 3.
- `model.kwargs.batch_encoder` stays at `pertsets.yaml`'s own default (`False`), unlike the
  Colab's `True`. Confirmed directly (not just assumed): Arc's raw source data for this cell line
  (huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint,
  `K562_essential_normalized_singlecell_01.h5ad`) does carry a real `obs['gem_group']` batch
  column (48 distinct values, ~2000-5000 cells each) — batch_encoder=True is meaningful for
  Arc's own pipeline. scPertEval's own processed file
  (`data/source/replogle22k562_processed_complete.h5ad`) never carries that column forward
  (`obs` is `perturbation` only), so our own `prepare_dataset()` below sets a constant
  `obs['gem_group']="1"` placeholder — encoding it would contribute nothing. Recovering the real
  values would require barcode-matching against Arc's raw file, out of scope here.
- `training.val_freq`/`batch_size`/`lr` take the Colab's own values (2000/64/1e-3) rather than
  state's bare library defaults (2000/16/1e-4 — val_freq happens to already agree) — the Colab
  is the validated reference run, not just the framework's blanket defaults.
- `training.max_steps` does NOT take the Colab's value (80000). The preprint itself gives no
  ST-specific training-duration figures at all (section 5.6.2 covers only architecture/
  fine-tuning mechanics; the only concrete step/epoch/optimizer numbers in the paper, section
  5.6.3-5.6.4, are for the unrelated 600M-parameter SE pretraining model) — so 80000 was only
  ever Colab-derived, not paper-grounded. A real diagnostic run at hidden_dim=328 (the old
  default; 10 epochs = 470 steps at 47 steps/epoch, validating every half-epoch, on the actual
  full-scale data) showed val_loss flat/noisy from step ~287 (epoch ~6) onward — clear evidence
  of a plateau far short of 80000. 10000 is used instead — note the exact steps/epoch, and so
  the exact epoch-equivalent of 10000, shifts with cell_set_len (fewer cells per set means more
  sentences per epoch — see the cell_set_len bullet above); re-check the plateau empirically
  whenever hidden_dim/cell_set_len change rather than assuming the old run's step count still
  applies. Even so, 10000 is already the largest training budget of any of the 6 benchmarked
  models on any reasonable epoch accounting (GEARS 20 epochs, scGPT 15 epochs fine-tuning a
  pretrained checkpoint, PRESAGE early-stopped well under 20 in practice, scLAMBDA 200 epochs,
  TabICL 0 — frozen in-context learner), so there's no case for going higher without new
  evidence the plateau doesn't hold. `training.ckpt_every_n_steps` is passed through only if
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
  --batch-size 4`). `--num-workers` doesn't need lowering for smoke runs specifically — see the
  `main()` argument's own docstring for why (predict's own dataloader workers are always forced
  to 0 regardless of this value, which is what smoke-scale `--num-workers 0` was working around).
- `training.val_freq` defaults to 47, not the Colab's 2000 — one full epoch's worth of steps at
  cell_set_len=64 (measured: 47 steps/epoch), not an arbitrary smaller number. This matters for
  two reasons, not just giving EarlyStopping (see next bullet) enough resolution: PRESAGE and
  scGPT — the two other benchmarked models with real early stopping — both validate exactly
  once per epoch (PRESAGE: no val_check_interval/check_val_every_n_epoch override anywhere in
  its own official configs, so it uses Lightning's per-epoch default; scGPT: a plain
  `for epoch in range(...): train_one_epoch(); eval_perturb(val_loader)` loop, once per epoch,
  no sub-epoch checks). STATE's val_freq is a *step* count, not an epoch count — at the Colab's
  2000 (or even the previously-tried 200), checks land mid-epoch (confirmed directly in
  metrics.csv: `epoch=4 step=206`, not aligned to any epoch boundary), so `patience` wouldn't
  mean "N epochs of no improvement" the way it does for PRESAGE/scGPT. At val_freq=47, checks
  land on true epoch boundaries, making `patience` directly comparable in epoch terms.
- STATE has no EarlyStopping callback anywhere in the installed library (arc-state==0.11.1) —
  confirmed directly against `state/_cli/_tx/_train.py`/`state/tx/utils/__init__.py`;
  `get_checkpoint_callbacks()` builds only a `ModelCheckpoint(monitor="val_loss", ...)`. Every
  other benchmarked model that validates during training has one (see
  .claude/model-checkpoint-selection.md) except STATE, so one is added here — via
  `state/patch/sitecustomize.py`, a Python startup hook, not an edit to the installed package.
  `state tx train` runs as a subprocess (see below), so a plain monkeypatch from this script
  wouldn't reach it; sitecustomize.py is auto-imported by Python at interpreter startup for any
  process that has its directory on PYTHONPATH, which `main()` sets only for the `state tx
  train` subprocess call (not `state tx predict`, and not any bare/manual `state` CLI
  invocation outside this script). See that file's own docstring for the exact mechanism.
  `--early-stop-patience` is passed through as an environment variable
  (`STATE_EARLY_STOP_PATIENCE`) since sitecustomize.py is auto-imported with no way to receive
  arguments directly. `min_delta` is deliberately left at Lightning's own default (0.0, i.e. any
  strict val_loss improvement resets the wait counter) — an 0.01 override was tried and
  simulated directly against a real measured val_loss trend (the cell_set_len=64 full-scale
  diagnostic), and produced the exact same stop point as 0.0 at every patience value tested,
  since the real step-to-step "new low" moves in that data are almost always bigger than 0.01
  even amid noise. It wasn't filtering anything, so it was dropped rather than kept as a knob
  doing nothing. `patience=10` (epochs, not steps, given val_freq=47 above) matches PRESAGE's
  and scGPT's own patience=10 — not yet validated against a real diagnostic at this cadence.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

ad.settings.allow_write_nullable_strings = True

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
PERT_DATA_DIR = HERE / "pert_data"
RUN_DIR = HERE / "runs"
OUT_DIR = HERE / "predictions"
METRICS_DIR = HERE / "metrics"
PATCH_DIR = HERE / "patch"  # contains sitecustomize.py — see its own docstring


def prepare_dataset(raw: Path, dataset_dir: Path, cell_type: str, num_hvgs: int) -> Path:
    """Write our shared RAW smoke data in cell-load's expected form: a dataset directory
    containing one h5ad, with obs['gem_group'] (cell-load's default batch_col — a single
    placeholder value, we have no real batch/plate info) and obsm['X_hvg'] (HVGs computed
    directly on the already-normalized data, no redundant normalize_total/log1p — see the
    module docstring).

    out_path is fold-independent (nothing above depends on which fold is running) but this
    used to unconditionally recompute and rewrite it on every single fold invocation —
    redoing the real HVG-selection cost every time, and, worse, writing directly to a path
    every fold shares: anndata's h5ad write isn't atomic, so a fold reading this file while
    another fold's write is still in progress could see a truncated, unreadable file. Not
    hypothetical: reproduced exactly this by accident, killing a job mid-write here. Now
    skips rebuilding if out_path already exists, and writes via a temp-file-then-atomic-
    rename when it does rebuild — through a tempfile.mkstemp()-claimed unique path, not a
    fixed "out_path + suffix" one, so two folds racing on a cold cache can't both write
    through the same temp path concurrently (which would reintroduce the exact corruption
    this is meant to avoid, just one level down).
    """
    out_path = dataset_dir / f"{cell_type}.h5ad"
    if out_path.exists():
        return out_path

    adata = ad.read_h5ad(raw)
    adata.obs["gem_group"] = "1"

    sc.pp.highly_variable_genes(adata, n_top_genes=num_hvgs)
    hvg = adata[:, adata.var["highly_variable"]].X
    adata.obsm["X_hvg"] = hvg.toarray() if hasattr(hvg, "toarray") else np.asarray(hvg)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_out_path = tempfile.mkstemp(dir=str(dataset_dir), prefix=out_path.name + ".", suffix=".tmp")
    os.close(fd)  # just reserving a unique name; write_h5ad needs to open the path itself
    adata.write_h5ad(tmp_out_path)
    os.replace(tmp_out_path, out_path)
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="training.max_steps (default: 10000, full-scale — ~213 epochs at 47 steps/epoch with cell_set_len=64, see module docstring)",
    )
    # training.val_freq is the *actual* checkpoint-cadence knob (every_n_train_steps=
    # val_freq in state's own ModelCheckpoint callback — see --ckpt-every-n-steps below),
    # not just a validation-logging frequency. The Colab uses the library's own default
    # (2000) as-is; set here to 47 (one full epoch's worth of steps at cell_set_len=64) so
    # EarlyStopping's `patience` means "N epochs with no improvement" the same way it does
    # for PRESAGE/scGPT, not an arbitrary sub-epoch step count — see module docstring.
    parser.add_argument("--val-freq", type=int, default=47, help="training.val_freq (default: 47, one epoch at cell_set_len=64 — matches PRESAGE/scGPT's per-epoch validation cadence so EarlyStopping's patience is directly comparable, see module docstring)")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="EarlyStopping patience in validation checks (default: 10 epochs at val_freq=47, matching PRESAGE's and scGPT's own patience=10 — min_delta left at Lightning's own default, see module docstring and state/patch/sitecustomize.py)",
    )
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
    # perturbation.yaml's own data-config default is 12; the Colab explicitly overrides to 4.
    # This value only governs *training*'s dataloader workers — state tx predict deadlocks
    # with worker processes at all (confirmed at both smoke scale with 12 and full scale with
    # this default of 4: 0% GPU, no CPU progress, indefinitely), so predict's own workers are
    # unconditionally forced to 0 below regardless of what's passed here (see the
    # data_module.torch patch after the train subprocess call).
    parser.add_argument("--num-workers", type=int, default=4, help="data.kwargs.num_workers (default: 4, per the Colab; training only — see below)")
    parser.add_argument("--batch-size", type=int, default=64, help="training.batch_size (default: 64, per the Colab)")
    parser.add_argument("--lr", type=float, default=1e-3, help="training.lr (default: 1e-3, per the Colab; library default is 1e-4)")
    # model.kwargs.cell_set_len: how many cells the transformer backbone processes as one
    # set (n_positions/max_position_embeddings in pertsets.yaml). cell_load's own sampler
    # (data_modules/samplers.py) chunks each perturbation's cells into consecutive,
    # non-overlapping blocks of this size; any leftover partial block gets padded up to full
    # size by resampling *from that same leftover pool* with replacement — so cell_set_len
    # directly trades off against how much of what the model trains on is a duplicated cell
    # rather than a fresh one (22.5% at 64 vs. 9.8% at 32, measured on replogle22k562). Despite
    # that, 32 measurably trains worse (val_loss plateau ~1.2 vs. ~0.87-0.89 at 64, two direct
    # full-scale diagnostics with hidden_dim=328 held fixed) — see module docstring. Kept at the
    # Colab's own 64, not the preprint's Table 3 value of 32.
    parser.add_argument("--cell-set-len", type=int, default=64, help="model.kwargs.cell_set_len (default: 64, matches the Colab and Arc's own real k562 checkpoints — Table 3's 32 was tested and rejected, see module docstring)")
    # pertsets.yaml's own default (328, 37.7M params measured at cell_set_len=32). The
    # preprint's Table 3 claims 128 (~10M params) for Replogle-Nadig, but that does not hold up
    # empirically — see module docstring (real diagnostic + Arc's own released k562 checkpoints
    # both land on 328, not 128) — so the default here stays at pertsets.yaml's own value rather
    # than overriding to Table 3's.
    parser.add_argument("--hidden-dim", type=int, default=328, help="model.kwargs.hidden_dim (default: 328, pertsets.yaml's own default — see module docstring for why Table 3's 128 was tested and rejected)")
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
        f"model.kwargs.hidden_dim={args.hidden_dim}",
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

    # PYTHONPATH/STATE_EARLY_STOP_PATIENCE only apply to this one subprocess call — predict
    # below runs with the normal, unmodified environment. See state/patch/sitecustomize.py and
    # the module docstring's EarlyStopping bullet for why this is how the callback gets added.
    train_env = os.environ.copy()
    train_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PATCH_DIR), train_env.get("PYTHONPATH")]))
    train_env["STATE_EARLY_STOP_PATIENCE"] = str(args.early_stop_patience)
    subprocess.run(["state", "tx", "train", *train_overrides], check=True, env=train_env)

    # Lightning's CSVLogger already writes everything needed (val_loss/train_loss/step/epoch)
    # to run_output_dir/version_0/metrics.csv — just relocate it to match predictions/ and
    # saved_models/'s <dataset>/fold_<i> convention, no capture/parsing needed (unlike
    # gears/sclambda/scgpt, which don't produce a structured metrics file on their own).
    metrics_src = RUN_DIR / run_name / "version_0" / "metrics.csv"
    metrics_dst = METRICS_DIR / args.dataset / f"fold_{args.fold}.csv"
    metrics_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(metrics_src, metrics_dst)

    run_output_dir = RUN_DIR / run_name
    # state tx predict deserializes this saved PerturbationDataModule config (a plain dict —
    # cell_load's own PerturbationDataModule.save_state()/load_state()) rather than rebuilding
    # the data module from config.yaml, so whatever num_workers training used is inherited
    # unchanged (see the --num-workers override above). Confirmed empirically on the real
    # full-scale data: num_workers=4 (this script's own default) deadlocks state tx predict
    # indefinitely (0% GPU utilization, no CPU progress) — a full-scale instance of the same
    # dataloader-worker deadlock already documented at smoke scale for num_workers=12. Training
    # itself is unaffected by this (runs fine at num_workers=4), so rather than dropping
    # num_workers to 0 for training too (and losing real throughput there), patch just the
    # saved dict's num_workers to 0 before predict loads it.
    data_module_path = run_output_dir / "data_module.torch"
    data_module_state = torch.load(data_module_path, map_location="cpu")
    data_module_state["num_workers"] = 0
    torch.save(data_module_state, data_module_path)

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

    out_path = OUT_DIR / args.dataset / f"fold_{args.fold}.h5ad"
    pred_adata = ad.AnnData(
        X=np.asarray(pred.X, dtype=np.float32),
        obs={"perturbation": pred.obs["perturbation"].to_numpy()},
    )
    pred_adata.var_names = pd.Index(np.asarray(var_dims["gene_names"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_adata.write_h5ad(out_path)
    print(f"wrote {pred_adata.shape} predictions to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
