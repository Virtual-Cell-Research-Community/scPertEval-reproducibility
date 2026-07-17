# models/

Minimum steps to get each model's environment running and launch training + prediction.
Each model lives in its own isolated pixi environment (`gears`, `scgpt`, `presage`,
`sclambda`, `state`, `tabicl`) — there's no shared/default environment, so every command
below is run via `pixi run -e <model> ...` from this directory (`models/`).

This doesn't cover `compare.py`/`plots.py` — those are being folded into the project's
notebooks and are still in flux.

## 1. Fetch external assets (one-time per model that needs one)

```bash
pixi run -e scgpt fetch-checkpoint      # pretrained scGPT_human checkpoint (~gdown)
pixi run -e presage fetch-cache         # PRESAGE's gene-embedding cache (~3.4GB, Zenodo)
pixi run -e sclambda fetch-embeddings   # GenePT gene-embedding pickle (Zenodo)
```

`gears`, `state`, and `tabicl` fetch what they need automatically on first run (GEARS' own
GO-annotation reference, TabICL's pretrained checkpoint from Hugging Face). **`tabicl` also
needs PRESAGE's cache fetched** (it reuses the same gene-embedding pickles), even though it
runs in its own separate environment — run the `presage fetch-cache` line above regardless of
whether you're using PRESAGE itself.

Sanity-check any environment came up correctly:

```bash
pixi run -e <model> smoke-test
```

## 2. Prepare a dataset (one-time per dataset)

```bash
# fetch the source h5ad (see docs/user-guide/datasets.md for the full dataset list)
gsutil cp gs://scperteval/processed/replogle22k562_processed_complete.h5ad data/source/

# apply the Miller-consistent prep (drop uncovered perturbations, downsample, HVG panel)
pixi run -e gears python data/prepare_data.py --source replogle22k562

# 5-fold split
pixi run -e gears python data/prepare_split.py data/replogle22k562/raw.h5ad
```

Add `--smoke` to `prepare_data.py` for a small, fast smoke-scale subsample instead (writes to
`data/smoke_k562/` by default) — useful for checking a change end-to-end before committing to
a full-scale run.

Either script can be run from any model's environment — they only need the standard
scientific-Python stack, nothing model-specific.

## 3. Train + predict, one model/fold at a time

Every model exposes the same CLI shape:

```bash
pixi run -e gears    python gears/train_predict.py    --dataset replogle22k562 --fold 0
pixi run -e scgpt    python scgpt/train_predict.py    --dataset replogle22k562 --fold 0
pixi run -e presage  python presage/train_predict.py  --dataset replogle22k562 --fold 0
pixi run -e sclambda python sclambda/train_predict.py --dataset replogle22k562 --fold 0
pixi run -e state    python state/train_predict.py    --dataset replogle22k562 --fold 0
pixi run -e tabicl   python tabicl/train_predict.py   --dataset replogle22k562 --fold 0
```

All flags beyond `--dataset`/`--fold` default to each model's own full-scale values (see each
script's own docstring/`--help` for what those are and why) — pass smaller ones (e.g.
`--epochs`, `--max-steps`) for a quick smoke-scale check instead of a real run.

**`--batch-size` in particular needs lowering at smoke scale, for most models** (GEARS, scGPT,
PRESAGE, STATE all expose it — default 32/64/16/64 respectively, all calibrated for full-scale
data). The smoke dataset just doesn't have enough samples for those defaults: PRESAGE trains on
pseudobulk (one row per perturbation × cell type), so its default of 16 hard-crashes on
`smoke_k562` ("no training batches" — Lightning silently trains on zero batches rather than
erroring earlier); `--batch-size 2` works there. The others degrade more gracefully but still
warrant a smaller value at smoke scale for the same reason.

## 4. Train + predict, the full sweep (all models × all 5 folds)

```bash
models/slurm/submit_all.sh                              # all 6 models, replogle22k562
models/slurm/submit_all.sh replogle22k562 gears scgpt    # just these two
```

Submits one independent 5-task SLURM array job per model (`models/slurm/train.sbatch`).
