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

`gears` and `tabicl` fetch what they need automatically on first run (GEARS' own GO-annotation
reference from Harvard Dataverse; TabICL's pretrained checkpoint from Hugging Face, via
`huggingface_hub`'s own lock-protected cache — safe under concurrent folds by construction).
`state` fetches nothing at our settings — we always train from scratch (`checkpoint: null`),
so no Hugging Face checkpoint is ever pulled. **`tabicl` also needs PRESAGE's cache fetched**
(it reuses the same gene-embedding pickles), even though it runs in its own separate
environment — run the `presage fetch-cache` line above regardless of whether you're using
PRESAGE itself.

GEARS' own downloads are the one spot worth knowing about if you're launching multiple folds
at once (`submit_all.sh` does, via a 5-task SLURM array): on a cold cache, all 5 fold
processes race to fetch the same reference files into the same shared `gears/pert_data/`
directory. This is patched to be crash-safe in `gears/patch.py` (`_atomic_download`,
`_atomic_tar_download_and_extract` — downloads land in a private temp file/dir first, only
atomically renamed into place once complete), so a fold getting preempted or network-dropped
mid-download can't corrupt another fold's already-good copy.

Sanity-check any environment came up correctly:

```bash
pixi run -e <model> smoke-test
```

## 2. Prepare a dataset (one-time per dataset)

The ranking-sensitivity study starts from the public canonical raw-count objects produced by
`workflows.data`. Model preparation writes a separate normalized/log1p view under
`models/data/`; it never changes the canonical file.

```bash
# Apply the model-only normalization and Miller-consistent preparation. The selected public
# study models are GEARS and STATE, so only GEARS' external gene reference constrains the
# shared perturbation cohort.
uv run python -m models.data.prepare_data \
  --input data/processed/replogle-weissman-2022-k562-essential/adata.h5ad \
  --dataset-name replogle-weissman-2022-k562-essential \
  --cell-type K562 \
  --compatible-models gears state

# Freeze the five-fold split and its checksum-bearing archival manifest.
uv run python -m models.data.prepare_split \
  models/data/replogle-weissman-2022-k562-essential/raw.h5ad \
  --dataset-id replogle-weissman-2022-k562-essential
```

The older `--source replogle22k562` interface remains available only for reproducing the
legacy hosted, already-normalized object.

Add `--smoke` to `prepare_data.py` for a small, fast smoke-scale subsample instead — useful for
checking a change end-to-end before committing to a full-scale run.

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

## 5. Baselines that need no training

```bash
pixi run -e gears python baselines/baselines.py --dataset replogle22k562
```

Predicts `no_change`, `mean`, `linear`, and `interpolated` in one process (reads `raw.h5ad` once,
loops every fold). Runs from any model's environment, same as `prepare_data.py`/`prepare_split.py`
above.
Writes `baselines/<name>/predictions/<dataset>/fold_<i>.h5ad` per baseline — same per-fold shape
as `models/*/train_predict.py`, just nested under `baselines/` instead of sitting alongside
`gears/`, `scgpt/`, ... since these aren't trained models.

```bash
cd models && sbatch --export=ALL,MODELS_DIR="$(pwd)" slurm/baselines.sbatch
```

One non-array CPU job (`models/slurm/baselines.sbatch`).

## 6. Context-transfer model views and STATE

Context-transfer experiments use a combined, bounded model view produced by
`models.data.prepare_context`. Its `obs["model_perturbation"]` is the condition
STATE sees, while `obs["evaluation_condition"]` includes the context and is the
identifier stored in prediction bundles.

STATE uses cell-load's few-shot split rather than zero-shot splitting. Every
treated condition in the held-out context is assigned to the test split, while
its control cells remain available to the model. The experiment manifest also
freezes a 10% condition subset used for cross-context validation in the
training contexts.

```bash
cd models
pixi run -e state python state/train_predict.py \
  --dataset replogle-nadig-four-context-60-cell \
  --split-manifest ../configs/experiments/replogle-nadig-four-context-60-cell.json \
  --fold 0 \
  --perturbation-key model_perturbation
```

The same command shape applies to `sciplex3-nine-drug-context-transfer`.
STATE retains its native per-cell predictions. No-change and fold-mean context
baselines emit fixed 120-cell populations so they participate in both profile
and distributional sensitivity analyses.
