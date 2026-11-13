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

On Helix, fetch the currently missing scGPT and PRESAGE/TabICL assets as restart-safe CPU
jobs from the repository root:

```bash
sbatch scripts/fetch_perturbation_model_assets_helix.sbatch
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
at once (`submit_all.sh` does, via a 15-task fold-by-seed SLURM array): on a cold cache, all
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
# Materialize the exact frozen GWPS cohort, then freeze its five-fold split.
sbatch ../scripts/prepare_k562_gwps_model_view_helix.sbatch
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
pixi run -e gears    python gears/train_predict.py    --dataset replogle-weissman-2022-k562-gwps --fold 0 --seed 42
pixi run -e scgpt    python scgpt/train_predict.py    --dataset replogle-weissman-2022-k562-gwps --fold 0 --seed 42
pixi run -e presage  python presage/train_predict.py  --dataset replogle-weissman-2022-k562-gwps --fold 0 --seed 42
pixi run -e sclambda python sclambda/train_predict.py --dataset replogle-weissman-2022-k562-gwps --fold 0 --seed 42
pixi run -e state    python state/train_predict.py    --dataset replogle-weissman-2022-k562-gwps --fold 0 --seed 42
pixi run -e tabicl   python tabicl/train_predict.py   --dataset replogle-weissman-2022-k562-gwps --fold 0 --seed 42
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

Each adapter writes to `runs/<dataset>/<model>/fold-<n>/seed-<seed>/`, including its resolved
configuration, checkpoint, training metrics, validated prediction manifest, checksums, and a
completion marker. Re-running the same command skips a checksum-valid completed run.

The scLAMBDA adapter retains upstream `Model.train` and `Model.predict`, but replaces its
quadratic initialization with an arithmetic-preserving categorical embedding table. Upstream
otherwise scans all cells once per perturbation and allocates two unused full-cell embedding
arrays when target-coordinate prediction is disabled. Validation and mean prediction use
scLAMBDA's native `ctrl_size` option with 300 controls, matching the fixed validation population
in its context-model implementation. The adapter also passes the declared training seed into
scLAMBDA rather than allowing the library's constructor default to collapse all three seeds.

## 4. Train + predict, the full sweep

On a workstation, run one model serially on one explicitly selected GPU. The
launcher preserves one append-only log per fold/seed, stops on the first
failure, and relies on the shared run contract to skip checksum-valid results:

```bash
GPU=0 \
RUNS_ROOT=/path/on/shared-storage/runs \
scripts/run_perturbation_model_workstation.sh sclambda
```

Run it inside tmux or another persistent session. `FOLDS` and `SEEDS` can
restrict a scaling trial. `FORCE_INCOMPLETE=1` only adds `--force` when a run
directory has an old resolved configuration but no completion record; it does
not force already completed runs.

On Slurm, submit the same frozen grid as arrays:

```bash
models/slurm/submit_all.sh                 # five folds x three seeds; TabICLv2 once per fold
models/slurm/submit_all.sh --test-only     # validate scheduling without submitting
models/slurm/submit_all.sh --smoke         # fold 0, seed 42, reduced training budgets
models/slurm/submit_all.sh --smoke --afterok 123456  # wait for an asset/preparation job
```

The stochastic models use 15-task arrays (five folds × seeds 42, 43, and 44). TabICLv2 is
training-free and deterministic, so it uses one five-fold array at archival seed 42. `--smoke`
submits only fold 0 at seed 42 with model-specific truncated budgets. Full arrays default to
one concurrent task per model (`MAX_CONCURRENT=1`) so no model family launches several 48 GiB
SDS reads at once. Set `MAX_CONCURRENT` explicitly only after confirming storage and node
capacity. Full tasks request 192 GiB RAM because the GWPS CSR arrays alone occupy about 47 GiB
before AnnData and model-specific working copies; scLAMBDA receives a 256 GiB override for its
dense training tensors. The previous 64 GiB request was unsafe.

A full-data scLAMBDA scaling run on a 96 GiB RTX PRO 6000 validated the
categorical-embedding initialization end to end: 1,333,016 cells and 8,248
genes fit in approximately 46 GiB GPU memory, training took about 10.4 seconds
per epoch, and the 10-epoch run completed in about 17 minutes including the
SDS read, initialization, validation, and prediction write. Its output covered
all 1,387 fold-0 test perturbations and passed both the prediction contract and
artifact-checksum validation. A 200-epoch production run is therefore expected
to take roughly 50--60 minutes per fold/seed on that workstation class.

The complete production sweep finished on Beauty GPU 0 on 2026-11-13 at code
revision `0eb13a2`: five folds times seeds 42, 43, and 44, for 15 runs total.
Each archived prediction has shape 1,387 by 8,248. A final audit revalidated
every completion record, artifact checksum, frozen test fold, gene universe,
and numeric prediction bundle; all 15 passed. The persistent launcher exited
with status 0 after the last run.

## 5. Baselines that need no training

```bash
uv run --frozen python -m models.baselines.perturbation_baselines \
  --dataset replogle-weissman-2022-k562-gwps
```

Predicts the ranked `no-change`, `fold-mean`, and bilinear-PCA `linear` baselines in one process
(reads `raw.h5ad` once, loops every fold). Each prediction is validated against the frozen fold
and gene universe, then written to the same checksum-addressed `runs/<dataset>/<model>/fold-<n>/`
layout as the substantive models. Because these methods perform no model selection, fold-mean,
linear, and the oracle use the complete train-plus-validation development pool, matching the
frozen development-pool policy. Re-running skips only checksum-valid completed bundles.

The privileged `interpolated-technical-duplicate` oracle is deliberately excluded from the
ranked defaults. Materialize it separately when the positive control is needed:

```bash
uv run --frozen python -m models.baselines.perturbation_baselines \
  --dataset replogle-weissman-2022-k562-gwps \
  --models interpolated-technical-duplicate
```

All baselines are deterministic and therefore have one archival artifact per fold at seed 42;
downstream comparisons may broadcast their fixed scores across seed strata but must not treat
those values as independent replicates.

```bash
cd models && sbatch --export=ALL,MODELS_DIR="$(pwd)" slurm/baselines.sbatch
```

One non-array CPU job (`models/slurm/baselines.sbatch`); it materializes only the three ranked
baselines and uses the shared uv lockfile.

## 6. Context-transfer model views and STATE

Context-transfer experiments use a combined, bounded model view produced by
`models.data.prepare_context`. Its `obs["model_perturbation"]` is the condition
STATE sees, while `obs["evaluation_condition"]` includes the context and is the
identifier stored in prediction bundles.

Materialize the frozen 50-cell genetic view from canonical raw-count objects:

```bash
uv run python -m models.data.prepare_context \
  --config configs/experiments/replogle-nadig-four-context-50-cell.json \
  --input replogle-weissman-2022-k562-essential=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/derived/replogle-weissman-2022-k562-essential/adata.h5ad \
  --input replogle-weissman-2022-rpe1=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/derived/replogle-weissman-2022-rpe1/adata.h5ad \
  --input nadig-oconner-2024-hepg2=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/derived/nadig-oconner-2024-hepg2/adata.h5ad \
  --input nadig-oconner-2024-jurkat=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/derived/nadig-oconner-2024-jurkat/adata.h5ad \
  --input-registry replogle-weissman-2022-k562-essential=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/registry/replogle-weissman-2022-k562-essential.toml \
  --input-registry replogle-weissman-2022-rpe1=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/registry/replogle-weissman-2022-rpe1.toml \
  --input-registry nadig-oconner-2024-hepg2=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/registry/nadig-oconner-2024-hepg2.toml \
  --input-registry nadig-oconner-2024-jurkat=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/registry/nadig-oconner-2024-jurkat.toml \
  --output models/data/replogle-nadig-four-context-50-cell/raw.h5ad \
  --counts-output models/data/replogle-nadig-four-context-50-cell/counts.h5ad
```

Materialize sci-Plex3. This command applies the declared global union of 8,192
HVGs and the top 100 DE genes per drug-dose comparison; it does not alter the
canonical raw-count object:

```bash
uv run python -m models.data.prepare_context \
  --config configs/experiments/sciplex3-nine-drug-context-transfer.json \
  --input srivatsan-trapnell-2020-sciplex3=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/derived/srivatsan-trapnell-2020-sciplex3/adata.h5ad \
  --input-registry srivatsan-trapnell-2020-sciplex3=/home/pschaefer/sds-hd/sd22b002/projects/PerturbationData/registry/srivatsan-trapnell-2020-sciplex3.toml \
  --output models/data/sciplex3-nine-drug-context-transfer/raw.h5ad \
  --counts-output models/data/sciplex3-nine-drug-context-transfer/counts.h5ad
```

Each command writes aligned normalized/log1p `raw.h5ad` and raw-count
`counts.h5ad` model views, plus `raw.manifest.json`, exact
`raw.cell-counts.tsv`, and—where applicable—`raw.gene-panel.tsv` plus the
per-condition ranked `raw.gene-panel-de.tsv`. Both H5ADs contain the same cells
and genes in the same order; only their expression representation differs.
Validated SHA-256 values are reused from PerturbationData's registry while input
paths, sizes, and modification metadata are checked before and after the atomic
output writes. Without `--input-registry`, files are hashed directly.

STATE uses cell-load's few-shot split rather than zero-shot splitting. Every
treated condition in the held-out context is assigned to the test split, while
its control cells remain available to the model. The experiment manifest also
freezes a 10% condition subset used for cross-context validation in the
training contexts.

```bash
cd models
pixi run -e state python state/train_predict.py \
  --dataset replogle-nadig-four-context-50-cell \
  --split-manifest ../configs/experiments/replogle-nadig-four-context-50-cell.json \
  --fold 0 \
  --perturbation-key model_perturbation
```

The same command shape applies to `sciplex3-nine-drug-context-transfer` and to
the jointly conditioned context adapters: `cpa`, `biolord`, `scdisinfact`, and
`trvae`. trVAE is included only for the chemical track because its joint genetic
scaling smoke exceeded the prespecified compute threshold. STATE, CPA, bioLORD,
and trVAE consume `raw.h5ad`;
scDisInFact consumes `counts.h5ad`. Each adapter records its native prediction
space and the deterministic transform into the shared log1p evaluation space.
STATE retains its native per-cell predictions. No-change and fold-mean context
baselines emit populations matched to each observed treated-cell count so they participate in both profile
and distributional sensitivity analyses. The context linear baseline is a
single joint ridge model over condition identities and control-state PCs and
uses only held-out-context control cells.

All current context runners use the shared generated layout:

```text
runs/<experiment>/<model>/<fold-id>/seed-<seed>/
  resolved-run.json
  checkpoints/
  logs/
  predictions/predictions.h5ad
  predictions/predictions.manifest.json
  completion.json
```

`completion.json` is written last and checksums every required artifact. A
matching rerun is skipped only when its resolved configuration and all artifact
checksums still validate; an interrupted run remains resumable.

If scDisInFact has already written its checkpoint, losses, and predictions but
was interrupted during manifest or completion-record creation, rerun the same
command with `--postprocess-existing`. This validates and finalizes those saved
artifacts without repeating training.
