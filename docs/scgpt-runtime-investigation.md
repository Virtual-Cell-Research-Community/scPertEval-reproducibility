# Why the GWPS scGPT run is slow

## Bottom line

The active run is not comparable to Hugo's published scPertEval timing or to the Miller et al.
benchmark at equal scale.

- The active run fine-tunes on the full frozen GWPS model view: 1,333,016 cells,
  6,935 perturbations, and 8,248 genes. Training is cell-level, so every epoch traverses roughly
  the training-cell fraction of that object. On Beast's NVIDIA RTX A6000, the first three
  measured epochs took 9,903, 9,897, and 9,881 seconds (about 2 h 45 min each).
- Hugo's published scPertEval runs used the much smaller **K562 essential** screen and ran the
  five folds concurrently on Slurm. A complete 15-epoch fold averaged 2 h 1 min on H200 and
  5 h 10 min on L40S; this was one fixed-seed run per fold, not three seeds.
- Miller et al. did include the GWPS dataset, but their preparation explicitly retained at
  most 2,500 perturbations, capped cells per perturbation, used 10 epochs and a 1,200-token
  sequence, and launched the five folds as separate GPU jobs. It was not the full 6,935-
  perturbation, 15-epoch, three-seed workload planned here.

The current plan therefore multiplies a much larger per-fold job by 15 fold/seed runs. The
observed runtime is unsurprising; blindly launching all 15 would be a multi-week sequential
workload.

## What the active adapter computes

The current adapter follows scGPT's official perturbation tutorial structure: a pretrained
whole-human `TransformerGenerator` is fine-tuned on control-cell-in/perturbed-cell-out pairs.
For every training cell it samples up to 1,536 genes, runs a 12-layer, 512-dimensional
transformer, and computes masked MSE. It uses batch size 64, validates every epoch, allows 15
epochs, and stops only after 10 consecutive non-improving validation-Pearson epochs. See the
[current adapter](https://github.com/Virtual-Cell-Research-Community/scPertEval-reproducibility/blob/75f9b2e9fdab1f9a8b4c48a7a2ee9e9303b86714/models/scgpt/train_predict.py).

Two scaling facts matter:

1. Epoch time scales approximately with the number of training cells because the loader emits
   one graph per cell.
2. Transformer attention cost grows approximately quadratically with sequence length. The
   8,248-gene panel is randomly reduced to 1,536 genes per batch, so the full panel is not fed
   through attention at once, but 1,536 is still a long sequence.

The active job is already reading its prepared cache from Beast-local NVMe and keeps the GPU at
roughly 100% utilization. The present bottleneck is therefore substantive model compute over a
large number of cells, not the earlier slow SDS/SSHFS read.

## What Hugo ran

The official scPertEval model documentation says all six models were benchmarked on
`Replogle22 K562 essential`, not GWPS. The materialized canonical essential view contains
267,974 cells, about one fifth as many as the active GWPS view. Its preparation downsamples each non-control condition
to the rounded mean condition size, caps controls at 8,192 cells, and uses 8,192 HVGs union
perturbation targets. The pre-filter cohort was 1,789 targets. See
[Models introduction and training detail](https://scperteval.readthedocs.io/en/stable/benchmark/models.html#data-preparation).

The same page reports these measured scGPT resources for a **whole 15-epoch fold**:

| GPU | folds | mean elapsed per fold | mean MaxRSS | progress |
|---|---:|---:|---:|---:|
| H200 | 0, 2, 3, 4 | 2 h 1 min | 55.1 GB | 15/15 epochs |
| L40S | 1 | 5 h 10 min | 57.2 GB | 15/15 epochs |

The public runner fixes the seed to 42, and the Slurm launcher submits one five-task array over
folds 0–4. It does not run three seed replicates. See the pinned
[scGPT runner](https://github.com/Virtual-Cell-Research-Community/scPertEval-models/blob/430cf437feba9c414d697de0de2b414ab1d8495e/scgpt/train_predict.py),
[Slurm job](https://github.com/Virtual-Cell-Research-Community/scPertEval-models/blob/430cf437feba9c414d697de0de2b414ab1d8495e/slurm/train.sbatch), and
[launcher documentation](https://github.com/Virtual-Cell-Research-Community/scPertEval-models/blob/430cf437feba9c414d697de0de2b414ab1d8495e/README.md#4-train--predict-the-full-sweep-all-models--all-5-folds).

Thus Hugo's wall-clock strategy was approximately “five smaller folds in parallel,” not “15
GWPS runs sequentially.”

## What Miller et al. ran

The peer-reviewed paper is [Miller et al., *Nature Biotechnology* (2026),
doi:10.1038/s41587-026-03307-w](https://doi.org/10.1038/s41587-026-03307-w). Its released code is
the primary source for the exact training workload.

### Dataset reduction

For K562 GWPS, the released preparation sets `MAX_PERTS = 2500`. It keeps a fixed priority list
and randomly samples the remaining perturbations to reach 2,500, rather than using the whole
screen. It then:

- removes conditions with fewer than 12 cells;
- caps each non-control condition at the rounded mean condition size;
- caps controls at 8,192 cells;
- retains 8,192 HVGs union perturbation-target genes;
- constructs five deterministic 70% train / 10% validation / 20% test folds.

See the pinned
[GWPS preparation script](https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines/blob/15361aec8392fb45a75c8c9da88ca5e6510fca97/data/replogle22k562gwps/get_data.py).

### scGPT budget

Miller's released scGPT configuration uses:

- 10 epochs, with no early stopping in the wrapper;
- batch size 32;
- maximum sequence length 1,200;
- the pretrained scGPT model;
- one model per fold.

See the pinned
[scGPT configuration](https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines/blob/15361aec8392fb45a75c8c9da88ca5e6510fca97/cellsimbench/configs/model/scgpt.yaml) and
[training loop](https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines/blob/15361aec8392fb45a75c8c9da88ca5e6510fca97/docker/scgpt/scgpt_wrapper.py).

Their benchmark launcher creates five jobs for a single-gene dataset and schedules them across
available GPUs. The repository recommends at least five 24-GB GPUs (an 8×A10 instance), 384 GB
CPU RAM, and 2 TB storage for full reproduction. This is a **recommended configuration**, not
evidence of the exact hardware or runtime used for the published run. The paper/repository does
not provide a defensible per-fold scGPT wall time. See the pinned
[reproduction README](https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines/blob/15361aec8392fb45a75c8c9da88ca5e6510fca97/README.md).

The Miller workload is therefore materially smaller than ours in four independent ways:

| choice | current study | Miller release |
|---|---:|---:|
| GWPS perturbations | 6,935 | at most 2,500 |
| max epochs | 15, patience 10 | 10, fixed |
| max sequence length | 1,536 | 1,200 |
| replicated runs | 5 folds × 3 seeds | 5 folds once |

The sequence-length difference alone makes the current attention matrices about
`(1536 / 1200)^2 = 1.64×` larger per example. Combined with roughly 2.8× as many perturbations,
a 1.5× larger epoch budget, and 3× as many seed replicates, the total planned computation is
not in the same regime.

## Implications for this benchmark

It would be invalid to give scGPT a smaller perturbation cohort than the other models while
still comparing their rankings. The scientifically clean options are:

1. **Keep the full common cohort and parallelize.** Run fold/seed jobs on separate high-end
   GPUs, ideally H200/A100-class Helix nodes with enough host RAM. This preserves the current
   design but remains expensive.
2. **Adopt a smaller common cohort for every model.** A Miller-like 2,500-perturbation cohort
   would make scGPT much cheaper, but already-completed models would need to be rerun on that
   same cohort.
3. **Use the Miller scGPT training budget for every scGPT run.** Ten epochs and sequence length
   1,200 are defensible published choices and would reduce compute, but they do not remove the
   dominant cell-count cost. The model still must use the same perturbation cohort as its
   competitors.
4. **Run a measured scaling pilot before the sweep.** Benchmark a short, restartable epoch on
   H200/A100 and test sequence length 1,200 plus the largest safe batch. Freeze the choice
   before running all folds/seeds; do not tune it using test performance.

Given that the analysis is intended to study evaluation-induced ranking changes rather than
establish a definitive new model leaderboard, option 3 plus option 4 is the least disruptive
next step. The current A6000 fold should not be multiplied across 15 jobs without that decision.

## Remaining uncertainty

- Miller et al. publish recommended reproduction hardware but not the exact GPU model or
  per-fold scGPT runtime used for the paper.
- A precise speedup on Helix cannot be inferred from GPU peak FLOPS alone because data loading,
  host RAM, and validation also contribute. It should be measured with a one-epoch pilot.
- Changing epoch count or sequence length can change model quality. Those choices should be
  fixed from a published configuration or a training-only criterion, never selected from test
  scores.
