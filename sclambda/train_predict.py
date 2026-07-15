"""Smoke test: train scLAMBDA on a small subsample and write a scPertEval-compatible
predictions.h5ad.

Uses our own 5-fold cross-validation split (models/data/prepare_split.py) rather than
scLAMBDA's own ``data_split()`` (a single random draw, same shortcoming as GEARS' own
``simulation`` split — see prepare_split.py's docstring). scLAMBDA's own condition-string
convention (``"ctrl"`` / ``"<GENE>+ctrl"``) is identical to GEARS', so this reuses
models/data/*_folds/fold_<i>.pkl directly — no new fold-file format needed, unlike PRESAGE's
own convention (see prepare_split.py's ``to_presage_custom_split``).

scLAMBDA's own ``Model`` never computes gene embeddings itself — it just takes a
``gene_embeddings: dict[str, np.array]``. The Genentech/PRESAGE-adjacent Miller reimplementation
(shiftbioscience/Perturbation-Models-Outperform-Baselines) generates these live via the OpenAI
embeddings API (requiring an OPENAI_API_KEY); the official demo instead loads a precomputed
pickle that traces back to the GenePT project (github.com/yiqunchen/GenePT) — OpenAI
text-embedding-3-large embeddings of NCBI gene summaries, precomputed once for ~134k gene
symbols and deposited on Zenodo (DOI 10.5281/zenodo.10833191), fetched here via
`pixi run -e sclambda fetch-embeddings`. No API key needed.

``use_tg_coord`` (an auxiliary per-target-gene decoder head, intended to sharpen combo-gene
predictions) is left at its default (``False``): neither of scLAMBDA's own demo notebooks turn
it on, so this mirrors what's actually been validated by the authors rather than opting into an
untested code path.

Everything else — ``Model.__init__``, ``.train()``, ``.predict()`` — is the official code,
unmodified.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ad.settings.allow_write_nullable_strings = True

from sclambda.model import Model  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
# Populated by `pixi run -e sclambda fetch-embeddings` (models/pixi.toml) — not downloaded
# here, so this script has no direct download dependency of its own.
GENEPT_PICKLE = (
    HERE / "cache" / "GenePT_emebdding_v2" / "GenePT_gene_protein_embedding_model_3_text.pickle."
)
OUT_DIR = HERE / "smoke_data"
SAVED_MODELS_DIR = HERE / "saved_models"
# scLAMBDA's own `Model.train()` only ever updates `self.model_best` inside its
# `(epoch+1) % 10 == 0` validation checkpoints (or, if there are no validation perturbations
# at all, on the very last epoch) — below 10 epochs it's never assigned and
# `self.Net = self.model_best` at the end of training raises AttributeError. 10 is the minimum
# that guarantees at least one checkpoint.
MIN_EPOCHS = 10
# training_epochs=200 is scLAMBDA's own Model.__init__ default (sclambda/model.py) and
# matches cellsimbench's own sclambda.yaml exactly — not an independently tuned value,
# the benchmark just uses the library's stock default. Smoke scale needs a much lower
# value passed explicitly (still >= MIN_EPOCHS above).
DEFAULT_EPOCHS = 200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="smoke_k562",
        help="models/data/<dataset>_raw.h5ad + <dataset>_folds/ to train/predict on (default: smoke_k562)",
    )
    parser.add_argument("--fold", type=int, default=0, help="which fold_<i>.pkl to train/predict on")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"training epochs (default: {DEFAULT_EPOCHS}, full-scale; must be >= {MIN_EPOCHS}, see above)",
    )
    args = parser.parse_args()
    if args.epochs < MIN_EPOCHS:
        raise SystemExit(f"--epochs must be >= {MIN_EPOCHS} (scLAMBDA never checkpoints below that, see module docstring)")

    fold_path = DATA_DIR / f"{args.dataset}_folds" / f"fold_{args.fold}.pkl"
    out_path = OUT_DIR / f"{args.dataset}_predictions_fold{args.fold}.h5ad"

    if not GENEPT_PICKLE.exists():
        raise FileNotFoundError(f"{GENEPT_PICKLE} missing — run `pixi run -e sclambda fetch-embeddings` first")

    adata = ad.read_h5ad(DATA_DIR / f"{args.dataset}_raw.h5ad")

    with open(fold_path, "rb") as f:
        fold = pickle.load(f)
    split_map = {c: "train" for c in fold["train"]}
    split_map.update({c: "val" for c in fold["val"]})
    split_map.update({c: "test" for c in fold["test"]})
    adata.obs["split"] = adata.obs["condition"].map(split_map)

    with open(GENEPT_PICKLE, "rb") as f:
        genept = pickle.load(f)
    pert_genes = sorted({c.split("+")[0] for c in adata.obs["condition"].unique() if c != "ctrl"})
    gene_embeddings = {g: np.asarray(genept[g], dtype=np.float32) for g in pert_genes}

    model = Model(
        adata,
        gene_embeddings,
        split_name="split",
        training_epochs=args.epochs,
        # Namespaced by dataset too — otherwise a smoke run and a full run at the same fold
        # index would silently overwrite each other's checkpoint.
        model_path=str(SAVED_MODELS_DIR / args.dataset / f"fold_{args.fold}"),
        # multi_gene stays at its default (True) despite this being a single-gene dataset:
        # scLAMBDA's README's multi_gene=False path assumes bare-gene-symbol conditions
        # ("gene_a", no suffix). Ours are "<GENE>+ctrl" (the GEARS convention, shared across
        # gears/scgpt/presage/sclambda so all four read the same fold files) — with
        # multi_gene=False, predict()/generate()/the validation loop all do
        # `self.gene_emb[i]`, i.e. look up the *whole* condition string as a dict key, which
        # KeyErrors against a gene_embeddings dict keyed by bare symbols. multi_gene=True
        # instead splits on "+" and sums (gene_emb['GENE'] + gene_emb['ctrl'] == gene_emb['GENE']
        # since 'ctrl' embeds to zero) — the correct behavior for our convention.
    )
    model.train()

    test_conditions = fold["test"]
    print(f"test perturbations ({len(test_conditions)}): {test_conditions}")
    preds = model.predict(test_conditions, return_type="mean")

    rows = [preds[c] for c in test_conditions]
    perturbations = [c.split("+")[0] for c in test_conditions]
    pred_adata = ad.AnnData(X=np.vstack(rows).astype(np.float32), obs={"perturbation": perturbations})
    pred_adata.var_names = pd.Index(np.asarray(adata.var_names))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_adata.write_h5ad(out_path)
    print(f"wrote {pred_adata.shape} predictions to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
