"""Smoke test: fine-tune the pretrained scGPT `whole-human` checkpoint for perturbation
prediction and write a scPertEval-compatible predictions.h5ad.

Adapted from bowang-lab/scGPT's tutorials/Tutorial_Perturbation.ipynb, which is stale in two
ways this script fixes:
  - It imports `torchtext.vocab.Vocab` / `torchtext._torchtext.VocabPybind` directly, which
    hits the same compiled-extension ABI mismatch (modern torch vs. old torchtext) that this
    repo's own `scgpt.tokenizer.vocab_compat` was written to work around. We only ever load a
    pretrained vocab, so we use `GeneVocab.from_file` (goes through the compat layer) and skip
    the torchtext imports entirely.
  - `eval_perturb` casts to the removed `np.float` alias; replaced with `np.float64`.

Otherwise the logic (data via GEARS' PertData, partial-prefix loading of the pretrained
encoder, the `TransformerGenerator` fine-tuning loop, per-epoch validation with
patience-based early stopping and best-model selection by validation Pearson correlation,
and predicting via pooled control-cell graphs) is unchanged from the tutorial — including
its own `shiftbioscience/Perturbation-Models-Outperform-Baselines` and
`const-ae/linear_perturbation_prediction-Paper` reimplementations, which both keep the same
validation/early-stopping/best-model structure.

Reuses the exact same subsampled raw data and 5-fold split as models/gears/train_predict.py
(models/data/prepare_split.py's fold_<i>.pkl, via GEARS' own split="custom") so the two
models' predictions are comparable in a shared scoring notebook.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch import nn

ad.settings.allow_write_nullable_strings = True

# See models/gears/train_predict.py for why this is needed: cell-gears indexes a scipy
# sparse matrix with a raw pandas boolean Series, and pandas dropped `Series.nonzero()`.
if not hasattr(pd.Series, "nonzero"):
    pd.Series.nonzero = lambda self: self.to_numpy().nonzero()

from gears import PertData  # noqa: E402
from gears.utils import create_cell_graph_dataset_for_prediction  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

import scgpt as scg  # noqa: E402
from scgpt.model import TransformerGenerator  # noqa: E402
from scgpt.loss import masked_mse_loss  # noqa: E402
from scgpt.tokenizer.gene_tokenizer import GeneVocab  # noqa: E402
from scgpt.utils import (  # noqa: E402
    compute_perturbation_metrics,
    load_pretrained,
    map_raw_id_to_vocab_id,
    set_seed,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
PERT_DATA_DIR = HERE / "pert_data"
# Populated by `pixi run -e scgpt fetch-checkpoint` (models/pixi.toml) — not downloaded
# here, so this script has no gdown dependency of its own.
CHECKPOINT_DIR = HERE / "checkpoints" / "scGPT_human"
OUT_DIR = HERE / "smoke_data"
SAVED_MODELS_DIR = HERE / "saved_models"

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0
# nn.Embedding(3, d_model, padding_idx=pert_pad_id) permanently zeroes (never trains) the
# embedding row at this index. GEARS' per-gene pert flag (cell-gears==0.0.2's
# create_cell_graph, the version scGPT's data pipeline uses) only ever takes values 0
# ("not perturbed") or 1 ("perturbed") — never 2. The tutorial's default of 0 therefore
# freezes the embedding for the overwhelmingly common "not perturbed" case; 2 is an index
# that never occurs in real data, so both real states (0 and 1) stay trainable.
pert_pad_id = 2
include_zero_gene = "all"
max_seq_len = 1536
load_param_prefixs = ["encoder", "value_encoder", "transformer_encoder"]

lr = 1e-4
schedule_interval = 1
amp = True
# Matches the tutorial's own early_stop = 10.
EARLY_STOP = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="smoke_k562",
        help="models/data/<dataset>_raw.h5ad + <dataset>_folds/ to train/predict on (default: smoke_k562)",
    )
    parser.add_argument("--fold", type=int, default=0, help="which fold_<i>.pkl to train/predict on")
    # The tutorial's own value (epochs = 15), with its patience=10 early stopping doing
    # the real work — cellsimbench's own scgpt.yaml uses a fixed max_epochs=10 with no
    # early stopping at all, a different structure, not something to match here (see
    # module docstring). Smoke scale needs a much lower value passed explicitly.
    parser.add_argument("--epochs", type=int, default=15, help="fine-tuning epochs (default: 15, full-scale)")
    # Already matches both smoke and full scale (cell-level batching) and the tutorial's
    # own value — no smoke/full distinction needed, just exposed for configurability.
    parser.add_argument("--batch-size", type=int, default=64, help="training batch size (default: 64)")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="eval batch size (default: 64)")
    args = parser.parse_args()
    raw = DATA_DIR / f"{args.dataset}_raw.h5ad"
    fold_path = DATA_DIR / f"{args.dataset}_folds" / f"fold_{args.fold}.pkl"
    out_path = OUT_DIR / f"{args.dataset}_predictions_fold{args.fold}.h5ad"

    if not (CHECKPOINT_DIR / "best_model.pt").exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_DIR} has no checkpoint — run `pixi run -e scgpt fetch-checkpoint` first"
        )

    pert_data = PertData(str(PERT_DATA_DIR))
    pert_data.new_data_process(dataset_name=args.dataset, adata=ad.read_h5ad(raw))
    # scgpt pins cell-gears<0.0.3, which predates PertData.prepare_split's split="custom"
    # option (added later in cell-gears 0.1.2, the version models/gears's own environment
    # uses) — replicate that branch's own body directly: it only ever sets these two
    # attributes before get_dataloader() reads them, nothing else.
    pert_data.split = "custom"
    with open(fold_path, "rb") as f:
        pert_data.set2conditions = pickle.load(f)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.eval_batch_size)

    model_config_file = CHECKPOINT_DIR / "args.json"
    model_file = CHECKPOINT_DIR / "best_model.pt"
    vocab_file = CHECKPOINT_DIR / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[pad_token])

    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids_in_vocab = np.array([1 if g in vocab else -1 for g in genes])
    scg.logger.info(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(genes)} genes in vocab")
    gene_ids = np.array([vocab[g] if g in vocab else vocab[pad_token] for g in genes], dtype=int)
    n_genes = len(genes)

    with open(model_config_file) as f:
        model_configs = json.load(f)
    embsize, nhead, d_hid, nlayers, n_layers_cls = (
        model_configs["embsize"],
        model_configs["nheads"],
        model_configs["d_hid"],
        model_configs["nlayers"],
        model_configs["n_layers_cls"],
    )

    model = TransformerGenerator(
        len(vocab),
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=0,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        # Real FA2 (models/pixi.toml pins pytorch-gpu=2.8.* to match a prebuilt
        # flash-attn wheel). The checkpoint was trained with FA1's "direct" Wqkv
        # layout; load_pretrained detects the FA1-vs-FA2 naming mismatch itself
        # (get_flash_attn_parameter_rename_rules) and renames to FA2's wrapped
        # "self_attn._impl.Wqkv" layout.
        use_fast_transformer=True,
        fast_transformer_backend="flash",
    )
    pretrained_dict = torch.load(model_file, map_location=device, weights_only=True)
    model = load_pretrained(model, pretrained_dict, prefix=load_param_prefixs, verbose=True)
    model.to(device)

    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def train_one_epoch(epoch: int) -> None:
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch_data in pert_data.dataloader["train_loader"]:
            bs = len(batch_data.y)
            batch_data.to(device)
            x = batch_data.x
            ori_gene_values = x[:, 0].view(bs, n_genes)
            pert_flags = x[:, 1].long().view(bs, n_genes)
            target_values = batch_data.y

            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:max_seq_len]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_values[:, input_gene_ids]
            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids).repeat(bs, 1)
            src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                )
                loss = criterion(output_dict["mlm_output"], target_values, torch.ones_like(input_values, dtype=torch.bool))

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        scg.logger.info(f"epoch {epoch} mean train loss {total_loss / max(n_batches, 1):.4f}")

    def eval_perturb(loader: DataLoader) -> dict[str, np.ndarray]:
        """Mirrors the tutorial's own eval_perturb: run in inference mode over a loader,
        collecting predictions/truth (all genes and DE-subset) for compute_perturbation_metrics."""
        model.eval()
        pert_cat, pred, truth, pred_de, truth_de = [], [], [], [], []
        with torch.no_grad():
            for batch_data in loader:
                batch_data.to(device)
                pert_cat.extend(batch_data.pert)
                p = model.pred_perturb(batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp)
                t = batch_data.y
                pred.extend(p.cpu())
                truth.extend(t.cpu())
                for i, de_idx in enumerate(batch_data.de_idx):
                    pred_de.append(p[i, de_idx])
                    truth_de.append(t[i, de_idx])
        pred, truth = torch.stack(pred), torch.stack(truth)
        pred_de, truth_de = torch.stack(pred_de), torch.stack(truth_de)
        # The tutorial casts to the removed np.float alias; np.float64 is the equivalent
        # replacement (see module docstring).
        return {
            "pert_cat": np.array(pert_cat),
            "pred": pred.numpy().astype(np.float64),
            "truth": truth.numpy().astype(np.float64),
            "pred_de": pred_de.cpu().numpy().astype(np.float64),
            "truth_de": truth_de.cpu().numpy().astype(np.float64),
        }

    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    best_val_corr = 0.0
    best_model = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_one_epoch(epoch)
        val_res = eval_perturb(pert_data.dataloader["val_loader"])
        val_score = compute_perturbation_metrics(val_res, ctrl_adata)["pearson"]
        scg.logger.info(f"epoch {epoch} done in {time.time() - start:.1f}s, val pearson {val_score:.4f}")
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP:
                scg.logger.info(f"early stop at epoch {epoch}")
                break
        scheduler.step()

    # best_model can only stay None if validation Pearson never exceeds 0 in any epoch —
    # the tutorial itself has no guard against this either.
    model = best_model

    # Namespaced by dataset too — otherwise a smoke run and a full run at the same fold
    # index would silently overwrite each other's checkpoint.
    fold_dir = SAVED_MODELS_DIR / args.dataset / f"fold_{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "model.pt")

    def predict(pert_list: list[list[str]], pool_size: int = 300) -> dict[str, np.ndarray]:
        # 300 matches GEARS' own hardcoded num_samples default (gears.utils.
        # create_cell_graph_dataset_for_prediction) so both models average
        # over the same number of resampled control cells.
        ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        gene_list = pert_data.gene_names.values.tolist()
        model.eval()
        results = {}
        with torch.no_grad():
            for pert in pert_list:
                cell_graphs = create_cell_graph_dataset_for_prediction(
                    pert, ctrl_adata, gene_list, device, num_samples=pool_size
                )
                loader = DataLoader(cell_graphs, batch_size=args.eval_batch_size, shuffle=False)
                preds = []
                for batch_data in loader:
                    preds.append(
                        model.pred_perturb(batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp)
                    )
                preds = torch.cat(preds, dim=0)
                results["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)
        return results

    test_perts = sorted(pert_data.set2conditions["test"])
    single_gene_perts = [c.split("+")[0] for c in test_perts if c != "ctrl"]
    scg.logger.info(f"test perturbations ({len(single_gene_perts)}): {single_gene_perts}")
    preds = predict([[g] for g in single_gene_perts])

    rows = [preds[gene] for gene in single_gene_perts]
    pred_adata = ad.AnnData(
        X=np.vstack(rows).astype(np.float32), obs={"perturbation": single_gene_perts}
    )
    pred_adata.var_names = pd.Index(np.asarray(pert_data.gene_names.values))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_adata.write_h5ad(out_path)
    print(f"wrote {pred_adata.shape} predictions to {out_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.exit(main())
