"""Train the public STRING+GO TxPert variant on one frozen perturbation fold.

The public TxPert release contains the model, datasets, optimization logic, and inference
entry point, but omits its ``Trainer.fit`` entry point. This adapter reconstructs that small
piece from the released public checkpoint: 50 epochs, AdamW at 1e-3, five warm-up epochs,
cosine decay, and validation Pearson-delta selection with patience 40. Test responses are
never evaluated during training (the upstream validation hook evaluates both validation and
test on every epoch); only the frozen validation perturbations drive selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from workflows.model_runs import add_point_run_arguments, begin_population_model_run

IMPLEMENTATION_REVISION = "valence-labs/TxPert@08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
ENVIRONMENT = "models/pixi.toml#txpert-public"

MODEL_ARGS = {
    "model_type": "txpert",
    "no_basal_model": False,
    "no_pert_model": False,
    "hidden_dim": 512,
    "latent_dim": 64,
    "dropout": 0.2,
    "use_batch_norm": True,
    "use_layer_norm": False,
    "omit_cntr": False,
    "mse_weight": 1.0,
    "slow_benchmark": False,
    "cntr_model": {
        "model_type": "mlp",
        "rank": 16,
        "decode_latent": False,
        "use_batch_norm": True,
        "use_layer_norm": False,
    },
    "pert_model": {
        "model_type": "exphormer",
        "layer_type": "exphormer_w_mpnn",
        "num_layers": 4,
        "hidden_dim": 128,
        "skip": "skip_cat",
        "dropout": 0.1,
        "num_heads": 2,
        "concat": True,
        "add_self_loops": True,
        "expander_degree": 3,
        "add_reverse_edges": True,
        "use_edge_weight": False,
        "use_struc_feat": False,
        "pos_enc": "none",
        "union_edge_type": "multihot",
        "edge_feat_map_type": "linear",
    },
}

GRAPH_ARGS = {
    "graph_cfg": {
        "graph0": {
            "graph_type": "string",
            "reduce2perts": True,
            "norm_weights": True,
            "mode": "top_20",
        },
        "graph1": {
            "graph_type": "go",
            "reduce2perts": True,
            "norm_weights": False,
            "mode": "top_20",
        },
    }
}


def to_txpert_log1p_4000(matrix):
    """Map the shared 10k-log1p model view to TxPert's native 4k-log1p view."""
    if sp.issparse(matrix):
        result = matrix.astype(np.float32, copy=True)
        result.data = np.log1p(np.expm1(result.data) * 0.4).astype(np.float32)
        return result
    values = np.asarray(matrix, dtype=np.float32)
    return np.log1p(np.expm1(values) * 0.4).astype(np.float32)


def txpert_annotations(data: ad.AnnData) -> None:
    """Add exactly the metadata fields used by the official TxPert dataset class."""
    labels = data.obs["perturbation"].astype(str)
    controls = data.obs["is_control"].astype(bool)
    conditions = labels.where(~controls, "ctrl") + np.where(controls, "", "+ctrl")
    data.obs["condition"] = conditions.to_numpy()
    data.obs["condition_name"] = ("K562_" + conditions + "_1+1").to_numpy()
    data.obs["control"] = controls.astype(np.int8).to_numpy()
    data.obs["cell_line"] = "K562"
    data.obs["batch"] = data.obs["batch"].astype(str).to_numpy()


def _split_data(data: ad.AnnData, fold: dict) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData]:
    labels = data.obs["perturbation"].astype(str)
    controls = data.obs["is_control"].astype(bool)
    control = data[controls].copy()
    train = data[(~controls & labels.isin(fold["train"])).to_numpy()].copy()
    validation = data[(~controls & labels.isin(fold["val"])).to_numpy()].copy()
    test = data[(~controls & labels.isin(fold["test"])).to_numpy()].copy()
    return train, validation, test, control


def _condition_ids(data: ad.AnnData, pert2id: dict[str, int]) -> pd.Series:
    return pd.Series([[pert2id[label]] for label in data.obs["perturbation"].astype(str)], index=data.obs_names)


def _make_dataset(data, controls, pert2id, *, device, include_controls):
    from gspp.constants import ObsmKey
    from gspp.data.datamodule import XcelltypePerturbDataset

    control_names = controls.obs["condition_name"] if include_controls else None
    dataset = XcelltypePerturbDataset(
        data,
        controls,
        _condition_ids(data, pert2id),
        data.obs["condition_name"],
        device=device,
        train_cond_names=control_names,
        match_cntr=True,
        avg_cntr=True,
        obsm_key=ObsmKey.RAW,
    )
    if include_controls:
        # The pinned upstream dataset extends the training set with literal ``["ctrl"]``
        # labels, while TxPert indexes the graph embedding tensor with every label. The
        # released public graph reserves the final numeric node for control, so restore that
        # intended ID here. Without this correction, upstream fails on its first control cell
        # by attempting to index a tensor with the characters in the string ``"ctrl"``.
        dataset.pert_conditions = pd.Series(
            list(dataset.pert_conditions.iloc[: data.n_obs])
            + [[pert2id["ctrl"]]] * controls.n_obs
        )
    return dataset


def _build_datamodule(data: ad.AnnData, fold: dict, pert2id: dict[str, int], batch_size: int, device: str):
    import lightning as light
    from gspp.data.data_utils import collate_fn
    from torch.utils.data import DataLoader

    train, validation, test, controls = _split_data(data, fold)

    class FrozenTxPertDataModule(light.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.adata = data
            self.control_data = controls
            self.id2pert = {value: key for key, value in pert2id.items()}
            self.subgroup = {"test_subgroup": {}}
            self.train_data = _make_dataset(train, controls, pert2id, device=device, include_controls=True)
            self.val_data = _make_dataset(validation, controls, pert2id, device=device, include_controls=False)
            self.test_data = _make_dataset(test, controls, pert2id, device=device, include_controls=False)

        def train_dataloader(self):
            return DataLoader(
                self.train_data,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True,
            )

        def val_dataloader(self):
            return DataLoader(self.val_data, batch_size=batch_size, collate_fn=collate_fn)

        def test_dataloader(self):
            return DataLoader(self.test_data, batch_size=batch_size, collate_fn=collate_fn)

        def predict_dataloader(self):
            return self.test_dataloader()

    return FrozenTxPertDataModule()


def _predict(model, loader, id2pert, genes: pd.Index, device: str) -> ad.AnnData:
    import torch

    outputs = []
    labels = []
    batches = []
    model.eval()
    model.to(device)
    for batch in loader:
        with torch.inference_mode():
            prediction = model.sample_inference(batch.control, batch.pert_idxs, batch.p, batch.cell_types)
        outputs.append(prediction.cpu().numpy().astype(np.float32))
        labels.extend(id2pert[int(indices[0])] for indices in batch.pert_idxs)
        batches.extend(map(str, batch.experimental_batches))
    values = np.concatenate(outputs)
    obs = pd.DataFrame(
        {"perturbation": labels, "batch": batches},
        index=[f"txpert-{index}" for index in range(len(labels))],
    )
    return ad.AnnData(X=values, obs=obs, var=pd.DataFrame(index=genes.copy()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="replogle22k562-essential")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--early-stop-patience", type=int, default=40)
    add_point_run_arguments(parser)
    args = parser.parse_args()

    import lightning as light
    import torch
    from gspp import constants
    from gspp.data.graphmodule import GSPGraph
    from gspp.predictor import PertPredictor
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    class ValidationOnlyPredictor(PertPredictor):
        def on_validation_epoch_end(self):
            self.general_validation_epoch(constants.VAL, self.trainer.datamodule.val_dataloader())

    # Preserve the released five-epoch warm-up for real 50-epoch runs while keeping deliberately
    # truncated smoke tests valid: SequentialLR cannot hand off to a zero-length cosine schedule.
    warmup_epochs = min(5, max(args.max_epochs - 1, 0))
    configuration = {
        "model": copy.deepcopy(MODEL_ARGS),
        "graphs": copy.deepcopy(GRAPH_ARGS),
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "early_stop_patience": args.early_stop_patience,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler": {
            "warmup_epochs": warmup_epochs,
            "type": "cosine",
            "total_epochs": args.max_epochs,
        },
        "native_input": "normalize_total_4000_log1p",
        "training_entry_point": "reconstructed from released checkpoint metadata",
    }
    run = begin_population_model_run(
        args,
        repository_root=REPOSITORY_ROOT,
        model_id="txpert-public",
        environment=ENVIRONMENT,
        configuration=configuration,
    )
    if run.skipped:
        print(f"verified completed run; skipping {run.handle.directory}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    light.seed_everything(args.seed, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = ad.read_h5ad(run.raw_path)
    data.X = to_txpert_log1p_4000(data.X)
    txpert_annotations(data)
    split = json.loads(run.split_manifest_path.read_text())
    fold = split["folds"][args.fold]
    targets = sorted(map(str, split["perturbations"]))
    pert2id = {gene: index for index, gene in enumerate(targets)}
    pert2id["ctrl"] = len(pert2id)
    datamodule = _build_datamodule(data, fold, pert2id, args.batch_size, device)
    graph = GSPGraph(pert2id=pert2id, gene2id={gene: index for index, gene in enumerate(data.var_names)}, **copy.deepcopy(GRAPH_ARGS))

    init_args = {
        "input_dim": data.n_vars,
        "output_dim": data.n_vars,
        "adata_output_dim": data.n_vars,
        "graph": graph,
        "device": device,
        "lr": 0.001,
        "min_lr": 0.0,
        "weight_decay": 0.0,
        "lr_scheduler_args": {
            "warmup_epochs": warmup_epochs,
            "type": "cosine",
            "total_epochs": args.max_epochs,
        },
        "match_cntr_for_eval": True,
    }

    def predictor_args() -> dict:
        # PertPredictor destructively pops the nested model configuration during
        # construction, so every fresh model (including checkpoint reloads) needs a copy.
        return {**init_args, "model_args": copy.deepcopy(MODEL_ARGS)}

    run.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=run.checkpoint_dir,
        filename="best-{epoch:02d}-{val_pearson_delta:.4f}",
        monitor="val_pearson_delta",
        mode="max",
        save_top_k=1,
        save_last=True,
        every_n_epochs=min(10, args.max_epochs),
    )
    trainer = light.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_epochs=args.max_epochs,
        deterministic=True,
        logger=False,
        callbacks=[
            checkpoint,
            EarlyStopping(
                monitor="val_pearson_delta",
                mode="max",
                patience=args.early_stop_patience,
            ),
        ],
    )
    model = ValidationOnlyPredictor(**predictor_args())
    trainer.fit(model, datamodule=datamodule)
    selected_path = Path(checkpoint.best_model_path or checkpoint.last_model_path)
    selected = ValidationOnlyPredictor.load_from_checkpoint(selected_path, **predictor_args())
    prediction = _predict(
        selected,
        datamodule.predict_dataloader(),
        datamodule.id2pert,
        data.var_names,
        device,
    )
    run.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(run.prediction_path)
    run.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "selected_checkpoint": selected_path.name,
                "best_val_pearson_delta": float(checkpoint.best_model_score),
                "epochs_completed": int(trainer.current_epoch),
                "training_cells": len(datamodule.train_data),
                "validation_cells": len(datamodule.val_data),
                "test_queries": len(datamodule.test_data),
            }
        ]
    ).to_csv(run.metrics_path, index=False)
    run.complete(
        implementation_revision=IMPLEMENTATION_REVISION,
        checkpoints=[selected_path],
        evaluation_transform="renormalize-log1p-to-10000",
    )
    print(f"completed {run.handle.directory}")


if __name__ == "__main__":
    main()
